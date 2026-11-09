export const QK_NORM_MODEL_TYPES = new Set(['qwen3', 'qwen3_moe', 'qwen3_next', 'olmo2', 'gemma3', 'gemma3_text'])
export const POST_NORM_MODEL_TYPES = new Set(['olmo2'])
export const SANDWICH_NORM_MODEL_TYPES = new Set(['gemma2', 'gemma3', 'gemma3_text'])

export type AttentionKind = 'MHA' | 'GQA' | 'MQA' | 'MLA'

export interface MLAConfig {
  q_lora_rank: number | null
  kv_lora_rank: number
  qk_nope_head_dim: number
  qk_rope_head_dim: number
  v_head_dim: number
}

export interface Attention {
  kind: AttentionKind
  num_heads: number
  num_kv_heads: number
  head_dim: number
  sliding_window: number | null
  sliding_pattern: string | null
  qk_norm: boolean
  attention_bias: boolean
  mla: MLAConfig | null
}

export interface MoE {
  num_experts: number
  experts_per_tok: number
  shared_experts: number
  moe_intermediate_size: number
  dense_layers: number
}

export interface ArchSpec {
  name: string
  model_type: string
  architecture: string
  hidden_size: number
  num_layers: number
  vocab_size: number | null
  context_length: number
  intermediate_size: number
  activation: string
  attention: Attention
  moe: MoE | null
  norm_type: string
  norm_mode: 'pre' | 'post' | 'sandwich'
  positional: 'rope' | 'learned' | 'alibi' | 'nope'
  rope_theta: number | null
  hybrid_note: string | null
  parallel_block: boolean
  tied_embeddings: boolean
  total_params: number
  active_params: number
  embed_params: number
  kv_cache_per_token_bytes: number
  is_moe?: boolean
}

function getVal<T>(cfg: Record<string, any>, ...keys: string[]): T | null {
  for (const k of keys) {
    if (cfg[k] !== undefined && cfg[k] !== null) return cfg[k]
  }
  return null
}

function hybridNote(cfg: Record<string, any>, layers: number): string | null {
  const lt = cfg.layer_types
  if (lt && Array.isArray(lt)) {
    const counts: Record<string, number> = {}
    for (const v of lt) counts[v] = (counts[v] || 0) + 1
    const exotic: Record<string, number> = {}
    for (const [k, n] of Object.entries(counts)) {
      if (k !== 'full_attention' && k !== 'sliding_attention' && k !== 'attention') {
        exotic[k] = n
      }
    }
    if (Object.keys(exotic).length > 0) {
      const exoticSum = Object.values(exotic).reduce((a, b) => a + b, 0)
      const attn = layers - exoticSum
      const parts = Object.entries(exotic).map(([k, n]) => `${n} ${k.replace(/_/g, ' ')}`).join(' + ')
      return `${parts} : ${attn} attention layers`
    }
  }
  const la = cfg.linear_attn_config
  if (la && typeof la === 'object' && Array.isArray(la.kda_layers)) {
    return `${la.kda_layers.length} linear-attention (KDA) : ${(la.full_attn_layers || []).length} full attention layers`
  }
  if (Array.isArray(cfg.full_attn_idxs)) {
    const full = cfg.full_attn_idxs.length
    return `${layers - full} conv-mixer : ${full} full attention layers`
  }
  if (Array.isArray(cfg.attn_type_list)) {
    const al = cfg.attn_type_list
    const linear = al.filter((x: any) => x === 0).length
    const softmax = al.filter((x: any) => x === 1).length
    return `${linear} lightning (linear) : ${softmax} softmax attention layers`
  }
  if (Object.keys(cfg).some(k => k.startsWith('linear_'))) {
    return 'hybrid: linear-attention mixer on most layers'
  }
  if (Object.keys(cfg).some(k => k.startsWith('mamba_') || k.startsWith('ssm_'))) {
    return 'hybrid: Mamba/SSM mixer layers'
  }
  return null
}

function attnParams(s: ArchSpec): number {
  const a = s.attention
  const h = s.hidden_size
  if (a.mla) {
    const m = a.mla
    const qkDim = m.qk_nope_head_dim + m.qk_rope_head_dim
    const q = m.q_lora_rank
      ? h * m.q_lora_rank + m.q_lora_rank + m.q_lora_rank * a.num_heads * qkDim
      : h * a.num_heads * qkDim
    const kv = h * (m.kv_lora_rank + m.qk_rope_head_dim) + m.kv_lora_rank + m.kv_lora_rank * a.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
    const o = a.num_heads * m.v_head_dim * h
    return q + kv + o
  }
  const qDim = a.num_heads * a.head_dim
  const kvDim = a.num_kv_heads * a.head_dim
  let p = h * qDim + 2 * h * kvDim + qDim * h
  if (a.attention_bias) p += qDim + 2 * kvDim + h
  if (a.qk_norm) p += 2 * a.head_dim
  return p
}

function ffnParams(s: ArchSpec, intermediate: number): number {
  const h = s.hidden_size
  const gated = s.activation === 'silu' || s.activation === 'swiglu' || s.activation.includes('gelu_pytorch_tanh')
  return gated ? 3 * h * intermediate : 2 * h * intermediate + intermediate + h
}

function estimateParams(s: ArchSpec): void {
  const h = s.hidden_size
  const vocab = s.vocab_size || 0
  const embed = vocab * h + (s.positional === 'learned' ? s.context_length * h : 0)
  const head = s.tied_embeddings ? 0 : vocab * h
  const normsPerLayer = 2 * h
  const finalNorm = h

  const attn = attnParams(s)
  let total = 0
  let active = 0
  for (let layer = 0; layer < s.num_layers; layer++) {
    total += attn + normsPerLayer
    active += attn + normsPerLayer
    if (s.moe && layer >= s.moe.dense_layers) {
      const m = s.moe
      const router = h * m.num_experts
      const expert = ffnParams(s, m.moe_intermediate_size)
      const shared = m.shared_experts ? ffnParams(s, m.moe_intermediate_size * m.shared_experts) : 0
      total += router + m.num_experts * expert + shared
      active += router + m.experts_per_tok * expert + shared
    } else {
      const dense = ffnParams(s, s.intermediate_size)
      total += dense
      active += dense
    }
  }

  s.embed_params = embed
  s.total_params = embed + head + total + finalNorm
  s.active_params = embed + head + active + finalNorm

  const a = s.attention
  const perLayer = a.mla ? a.mla.kv_lora_rank + a.mla.qk_rope_head_dim : 2 * a.num_kv_heads * a.head_dim
  s.kv_cache_per_token_bytes = 2 * s.num_layers * perLayer
}

export function parseConfig(rawCfg: Record<string, any>, name = 'model'): ArchSpec {
  let cfg = rawCfg
  const outerArch = (cfg.architectures && cfg.architectures[0]) || '?'
  if (cfg.text_config) {
    cfg = { ...cfg.text_config, architectures: cfg.architectures }
  }

  const modelType = cfg.model_type || 'unknown'
  const hidden = getVal<number>(cfg, 'hidden_size', 'n_embd', 'd_model', 'dim', 'hidden_dim')
  const layers = getVal<number>(cfg, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers')
  let heads = getVal<number>(cfg, 'num_attention_heads', 'n_head', 'num_heads', 'n_heads')

  if (hidden === null || layers === null) {
    throw new Error(`config for '${name}' has no recognizable hidden-size/layer fields (model_type='${modelType}')`)
  }
  if (heads === null) {
    heads = Math.max(Math.floor(hidden / 128), 1)
  }

  let kvHeads = getVal<number>(cfg, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads') ?? heads
  if (cfg.multi_query) kvHeads = 1
  const headDim = getVal<number>(cfg, 'head_dim') ?? Math.floor(hidden / heads)
  const vocab = getVal<number>(cfg, 'vocab_size', 'padded_vocab_size')
  const ctx = getVal<number>(cfg, 'max_position_embeddings', 'n_positions', 'n_ctx', 'max_seq_len', 'max_sequence_length', 'seq_length', 'model_max_length') ?? 0
  const inter = getVal<number>(cfg, 'intermediate_size', 'n_inner', 'ffn_hidden_size', 'ffn_dim', 'block_ff_dim') ?? 4 * hidden
  const act = getVal<string>(cfg, 'hidden_act', 'hidden_activation', 'activation_function') ?? (cfg.block_use_swiglu ? 'silu' : 'gelu')

  let mla: MLAConfig | null = null
  const mlaKeys = ['kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim']
  let kind: AttentionKind = 'MHA'
  if (mlaKeys.every(k => cfg[k] !== undefined && cfg[k] !== null)) {
    kind = 'MLA'
    mla = {
      q_lora_rank: cfg.q_lora_rank ?? null,
      kv_lora_rank: cfg.kv_lora_rank,
      qk_nope_head_dim: cfg.qk_nope_head_dim,
      qk_rope_head_dim: cfg.qk_rope_head_dim,
      v_head_dim: cfg.v_head_dim,
    }
  } else if (kvHeads === 1) {
    kind = 'MQA'
  } else if (kvHeads < heads) {
    kind = 'GQA'
  }

  const sliding = (cfg.use_sliding_window ?? true) ? getVal<number>(cfg, 'sliding_window') : null
  let pattern: string | null = null
  if (sliding) {
    if (cfg.sliding_window_pattern) {
      pattern = `${cfg.sliding_window_pattern - 1} local : 1 global`
    } else if (Array.isArray(cfg.layer_types)) {
      const lt = cfg.layer_types
      const local = lt.filter((x: any) => x === 'sliding_attention').length
      const global = lt.filter((x: any) => x === 'full_attention').length
      pattern = `${local} local / ${global} global`
    } else {
      pattern = 'all layers'
    }
  }

  const attention: Attention = {
    kind,
    num_heads: heads,
    num_kv_heads: kvHeads,
    head_dim: headDim,
    sliding_window: sliding,
    sliding_pattern: pattern,
    qk_norm: QK_NORM_MODEL_TYPES.has(modelType),
    attention_bias: Boolean(cfg.attention_bias ?? (modelType === 'gpt2')),
    mla,
  }

  let moe: MoE | null = null
  const nExperts = getVal<number>(cfg, 'num_experts', 'n_routed_experts', 'num_local_experts', 'moe_num_experts')
  if (nExperts && nExperts > 1) {
    let shared = getVal<number>(cfg, 'n_shared_experts', 'num_shared_experts') ?? 0
    if (!shared && (cfg.shared_expert_intermediate_size || cfg.shared_intermediate_size)) {
      shared = 1
    }
    moe = {
      num_experts: nExperts,
      experts_per_tok: getVal<number>(cfg, 'num_experts_per_tok', 'experts_per_token', 'num_experts_per_token', 'moe_topk', 'moe_top_k', 'num_selected_experts') ?? 1,
      shared_experts: shared,
      moe_intermediate_size: getVal<number>(cfg, 'moe_intermediate_size') ?? inter,
      dense_layers: cfg.first_k_dense_replace || 0,
    }
  }

  const theta = getVal<number>(cfg, 'rope_theta', 'rotary_emb_base', 'rope_embedding_base')
  let positional: 'rope' | 'learned' | 'alibi' | 'nope' = 'nope'
  if (cfg.alibi) {
    positional = 'alibi'
  } else if (theta !== null) {
    positional = 'rope'
  } else if ('alibi' in cfg || cfg.rotary) {
    positional = 'rope'
  } else if (modelType === 'gpt2') {
    positional = 'learned'
  }

  const spec: ArchSpec = {
    name,
    model_type: modelType,
    architecture: outerArch,
    hidden_size: hidden,
    num_layers: layers,
    vocab_size: vocab,
    context_length: ctx,
    intermediate_size: inter,
    activation: act,
    attention,
    moe,
    norm_type: 'layer_norm_epsilon' in cfg ? 'layernorm' : 'rmsnorm',
    norm_mode: POST_NORM_MODEL_TYPES.has(modelType) ? 'post' : SANDWICH_NORM_MODEL_TYPES.has(modelType) ? 'sandwich' : 'pre',
    positional,
    rope_theta: theta,
    hybrid_note: hybridNote(cfg, layers),
    parallel_block: Boolean(cfg.parallel_attn),
    tied_embeddings: Boolean(cfg.tie_word_embeddings ?? (modelType === 'gpt2' || modelType.startsWith('gemma'))),
    total_params: 0,
    active_params: 0,
    embed_params: 0,
    kv_cache_per_token_bytes: 0,
    is_moe: Boolean(moe),
  }

  try {
    estimateParams(spec)
  } catch {
    spec.total_params = 0
    spec.active_params = 0
  }

  return spec
}
