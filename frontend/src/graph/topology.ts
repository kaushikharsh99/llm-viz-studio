import type {
  ArchGraph,
  GraphNode,
  GraphEdge,
  GraphLevel,
  NodeKind,
  ConfidenceLevel,
  TensorInfo,
  TensorRef,
  MetadataCoverage,
  ModelHierarchyItem,
} from './types'
import { annotate } from './mapping'
import type { ArchSpec } from '../engine/spec'

const MOE_TYPES = new Set([
  'deepseek_v3',
  'deepseek_v2',
  'qwen3_moe',
  'qwen3_next',
  'mixtral',
  'glm4_moe',
  'dbrx',
  'arctic',
])

function tref(t: TensorInfo): TensorRef {
  return {
    name: t.name,
    shape: t.shape,
    dtype: t.dtype,
    numel: t.numel,
    bytes: t.bytes,
    shard: t.shard,
  }
}

function findTensors(tensors: TensorInfo[], layer: number | null, module: string): TensorInfo[] {
  return tensors.filter(
    t => t.layer === layer && (t.module === module || t.category === module) && t.category !== 'quant_scale'
  )
}

function sumParams(ts: TensorInfo[]): number | null {
  if (ts.length === 0) return null
  let total = 0
  let hasVerified = false
  for (const t of ts) {
    if (t.numel !== null && t.numel !== undefined) {
      total += t.numel
      hasVerified = true
    }
  }
  return hasVerified ? total : null
}

function getShapeStr(shape?: number[] | null): string | null {
  if (!shape || shape.length === 0) return null
  return `[${shape.join(' × ')}]`
}

/**
 * Build the full model hierarchy tree for the Left Panel
 */
export function buildHierarchy(
  spec: ArchSpec,
  tensors: TensorInfo[],
  isMoe: boolean,
  firstMoeLayer: number
): ModelHierarchyItem[] {
  const root: ModelHierarchyItem[] = []

  // 1. Embeddings
  const embTensors = findTensors(tensors, null, 'embedding')
  root.push({
    id: 'tree_embed',
    label: 'Token Embeddings',
    kind: 'embedding',
    level: 0,
    nodeId: 'embed',
    params: sumParams(embTensors),
  })

  // 2. Layers
  const layerItems: ModelHierarchyItem[] = []
  for (let l = 0; l < spec.num_layers; l++) {
    const isLayerMoe = isMoe && l >= firstMoeLayer
    const lAttn = tensors.filter(t => t.layer === l && t.category === 'attention')
    const lMlp = tensors.filter(t => t.layer === l && (t.category === 'mlp' || t.category === 'moe'))
    const lNorm = tensors.filter(t => t.layer === l && t.category === 'norm')
    const layerTotal = sumParams([...lAttn, ...lMlp, ...lNorm])

    layerItems.push({
      id: `tree_layer_${l}`,
      label: `Layer ${l} (${isLayerMoe ? 'MoE' : 'Dense'})`,
      kind: isLayerMoe ? 'moe_block' : 'layer_block',
      level: 1,
      layer: l,
      nodeId: `layer_${l}`,
      params: layerTotal,
      children: [
        {
          id: `tree_layer_${l}_norm1`,
          label: 'Pre-Attention Norm',
          kind: 'norm',
          level: 2,
          layer: l,
          nodeId: 'norm_1',
          params: sumParams(findTensors(tensors, l, 'input_layernorm')),
        },
        {
          id: `tree_layer_${l}_attn`,
          label: `${spec.attention?.kind || 'Self'}-Attention`,
          kind: 'attention_block',
          level: 2,
          layer: l,
          nodeId: 'attn_block',
          params: sumParams(lAttn),
        },
        {
          id: `tree_layer_${l}_norm2`,
          label: 'Pre-FFN Norm',
          kind: 'norm',
          level: 2,
          layer: l,
          nodeId: 'norm_2',
          params: sumParams(findTensors(tensors, l, 'post_attention_layernorm')),
        },
        {
          id: `tree_layer_${l}_ffn`,
          label: isLayerMoe ? `MoE Router & Experts` : `Feed-Forward (SwiGLU)`,
          kind: isLayerMoe ? 'moe_block' : 'ffn_block',
          level: 2,
          layer: l,
          nodeId: 'ffn_block',
          params: sumParams(lMlp),
        },
      ],
    })
  }

  root.push({
    id: 'tree_layers',
    label: `Transformer Layers (${spec.num_layers})`,
    kind: 'layer_group',
    level: 0,
    nodeId: 'layers_all',
    children: layerItems,
  })

  // 3. Final Norm
  const fnTensors = findTensors(tensors, null, 'final_norm')
  root.push({
    id: 'tree_final_norm',
    label: 'Final Normalization',
    kind: 'norm',
    level: 0,
    nodeId: 'final_norm',
    params: sumParams(fnTensors),
  })

  // 4. LM Head
  const headTensors = findTensors(tensors, null, 'lm_head')
  root.push({
    id: 'tree_lm_head',
    label: 'LM Head',
    kind: 'head',
    level: 0,
    nodeId: 'lm_head',
    params: sumParams(headTensors),
  })

  return root
}

/**
 * Reconstruct accurate computational dataflow topology for any requested granularity Level (0, 1, 2, 3)
 */
export function buildGraph(
  modelId: string,
  spec: ArchSpec,
  rawTensors: TensorInfo[],
  available: boolean,
  note?: string | null,
  activeLayer?: number | null,
  level: GraphLevel = 0,
  coverage?: MetadataCoverage
): ArchGraph {
  const tensors = annotate(rawTensors)
  const mt = (spec.model_type || '').toLowerCase()
  const isMoe = Boolean(spec.moe) || MOE_TYPES.has(mt)

  // Identify first MoE layer (for DeepSeek-V3 dense layers 0..2 vs MoE layers 3..60)
  let firstMoeLayer = 0
  if (isMoe) {
    const moeTensorLayers = tensors
      .filter(t => t.layer !== null && (t.module.includes('moe') || t.category === 'moe'))
      .map(t => t.layer!)
    if (moeTensorLayers.length > 0) {
      firstMoeLayer = Math.min(...moeTensorLayers)
    } else if (spec.moe?.dense_layers) {
      firstMoeLayer = spec.moe.dense_layers
    }
  }

  // Determine available layers
  const layerSet = new Set<number>()
  for (const t of tensors) {
    if (t.layer !== null && t.layer !== undefined) layerSet.add(t.layer)
  }
  const layers = layerSet.size > 0
    ? Array.from(layerSet).sort((a, b) => a - b)
    : Array.from({ length: spec.num_layers }, (_, i) => i)

  // Determine active layer for Levels 1, 2, 3
  const currentLayer =
    activeLayer !== undefined && activeLayer !== null && activeLayer >= 0
      ? activeLayer
      : (isMoe && firstMoeLayer < layers.length ? firstMoeLayer : (layers[0] ?? 0))

  const isCurrentLayerMoe = isMoe && currentLayer >= firstMoeLayer

  const verifiedParams = tensors
    .filter(t => t.category !== 'quant_scale')
    .reduce((acc, t) => acc + (t.numel || 0), 0)

  const g: ArchGraph = {
    model_id: modelId,
    model_type: spec.model_type,
    architecture: spec.architecture,
    level,
    active_layer: currentLayer,
    confidence: available ? 'verified' : 'config',
    metadata_source: available ? 'safetensors' : 'config_only',
    metadata_note: note ?? null,
    coverage,
    total_params: spec.total_params,
    verified_params: verifiedParams > 0 ? verifiedParams : null,
    tied_note:
      available && !tensors.some(t => t.name.includes('lm_head'))
        ? 'lm_head absent from checkpoint (tied embeddings) — verified parameter total excludes it'
        : null,
    nodes: [],
    edges: [],
    layers,
    unknowns: Array.from(new Set(tensors.filter(t => t.module === 'unknown').map(t => t.name))).slice(0, 50),
  }

  function addNode(
    id: string,
    label: string,
    kind: NodeKind,
    opts: {
      parent?: string | null
      layer?: number | null
      module?: string
      submodule?: string
      ts?: TensorInfo[]
      conf?: ConfidenceLevel
      ev?: string
      in_dim?: number | string | null
      out_dim?: number | string | null
      shape_str?: string | null
      collapsed_count?: number | null
      details?: Record<string, any>
    } = {}
  ): GraphNode {
    const ts = opts.ts || []
    const nodeParams = sumParams(ts)
    const nodeDtype = ts.length > 0 && ts[0].dtype ? ts[0].dtype : null

    const node: GraphNode = {
      id,
      label,
      kind,
      level,
      parent: opts.parent ?? null,
      layer: opts.layer ?? null,
      module: opts.module,
      submodule: opts.submodule,
      params: nodeParams,
      dtype: nodeDtype,
      in_dim: opts.in_dim ?? null,
      out_dim: opts.out_dim ?? null,
      shape_str: opts.shape_str ?? (ts.length > 0 ? getShapeStr(ts[0].shape) : null),
      confidence: opts.conf ?? (ts.length > 0 && ts.some(t => t.is_verified_shape) ? 'verified' : (available ? 'inferred' : 'config')),
      evidence: opts.ev ?? (ts.length > 0 ? ts[0].name : (available ? 'Inferred from architecture config' : 'Config only')),
      tensors: ts.slice(0, 30).map(tref),
      collapsed_count: opts.collapsed_count ?? null,
      details: opts.details,
    }
    g.nodes.push(node)
    return node
  }

  function addEdge(
    src: string,
    dst: string,
    kind: 'data' | 'residual' | 'weight' | 'control' = 'data',
    label?: string,
    conf: ConfidenceLevel = 'verified'
  ): GraphEdge {
    const id = `e_${src}_${dst}_${kind}`
    const edge: GraphEdge = { id, src, dst, kind, label, confidence: conf }
    g.edges.push(edge)
    return edge
  }

  const h = spec.hidden_size
  const v = spec.vocab_size
  const headDim = spec.attention?.head_dim || (spec.attention?.num_heads ? Math.floor(h / spec.attention.num_heads) : 128)
  const qHeads = spec.attention?.num_heads || 32
  const kvHeads = spec.attention?.num_kv_heads || qHeads
  const intermediate = spec.intermediate_size

  // =========================================================================
  // LEVEL 0: WHOLE MODEL MACRO DATAFLOW
  // =========================================================================
  if (level === 0) {
    // 1. Input tokens
    addNode('in_tokens', 'Input Tokens', 'io', {
      shape_str: '[batch_size, seq_len]',
      ev: 'Input token IDs [int64]',
      conf: 'verified',
    })

    // 2. Token Embedding
    const embTs = findTensors(tensors, null, 'embedding')
    addNode('embed', 'Token Embeddings', 'embedding', {
      ts: embTs,
      in_dim: v,
      out_dim: h,
      shape_str: `[${v} × ${h}]`,
      ev: embTs.length > 0 ? embTs[0].name : 'model.embed_tokens.weight',
      conf: embTs.length > 0 ? 'verified' : 'config',
    })
    addEdge('in_tokens', 'embed')

    let lastNodeId = 'embed'

    // 3. Repeated Layer Blocks (grouped by homogeneity)
    if (isMoe && firstMoeLayer > 0) {
      // Heterogeneous architecture (e.g. DeepSeek-V3: dense 0..2, MoE 3..60)
      addNode('layers_dense', `Dense Transformer Layers (0..${firstMoeLayer - 1})`, 'layer_group', {
        collapsed_count: firstMoeLayer,
        ev: `${firstMoeLayer} dense decoder blocks`,
        conf: available ? 'verified' : 'config',
      })
      addEdge(lastNodeId, 'layers_dense')
      lastNodeId = 'layers_dense'

      const moeCount = spec.num_layers - firstMoeLayer
      addNode('layers_moe', `MoE Transformer Layers (${firstMoeLayer}..${spec.num_layers - 1})`, 'layer_group', {
        collapsed_count: moeCount,
        ev: `${moeCount} Mixture-of-Experts decoder blocks (${spec.moe?.num_experts || 64} experts, top-${spec.moe?.experts_per_tok || 8})`,
        conf: available ? 'verified' : 'config',
      })
      addEdge(lastNodeId, 'layers_moe')
      lastNodeId = 'layers_moe'
    } else {
      // Homogeneous architecture
      addNode('layers_all', `${spec.num_layers}× Transformer Layers (0..${spec.num_layers - 1})`, 'layer_group', {
        collapsed_count: spec.num_layers,
        ev: `${spec.num_layers} sequential identical decoder blocks`,
        conf: available ? 'verified' : 'config',
      })
      addEdge(lastNodeId, 'layers_all')
      lastNodeId = 'layers_all'
    }

    // 4. Final Normalization
    const finTs = findTensors(tensors, null, 'final_norm')
    addNode('final_norm', `${spec.norm_type || 'RMSNorm'} (Final)`, 'norm', {
      ts: finTs,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
      ev: finTs.length > 0 ? finTs[0].name : 'model.norm.weight',
      conf: finTs.length > 0 ? 'verified' : 'config',
    })
    addEdge(lastNodeId, 'final_norm')

    // 5. LM Head
    const headTs = findTensors(tensors, null, 'lm_head')
    addNode('lm_head', 'LM Head (Logits Projection)', 'head', {
      ts: headTs,
      in_dim: h,
      out_dim: v,
      shape_str: `[${h} × ${v}]`,
      ev: headTs.length > 0 ? headTs[0].name : (available ? 'Tied with embed_tokens' : 'lm_head.weight'),
      conf: headTs.length > 0 ? 'verified' : 'inferred',
    })
    addEdge('final_norm', 'lm_head')

    // 6. Output Logits
    addNode('out_logits', 'Output Logits', 'io', {
      shape_str: `[batch_size, seq_len, ${v}]`,
      ev: 'Vocabulary probability distribution before sampling',
      conf: 'verified',
    })
    addEdge('lm_head', 'out_logits')
  }

  // =========================================================================
  // LEVEL 1: LAYER OVERVIEW (MACRO COMPUTATIONAL FLOW)
  // =========================================================================
  else if (level === 1) {
    const L = currentLayer

    // 1. Layer Input
    addNode(`l${L}_in`, `Layer ${L} Input`, 'io', {
      layer: L,
      shape_str: `[batch, seq, ${h}]`,
      ev: L === 0 ? 'From Token Embeddings' : `From Layer ${L - 1} Output`,
    })

    // 2. Pre-Attention Norm
    const norm1Ts = findTensors(tensors, L, 'input_layernorm')
    addNode(`l${L}_norm1`, `${spec.norm_type || 'RMSNorm'} 1 (Pre-Attention)`, 'norm', {
      layer: L,
      ts: norm1Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
      ev: norm1Ts.length > 0 ? norm1Ts[0].name : 'model.layers.N.input_layernorm.weight',
    })
    addEdge(`l${L}_in`, `l${L}_norm1`)

    // 3. Attention Block
    const attnTs = tensors.filter(t => t.layer === L && t.category === 'attention')
    addNode(`l${L}_attn`, `${spec.attention?.kind || 'GQA'} Attention Block`, 'attention_block', {
      layer: L,
      ts: attnTs,
      in_dim: h,
      out_dim: h,
      ev: `Self-Attention: ${qHeads} Q heads, ${kvHeads} KV heads, head_dim ${headDim}`,
      collapsed_count: 6,
    })
    addEdge(`l${L}_norm1`, `l${L}_attn`)

    // 4. Residual Add 1 (⊕)
    addNode(`l${L}_res1`, '⊕ Residual Add 1', 'residual', {
      layer: L,
      shape_str: `[batch, seq, ${h}]`,
      ev: 'Elementwise sum: Layer Input + Attention Output',
    })
    addEdge(`l${L}_attn`, `l${L}_res1`, 'data')
    addEdge(`l${L}_in`, `l${L}_res1`, 'residual', 'skip connection')

    // 5. Pre-FFN Norm
    const norm2Ts = findTensors(tensors, L, 'post_attention_layernorm')
    addNode(`l${L}_norm2`, `${spec.norm_type || 'RMSNorm'} 2 (Pre-FFN)`, 'norm', {
      layer: L,
      ts: norm2Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
      ev: norm2Ts.length > 0 ? norm2Ts[0].name : 'model.layers.N.post_attention_layernorm.weight',
    })
    addEdge(`l${L}_res1`, `l${L}_norm2`)

    // 6. FFN or MoE Block
    if (isCurrentLayerMoe) {
      const moeTs = tensors.filter(t => t.layer === L && (t.category === 'moe' || t.module.includes('moe')))
      const numExp = spec.moe?.num_experts || 64
      const topK = spec.moe?.experts_per_tok || 8
      addNode(`l${L}_ffn`, `MoE Block (${numExp} Experts, Top-${topK})`, 'moe_block', {
        layer: L,
        ts: moeTs,
        in_dim: h,
        out_dim: h,
        ev: `Mixture-of-Experts: router + ${numExp} routed experts + shared experts`,
        collapsed_count: numExp,
      })
      addEdge(`l${L}_norm2`, `l${L}_ffn`)
    } else {
      const mlpTs = tensors.filter(t => t.layer === L && t.category === 'mlp')
      addNode(`l${L}_ffn`, `SwiGLU Feed-Forward Block`, 'ffn_block', {
        layer: L,
        ts: mlpTs,
        in_dim: h,
        out_dim: h,
        shape_str: `[${h} → ${intermediate} → ${h}]`,
        ev: `Gated FFN: gate_proj ⊗ up_proj -> act(${spec.activation || 'silu'}) -> down_proj`,
        collapsed_count: 3,
      })
      addEdge(`l${L}_norm2`, `l${L}_ffn`)
    }

    // 7. Residual Add 2 (⊕)
    addNode(`l${L}_res2`, '⊕ Residual Add 2', 'residual', {
      layer: L,
      shape_str: `[batch, seq, ${h}]`,
      ev: 'Elementwise sum: Residual Add 1 + FFN Output',
    })
    addEdge(`l${L}_ffn`, `l${L}_res2`, 'data')
    addEdge(`l${L}_res1`, `l${L}_res2`, 'residual', 'skip connection')

    // 8. Layer Output
    addNode(`l${L}_out`, `Layer ${L} Output`, 'io', {
      layer: L,
      shape_str: `[batch, seq, ${h}]`,
      ev: L === spec.num_layers - 1 ? 'To Final Normalization' : `To Layer ${L + 1} Input`,
    })
    addEdge(`l${L}_res2`, `l${L}_out`)
  }

  // =========================================================================
  // LEVEL 2: MODULE INTERNALS (ATTENTION & FFN / MoE)
  // =========================================================================
  else if (level === 2) {
    const L = currentLayer

    // Input & Pre-Attention Norm
    addNode(`l${L}_in`, `Layer ${L} Input`, 'io', { layer: L, shape_str: `[batch, seq, ${h}]` })
    const norm1Ts = findTensors(tensors, L, 'input_layernorm')
    addNode(`l${L}_norm1`, `${spec.norm_type || 'RMSNorm'} 1`, 'norm', {
      layer: L,
      ts: norm1Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
    })
    addEdge(`l${L}_in`, `l${L}_norm1`)

    // Attention Internals
    const isMLA = spec.attention?.kind === 'MLA' || tensors.some(t => t.layer === L && t.module === 'attention_q_a')
    const qNormTs = findTensors(tensors, L, 'q_norm')
    const kNormTs = findTensors(tensors, L, 'k_norm')

    if (isMLA) {
      // DeepSeek Multi-Head Latent Attention
      const qaTs = findTensors(tensors, L, 'attention_q_a')
      const qbTs = findTensors(tensors, L, 'attention_q_b')
      const kvaTs = findTensors(tensors, L, 'attention_kv_a')
      const kvbTs = findTensors(tensors, L, 'attention_kv_b')
      const oTs = findTensors(tensors, L, 'o_proj')

      addNode(`l${L}_qa`, 'Q Latent ↓ (q_a_proj)', 'linear', { layer: L, ts: qaTs, in_dim: h, ev: 'Compress Q to latent dim' })
      addNode(`l${L}_qb`, 'Q Restore ↑ (q_b_proj)', 'linear', { layer: L, ts: qbTs, out_dim: qHeads * headDim, ev: 'Restore Q heads' })
      addNode(`l${L}_kva`, 'KV Latent ↓ (kv_a_proj)', 'linear', { layer: L, ts: kvaTs, in_dim: h, ev: 'Compress KV cache' })
      addNode(`l${L}_kvb`, 'KV Restore ↑ (kv_b_proj)', 'linear', { layer: L, ts: kvbTs, out_dim: kvHeads * headDim, ev: 'Restore KV heads' })

      addEdge(`l${L}_norm1`, `l${L}_qa`)
      addEdge(`l${L}_qa`, `l${L}_qb`)
      addEdge(`l${L}_norm1`, `l${L}_kva`)
      addEdge(`l${L}_kva`, `l${L}_kvb`)

      addNode(`l${L}_attn_scores`, 'MatMul (Q · Kᵀ / √d)', 'math_op', { layer: L, shape_str: `[${qHeads}, seq, seq]` })
      addEdge(`l${L}_qb`, `l${L}_attn_scores`)
      addEdge(`l${L}_kvb`, `l${L}_attn_scores`)

      addNode(`l${L}_softmax`, 'Softmax + Causal Mask', 'math_op', { layer: L, shape_str: `[${qHeads}, seq, seq]` })
      addEdge(`l${L}_attn_scores`, `l${L}_softmax`)

      addNode(`l${L}_attn_ctx`, 'MatMul (Scores · V)', 'math_op', { layer: L, shape_str: `[${qHeads}, seq, ${headDim}]` })
      addEdge(`l${L}_softmax`, `l${L}_attn_ctx`)
      addEdge(`l${L}_kvb`, `l${L}_attn_ctx`)

      addNode(`l${L}_o_proj`, 'O Projection (o_proj)', 'linear', { layer: L, ts: oTs, in_dim: qHeads * headDim, out_dim: h })
      addEdge(`l${L}_attn_ctx`, `l${L}_o_proj`)
    } else {
      // Standard GQA / MHA Attention
      const qTs = findTensors(tensors, L, 'q_proj')
      const kTs = findTensors(tensors, L, 'k_proj')
      const vTs = findTensors(tensors, L, 'v_proj')
      const oTs = findTensors(tensors, L, 'o_proj')

      addNode(`l${L}_q_proj`, 'Q Projection (q_proj)', 'linear', {
        layer: L,
        ts: qTs,
        in_dim: h,
        out_dim: qHeads * headDim,
        shape_str: `[${h} × ${qHeads * headDim}]`,
      })
      addNode(`l${L}_k_proj`, 'K Projection (k_proj)', 'linear', {
        layer: L,
        ts: kTs,
        in_dim: h,
        out_dim: kvHeads * headDim,
        shape_str: `[${h} × ${kvHeads * headDim}]`,
      })
      addNode(`l${L}_v_proj`, 'V Projection (v_proj)', 'linear', {
        layer: L,
        ts: vTs,
        in_dim: h,
        out_dim: kvHeads * headDim,
        shape_str: `[${h} × ${kvHeads * headDim}]`,
      })

      addEdge(`l${L}_norm1`, `l${L}_q_proj`)
      addEdge(`l${L}_norm1`, `l${L}_k_proj`)
      addEdge(`l${L}_norm1`, `l${L}_v_proj`)

      let qAfterNode = `l${L}_q_proj`
      let kAfterNode = `l${L}_k_proj`

      // Q/K Norm (e.g. Qwen, Gemma, Olmo)
      if (qNormTs.length > 0 || kNormTs.length > 0) {
        addNode(`l${L}_q_norm`, 'Q RMSNorm', 'norm', { layer: L, ts: qNormTs, shape_str: `[${headDim}]` })
        addNode(`l${L}_k_norm`, 'K RMSNorm', 'norm', { layer: L, ts: kNormTs, shape_str: `[${headDim}]` })
        addEdge(`l${L}_q_proj`, `l${L}_q_norm`)
        addEdge(`l${L}_k_proj`, `l${L}_k_norm`)
        qAfterNode = `l${L}_q_norm`
        kAfterNode = `l${L}_k_norm`
      }

      // RoPE Positional Encoding
      addNode(`l${L}_rope_q`, 'RoPE (Query)', 'attention_op', { layer: L, ev: `Rotary embedding (θ=${spec.rope_theta || 10000})` })
      addNode(`l${L}_rope_k`, 'RoPE (Key)', 'attention_op', { layer: L, ev: `Rotary embedding (θ=${spec.rope_theta || 10000})` })
      addEdge(qAfterNode, `l${L}_rope_q`)
      addEdge(kAfterNode, `l${L}_rope_k`)

      // Attention Matrix & Softmax
      addNode(`l${L}_attn_scores`, 'Attention Scores (Q · Kᵀ / √d)', 'math_op', {
        layer: L,
        shape_str: `[${qHeads}, seq, seq]`,
        ev: `Scaled dot-product (head_dim=${headDim})`,
      })
      addEdge(`l${L}_rope_q`, `l${L}_attn_scores`)
      addEdge(`l${L}_rope_k`, `l${L}_attn_scores`)

      addNode(`l${L}_softmax`, 'Softmax + Causal Mask', 'math_op', {
        layer: L,
        shape_str: `[${qHeads}, seq, seq]`,
        ev: 'Row-wise softmax probability distribution',
      })
      addEdge(`l${L}_attn_scores`, `l${L}_softmax`)

      addNode(`l${L}_attn_ctx`, 'Value Aggregation (Scores · V)', 'math_op', {
        layer: L,
        shape_str: `[batch, seq, ${qHeads * headDim}]`,
        ev: 'Weighted sum of value vectors',
      })
      addEdge(`l${L}_softmax`, `l${L}_attn_ctx`)
      addEdge(`l${L}_v_proj`, `l${L}_attn_ctx`)

      addNode(`l${L}_o_proj`, 'Output Projection (o_proj)', 'linear', {
        layer: L,
        ts: oTs,
        in_dim: qHeads * headDim,
        out_dim: h,
        shape_str: `[${qHeads * headDim} × ${h}]`,
      })
      addEdge(`l${L}_attn_ctx`, `l${L}_o_proj`)
    }

    // Residual Add 1 (⊕)
    addNode(`l${L}_res1`, '⊕ Residual Add 1', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
    addEdge(`l${L}_o_proj`, `l${L}_res1`, 'data')
    addEdge(`l${L}_in`, `l${L}_res1`, 'residual', 'skip connection')

    // Pre-FFN Norm
    const norm2Ts = findTensors(tensors, L, 'post_attention_layernorm')
    addNode(`l${L}_norm2`, `${spec.norm_type || 'RMSNorm'} 2`, 'norm', {
      layer: L,
      ts: norm2Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
    })
    addEdge(`l${L}_res1`, `l${L}_norm2`)

    // FFN / MoE Internals
    if (isCurrentLayerMoe) {
      const numExp = spec.moe?.num_experts || 64
      const topK = spec.moe?.experts_per_tok || 8
      const routerTs = findTensors(tensors, L, 'moe_router')
      const sharedGateTs = findTensors(tensors, L, 'moe_shared_gate')
      const sharedUpTs = findTensors(tensors, L, 'moe_shared_up')
      const sharedDownTs = findTensors(tensors, L, 'moe_shared_down')

      // Router
      addNode(`l${L}_router`, `MoE Gating Router (Top-${topK} of ${numExp})`, 'router', {
        layer: L,
        ts: routerTs,
        in_dim: h,
        out_dim: numExp,
        shape_str: `[${h} × ${numExp}]`,
        ev: `Gating router selects ${topK} active experts per token`,
      })
      addEdge(`l${L}_norm2`, `l${L}_router`)

      // Routed Experts Group
      const expertTs = tensors.filter(t => t.layer === L && t.category === 'moe' && t.expert !== null)
      addNode(`l${L}_routed_experts`, `${numExp} Routed Experts (${topK} Active)`, 'expert', {
        layer: L,
        ts: expertTs,
        in_dim: h,
        out_dim: h,
        collapsed_count: numExp,
        ev: `Each expert: Gate/Up (${spec.moe?.moe_intermediate_size || intermediate}) → SiLU → Down (${h})`,
      })
      addEdge(`l${L}_router`, `l${L}_routed_experts`, 'control', 'dispatch tokens')
      addEdge(`l${L}_norm2`, `l${L}_routed_experts`, 'data')

      // Shared Experts (if present)
      let hasShared = sharedGateTs.length > 0 || sharedUpTs.length > 0 || sharedDownTs.length > 0
      if (hasShared) {
        addNode(`l${L}_shared_experts`, 'Shared Experts (Always Active)', 'expert', {
          layer: L,
          ts: [...sharedGateTs, ...sharedUpTs, ...sharedDownTs],
          in_dim: h,
          out_dim: h,
          ev: 'Shared FFN pathways processing all tokens in parallel with routed experts',
        })
        addEdge(`l${L}_norm2`, `l${L}_shared_experts`, 'data')
      }

      // MoE Combine
      addNode(`l${L}_moe_combine`, 'MoE Weighted Combination', 'math_op', {
        layer: L,
        shape_str: `[batch, seq, ${h}]`,
        ev: 'Weighted sum of expert outputs by softmax router probabilities',
      })
      addEdge(`l${L}_routed_experts`, `l${L}_moe_combine`)
      if (hasShared) {
        addEdge(`l${L}_shared_experts`, `l${L}_moe_combine`)
      }

      // Residual Add 2 (⊕)
      addNode(`l${L}_res2`, '⊕ Residual Add 2', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
      addEdge(`l${L}_moe_combine`, `l${L}_res2`, 'data')
      addEdge(`l${L}_res1`, `l${L}_res2`, 'residual', 'skip connection')
    } else {
      // Dense SwiGLU FFN
      const gateTs = findTensors(tensors, L, 'gate_proj')
      const upTs = findTensors(tensors, L, 'up_proj')
      const downTs = findTensors(tensors, L, 'down_proj')

      addNode(`l${L}_gate_proj`, 'Gate Projection (gate_proj)', 'linear', {
        layer: L,
        ts: gateTs,
        in_dim: h,
        out_dim: intermediate,
        shape_str: `[${h} × ${intermediate}]`,
      })
      addNode(`l${L}_up_proj`, 'Up Projection (up_proj)', 'linear', {
        layer: L,
        ts: upTs,
        in_dim: h,
        out_dim: intermediate,
        shape_str: `[${h} × ${intermediate}]`,
      })
      addEdge(`l${L}_norm2`, `l${L}_gate_proj`)
      addEdge(`l${L}_norm2`, `l${L}_up_proj`)

      addNode(`l${L}_silu`, `${(spec.activation || 'silu').toUpperCase()} Activation`, 'math_op', {
        layer: L,
        shape_str: `[batch, seq, ${intermediate}]`,
        ev: 'Non-linear gating function',
      })
      addEdge(`l${L}_gate_proj`, `l${L}_silu`)

      addNode(`l${L}_hadamard`, 'Hadamard Product (Gate ⊗ Up)', 'math_op', {
        layer: L,
        shape_str: `[batch, seq, ${intermediate}]`,
        ev: 'Elementwise multiplication of activated gate and up projection',
      })
      addEdge(`l${L}_silu`, `l${L}_hadamard`)
      addEdge(`l${L}_up_proj`, `l${L}_hadamard`)

      addNode(`l${L}_down_proj`, 'Down Projection (down_proj)', 'linear', {
        layer: L,
        ts: downTs,
        in_dim: intermediate,
        out_dim: h,
        shape_str: `[${intermediate} × ${h}]`,
      })
      addEdge(`l${L}_hadamard`, `l${L}_down_proj`)

      // Residual Add 2 (⊕)
      addNode(`l${L}_res2`, '⊕ Residual Add 2', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
      addEdge(`l${L}_down_proj`, `l${L}_res2`, 'data')
      addEdge(`l${L}_res1`, `l${L}_res2`, 'residual', 'skip connection')
    }

    // Layer Output
    addNode(`l${L}_out`, `Layer ${L} Output`, 'io', { layer: L, shape_str: `[batch, seq, ${h}]` })
    addEdge(`l${L}_res2`, `l${L}_out`)
  }

  // =========================================================================
  // LEVEL 3: OPERATORS & TENSORS (PARAMETRIC & WEIGHT DETAILS)
  // =========================================================================
  else if (level === 3) {
    const L = currentLayer

    // In Level 3, we show the full mathematical dataflow with explicit tensor weight nodes
    addNode(`l${L}_in`, `Layer ${L} Input Vector`, 'io', { layer: L, shape_str: `[batch, seq, ${h}]` })

    // Input Norm
    const norm1Ts = findTensors(tensors, L, 'input_layernorm')
    addNode(`l${L}_norm1`, `${spec.norm_type || 'RMSNorm'} 1 Weight`, 'norm', {
      layer: L,
      ts: norm1Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
      ev: norm1Ts.length > 0 ? norm1Ts[0].name : 'model.layers.N.input_layernorm.weight',
    })
    addEdge(`l${L}_in`, `l${L}_norm1`)

    // Attention Projections with exact weight tensors
    const qTs = findTensors(tensors, L, 'q_proj')
    const kTs = findTensors(tensors, L, 'k_proj')
    const vTs = findTensors(tensors, L, 'v_proj')
    const oTs = findTensors(tensors, L, 'o_proj')

    addNode(`l${L}_q_weight`, 'q_proj.weight', 'tensor', {
      layer: L,
      ts: qTs,
      in_dim: h,
      out_dim: qHeads * headDim,
      shape_str: qTs.length > 0 && qTs[0].shape ? getShapeStr(qTs[0].shape) : `[${h} × ${qHeads * headDim}]`,
      ev: qTs.length > 0 ? `${qTs[0].name} (${qTs[0].dtype || '?'})` : 'Weight tensor',
    })
    addNode(`l${L}_k_weight`, 'k_proj.weight', 'tensor', {
      layer: L,
      ts: kTs,
      in_dim: h,
      out_dim: kvHeads * headDim,
      shape_str: kTs.length > 0 && kTs[0].shape ? getShapeStr(kTs[0].shape) : `[${h} × ${kvHeads * headDim}]`,
      ev: kTs.length > 0 ? `${kTs[0].name} (${kTs[0].dtype || '?'})` : 'Weight tensor',
    })
    addNode(`l${L}_v_weight`, 'v_proj.weight', 'tensor', {
      layer: L,
      ts: vTs,
      in_dim: h,
      out_dim: kvHeads * headDim,
      shape_str: vTs.length > 0 && vTs[0].shape ? getShapeStr(vTs[0].shape) : `[${h} × ${kvHeads * headDim}]`,
      ev: vTs.length > 0 ? `${vTs[0].name} (${vTs[0].dtype || '?'})` : 'Weight tensor',
    })

    addEdge(`l${L}_norm1`, `l${L}_q_weight`)
    addEdge(`l${L}_norm1`, `l${L}_k_weight`)
    addEdge(`l${L}_norm1`, `l${L}_v_weight`)

    addNode(`l${L}_rope_q`, 'RoPE (Query Positional)', 'attention_op', { layer: L, shape_str: `[${qHeads}, seq, ${headDim}]` })
    addNode(`l${L}_rope_k`, 'RoPE (Key Positional)', 'attention_op', { layer: L, shape_str: `[${kvHeads}, seq, ${headDim}]` })
    addEdge(`l${L}_q_weight`, `l${L}_rope_q`)
    addEdge(`l${L}_k_weight`, `l${L}_rope_k`)

    addNode(`l${L}_scores`, 'Scaled Q·Kᵀ MatMul', 'math_op', { layer: L, shape_str: `[${qHeads}, seq, seq]` })
    addEdge(`l${L}_rope_q`, `l${L}_scores`)
    addEdge(`l${L}_rope_k`, `l${L}_scores`)

    addNode(`l${L}_softmax`, 'Softmax + Causal Mask', 'math_op', { layer: L, shape_str: `[${qHeads}, seq, seq]` })
    addEdge(`l${L}_scores`, `l${L}_softmax`)

    addNode(`l${L}_ctx`, 'Scores · V Context MatMul', 'math_op', { layer: L, shape_str: `[batch, seq, ${qHeads * headDim}]` })
    addEdge(`l${L}_softmax`, `l${L}_ctx`)
    addEdge(`l${L}_v_weight`, `l${L}_ctx`)

    addNode(`l${L}_o_weight`, 'o_proj.weight', 'tensor', {
      layer: L,
      ts: oTs,
      in_dim: qHeads * headDim,
      out_dim: h,
      shape_str: oTs.length > 0 && oTs[0].shape ? getShapeStr(oTs[0].shape) : `[${qHeads * headDim} × ${h}]`,
      ev: oTs.length > 0 ? `${oTs[0].name} (${oTs[0].dtype || '?'})` : 'Weight tensor',
    })
    addEdge(`l${L}_ctx`, `l${L}_o_weight`)

    // Residual Add 1
    addNode(`l${L}_res1`, '⊕ Residual Add 1', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
    addEdge(`l${L}_o_weight`, `l${L}_res1`, 'data')
    addEdge(`l${L}_in`, `l${L}_res1`, 'residual', 'skip connection')

    // Norm 2
    const norm2Ts = findTensors(tensors, L, 'post_attention_layernorm')
    addNode(`l${L}_norm2`, `${spec.norm_type || 'RMSNorm'} 2 Weight`, 'norm', {
      layer: L,
      ts: norm2Ts,
      in_dim: h,
      out_dim: h,
      shape_str: `[${h}]`,
    })
    addEdge(`l${L}_res1`, `l${L}_norm2`)

    // FFN
    if (isCurrentLayerMoe) {
      const routerTs = findTensors(tensors, L, 'moe_router')
      const numExp = spec.moe?.num_experts || 64
      const topK = spec.moe?.experts_per_tok || 8
      addNode(`l${L}_router_w`, 'mlp.gate.weight (Router)', 'router', {
        layer: L,
        ts: routerTs,
        in_dim: h,
        out_dim: numExp,
        shape_str: routerTs.length > 0 && routerTs[0].shape ? getShapeStr(routerTs[0].shape) : `[${h} × ${numExp}]`,
      })
      addEdge(`l${L}_norm2`, `l${L}_router_w`)

      const expertTs = tensors.filter(t => t.layer === L && t.category === 'moe' && t.expert !== null)
      addNode(`l${L}_expert_tensors`, `Routed Experts Tensors (×${numExp})`, 'expert', {
        layer: L,
        ts: expertTs,
        collapsed_count: numExp,
        ev: `Each expert contains gate_proj, up_proj, down_proj weights (${expertTs.length} total tensors)`,
      })
      addEdge(`l${L}_router_w`, `l${L}_expert_tensors`, 'control', `Top-${topK}`)
      addEdge(`l${L}_norm2`, `l${L}_expert_tensors`, 'data')

      addNode(`l${L}_combine`, 'Expert Accumulator', 'math_op', { layer: L, shape_str: `[batch, seq, ${h}]` })
      addEdge(`l${L}_expert_tensors`, `l${L}_combine`)

      addNode(`l${L}_res2`, '⊕ Residual Add 2', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
      addEdge(`l${L}_combine`, `l${L}_res2`, 'data')
      addEdge(`l${L}_res1`, `l${L}_res2`, 'residual', 'skip connection')
    } else {
      const gateTs = findTensors(tensors, L, 'gate_proj')
      const upTs = findTensors(tensors, L, 'up_proj')
      const downTs = findTensors(tensors, L, 'down_proj')

      addNode(`l${L}_gate_w`, 'gate_proj.weight', 'tensor', {
        layer: L,
        ts: gateTs,
        shape_str: gateTs.length > 0 && gateTs[0].shape ? getShapeStr(gateTs[0].shape) : `[${h} × ${intermediate}]`,
      })
      addNode(`l${L}_up_w`, 'up_proj.weight', 'tensor', {
        layer: L,
        ts: upTs,
        shape_str: upTs.length > 0 && upTs[0].shape ? getShapeStr(upTs[0].shape) : `[${h} × ${intermediate}]`,
      })
      addEdge(`l${L}_norm2`, `l${L}_gate_w`)
      addEdge(`l${L}_norm2`, `l${L}_up_w`)

      addNode(`l${L}_act`, `${(spec.activation || 'silu').toUpperCase()} Activation`, 'math_op', { layer: L })
      addEdge(`l${L}_gate_w`, `l${L}_act`)

      addNode(`l${L}_mul`, '⊗ Elementwise Multiply', 'math_op', { layer: L })
      addEdge(`l${L}_act`, `l${L}_mul`)
      addEdge(`l${L}_up_w`, `l${L}_mul`)

      addNode(`l${L}_down_w`, 'down_proj.weight', 'tensor', {
        layer: L,
        ts: downTs,
        shape_str: downTs.length > 0 && downTs[0].shape ? getShapeStr(downTs[0].shape) : `[${intermediate} × ${h}]`,
      })
      addEdge(`l${L}_mul`, `l${L}_down_w`)

      addNode(`l${L}_res2`, '⊕ Residual Add 2', 'residual', { layer: L, shape_str: `[batch, seq, ${h}]` })
      addEdge(`l${L}_down_w`, `l${L}_res2`, 'data')
      addEdge(`l${L}_res1`, `l${L}_res2`, 'residual', 'skip connection')
    }

    addNode(`l${L}_out`, `Layer ${L} Output Vector`, 'io', { layer: L, shape_str: `[batch, seq, ${h}]` })
    addEdge(`l${L}_res2`, `l${L}_out`)
  }

  // Safety check: ensure no dangling edges
  const nodeIds = new Set(g.nodes.map(n => n.id))
  g.edges = g.edges.filter(e => nodeIds.has(e.src) && nodeIds.has(e.dst))

  return g
}
