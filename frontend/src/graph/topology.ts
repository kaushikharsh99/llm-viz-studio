import type { ArchGraph, GraphNode, GraphEdge, TensorInfo, TensorRef } from './types'
import { annotate } from './mapping'
import type { ArchSpec } from '../engine/spec'

const MOE_TYPES = new Set(['deepseek_v3', 'deepseek_v2', 'qwen3_moe', 'qwen3_next', 'mixtral', 'glm4_moe'])

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

function group(tensors: TensorInfo[], layer: number | null, module: string): TensorInfo[] {
  return tensors.filter(t => t.layer === layer && t.module === module && t.category !== 'quant_scale')
}

function sumParams(ts: TensorInfo[]): number {
  return ts.reduce((acc, t) => acc + (t.numel || 0), 0)
}

function getDim(ts: TensorInfo[], idx = 0): number | null {
  for (const t of ts) {
    if (t.shape && t.shape.length > idx) return t.shape[idx]
  }
  return null
}

export function buildGraph(
  modelId: string,
  spec: ArchSpec,
  rawTensors: TensorInfo[],
  available: boolean,
  note?: string | null,
  repLayer?: number | null
): ArchGraph {
  const tensors = annotate(rawTensors)
  const mt = (spec.model_type || '').toLowerCase()
  const isMoe = Boolean(spec.moe) || MOE_TYPES.has(mt)
  const layerSet = new Set<number>()
  for (const t of tensors) {
    if (t.layer !== null && t.layer !== undefined) layerSet.add(t.layer)
  }
  const layers = layerSet.size > 0
    ? Array.from(layerSet).sort((a, b) => a - b)
    : Array.from({ length: spec.num_layers }, (_, i) => i)

  const g: ArchGraph = {
    model_id: modelId,
    model_type: spec.model_type,
    architecture: spec.architecture,
    confidence: available ? 'verified' : 'config_only',
    metadata_source: available ? 'safetensors' : 'config_only',
    metadata_note: note ?? null,
    total_params: spec.total_params,
    verified_params: tensors
      .filter(t => t.category !== 'quant_scale')
      .reduce((acc, t) => acc + (t.numel || 0), 0),
    tied_note: available && !tensors.some(t => t.name.includes('lm_head'))
      ? 'lm_head absent from checkpoint (tied embeddings) — verified total excludes it'
      : null,
    nodes: [],
    edges: [],
    layers,
    unknowns: Array.from(new Set(tensors.filter(t => t.module === 'unknown').map(t => t.name))).slice(0, 50),
  }

  function N(
    id: string,
    label: string,
    kind: string,
    opts: {
      level?: number
      parent?: string | null
      layer?: number | null
      ts?: TensorInfo[]
      conf?: string
      ev?: string
      in_dim?: number | null
      out_dim?: number | null
      collapsed?: number | null
    } = {}
  ): void {
    const ts = opts.ts || []
    g.nodes.push({
      id,
      label,
      kind,
      level: opts.level ?? 1,
      parent: opts.parent ?? null,
      layer: opts.layer ?? null,
      params: sumParams(ts),
      dtype: ts.length > 0 ? ts[0].dtype : null,
      in_dim: opts.in_dim ?? null,
      out_dim: opts.out_dim ?? null,
      confidence: opts.conf ?? 'verified',
      evidence: opts.ev ?? '',
      tensors: ts.slice(0, 20).map(tref),
      collapsed_count: opts.collapsed ?? null,
    })
  }

  function E(src: string, dst: string, kind: 'data' | 'residual' = 'data', conf = 'verified'): void {
    g.edges.push({ src, dst, kind, confidence: conf })
  }

  const emb = group(tensors, null, 'embedding')
  N('embed', 'Token embeddings', 'embedding', {
    ts: emb,
    ev: emb.length > 0 ? 'embed_tokens.weight' : 'config only',
    in_dim: spec.vocab_size,
    out_dim: spec.hidden_size,
    conf: emb.length > 0 ? 'verified' : 'config',
  })

  N('blocks', `Transformer layers ×${layers.length}`, 'block', {
    conf: available ? 'verified' : 'config',
    collapsed: layers.length,
    ev: 'repeated block — expand one layer below',
  })

  let rep = repLayer !== undefined && repLayer !== null
    ? repLayer
    : (layers.length > 0 ? layers[0] : 0)
  if (repLayer === undefined || repLayer === null) {
    if (isMoe) {
      const moeLayers = Array.from(
        new Set(
          tensors
            .filter(t => t.layer !== null && t.layer !== undefined && (t.category === 'moe_router' || t.category === 'moe_expert'))
            .map(t => t.layer!)
        )
      ).sort((a, b) => a - b)
      if (moeLayers.length > 0) rep = moeLayers[0]
    }
  }

  const bp = `block:${rep}`
  N(bp, `Layer ${rep} (representative)`, 'block', {
    level: 2,
    parent: 'blocks',
    layer: rep,
    conf: available ? 'verified' : 'partial',
    ev: available ? 'tensors present' : 'no tensor evidence',
  })

  E('embed', 'blocks')
  E('blocks', 'final_norm')
  E(bp, 'blocks')

  const fin = group(tensors, null, 'final_norm')
  N('final_norm', 'Final normalization', 'norm', {
    ts: fin,
    conf: fin.length > 0 ? 'verified' : 'config',
  })

  const head = group(tensors, null, 'lm_head')
  N('lm_head', 'LM head', 'head', {
    ts: head,
    in_dim: spec.hidden_size,
    out_dim: spec.vocab_size,
    conf: head.length > 0 ? 'verified' : 'inferred',
    ev: available && head.length === 0 ? 'tied embeddings' : '',
  })

  const pre = `${bp}:`
  if (available && tensors.some(t => t.layer === rep)) {
    const n1 = group(tensors, rep, 'input_norm')
    const n2 = group(tensors, rep, 'post_attn_norm')
    N(pre + 'norm1', 'RMSNorm/LayerNorm 1', 'norm', {
      level: 2, parent: bp, layer: rep, ts: n1, conf: n1.length > 0 ? 'verified' : 'inferred',
    })
    N(pre + 'norm2', 'RMSNorm/LayerNorm 2', 'norm', {
      level: 2, parent: bp, layer: rep, ts: n2, conf: n2.length > 0 ? 'verified' : 'inferred',
    })

    const q = group(tensors, rep, 'attention')
    const qa = [...group(tensors, rep, 'attention_q_a'), ...group(tensors, rep, 'attention_q_b')]
    const kva = group(tensors, rep, 'attention_kv_a')
    const kvb = group(tensors, rep, 'attention_kv_b')
    const op = group(tensors, rep, 'attention_o')
    const fused = group(tensors, rep, 'attention_qkv_fused')
    const qkNorm = group(tensors, rep, 'attention_qk_norm')

    if (spec.attention.kind === 'MLA' || qa.length > 0) {
      N(pre + 'attn_qa', 'Q latent ↓ (q_a)', 'attention', { level: 2, parent: bp, layer: rep, ts: qa, in_dim: spec.hidden_size })
      N(pre + 'attn_kva', 'KV latent ↓ (kv_a, MQA)', 'attention', { level: 2, parent: bp, layer: rep, ts: kva })
      N(pre + 'attn_qb', 'Q ↑ (q_b)', 'attention', { level: 2, parent: bp, layer: rep, ts: qa })
      N(pre + 'attn_kvb', 'KV ↑ (kv_b)', 'attention', { level: 2, parent: bp, layer: rep, ts: kvb })
      N(pre + 'attn_o', 'O projection', 'attention', { level: 2, parent: bp, layer: rep, ts: op })
      E(pre + 'norm1', pre + 'attn_qa')
      E(pre + 'norm1', pre + 'attn_kva')
      E(pre + 'attn_qb', pre + 'attn_o')
      E(pre + 'attn_kvb', pre + 'attn_o')
    } else if (fused.length > 0) {
      N(pre + 'attn_qkv', 'Fused QKV projection', 'attention', { level: 2, parent: bp, layer: rep, ts: fused })
      N(pre + 'attn_o', 'O projection', 'attention', { level: 2, parent: bp, layer: rep, ts: op })
      E(pre + 'norm1', pre + 'attn_qkv')
      E(pre + 'attn_qkv', pre + 'attn_o')
    } else if (q.length > 0) {
      N(pre + 'attn_qkv', 'Q/K/V projections', 'attention', {
        level: 2, parent: bp, layer: rep, ts: q, in_dim: spec.hidden_size, out_dim: getDim(q),
      })
      N(pre + 'attn_o', 'O projection', 'attention', { level: 2, parent: bp, layer: rep, ts: op })
      E(pre + 'norm1', pre + 'attn_qkv')
      E(pre + 'attn_qkv', pre + 'attn_o')
    } else {
      N(pre + 'attn', 'Attention (no tensor evidence)', 'attention', { level: 2, parent: bp, layer: rep, conf: 'unknown' })
      E(pre + 'norm1', pre + 'attn')
    }

    if (spec.positional === 'rope') {
      N(pre + 'rope', 'RoPE', 'op', { level: 2, parent: bp, layer: rep, conf: 'inferred', ev: 'positional=rope in config' })
    }
    N(pre + 'softmax', 'Softmax attention', 'op', { level: 2, parent: bp, layer: rep, conf: 'inferred', ev: 'standard attention rule' })
    if (qkNorm.length > 0) {
      N(pre + 'qk_norm', 'Q/K norm', 'norm', { level: 2, parent: bp, layer: rep, ts: qkNorm })
    }

    E(bp, pre + 'norm1', 'residual', 'inferred')

    const mlp = group(tensors, rep, 'mlp')
    const gateUp = group(tensors, rep, 'mlp_gate_up_fused')
    const router = group(tensors, rep, 'router')
    const shared = group(tensors, rep, 'shared_expert')
    const experts = tensors.filter(t => t.layer === rep && t.module === 'moe_expert')

    if (isMoe && (router.length > 0 || experts.length > 0)) {
      N(pre + 'router', 'MoE router (top-k)', 'moe', {
        level: 2, parent: bp, layer: rep, ts: router, conf: router.length > 0 ? 'verified' : 'inferred',
      })
      const expCount = spec.moe ? spec.moe.num_experts : new Set(experts.map(t => t.expert)).size
      N(pre + 'experts', `Routed experts ×${expCount}`, 'moe', {
        level: 2, parent: bp, layer: rep, ts: experts.slice(0, 20), conf: experts.length > 0 ? 'verified' : 'inferred',
        ev: `${new Set(experts.map(t => t.expert)).size} expert tensors in layer ${rep} (showing first 20)`,
      })
      if (shared.length > 0) {
        N(pre + 'shared', 'Shared experts', 'moe', { level: 2, parent: bp, layer: rep, ts: shared, conf: 'verified' })
      }
      E(pre + 'norm2', pre + 'router')
      E(pre + 'router', pre + 'experts')
      if (shared.length > 0) {
        E(pre + 'norm2', pre + 'shared')
      }
    } else if (mlp.length > 0 || gateUp.length > 0) {
      const mlpLabel = new Set(mlp.map(t => t.module)).size > 0 ? 'SwiGLU MLP (gate/up/down)' : 'Fused MLP'
      N(pre + 'mlp', mlpLabel, 'mlp', {
        level: 2, parent: bp, layer: rep, ts: [...mlp, ...gateUp], conf: 'verified',
        in_dim: spec.hidden_size, out_dim: getDim([...mlp, ...gateUp]),
      })
      N(pre + 'silu', 'SiLU gating', 'op', {
        level: 2, parent: bp, layer: rep, conf: 'inferred', ev: `activation=${spec.activation}`,
      })
      E(pre + 'norm2', pre + 'mlp')
      E(pre + 'mlp', pre + 'silu')
    } else {
      N(pre + 'mlp', 'Feed-forward (no tensor evidence)', 'mlp', { level: 2, parent: bp, layer: rep, conf: 'unknown' })
      E(pre + 'norm2', pre + 'mlp')
    }

    const unk = tensors.filter(t => t.layer === rep && t.module === 'unknown')
    if (unk.length > 0) {
      N(pre + 'unknown', `Unmapped tensors ×${unk.length}`, 'unknown', {
        level: 2, parent: bp, layer: rep, ts: unk, conf: 'unknown',
      })
    }
    g.confidence = available ? 'verified' : 'partial'
  } else {
    g.confidence = 'partial'
    if (!available) {
      g.metadata_note = (note || 'safetensors unavailable') + ' — config-only graph; block internals unavailable'
    }
  }

  const ids = new Set(g.nodes.map(n => n.id))
  for (const e of g.edges) {
    if (!ids.has(e.src) || !ids.has(e.dst)) {
      g.metadata_note = (g.metadata_note || '') + ` | dangling edge ${e.src}->${e.dst}`
    }
  }

  return g
}
