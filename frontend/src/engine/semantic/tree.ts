import type { ArchSpec } from '../spec'
import type { TensorInfo, TensorRef } from '../../graph/types'
import type { SemanticNode, SemanticNodeKind } from './types'

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

function findTensors(tensors: TensorInfo[], layer: number | null, moduleOrCat: string): TensorInfo[] {
  return tensors.filter(
    t => t.layer === layer && (t.module === moduleOrCat || t.category === moduleOrCat) && t.category !== 'quant_scale'
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
 * Build the complete semantic architecture tree for a model
 */
export function buildSemanticTree(
  spec: ArchSpec,
  rawTensors: TensorInfo[] = [],
  activeLayer = 0
): SemanticNode {
  const h = spec.hidden_size
  const v = spec.vocab_size || 32000
  const qHeads = spec.attention?.num_heads || 32
  const headDim = spec.attention?.head_dim || Math.floor(h / qHeads)
  const kvHeads = spec.attention?.num_kv_heads || qHeads
  const intermediate = spec.intermediate_size
  const isMoe = Boolean(spec.moe)
  const isMLA = spec.attention?.kind === 'MLA'
  const L = activeLayer

  const tensors = rawTensors

  // 1. Token Embeddings
  const embTs = findTensors(tensors, null, 'embedding')
  const embNode: SemanticNode = {
    id: 'embed_tokens',
    kind: 'embedding',
    label: 'Token embedding layer',
    sublabel: `Vocabulary: ${v.toLocaleString()} tokens → ${h} hidden`,
    badge: 'EMBED',
    tensors: embTs.map(tref),
    params: sumParams(embTs) || (v * h),
    in_dim: v,
    out_dim: h,
    shape_str: embTs.length > 0 && embTs[0].shape ? getShapeStr(embTs[0].shape) : `[${v} × ${h}]`,
    dtype: embTs[0]?.dtype || null,
    is_expandable: true,
    default_detail_level: 1,
    description: 'Maps discrete input token IDs to dense continuous representation vectors in latent hidden space.',
    evidence: embTs[0]?.name || 'model.embed_tokens.weight',
  }

  // 2. Pre-Attention Norm
  const norm1Ts = findTensors(tensors, L, 'input_layernorm')
  const norm1Node: SemanticNode = {
    id: `layer_${L}_norm1`,
    kind: 'norm_1',
    label: spec.norm_type === 'rmsnorm' ? 'RMSNorm 1' : 'LayerNorm 1',
    sublabel: `Pre-Attention Normalization [${h}]`,
    badge: 'NORM',
    layerIndex: L,
    tensors: norm1Ts.map(tref),
    params: sumParams(norm1Ts) || h,
    in_dim: h,
    out_dim: h,
    shape_str: `[${h}]`,
    dtype: norm1Ts[0]?.dtype || null,
    is_expandable: false,
    default_detail_level: 1,
    description: 'Normalizes input activation variance before the attention block (Pre-LN formulation).',
    evidence: norm1Ts[0]?.name || 'model.layers.0.input_layernorm.weight',
  }

  // 3. Self-Attention Block & Children
  const attnTs = tensors.filter(t => t.layer === L && t.category === 'attention')
  const qTs = findTensors(tensors, L, 'q_proj')
  const kTs = findTensors(tensors, L, 'k_proj')
  const vTs = findTensors(tensors, L, 'v_proj')
  const oTs = findTensors(tensors, L, 'o_proj')
  const qNormTs = findTensors(tensors, L, 'q_norm')
  const kNormTs = findTensors(tensors, L, 'k_norm')

  const attnChildren: SemanticNode[] = []

  if (isMLA) {
    // DeepSeek Multi-Head Latent Attention internals
    const qaTs = findTensors(tensors, L, 'attention_q_a')
    const qbTs = findTensors(tensors, L, 'attention_q_b')
    const kvaTs = findTensors(tensors, L, 'attention_kv_a')
    const kvbTs = findTensors(tensors, L, 'attention_kv_b')

    attnChildren.push(
      {
        id: `layer_${L}_qa`,
        kind: 'mla_latent',
        label: 'Q Latent Down-Projection (q_a_proj)',
        badge: 'LINEAR',
        layerIndex: L,
        tensors: qaTs.map(tref),
        params: sumParams(qaTs),
        in_dim: h,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Compresses query representations into low-dimensional latent space.',
      },
      {
        id: `layer_${L}_qb`,
        kind: 'q_proj',
        label: 'Q Latent Up-Projection (q_b_proj)',
        badge: 'LINEAR',
        layerIndex: L,
        tensors: qbTs.map(tref),
        params: sumParams(qbTs),
        out_dim: qHeads * headDim,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Restores compressed query representation into individual attention heads.',
      },
      {
        id: `layer_${L}_kva`,
        kind: 'mla_latent',
        label: 'KV Latent Down-Projection (kv_a_proj)',
        badge: 'LINEAR',
        layerIndex: L,
        tensors: kvaTs.map(tref),
        params: sumParams(kvaTs),
        in_dim: h,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Compresses KV cache into shared low-rank latent representation (MQA decoupled RoPE).',
      },
      {
        id: `layer_${L}_kvb`,
        kind: 'v_proj',
        label: 'KV Latent Up-Projection (kv_b_proj)',
        badge: 'LINEAR',
        layerIndex: L,
        tensors: kvbTs.map(tref),
        params: sumParams(kvbTs),
        out_dim: kvHeads * headDim,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Restores compressed key and value heads during attention calculation.',
      }
    )
  } else {
    // Standard GQA / MHA projections
    attnChildren.push(
      {
        id: `layer_${L}_q_proj`,
        kind: 'q_proj',
        label: 'Query Projection (q_proj)',
        sublabel: `${h} → ${qHeads * headDim} (${qHeads} heads × ${headDim} dim)`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: qTs.map(tref),
        params: sumParams(qTs) || (h * qHeads * headDim),
        in_dim: h,
        out_dim: qHeads * headDim,
        shape_str: qTs[0]?.shape ? getShapeStr(qTs[0].shape) : `[${h} × ${qHeads * headDim}]`,
        dtype: qTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Projects normalized token representations into query vectors for attention routing.',
        evidence: qTs[0]?.name || 'model.layers.0.self_attn.q_proj.weight',
      },
      {
        id: `layer_${L}_k_proj`,
        kind: 'k_proj',
        label: 'Key Projection (k_proj)',
        sublabel: `${h} → ${kvHeads * headDim} (${kvHeads} KV groups × ${headDim} dim)`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: kTs.map(tref),
        params: sumParams(kTs) || (h * kvHeads * headDim),
        in_dim: h,
        out_dim: kvHeads * headDim,
        shape_str: kTs[0]?.shape ? getShapeStr(kTs[0].shape) : `[${h} × ${kvHeads * headDim}]`,
        dtype: kTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Projects normalized token representations into key vectors used for dot-product matching.',
        evidence: kTs[0]?.name || 'model.layers.0.self_attn.k_proj.weight',
      },
      {
        id: `layer_${L}_v_proj`,
        kind: 'v_proj',
        label: 'Value Projection (v_proj)',
        sublabel: `${h} → ${kvHeads * headDim} (${kvHeads} KV groups × ${headDim} dim)`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: vTs.map(tref),
        params: sumParams(vTs) || (h * kvHeads * headDim),
        in_dim: h,
        out_dim: kvHeads * headDim,
        shape_str: vTs[0]?.shape ? getShapeStr(vTs[0].shape) : `[${h} × ${kvHeads * headDim}]`,
        dtype: vTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Projects token representations into value vectors that are aggregated based on attention weights.',
        evidence: vTs[0]?.name || 'model.layers.0.self_attn.v_proj.weight',
      }
    )

    if (qNormTs.length > 0) {
      attnChildren.push({
        id: `layer_${L}_qk_norm`,
        kind: 'qk_norm',
        label: 'Query & Key RMSNorm (QK-Norm)',
        badge: 'NORM',
        layerIndex: L,
        tensors: [...qNormTs, ...kNormTs].map(tref),
        params: sumParams([...qNormTs, ...kNormTs]),
        shape_str: `[${headDim}]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Normalizes query and key head vectors to stabilize attention logits in large models.',
      })
    }

    if (spec.positional === 'rope') {
      attnChildren.push({
        id: `layer_${L}_rope`,
        kind: 'rope_op',
        label: 'Rotary Position Embedding (RoPE)',
        sublabel: `Base θ = ${spec.rope_theta || 10000}`,
        badge: 'ROTARY',
        layerIndex: L,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Rotates query and key vectors in complex 2D subspaces according to token position.',
      })
    }

    // Detail Level 3 Operations
    attnChildren.push(
      {
        id: `layer_${L}_attn_scores`,
        kind: 'attn_scores',
        label: 'Scaled Dot-Product (Q · Kᵀ / √d)',
        sublabel: `Matrix MatMul [${qHeads}, seq_len, seq_len]`,
        badge: 'MATMUL',
        layerIndex: L,
        shape_str: `[${qHeads}, seq, seq]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Computes unnormalized pairwise attention compatibility scores between queries and keys.',
      },
      {
        id: `layer_${L}_softmax`,
        kind: 'softmax_op',
        label: 'Causal Mask + Softmax',
        sublabel: 'Row-wise probability distribution [∑=1.0]',
        badge: 'SOFTMAX',
        layerIndex: L,
        shape_str: `[${qHeads}, seq, seq]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Applies lower-triangular causal attention mask and normalizes scores into attention weights.',
      },
      {
        id: `layer_${L}_attn_ctx`,
        kind: 'attn_context',
        label: 'Value Aggregation (Scores · V)',
        sublabel: `Weighted Context Sum [batch, seq, ${qHeads * headDim}]`,
        badge: 'MATMUL',
        layerIndex: L,
        shape_str: `[batch, seq, ${qHeads * headDim}]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Multiplies attention probability weights by value vectors to produce contextual head embeddings.',
      }
    )

    attnChildren.push({
      id: `layer_${L}_o_proj`,
      kind: 'o_proj',
      label: 'Output Projection (o_proj)',
      sublabel: `${qHeads * headDim} → ${h}`,
      badge: 'LINEAR',
      layerIndex: L,
      tensors: oTs.map(tref),
      params: sumParams(oTs) || (qHeads * headDim * h),
      in_dim: qHeads * headDim,
      out_dim: h,
      shape_str: oTs[0]?.shape ? getShapeStr(oTs[0].shape) : `[${qHeads * headDim} × ${h}]`,
      dtype: oTs[0]?.dtype || null,
      is_expandable: true,
      default_detail_level: 2,
      description: 'Combines multi-head contextual attention outputs back into the residual stream dimension.',
      evidence: oTs[0]?.name || 'model.layers.0.self_attn.o_proj.weight',
    })
  }

  const attnKindLabel = spec.attention?.kind === 'MLA'
    ? 'Multi-head Latent Attention'
    : spec.attention?.kind === 'GQA'
      ? 'Masked grouped-query attention'
      : spec.attention?.kind === 'MQA'
        ? 'Masked multi-query attention'
        : 'Masked multi-head attention'

  const attnNode: SemanticNode = {
    id: `layer_${L}_attention`,
    kind: 'attention',
    label: attnKindLabel,
    sublabel: `${qHeads} heads, ${kvHeads} KV groups, head_dim ${headDim}`,
    badge: 'ATTENTION',
    layerIndex: L,
    tensors: attnTs.map(tref),
    params: sumParams(attnTs),
    is_expandable: true,
    default_detail_level: 1,
    children: attnChildren,
    description: `Multi-head self-attention mechanism with ${qHeads} query heads and ${kvHeads} key/value groups.`,
    evidence: attnTs[0]?.name || 'model.layers.0.self_attn',
  }

  // 4. Residual Add 1
  const res1Node: SemanticNode = {
    id: `layer_${L}_res1`,
    kind: 'residual_add_1',
    label: '⊕ Residual Connection 1',
    sublabel: 'Layer input added directly to Attention output',
    badge: 'RESIDUAL',
    layerIndex: L,
    shape_str: `[batch, seq, ${h}]`,
    is_expandable: false,
    default_detail_level: 1,
    description: 'Preserves gradient highway and identity mapping around the multi-head attention block.',
  }

  // 5. Pre-FFN Norm
  const norm2Ts = findTensors(tensors, L, 'post_attention_layernorm')
  const norm2Node: SemanticNode = {
    id: `layer_${L}_norm2`,
    kind: 'norm_2',
    label: spec.norm_type === 'rmsnorm' ? 'RMSNorm 2' : 'LayerNorm 2',
    sublabel: `Pre-FFN Normalization [${h}]`,
    badge: 'NORM',
    layerIndex: L,
    tensors: norm2Ts.map(tref),
    params: sumParams(norm2Ts) || h,
    in_dim: h,
    out_dim: h,
    shape_str: `[${h}]`,
    dtype: norm2Ts[0]?.dtype || null,
    is_expandable: false,
    default_detail_level: 1,
    description: 'Normalizes activations before the feed-forward or Mixture-of-Experts network.',
    evidence: norm2Ts[0]?.name || 'model.layers.0.post_attention_layernorm.weight',
  }

  // 6. FFN / MoE Block & Children
  let ffnNode: SemanticNode

  if (isMoe) {
    const routerTs = findTensors(tensors, L, 'moe_router')
    const expertTs = tensors.filter(t => t.layer === L && t.category === 'moe' && t.expert !== null)
    const sharedGateTs = findTensors(tensors, L, 'moe_shared_gate')
    const sharedUpTs = findTensors(tensors, L, 'moe_shared_up')
    const sharedDownTs = findTensors(tensors, L, 'moe_shared_down')
    const numExperts = spec.moe?.num_experts || 64
    const topK = spec.moe?.experts_per_tok || 8

    const moeChildren: SemanticNode[] = [
      {
        id: `layer_${L}_router`,
        kind: 'moe_router',
        label: `Gating Router (Top-${topK} of ${numExperts})`,
        sublabel: `${h} → ${numExperts} Softmax routing probabilities`,
        badge: 'ROUTER',
        layerIndex: L,
        tensors: routerTs.map(tref),
        params: sumParams(routerTs) || (h * numExperts),
        in_dim: h,
        out_dim: numExperts,
        shape_str: routerTs[0]?.shape ? getShapeStr(routerTs[0].shape) : `[${h} × ${numExperts}]`,
        is_expandable: false,
        default_detail_level: 2,
        description: 'Computes gating distribution to select the top active experts for each token dynamically.',
      },
      {
        id: `layer_${L}_routed_experts`,
        kind: 'moe_expert_group',
        label: `${numExperts} Routed Experts`,
        sublabel: `${topK} active per token · SwiGLU FFN per expert`,
        badge: 'EXPERTS',
        layerIndex: L,
        tensors: expertTs.map(tref),
        params: sumParams(expertTs),
        is_expandable: true,
        default_detail_level: 2,
        description: 'Sparse expert pool. Each expert performs independent non-linear feature transformation.',
      },
    ]

    if (sharedGateTs.length > 0 || sharedUpTs.length > 0 || sharedDownTs.length > 0) {
      moeChildren.push({
        id: `layer_${L}_shared_experts`,
        kind: 'moe_shared',
        label: 'Shared Experts (Always Active)',
        sublabel: 'Processes all tokens in parallel with sparse routed experts',
        badge: 'SHARED',
        layerIndex: L,
        tensors: [...sharedGateTs, ...sharedUpTs, ...sharedDownTs].map(tref),
        params: sumParams([...sharedGateTs, ...sharedUpTs, ...sharedDownTs]),
        is_expandable: false,
        default_detail_level: 2,
        description: 'Dense shared expert pathway capturing common knowledge across all token types.',
      })
    }

    ffnNode = {
      id: `layer_${L}_moe`,
      kind: 'moe',
      label: 'MoE',
      sublabel: `${numExperts} experts (top-${topK} active)`,
      badge: 'MOE',
      layerIndex: L,
      tensors: [...routerTs, ...expertTs].map(tref),
      params: sumParams([...routerTs, ...expertTs]),
      is_expandable: true,
      default_detail_level: 1,
      children: moeChildren,
      description: `Sparse Mixture-of-Experts layer with ${numExperts} total experts and ${topK} routed per token.`,
    }
  } else {
    // Dense SwiGLU FFN
    const gateTs = findTensors(tensors, L, 'gate_proj')
    const upTs = findTensors(tensors, L, 'up_proj')
    const downTs = findTensors(tensors, L, 'down_proj')
    const mlpTs = [...gateTs, ...upTs, ...downTs]

    const ffnChildren: SemanticNode[] = [
      {
        id: `layer_${L}_gate_proj`,
        kind: 'gate_proj',
        label: 'Gate Projection (gate_proj)',
        sublabel: `${h} → ${intermediate}`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: gateTs.map(tref),
        params: sumParams(gateTs) || (h * intermediate),
        in_dim: h,
        out_dim: intermediate,
        shape_str: gateTs[0]?.shape ? getShapeStr(gateTs[0].shape) : `[${h} × ${intermediate}]`,
        dtype: gateTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Projects hidden dimension to intermediate size to generate non-linear gating activations.',
      },
      {
        id: `layer_${L}_act`,
        kind: 'act_op',
        label: `${(spec.activation || 'silu').toUpperCase()} Non-Linearity`,
        badge: 'ACTIVATION',
        layerIndex: L,
        shape_str: `[batch, seq, ${intermediate}]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Non-linear gating function applied to gate_proj output.',
      },
      {
        id: `layer_${L}_up_proj`,
        kind: 'up_proj',
        label: 'Up Projection (up_proj)',
        sublabel: `${h} → ${intermediate}`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: upTs.map(tref),
        params: sumParams(upTs) || (h * intermediate),
        in_dim: h,
        out_dim: intermediate,
        shape_str: upTs[0]?.shape ? getShapeStr(upTs[0].shape) : `[${h} × ${intermediate}]`,
        dtype: upTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Projects hidden dimension into the intermediate expanded feed-forward channel.',
      },
      {
        id: `layer_${L}_hadamard`,
        kind: 'hadamard_op',
        label: 'Elementwise Product (Gate ⊗ Up)',
        badge: 'MULTIPLY',
        layerIndex: L,
        shape_str: `[batch, seq, ${intermediate}]`,
        is_expandable: false,
        default_detail_level: 3,
        description: 'Multiplies activated gate projection with up projection point-by-point.',
      },
      {
        id: `layer_${L}_down_proj`,
        kind: 'down_proj',
        label: 'Down Projection (down_proj)',
        sublabel: `${intermediate} → ${h}`,
        badge: 'LINEAR',
        layerIndex: L,
        tensors: downTs.map(tref),
        params: sumParams(downTs) || (intermediate * h),
        in_dim: intermediate,
        out_dim: h,
        shape_str: downTs[0]?.shape ? getShapeStr(downTs[0].shape) : `[${intermediate} × ${h}]`,
        dtype: downTs[0]?.dtype || null,
        is_expandable: true,
        default_detail_level: 2,
        description: 'Compresses intermediate expanded representation back down to residual stream dimension.',
      },
    ]

    ffnNode = {
      id: `layer_${L}_ffn`,
      kind: 'ffn',
      label: 'Feed forward',
      sublabel: `SwiGLU: ${h} → ${intermediate} → ${h}`,
      badge: 'FFN',
      layerIndex: L,
      tensors: mlpTs.map(tref),
      params: sumParams(mlpTs) || (3 * h * intermediate),
      is_expandable: true,
      default_detail_level: 1,
      children: ffnChildren,
      description: 'SwiGLU gated feed-forward network combining gate projection, non-linear activation, and down projection.',
    }
  }

  // 7. Residual Add 2
  const res2Node: SemanticNode = {
    id: `layer_${L}_res2`,
    kind: 'residual_add_2',
    label: '⊕ Residual Connection 2',
    sublabel: 'Attention residual added directly to FFN output',
    badge: 'RESIDUAL',
    layerIndex: L,
    shape_str: `[batch, seq, ${h}]`,
    is_expandable: false,
    default_detail_level: 1,
    description: 'Residual skip connection preserving feature representations across the feed-forward block.',
  }

  // Active Layer Block
  const layerBlockNode: SemanticNode = {
    id: `layer_${L}_block`,
    kind: 'layer_block',
    label: `Transformer Block (Layer ${L})`,
    sublabel: `Norm1 → Attention → Res1 → Norm2 → FFN → Res2`,
    badge: 'BLOCK',
    layerIndex: L,
    is_expandable: true,
    default_detail_level: 1,
    children: [
      norm1Node,
      attnNode,
      res1Node,
      norm2Node,
      ffnNode,
      res2Node,
    ],
  }

  // Repeated Layers Container
  const layersContainer: SemanticNode = {
    id: 'layers_container',
    kind: 'layers_container',
    label: `${spec.num_layers} × Repeated Transformer Blocks`,
    badge: `×${spec.num_layers}`,
    is_expandable: true,
    default_detail_level: 1,
    children: [layerBlockNode],
    description: `Sequence of ${spec.num_layers} identical Transformer decoder layers.`,
  }

  // 8. Final Normalization
  const finTs = findTensors(tensors, null, 'final_norm')
  const finNormNode: SemanticNode = {
    id: 'final_norm',
    kind: 'final_norm',
    label: spec.norm_type === 'rmsnorm' ? 'Final RMSNorm' : 'Final LayerNorm',
    sublabel: `Final Output Normalization [${h}]`,
    badge: 'NORM',
    tensors: finTs.map(tref),
    params: sumParams(finTs) || h,
    in_dim: h,
    out_dim: h,
    shape_str: `[${h}]`,
    dtype: finTs[0]?.dtype || null,
    is_expandable: false,
    default_detail_level: 1,
    description: 'Final normalization layer applied to the hidden states before the linear language model head.',
    evidence: finTs[0]?.name || 'model.norm.weight',
  }

  // 9. Linear Output Layer (LM Head)
  const headTs = findTensors(tensors, null, 'lm_head')
  const lmHeadNode: SemanticNode = {
    id: 'lm_head',
    kind: 'lm_head',
    label: 'Linear output layer',
    sublabel: `Unembeds ${h} → ${v.toLocaleString()} logits`,
    badge: 'HEAD',
    tensors: headTs.map(tref),
    params: sumParams(headTs) || (h * v),
    in_dim: h,
    out_dim: v,
    shape_str: headTs[0]?.shape ? getShapeStr(headTs[0].shape) : `[${h} × ${v}]`,
    dtype: headTs[0]?.dtype || null,
    is_expandable: true,
    default_detail_level: 1,
    description: 'Projects the final hidden representation into logits over the entire vocabulary.',
    evidence: headTs[0]?.name || (embTs.length > 0 ? 'Tied with embed_tokens' : 'lm_head.weight'),
  }

  // Root Model Node
  return {
    id: 'model_root',
    kind: 'model',
    label: spec.name,
    badge: spec.architecture,
    is_expandable: true,
    default_detail_level: 1,
    children: [
      embNode,
      layersContainer,
      finNormNode,
      lmHeadNode,
    ],
    details: {
      total_params: spec.total_params,
      active_params: spec.active_params,
      num_layers: spec.num_layers,
      hidden_size: spec.hidden_size,
      vocab_size: spec.vocab_size,
      context_length: spec.context_length,
      attention: spec.attention,
      moe: spec.moe,
    },
  }
}
