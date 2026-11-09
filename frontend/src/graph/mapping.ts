import type { TensorInfo } from './types'

export interface MappingResult {
  layer: number | null
  module: string
  category: string
  expert: number | null
  submodule?: string
}

// Ordered rules: First match wins. Specific rules MUST precede generic ones!
const PATTERNS: [RegExp, string, string, string?][] = [
  // 1. Embeddings & Model-Level Heads / Norms
  [/^(?:model\.)?(?:embed_tokens|wte)\.weight$/, 'embedding', 'embedding', 'Token Embeddings'],
  [/^(?:lm_head|model\.lm_head)\.weight$/, 'lm_head', 'lm_head', 'LM Head'],
  [/^model\.(?:norm|final_layernorm)\.weight$/, 'final_norm', 'norm', 'Final Normalization'],

  // 2. Layer-Level Normalizations
  [/^model\.layers\.(?<layer>\d+)\.(?:input_layernorm|norm1)\.weight$/, 'input_layernorm', 'norm', 'Input LayerNorm / RMSNorm'],
  [/^model\.layers\.(?<layer>\d+)\.(?:post_attention_layernorm|norm2)\.weight$/, 'post_attention_layernorm', 'norm', 'Post-Attention LayerNorm / RMSNorm'],

  // 3. Attention Head Norms (e.g. Qwen, Olmo2, Gemma3)
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_norm\.weight$/, 'q_norm', 'norm', 'Query Normalization'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.k_norm\.weight$/, 'k_norm', 'norm', 'Key Normalization'],

  // 4. Specific Attention Projections (Must be listed individually to preserve distinction!)
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_proj\.(?:weight|bias)$/, 'q_proj', 'attention', 'Query Projection'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.k_proj\.(?:weight|bias)$/, 'k_proj', 'attention', 'Key Projection'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.v_proj\.(?:weight|bias)$/, 'v_proj', 'attention', 'Value Projection'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.o_proj\.(?:weight|bias)$/, 'o_proj', 'attention', 'Output Projection'],

  // 5. DeepSeek Multi-Head Latent Attention (MLA)
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_a_proj\.weight$/, 'attention_q_a', 'attention', 'Q Latent Down-Projection (q_a)'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_b_proj\.weight$/, 'attention_q_b', 'attention', 'Q Latent Up-Projection (q_b)'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.(?:q|kv)_a(?:_layernorm|_proj).*$/, 'attention_kv_a', 'attention', 'KV Latent Down-Projection (kv_a)'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.kv_b_proj\.weight$/, 'attention_kv_b', 'attention', 'KV Latent Up-Projection (kv_b)'],

  // 6. Fused Attention Projections
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.(?:qkv_proj|W_pack)\.(?:weight|bias)$/, 'attention_qkv_fused', 'attention', 'Fused QKV Projection'],

  // 7. Dense FFN Projections
  [/^model\.layers\.(?<layer>\d+)\.mlp\.gate_proj\.(?:weight|bias)$/, 'gate_proj', 'mlp', 'Gate Projection'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.up_proj\.(?:weight|bias)$/, 'up_proj', 'mlp', 'Up Projection'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.down_proj\.(?:weight|bias)$/, 'down_proj', 'mlp', 'Down Projection'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.gate_up_proj\.(?:weight|bias)$/, 'mlp_gate_up_fused', 'mlp', 'Fused Gate-Up Projection'],

  // 8. MoE Router
  [/^model\.layers\.(?<layer>\d+)\.mlp\.(?:gate|router)\.weight$/, 'moe_router', 'moe', 'MoE Gating Router'],

  // 9. MoE Shared Experts (e.g. DeepSeek, Qwen MoE)
  [/^model\.layers\.(?<layer>\d+)\.mlp\.shared_experts\.gate_proj\.weight$/, 'moe_shared_gate', 'moe', 'Shared Expert Gate'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.shared_experts\.up_proj\.weight$/, 'moe_shared_up', 'moe', 'Shared Expert Up'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.shared_experts\.down_proj\.weight$/, 'moe_shared_down', 'moe', 'Shared Expert Down'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.shared_experts\..*$/, 'moe_shared', 'moe', 'Shared Expert'],

  // 10. MoE Routed Experts
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.gate_proj\.weight$/, 'moe_expert_gate', 'moe', 'Routed Expert Gate'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.up_proj\.weight$/, 'moe_expert_up', 'moe', 'Routed Expert Up'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.down_proj\.weight$/, 'moe_expert_down', 'moe', 'Routed Expert Down'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.gate_up_proj.*$/, 'moe_expert_gate_up_fused', 'moe', 'Routed Expert Fused Gate-Up'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\..*$/, 'moe_expert', 'moe', 'Routed Expert'],
]

const LAYER_FALLBACK_RE = /^model\.layers\.(?<layer>\d+)\./

export function mapTensor(name: string): [number | null, string, string, number | null, string?] {
  for (const [pat, module, category, submodule] of PATTERNS) {
    const m = pat.exec(name)
    if (m) {
      const groups = m.groups || {}
      const layer = groups.layer !== undefined ? parseInt(groups.layer, 10) : null
      const expert = groups.expert !== undefined ? parseInt(groups.expert, 10) : null
      return [layer, module, category, expert, submodule]
    }
  }

  // Fallback check
  const m = LAYER_FALLBACK_RE.exec(name)
  const layer = m && m.groups?.layer !== undefined ? parseInt(m.groups.layer, 10) : null
  if (name.includes('weight_scale_inv') || name.includes('.scales')) {
    return [layer, 'quant_scale', 'quant_scale', null, 'Quantization Scale']
  }
  return [layer, 'unknown', 'unknown', null, 'Unmapped Tensor']
}

export function annotate(tensors: TensorInfo[]): TensorInfo[] {
  for (const t of tensors) {
    const [layer, module, category, expert] = mapTensor(t.name)
    t.layer = layer
    t.module = module
    t.category = category
    t.expert = expert
  }
  return tensors
}
