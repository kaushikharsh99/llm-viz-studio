import type { TensorInfo } from './types'

const PATTERNS: [RegExp, string, string][] = [
  [/^(?:model\.)?(?:embed_tokens|wte)\.weight$/, 'embedding', 'embedding'],
  [/^(?:lm_head|model\.lm_head)\.weight$/, 'lm_head', 'lm_head'],
  [/^model\.norm\.weight$/, 'final_norm', 'norm'],
  [/^model\.layers\.(?<layer>\d+)\.input_layernorm\.weight$/, 'input_norm', 'norm'],
  [/^model\.layers\.(?<layer>\d+)\.post_attention_layernorm\.weight$/, 'post_attn_norm', 'norm'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)$/, 'attention', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_a_proj\.weight$/, 'attention_q_a', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.q_b_proj\.weight$/, 'attention_q_b', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.(q|kv)_a(_layernorm|_proj).*/, 'attention_kv_a', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.kv_b_proj\.weight$/, 'attention_kv_b', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.o_proj\.weight$/, 'attention_o', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.(q|k|v|o)_norm\.weight$/, 'attention_qk_norm', 'norm'],
  [/^model\.layers\.(?<layer>\d+)\.self_attn\.qkv_proj\.weight$/, 'attention_qkv_fused', 'attention'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.(gate|up|down)_proj\.weight$/, 'mlp', 'mlp'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.gate_up_proj\.weight$/, 'mlp_gate_up_fused', 'mlp'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.(gate|up|down)_proj\.weight$/, 'expert', 'moe_expert'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$/, 'shared_expert', 'moe_shared'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.(gate|router)\.weight$/, 'router', 'moe_router'],
  [/^model\.layers\.(?<layer>\d+)\.mlp\.experts\.gate_up_proj.*/, 'expert_gate_up_fused', 'moe_expert'],
]

const LAYER_RE = /^model\.layers\.(?<layer>\d+)\./

export function mapTensor(name: string): [number | null, string, string, number | null] {
  for (const [pat, module, category] of PATTERNS) {
    const m = pat.exec(name)
    if (m) {
      const groups = m.groups || {}
      const layer = groups.layer !== undefined ? parseInt(groups.layer, 10) : null
      const expert = groups.expert !== undefined ? parseInt(groups.expert, 10) : null
      return [layer, module, category, expert]
    }
  }
  const m = LAYER_RE.exec(name)
  const layer = m && m.groups?.layer !== undefined ? parseInt(m.groups.layer, 10) : null
  if (name.includes('weight_scale_inv')) {
    return [layer, 'quant_scale', 'quant_scale', null]
  }
  return [layer, 'unknown', 'unknown', null]
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
