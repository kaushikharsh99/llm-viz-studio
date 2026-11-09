import { describe, it, expect } from 'vitest'
import { mapTensor } from './graph/mapping'

describe('Module Mapping Precedence and Recognition (Phase 2.2)', () => {
  it('maps embeddings and model-level components correctly', () => {
    expect(mapTensor('model.embed_tokens.weight')).toEqual([null, 'embedding', 'embedding', null, 'Token Embeddings'])
    expect(mapTensor('model.norm.weight')).toEqual([null, 'final_norm', 'norm', null, 'Final Normalization'])
    expect(mapTensor('lm_head.weight')).toEqual([null, 'lm_head', 'lm_head', null, 'LM Head'])
  })

  it('preserves distinct attention projections without swallowing o_proj', () => {
    const [layerQ, modQ, catQ] = mapTensor('model.layers.5.self_attn.q_proj.weight')
    expect(layerQ).toBe(5)
    expect(modQ).toBe('q_proj')
    expect(catQ).toBe('attention')

    const [layerK, modK, catK] = mapTensor('model.layers.5.self_attn.k_proj.weight')
    expect(layerK).toBe(5)
    expect(modK).toBe('k_proj')
    expect(catK).toBe('attention')

    const [layerV, modV, catV] = mapTensor('model.layers.5.self_attn.v_proj.weight')
    expect(layerV).toBe(5)
    expect(modV).toBe('v_proj')
    expect(catV).toBe('attention')

    // Critical precedence test: o_proj must map to o_proj, NOT generic attention
    const [layerO, modO, catO] = mapTensor('model.layers.5.self_attn.o_proj.weight')
    expect(layerO).toBe(5)
    expect(modO).toBe('o_proj')
    expect(catO).toBe('attention')
  })

  it('maps attention normalization (q_norm and k_norm)', () => {
    const [, modQn, catQn] = mapTensor('model.layers.0.self_attn.q_norm.weight')
    expect(modQn).toBe('q_norm')
    expect(catQn).toBe('norm')

    const [, modKn, catKn] = mapTensor('model.layers.0.self_attn.k_norm.weight')
    expect(modKn).toBe('k_norm')
    expect(catKn).toBe('norm')
  })

  it('maps layer-level normalizations', () => {
    const [l1, modNorm1] = mapTensor('model.layers.3.input_layernorm.weight')
    expect(l1).toBe(3)
    expect(modNorm1).toBe('input_layernorm')

    const [l2, modNorm2] = mapTensor('model.layers.3.post_attention_layernorm.weight')
    expect(l2).toBe(3)
    expect(modNorm2).toBe('post_attention_layernorm')
  })

  it('maps dense FFN projections distinctly', () => {
    expect(mapTensor('model.layers.2.mlp.gate_proj.weight')[1]).toBe('gate_proj')
    expect(mapTensor('model.layers.2.mlp.up_proj.weight')[1]).toBe('up_proj')
    expect(mapTensor('model.layers.2.mlp.down_proj.weight')[1]).toBe('down_proj')
    expect(mapTensor('model.layers.2.mlp.gate_up_proj.weight')[1]).toBe('mlp_gate_up_fused')
  })

  it('maps MoE router, shared experts, and routed experts with index', () => {
    // Router
    const [rLayer, rMod, rCat] = mapTensor('model.layers.7.mlp.gate.weight')
    expect(rLayer).toBe(7)
    expect(rMod).toBe('moe_router')
    expect(rCat).toBe('moe')

    // Shared experts
    const [, sGate] = mapTensor('model.layers.7.mlp.shared_experts.gate_proj.weight')
    expect(sGate).toBe('moe_shared_gate')
    const [, sUp] = mapTensor('model.layers.7.mlp.shared_experts.up_proj.weight')
    expect(sUp).toBe('moe_shared_up')
    const [, sDown] = mapTensor('model.layers.7.mlp.shared_experts.down_proj.weight')
    expect(sDown).toBe('moe_shared_down')

    // Routed individual expert
    const [expLayer, expMod, expCat, expIdx] = mapTensor('model.layers.12.mlp.experts.23.gate_proj.weight')
    expect(expLayer).toBe(12)
    expect(expMod).toBe('moe_expert_gate')
    expect(expCat).toBe('moe')
    expect(expIdx).toBe(23)
  })

  it('maps DeepSeek Multi-Head Latent Attention (MLA) tensors', () => {
    expect(mapTensor('model.layers.4.self_attn.q_a_proj.weight')[1]).toBe('attention_q_a')
    expect(mapTensor('model.layers.4.self_attn.q_b_proj.weight')[1]).toBe('attention_q_b')
    expect(mapTensor('model.layers.4.self_attn.kv_a_proj_with_mqa.weight')[1]).toBe('attention_kv_a')
    expect(mapTensor('model.layers.4.self_attn.kv_b_proj.weight')[1]).toBe('attention_kv_b')
  })

  it('handles unknown tensors and quantization scale fallbacks cleanly', () => {
    const [layer, mod, cat] = mapTensor('model.layers.1.self_attn.unknown_weight')
    expect(layer).toBe(1)
    expect(mod).toBe('unknown')
    expect(cat).toBe('unknown')

    const [qLayer, qMod, qCat] = mapTensor('model.layers.1.self_attn.q_proj.weight_scale_inv')
    expect(qLayer).toBe(1)
    expect(qMod).toBe('quant_scale')
    expect(qCat).toBe('quant_scale')
  })
})
