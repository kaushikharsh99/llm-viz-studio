import { describe, it, expect } from 'vitest'
import { parseConfig } from './engine/spec'
import { renderModel } from './engine/render/block'
import { mapTensor } from './graph/mapping'
import { buildGraph } from './graph/topology'

// Sample Qwen fixture
const qwenFixture = {
  architectures: ['Qwen2ForCausalLM'],
  model_type: 'qwen2',
  hidden_size: 1024,
  num_hidden_layers: 24,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  intermediate_size: 2816,
  vocab_size: 151936,
  max_position_embeddings: 32768,
  hidden_act: 'silu',
  rope_theta: 1000000.0,
}

const moeFixture = {
  architectures: ['Qwen3MoeForCausalLM'],
  model_type: 'qwen3_moe',
  hidden_size: 2048,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 4,
  intermediate_size: 5632,
  num_experts: 64,
  num_experts_per_tok: 4,
  vocab_size: 151936,
  max_position_embeddings: 32768,
  hidden_act: 'silu',
}

describe('TypeScript engine validation', () => {
  it('parses dense config and estimates parameters', () => {
    const spec = parseConfig(qwenFixture, 'qwen-test')
    expect(spec.hidden_size).toBe(1024)
    expect(spec.num_layers).toBe(24)
    expect(spec.attention.kind).toBe('GQA')
    expect(spec.total_params).toBeGreaterThan(100_000_000)
    expect(spec.is_moe).toBe(false)
  })

  it('parses MoE config with active params < total params', () => {
    const spec = parseConfig(moeFixture, 'moe-test')
    expect(spec.moe).not.toBeNull()
    expect(spec.moe?.num_experts).toBe(64)
    expect(spec.total_params).toBeGreaterThan(spec.active_params)
  })

  it('renders colorful Raschka architecture diagram SVG', () => {
    const spec = parseConfig(qwenFixture, 'qwen-test')
    const svg = renderModel(spec)
    expect(svg).toContain('<svg')
    expect(svg).toContain('llmviz-studio-ff664f') // coral accent
    expect(svg).toContain('FeedForward')
  })

  it('maps tensor names to modules correctly', () => {
    expect(mapTensor('model.embed_tokens.weight')[1]).toBe('embedding')
    expect(mapTensor('model.layers.0.self_attn.q_proj.weight')[1]).toBe('q_proj')
    expect(mapTensor('model.layers.3.mlp.experts.17.gate_proj.weight')[1]).toBe('moe_expert_gate')
    expect(mapTensor('lm_head.weight')[1]).toBe('lm_head')
  })

  it('builds canonical graph structure across levels', () => {
    const spec = parseConfig(qwenFixture, 'qwen-test')
    // Level 0: Whole Model
    const g0 = buildGraph('qwen-test', spec, [], false, 'config only', null, 0)
    expect(g0.metadata_source).toBe('config_only')
    const nodeIds0 = g0.nodes.map(n => n.id)
    expect(nodeIds0).toContain('embed')
    expect(nodeIds0).toContain('layers_all')
    expect(nodeIds0).toContain('final_norm')
    expect(nodeIds0).toContain('lm_head')
    expect(nodeIds0).toContain('out_logits')

    // Level 1: Layer Overview
    const g1 = buildGraph('qwen-test', spec, [], false, null, 0, 1)
    const nodeIds1 = g1.nodes.map(n => n.id)
    expect(nodeIds1).toContain('l0_norm1')
    expect(nodeIds1).toContain('l0_attn')
    expect(nodeIds1).toContain('l0_res1')
    expect(nodeIds1).toContain('l0_norm2')
    expect(nodeIds1).toContain('l0_ffn')
    expect(nodeIds1).toContain('l0_res2')

    // Level 2: Module Internals
    const g2 = buildGraph('qwen-test', spec, [], false, null, 0, 2)
    const nodeIds2 = g2.nodes.map(n => n.id)
    expect(nodeIds2).toContain('l0_q_proj')
    expect(nodeIds2).toContain('l0_rope_q')
    expect(nodeIds2).toContain('l0_attn_scores')
    expect(nodeIds2).toContain('l0_softmax')
    expect(nodeIds2).toContain('l0_gate_proj')
    expect(nodeIds2).toContain('l0_silu')
    expect(nodeIds2).toContain('l0_hadamard')
    expect(nodeIds2).toContain('l0_down_proj')
  })
})
