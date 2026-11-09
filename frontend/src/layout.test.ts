import { describe, it, expect } from 'vitest'
import { buildGraph } from './graph/topology'
import { layoutGraph } from './graph/layout'
import { parseConfig } from './engine/spec'

const fixture = {
  architectures: ['Qwen2ForCausalLM'],
  model_type: 'qwen2',
  hidden_size: 1024,
  num_hidden_layers: 12,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  intermediate_size: 2816,
  vocab_size: 151936,
}

describe('ELK.js Automatic Layout Engine', () => {
  it('computes valid (x, y) coordinates for all nodes in Level 0', async () => {
    const spec = parseConfig(fixture, 'test-model')
    const graph = buildGraph('test-model', spec, [], false, null, null, 0)
    const layout = await layoutGraph(graph)

    expect(layout.nodes.length).toBe(graph.nodes.length)
    expect(layout.width).toBeGreaterThan(0)
    expect(layout.height).toBeGreaterThan(0)

    for (const node of layout.nodes) {
      expect(typeof node.x).toBe('number')
      expect(typeof node.y).toBe('number')
      expect(node.width).toBeGreaterThan(0)
      expect(node.height).toBeGreaterThan(0)
    }

    // Embed should be placed above LM head in top-to-bottom layout
    const embed = layout.nodes.find(n => n.id === 'embed')!
    const head = layout.nodes.find(n => n.id === 'lm_head')!
    expect(head.y).toBeGreaterThan(embed.y)
  })

  it('computes valid coordinates for Level 2 module internals', async () => {
    const spec = parseConfig(fixture, 'test-model')
    const graph = buildGraph('test-model', spec, [], false, null, 0, 2)
    const layout = await layoutGraph(graph)

    expect(layout.nodes.length).toBeGreaterThan(10)
    const qProj = layout.nodes.find(n => n.id === 'l0_q_proj')!
    const res1 = layout.nodes.find(n => n.id === 'l0_res1')!
    expect(res1.y).toBeGreaterThan(qProj.y)
  })
})
