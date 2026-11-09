import { describe, it, expect } from 'vitest'
import { parseConfig } from './engine/spec'
import { buildSemanticTree } from './engine/semantic/tree'
import { computeDiagramLayout } from './engine/semantic/layout'
import { renderSemanticDiagram } from './engine/semantic/render'
import type { WorkspaceState } from './engine/semantic/types'

const qwenDenseFixture = {
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
  architectures: ['DeepseekV3ForCausalLM'],
  model_type: 'deepseek_v3',
  hidden_size: 2048,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 4,
  intermediate_size: 5632,
  num_experts: 64,
  num_experts_per_tok: 8,
  vocab_size: 129280,
  max_position_embeddings: 32768,
}

describe('Semantic Architecture Diagram Engine', () => {
  it('builds a semantic tree for dense model with all components', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])

    expect(tree.id).toBe('model_root')
    const childKinds = tree.children?.map(c => c.kind)
    expect(childKinds).toContain('embedding')
    expect(childKinds).toContain('layers_container')
    expect(childKinds).toContain('final_norm')
    expect(childKinds).toContain('lm_head')

    const layersContainer = tree.children?.find(c => c.kind === 'layers_container')
    const block = layersContainer?.children?.[0]
    expect(block?.kind).toBe('layer_block')

    const blockKinds = block?.children?.map(c => c.kind)
    expect(blockKinds).toContain('norm_1')
    expect(blockKinds).toContain('attention')
    expect(blockKinds).toContain('residual_add_1')
    expect(blockKinds).toContain('norm_2')
    expect(blockKinds).toContain('ffn')
    expect(blockKinds).toContain('residual_add_2')
  })

  it('renders Preset 1 (Overview) SVG preserving Raschka design tokens', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])
    const state: WorkspaceState = {
      preset: 1,
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: 0,
      selectedNodeId: null,
      focusedNodeId: null,
    }
    const layout = computeDiagramLayout(tree, spec, state)
    const svg = renderSemanticDiagram(layout, spec, state)

    expect(svg).toContain('<svg')
    expect(svg).toContain('grouped-query')
    expect(svg).toContain('Feed forward')
    expect(svg).toContain('RMSNorm 1')
    expect(svg).toContain('Linear output layer')
    expect(svg).toContain('Token embedding layer')
    expect(svg).toContain('24 ×') // Layer count
  })

  it('dynamically expands Attention in Preset 2 without overlapping other nodes', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])
    const state: WorkspaceState = {
      preset: 2, // Module detail
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: 0,
      selectedNodeId: null,
      focusedNodeId: null,
    }
    const layout = computeDiagramLayout(tree, spec, state)
    const svg = renderSemanticDiagram(layout, spec, state)

    expect(svg).toContain('Self-Attention Mechanism (Expanded)')
    expect(svg).toContain('Query Projection')
    expect(svg).toContain('Key Projection')
    expect(svg).toContain('Value Projection')
    expect(svg).toContain('Output Projection')

    // Height of expanded layout must be greater than Preset 1
    const p1State: WorkspaceState = { ...state, preset: 1 }
    const p1Layout = computeDiagramLayout(tree, spec, p1State)
    expect(layout.height).toBeGreaterThan(p1Layout.height)
  })

  it('supports Focus Mode on Attention subsystem', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])
    const state: WorkspaceState = {
      preset: 2,
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: 0,
      selectedNodeId: null,
      focusedNodeId: 'layer_0_attention',
    }
    const layout = computeDiagramLayout(tree, spec, state)
    const svg = renderSemanticDiagram(layout, spec, state)

    expect(svg).toContain('Isolated Component View')
    expect(svg).toContain('Query Projection')
    expect(layout.width).toBe(900)
  })

  it('renders Preset 3 (Deep operations & computational dataflow edges)', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])
    const state: WorkspaceState = {
      preset: 3, // Deep detail
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: 0,
      selectedNodeId: null,
      focusedNodeId: null,
    }
    const layout = computeDiagramLayout(tree, spec, state)
    const svg = renderSemanticDiagram(layout, spec, state)

    expect(layout.internalEdges).toBeDefined()
    expect(layout.internalEdges!.length).toBeGreaterThan(0)
    // Must contain deep attention operations
    expect(svg).toContain('Scaled Dot-Product')
    expect(svg).toContain('Causal Mask + Softmax')
    expect(svg).toContain('Value Aggregation')
    // Must contain deep FFN operations
    expect(svg).toContain('Elementwise Product')
    expect(svg).toContain('SILU Non-Linearity')
  })

  it('computes magnified side insets with projection rays and non-overlapping layout', () => {
    const spec = parseConfig(qwenDenseFixture, 'qwen-test')
    const tree = buildSemanticTree(spec, [])
    const state: WorkspaceState = {
      preset: 2, // Module Insets
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: 0,
      selectedNodeId: null,
      focusedNodeId: null,
    }
    const layout = computeDiagramLayout(tree, spec, state)

    // Side insets must be generated
    expect(layout.magnifiedInsets).toBeDefined()
    expect(layout.magnifiedInsets!.length).toBe(2) // Attention & FFN insets

    const attnInset = layout.magnifiedInsets!.find(i => i.kind === 'attention')
    const ffnInset = layout.magnifiedInsets!.find(i => i.kind === 'ffn')
    expect(attnInset).toBeDefined()
    expect(ffnInset).toBeDefined()

    // Bounding boxes in side insets must not overlap horizontally in the same row
    const qNode = attnInset!.children.find(c => c.kind === 'q_proj')
    const kNode = attnInset!.children.find(c => c.kind === 'k_proj')
    const vNode = attnInset!.children.find(c => c.kind === 'v_proj')
    expect(qNode).toBeDefined()
    expect(kNode).toBeDefined()
    expect(vNode).toBeDefined()

    // Q right edge must be strictly less than K left edge (zero horizontal overlap)
    expect(qNode!.rightX).toBeLessThan(kNode!.leftX)
    // K right edge must be strictly less than V left edge (zero horizontal overlap)
    expect(kNode!.rightX).toBeLessThan(vNode!.leftX)

    // SVG must render magnifying rays and translucent projection cone
    const svg = renderSemanticDiagram(layout, spec, state)
    expect(svg).toContain('<polygon points=')
    expect(svg).toContain('class="leader"')
    expect(svg).toContain('class="inset-accent"')
    expect(layout.width).toBe(1600)
  })
})
