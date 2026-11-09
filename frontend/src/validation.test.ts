import { describe, it, expect } from 'vitest'
import { parseConfig } from './engine/spec'
import { buildSemanticTree } from './engine/semantic/tree'
import { computeDiagramLayout } from './engine/semantic/layout'
import { renderSemanticDiagram } from './engine/semantic/render'
import type { WorkspaceState } from './engine/semantic/types'
import { generateDiagramSvg } from './shared/api/api'

// Fixtures for 4 key model families
const smolLM2Fixture = {
  architectures: ['LlamaForCausalLM'],
  model_type: 'llama',
  hidden_size: 2048,
  num_hidden_layers: 24,
  num_attention_heads: 32,
  num_key_value_heads: 32,
  intermediate_size: 5632,
  vocab_size: 49152,
  max_position_embeddings: 8192,
  hidden_act: 'silu',
  rope_theta: 100000.0,
}

const qwen25Fixture = {
  architectures: ['Qwen2ForCausalLM'],
  model_type: 'qwen2',
  hidden_size: 1536,
  num_hidden_layers: 28,
  num_attention_heads: 12,
  num_key_value_heads: 2,
  intermediate_size: 8960,
  vocab_size: 151936,
  max_position_embeddings: 32768,
  hidden_act: 'silu',
  rope_theta: 1000000.0,
}

const qwen3DenseFixture = {
  architectures: ['Qwen3ForCausalLM'],
  model_type: 'qwen3',
  hidden_size: 2048,
  num_hidden_layers: 32,
  num_attention_heads: 16,
  num_key_value_heads: 4,
  intermediate_size: 11008,
  vocab_size: 151936,
  max_position_embeddings: 32768,
  hidden_act: 'silu',
  rope_theta: 1000000.0,
}

const deepseekV3Fixture = {
  architectures: ['DeepseekV3ForCausalLM'],
  model_type: 'deepseek_v3',
  hidden_size: 2048,
  num_hidden_layers: 61,
  num_attention_heads: 16,
  num_key_value_heads: 16,
  intermediate_size: 10944,
  num_experts: 256,
  num_experts_per_tok: 8,
  vocab_size: 129280,
  max_position_embeddings: 163840,
  hidden_act: 'silu',
}

describe('Phase 12 Comprehensive Validation across 4 Model Families', () => {
  const models = [
    { name: 'SmolLM2-1.7B (Llama-style)', cfg: smolLM2Fixture },
    { name: 'Qwen 2.5 (Dense GQA)', cfg: qwen25Fixture },
    { name: 'Qwen 3 (Dense)', cfg: qwen3DenseFixture },
    { name: 'DeepSeek-V3 (MLA + MoE)', cfg: deepseekV3Fixture },
  ]

  for (const { name, cfg } of models) {
    describe(`Model: ${name}`, () => {
      const spec = parseConfig(cfg, name)
      const tree = buildSemanticTree(spec, [])

      it('generates valid Preset 1 (Overview) diagram', () => {
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
        expect(svg).toContain('</svg>')
        expect(svg).toContain('Linear output layer')
        expect(svg).toContain('Token embedding layer')
        expect(layout.height).toBeGreaterThan(600)
        expect(layout.width).toBeGreaterThan(800)
      })

      it('generates valid Preset 2 (Module Detail) diagram with expanded internal modules', () => {
        const state: WorkspaceState = {
          preset: 2,
          expandedIds: new Set(),
          collapsedIds: new Set(),
          selectedLayer: 0,
          selectedNodeId: null,
          focusedNodeId: null,
        }
        const layout = computeDiagramLayout(tree, spec, state)
        const svg = renderSemanticDiagram(layout, spec, state)

        expect(svg).toContain('Self-Attention Mechanism (Expanded)')
        expect(layout.height).toBeGreaterThan(1000)
      })

      it('generates valid Preset 3 (Deep Detail) diagram with operations and shapes', () => {
        const state: WorkspaceState = {
          preset: 3,
          expandedIds: new Set(),
          collapsedIds: new Set(),
          selectedLayer: 0,
          selectedNodeId: null,
          focusedNodeId: null,
        }
        const layout = computeDiagramLayout(tree, spec, state)
        const svg = renderSemanticDiagram(layout, spec, state)

        expect(svg).toContain('<svg')
        expect(layout.height).toBeGreaterThan(1200)
      })

      it('allows custom manual Attention expansion from Preset 1', () => {
        const state: WorkspaceState = {
          preset: 1,
          expandedIds: new Set(['layer_0_attention']),
          collapsedIds: new Set(),
          selectedLayer: 0,
          selectedNodeId: 'layer_0_attention',
          focusedNodeId: null,
        }
        const layout = computeDiagramLayout(tree, spec, state)
        const svg = renderSemanticDiagram(layout, spec, state)

        expect(svg).toContain('Self-Attention Mechanism (Expanded)')
        // FFN must remain collapsed in this custom state!
        expect(svg).not.toContain('Feed-Forward (Expanded)')
      })

      it('allows custom manual FFN expansion while keeping Attention collapsed', () => {
        const ffnId = spec.moe ? 'layer_0_moe' : 'layer_0_ffn'
        const state: WorkspaceState = {
          preset: 1,
          expandedIds: new Set([ffnId]),
          collapsedIds: new Set(),
          selectedLayer: 0,
          selectedNodeId: ffnId,
          focusedNodeId: null,
        }
        const layout = computeDiagramLayout(tree, spec, state)
        const svg = renderSemanticDiagram(layout, spec, state)

        // Attention should be collapsed in this custom state
        expect(svg).not.toContain('Self-Attention Mechanism (Expanded)')
        expect(svg).toContain('Expanded)')
      })

      it('renders isolated Focus Mode view cleanly', () => {
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
        expect(layout.width).toBe(900)
      })
    })
  }

  it('generates all 4 export modes successfully', async () => {
    const spec = parseConfig(qwen25Fixture, 'qwen-export-test')
    const tree = buildSemanticTree(spec, [])

    const p1Svg = renderSemanticDiagram(computeDiagramLayout(tree, spec, {
      preset: 1, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null
    }), spec, { preset: 1, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null }, { interactive: false })

    const p2Svg = renderSemanticDiagram(computeDiagramLayout(tree, spec, {
      preset: 2, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null
    }), spec, { preset: 2, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null }, { interactive: false })

    const p3Svg = renderSemanticDiagram(computeDiagramLayout(tree, spec, {
      preset: 3, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null
    }), spec, { preset: 3, expandedIds: new Set(), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null }, { interactive: false })

    const customWorkspaceSvg = renderSemanticDiagram(computeDiagramLayout(tree, spec, {
      preset: 1, expandedIds: new Set(['layer_0_attention']), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null
    }), spec, { preset: 1, expandedIds: new Set(['layer_0_attention']), collapsedIds: new Set(), selectedLayer: 0, selectedNodeId: null, focusedNodeId: null }, { interactive: false })

    for (const svg of [p1Svg, p2Svg, p3Svg, customWorkspaceSvg]) {
      expect(svg).toContain('<svg')
      expect(svg).toContain('</svg>')
      expect(svg).not.toContain('undefined')
      expect(svg).not.toContain('NaN')
    }
  })
})
