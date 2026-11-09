import type { TensorRef, MetadataCoverage } from '../../graph/types'

export type DetailPreset = 1 | 2 | 3

export type SemanticNodeKind =
  | 'model'
  | 'sample_input'
  | 'tokenized_text'
  | 'tokens_io'
  | 'embedding_group'
  | 'embedding'
  | 'pos_embedding'
  | 'layers_container'
  | 'layer_block'
  | 'residual_add_1'
  | 'residual_add_2'
  | 'norm_1'
  | 'norm_2'
  | 'attention'
  | 'attention_internals'
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'q_norm'
  | 'k_norm'
  | 'qk_norm'
  | 'rope_op'
  | 'attn_scores'
  | 'softmax_op'
  | 'attn_context'
  | 'mla_latent'
  | 'ffn'
  | 'ffn_internals'
  | 'gate_proj'
  | 'up_proj'
  | 'act_op'
  | 'hadamard_op'
  | 'down_proj'
  | 'moe'
  | 'moe_internals'
  | 'moe_router'
  | 'moe_expert_group'
  | 'moe_expert'
  | 'moe_shared'
  | 'final_norm'
  | 'lm_head'
  | 'output_logits'

export interface SemanticNode {
  id: string
  kind: SemanticNodeKind
  label: string
  sublabel?: string
  badge?: string
  parent?: string | null
  children?: SemanticNode[]
  tensors?: TensorRef[]
  params?: number | null
  in_dim?: number | string | null
  out_dim?: number | string | null
  shape_str?: string | null
  dtype?: string | null
  is_expandable: boolean
  default_detail_level: DetailPreset
  description?: string
  evidence?: string
  details?: Record<string, any>
  layerIndex?: number
  expertIndex?: number
}

export interface WorkspaceState {
  preset: DetailPreset
  expandedIds: Set<string>
  collapsedIds: Set<string>
  selectedLayer: number
  selectedNodeId: string | null
  focusedNodeId: string | null // For Focus Mode
}

export interface LayoutedNode extends Omit<SemanticNode, 'children'> {
  x: number
  y: number
  width: number
  height: number
  isExpanded: boolean
  children?: LayoutedNode[]
  // Layout metadata
  cx: number
  topY: number
  bottomY: number
  leftX: number
  rightX: number
}

export interface DataflowEdge {
  fromId: string
  toId: string
  fromX: number
  fromY: number
  toX: number
  toY: number
  label?: string
  badge?: string
  style?: 'solid' | 'dashed'
}

export interface DiagramLayout {
  width: number
  height: number
  viewBox: string
  rootNodes: LayoutedNode[]
  allNodes: Map<string, LayoutedNode>
  residualRails: {
    startCx: number
    startY: number
    railX: number
    endY: number
    endCx: number
  }[]
  internalEdges?: DataflowEdge[]
  callouts: {
    x: number
    y: number
    lines: [string, string][][]
    targetX: number
    targetY: number
  }[]
  outerContainer?: {
    x: number
    y: number
    w: number
    h: number
  }
  repeatedBlockContainer?: {
    x: number
    y: number
    w: number
    h: number
    numLayers: number
  }
}
