export type GraphLevel = 0 | 1 | 2 | 3

export type NodeKind =
  | 'embedding'
  | 'layer_group'
  | 'layer_block'
  | 'attention_block'
  | 'ffn_block'
  | 'moe_block'
  | 'linear'
  | 'norm'
  | 'residual'
  | 'math_op'
  | 'attention_op'
  | 'router'
  | 'expert'
  | 'head'
  | 'tensor'
  | 'io'
  | 'unknown'

export type ConfidenceLevel = 'verified' | 'inferred' | 'partial' | 'config' | 'unknown'

export interface TensorRef {
  name: string
  shape: number[] | null
  dtype: string | null
  numel: number | null
  bytes: number | null
  shard: string
}

export interface TensorInfo extends TensorRef {
  layer: number | null
  module: string
  category: string
  expert: number | null
  is_verified_shape: boolean
}

export interface MetadataCoverage {
  totalNamedTensors: number
  tensorsWithVerifiedShapes: number
  tensorsWithoutShapeMetadata: number
  shardsInspected: number
  shardsNotInspected: number
  confidence: ConfidenceLevel
  modelRevision: string
}

export interface GraphNode {
  id: string
  label: string
  kind: NodeKind
  level: GraphLevel
  parent?: string | null
  layer?: number | null
  module?: string
  submodule?: string
  params: number | null
  dtype?: string | null
  in_dim?: number | string | null
  out_dim?: number | string | null
  shape_str?: string | null
  confidence: ConfidenceLevel
  evidence: string
  tensors: TensorRef[]
  collapsed_count?: number | null
  is_expanded?: boolean
  details?: Record<string, any>
  // layout coordinates populated by ELK
  x?: number
  y?: number
  width?: number
  height?: number
}

export interface GraphEdge {
  id: string
  src: string
  dst: string
  srcHandle?: string
  dstHandle?: string
  label?: string
  kind: 'data' | 'residual' | 'weight' | 'control'
  confidence: ConfidenceLevel
}

export interface ArchGraph {
  model_id: string
  model_type: string
  architecture: string
  level: GraphLevel
  active_layer: number | null
  confidence: ConfidenceLevel
  metadata_source: 'safetensors' | 'config_only'
  metadata_note?: string | null
  coverage?: MetadataCoverage
  total_params: number
  verified_params: number | null
  tied_note?: string | null
  nodes: GraphNode[]
  edges: GraphEdge[]
  layers: number[]
  unknowns: string[]
}

export interface ModelHierarchyItem {
  id: string
  label: string
  kind: string
  level: number
  layer?: number | null
  nodeId: string
  params?: number | null
  children?: ModelHierarchyItem[]
}
