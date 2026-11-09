export interface TensorRef {
  name: string
  shape: number[]
  dtype: string
  numel: number
  bytes: number
  shard: string
}

export interface GraphNode {
  id: string
  label: string
  kind: string
  level: number
  parent?: string | null
  layer?: number | null
  params: number
  dtype?: string | null
  in_dim?: number | null
  out_dim?: number | null
  confidence: string
  evidence: string
  tensors: TensorRef[]
  collapsed_count?: number | null
}

export interface GraphEdge {
  src: string
  dst: string
  kind: 'data' | 'residual'
  confidence: string
}

export interface ArchGraph {
  model_id: string
  model_type: string
  architecture: string
  confidence: string
  metadata_source: 'safetensors' | 'config_only'
  metadata_note?: string | null
  total_params: number
  verified_params: number
  tied_note?: string | null
  nodes: GraphNode[]
  edges: GraphEdge[]
  layers: number[]
  unknowns: string[]
}

export interface TensorInfo extends TensorRef {
  layer?: number | null
  module: string
  category: string
  expert?: number | null
}
