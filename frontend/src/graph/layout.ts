import ELK from 'elkjs/lib/elk.bundled.js'
import type { ArchGraph, GraphNode, GraphEdge } from './types'

const elk = new ELK()

export interface LayoutedGraph {
  nodes: (GraphNode & { x: number; y: number; width: number; height: number })[]
  edges: GraphEdge[]
  width: number
  height: number
}

function getNodeDimensions(node: GraphNode): { width: number; height: number } {
  switch (node.kind) {
    case 'residual':
      return { width: 170, height: 48 }
    case 'io':
      return { width: 220, height: 52 }
    case 'norm':
      return { width: 250, height: 68 }
    case 'math_op':
    case 'attention_op':
      return { width: 250, height: 64 }
    case 'layer_group':
      return { width: 330, height: 80 }
    case 'attention_block':
    case 'ffn_block':
    case 'moe_block':
      return { width: 310, height: 88 }
    case 'router':
    case 'expert':
      return { width: 290, height: 84 }
    case 'linear':
    case 'tensor':
    case 'embedding':
    case 'head':
    default:
      return { width: 280, height: 84 }
  }
}

/**
 * Compute hierarchical directed graph layout using ELK.js
 */
export async function layoutGraph(graph: ArchGraph): Promise<LayoutedGraph> {
  if (graph.nodes.length === 0) {
    return { nodes: [], edges: [], width: 800, height: 600 }
  }

  const elkNodes = graph.nodes.map(n => {
    const dim = getNodeDimensions(n)
    return {
      id: n.id,
      width: dim.width,
      height: dim.height,
    }
  })

  const elkEdges = graph.edges.map(e => ({
    id: e.id,
    sources: [e.src],
    targets: [e.dst],
  }))

  const elkGraph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '36',
      'elk.layered.spacing.nodeNodeBetweenLayers': '52',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    },
    children: elkNodes,
    edges: elkEdges,
  }

  try {
    const res = await elk.layout(elkGraph)
    const posMap = new Map<string, { x: number; y: number; width: number; height: number }>()

    for (const child of res.children || []) {
      posMap.set(child.id, {
        x: child.x || 0,
        y: child.y || 0,
        width: child.width || 260,
        height: child.height || 70,
      })
    }

    const layoutedNodes = graph.nodes.map(n => {
      const pos = posMap.get(n.id) || { x: 0, y: 0, width: 260, height: 70 }
      return {
        ...n,
        x: pos.x,
        y: pos.y,
        width: pos.width,
        height: pos.height,
      }
    })

    return {
      nodes: layoutedNodes,
      edges: graph.edges,
      width: (res as any).width || 1000,
      height: (res as any).height || 800,
    }
  } catch {
    // Fallback if ELK layout throws
    let currentY = 0
    const fallbackNodes = graph.nodes.map((n, i) => {
      const dim = getNodeDimensions(n)
      const node = {
        ...n,
        x: 200,
        y: currentY,
        width: dim.width,
        height: dim.height,
      }
      currentY += dim.height + 40
      return node
    })
    return {
      nodes: fallbackNodes,
      edges: graph.edges,
      width: 800,
      height: currentY + 100,
    }
  }
}
