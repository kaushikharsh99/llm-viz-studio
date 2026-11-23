import { SVG, escapeHtml } from '../render/svg'
import { textWidth } from '../render/widths'
import type { DiagramLayout, LayoutedNode } from './types'

import { orbitProjection, type DiagramCamera } from './orbit'

type Point = [number, number, number]
type Projected = { x: number; y: number; depth: number }

const COLORS = {
  weights: ['#aab7ee', '#6175bc', '#425588'],
  activation: ['#a2ded0', '#48a78f', '#26775e'],
  operation: ['#f5d7a5', '#d99b51', '#9d652b'],
}

function palette(node: LayoutedNode) {
  if (node.tensors?.length || node.kind.endsWith('_proj') || ['embedding', 'lm_head'].includes(node.kind)) {
    return COLORS.weights
  }
  if (['sample_input', 'tokenized_text', 'tokens_io', 'attn_context', 'output_logits'].includes(node.kind)) {
    return COLORS.activation
  }
  return COLORS.operation
}

export function visibleTensorNodes(layout: DiagramLayout): LayoutedNode[] {
  const inFlowOrder = (nodes: LayoutedNode[]) => nodes.slice().sort((a, b) => b.y - a.y || a.x - b.x)
  return inFlowOrder(layout.rootNodes).flatMap(node => {
    if (node.children?.length && !layout.repeatedBlockContainer) return inFlowOrder(node.children)
    const inset = layout.magnifiedInsets?.find(candidate => candidate.sourceNodeId === node.id)
    return inset ? [node, ...inFlowOrder(inset.children)] : [node]
  })
}

function volume(node: LayoutedNode, layoutHeight: number) {
  const residual = node.kind.startsWith('residual_add')
  const width = residual ? 28 : node.width * 0.76
  const height = residual ? 22 : node.height * 0.68
  const color = palette(node)
  const depth = residual ? 28 : color === COLORS.weights ? 104 : color === COLORS.activation ? 72 : 52
  const familyOffset = color === COLORS.weights ? 30 : color === COLORS.activation ? -34 : 0
  const z = familyOffset + (node.kind === 'q_proj' ? -58 : node.kind === 'v_proj' ? 58 : 0)
  return { node, center: [node.cx, layoutHeight - (node.y + node.height / 2), z] as Point, width, height, depth, color }
}

export function renderTensorScene(
  layout: DiagramLayout,
  camera: DiagramCamera,
  selectedId: string | null,
  viewportScale: number,
): string {
  const svg = new SVG()
  const yaw = camera.yaw * Math.PI / 180
  const centerX = layout.width / 2
  const cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw)
  const project = orbitProjection(camera, centerX, layout.height / 2)
  const nodes = visibleTensorNodes(layout).map(node => volume(node, layout.height))
  const byId = new Map(nodes.map(block => [block.node.id, block]))
  const labelSize = Math.max(17, 11 / viewportScale)
  const cellSize = 23

  const line = (from: Projected, to: Projected, cls: string) =>
    svg.line(from.x, from.y, to.x, to.y, cls)

  // Connect the original dataflow to the centers of the spatial volumes. The
  // volume faces cover the parts of the connections inside each block.
  const connect = (from: Point, to: Point, selected: boolean) => {
    const a = project(from), b = project(to)
    const middle = project([from[0], (from[1] + to[1]) / 2, from[2]])
    svg.path(`M ${a.x} ${a.y} C ${middle.x} ${middle.y}, ${b.x} ${middle.y}, ${b.x} ${b.y}`,
      selected ? 'connection connection-selected' : 'connection')
  }
  svg.raw('<g class="diagram-scene-3d">')
  if (layout.repeatedBlockContainer) {
    const spine = layout.rootNodes.slice().reverse()
    spine.slice(1).forEach((node, i) => {
      const from = byId.get(spine[i].id)!
      const to = byId.get(node.id)!
      connect(from.center, to.center, from.node.id === selectedId || node.id === selectedId)
    })
  }
  for (const rail of layout.residualRails) {
    const points = [
      project([rail.startCx, layout.height - rail.startY, 0]),
      project([rail.railX, layout.height - rail.startY, 0]),
      project([rail.railX, layout.height - rail.endY, 0]),
      project([rail.endCx, layout.height - rail.endY, 0]),
    ]
    svg.raw(`<polyline class="connection residual-connection" points="${points.map(p => `${p.x},${p.y}`).join(' ')}"/>`)
  }
  for (const edge of layout.internalEdges ?? []) {
    const from = byId.get(edge.fromId)
    const to = byId.get(edge.toId)
    if (from && to) connect(from.center, to.center, from.node.id === selectedId || to.node.id === selectedId)
  }
  for (const inset of layout.magnifiedInsets ?? []) {
    const source = byId.get(inset.sourceNodeId)
    if (!source) continue
    const inputs = inset.children.filter(child => !inset.edges.some(edge => edge.toId === child.id))
    for (const input of inputs) connect(source.center, byId.get(input.id)!.center, source.node.id === selectedId)
  }

  const blocks = nodes.map(block => ({ ...block, projected: project(block.center) }))
  blocks.sort((a, b) => a.projected.depth - b.projected.depth)
  for (const block of blocks) {
    const { node, center: [x, y, z], width, height, depth, color } = block
    const left = x - width / 2, right = x + width / 2
    const top = y - height / 2, bottom = y + height / 2
    const back = z - depth / 2, front = z + depth / 2
    const faces: { corners: [Point, Point, Point, Point]; axis: number }[] = [
      { corners: [[left, top, front], [right, top, front], [right, bottom, front], [left, bottom, front]], axis: 1 },
      { corners: [[right, top, back], [left, top, back], [left, bottom, back], [right, bottom, back]], axis: 1 },
      { corners: [[left, top, back], [right, top, back], [right, top, front], [left, top, front]], axis: 0 },
      { corners: [[left, bottom, front], [right, bottom, front], [right, bottom, back], [left, bottom, back]], axis: 0 },
      { corners: [[right, top, front], [right, top, back], [right, bottom, back], [right, bottom, front]], axis: 2 },
      { corners: [[left, top, back], [left, top, front], [left, bottom, front], [left, bottom, back]], axis: 2 },
    ]
    const selected = node.id === selectedId
    svg.raw(`<g class="interactive-node tensor-volume${selected ? ' node-selected' : ''}" data-node-id="${escapeHtml(node.id)}" role="button" tabindex="0" aria-label="${escapeHtml(node.label)}" aria-pressed="${selected}"${node.is_expandable ? ` aria-expanded="${node.isExpanded}"` : ''}>`)
    svg.raw(`<title>${escapeHtml(node.shape_str ? `${node.label} · ${node.shape_str}` : node.label)}</title>`)
    for (const { corners, axis } of faces) {
      const p = corners.map(project)
      const signedArea = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x)
      if (signedArea <= 0.01) continue
      svg.raw(`<polygon class="tensor-face" points="${p.map(v => `${v.x},${v.y}`).join(' ')}" fill="${color[axis]}"/>`)
      const interpolate = (a: Projected, b: Projected, t: number): Projected =>
        ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, depth: 0 })
      const xCells = Math.max(1, Math.round((axis === 2 ? depth : width) / cellSize))
      const yCells = Math.max(1, Math.round((axis === 0 ? depth : height) / cellSize))
      for (let i = 1; i < xCells; i++) line(interpolate(p[0], p[1], i / xCells), interpolate(p[3], p[2], i / xCells), 'tensor-cell')
      for (let i = 1; i < yCells; i++) line(interpolate(p[0], p[3], i / yCells), interpolate(p[1], p[2], i / yCells), 'tensor-cell')
    }
    svg.raw('</g>')
  }
  svg.raw('</g><g class="tensor-labels">')
  const rowHeight = labelSize * 1.65
  const placed: { x: number; y: number; width: number }[] = []
  const ordered = blocks.slice().reverse().sort((a, b) =>
    Number(b.node.id === selectedId) - Number(a.node.id === selectedId))
  for (const { node, projected, width: blockWidth, depth } of ordered) {
    if (node.kind.startsWith('residual_add') && node.id !== selectedId) continue
    const label = node.label.replace(/\s*\([a-z0-9_]+\)$/i, '')
    const width = Math.min(layout.width - 24, textWidth(label, labelSize, true) + labelSize * 1.2)
    const xExtent = (Math.abs(cosYaw) * blockWidth + Math.abs(sinYaw) * depth) / 2
    const direction = projected.x >= centerX ? 1 : -1
    const clampX = (value: number) => Math.min(layout.width - width / 2 - 12, Math.max(width / 2 + 12, value))
    const candidates = [direction, -direction].map(side =>
      clampX(projected.x + side * (xExtent + width / 2 + labelSize * 1.3)))
    let row = 0
    let y = projected.y
    const available = () => candidates.find(x => !placed.some(other =>
      Math.abs(y - other.y) < rowHeight && Math.abs(x - other.x) < (width + other.width) / 2 + 7))
    let x = available()
    while (x === undefined) {
      row = row > 0 ? -row : 1 - row
      y = projected.y + row * rowHeight
      x = available()
    }
    placed.push({ x, y, width })
    svg.line(projected.x, projected.y, x, y, 'tensor-label-leader')
    svg.raw(`<g class="orbit-label-chip${node.id === selectedId ? ' is-selected' : ''}" data-node-id="${escapeHtml(node.id)}">`)
    svg.rect(x - width / 2, y - rowHeight / 2 + 2, width, rowHeight - 4, 5)
    svg.text(x, y + labelSize * 0.34, label, Math.min(labelSize, (width - labelSize * 0.6) / textWidth(label, 1, true)), '', 'middle', 700)
    svg.raw('</g>')
  }
  svg.raw('</g>')
  return svg.document(layout.width, layout.height, sceneStyles, 'tensor-scene', true)
}

const sceneStyles = `
  #tensor-scene .surface { fill: transparent; }
  #tensor-scene .connection { stroke: #789491; stroke-width: 3; fill: none; opacity: 0.7; pointer-events: none; }
  #tensor-scene .residual-connection { stroke-dasharray: 5 5; opacity: 0.65; }
  #tensor-scene .connection-selected { stroke: #0a7f71; stroke-width: 5; opacity: 1; }
  #tensor-scene .tensor-volume, #tensor-scene .orbit-label-chip { cursor: pointer; }
  #tensor-scene .tensor-face { stroke: #344963; stroke-width: 1.35; stroke-linejoin: round; }
  #tensor-scene .tensor-volume:is(:hover, :focus-visible, .node-selected) .tensor-face { stroke: #0b6e64; stroke-width: 3; }
  #tensor-scene .tensor-volume:focus-visible { outline: none; }
  #tensor-scene .tensor-cell { stroke: #fff; opacity: .55; stroke-width: .85; pointer-events: none; }
  #tensor-scene .tensor-label-leader { stroke: #75849a; stroke-width: 1.5; pointer-events: none; }
  #tensor-scene .orbit-label-chip rect { fill: #fff; stroke: #c2ccd9; }
  #tensor-scene .orbit-label-chip:is(:hover, .is-selected) rect { fill: #e0f5ef; stroke: #0b6e64; stroke-width: 2; }
  #tensor-scene .orbit-label-chip text { fill: #1a263a; font-family: system-ui, sans-serif; }
`
