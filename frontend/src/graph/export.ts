import type { ArchGraph, GraphNode } from './types'
import { escapeHtml } from '../engine/render/svg'
import { layoutGraph } from './layout'

const KIND_COLORS: Record<string, { fill: string; stroke: string; text: string; badge: string }> = {
  embedding: { fill: '#f0f9ff', stroke: '#0284c7', text: '#0369a1', badge: '#bae6fd' },
  linear: { fill: '#f8fafc', stroke: '#64748b', text: '#1e293b', badge: '#e2e8f0' },
  tensor: { fill: '#f8fafc', stroke: '#64748b', text: '#1e293b', badge: '#e2e8f0' },
  norm: { fill: '#fefce8', stroke: '#ca8a04', text: '#854d0e', badge: '#fef08a' },
  attention_block: { fill: '#faf5ff', stroke: '#9333ea', text: '#6b21a8', badge: '#e9d5ff' },
  ffn_block: { fill: '#f0fdf4', stroke: '#16a34a', text: '#15803d', badge: '#bbf7d0' },
  moe_block: { fill: '#fff1f2', stroke: '#e11d48', text: '#be123c', badge: '#fecdd3' },
  layer_group: { fill: '#f1f5f9', stroke: '#475569', text: '#0f172a', badge: '#cbd5e1' },
  residual: { fill: '#fff7ed', stroke: '#ea580c', text: '#c2410c', badge: '#fed7aa' },
  math_op: { fill: '#f8fafc', stroke: '#94a3b8', text: '#334155', badge: '#e2e8f0' },
  attention_op: { fill: '#f8fafc', stroke: '#94a3b8', text: '#334155', badge: '#e2e8f0' },
  router: { fill: '#fdf4ff', stroke: '#c026d3', text: '#86198f', badge: '#f5d0fe' },
  expert: { fill: '#fff1f2', stroke: '#e11d48', text: '#be123c', badge: '#fecdd3' },
  head: { fill: '#f8fafc', stroke: '#0f172a', text: '#0f172a', badge: '#e2e8f0' },
  io: { fill: '#f1f5f9', stroke: '#64748b', text: '#334155', badge: '#cbd5e1' },
  unknown: { fill: '#fef2f2', stroke: '#ef4444', text: '#b91c1c', badge: '#fecaca' },
}

function fmtParams(num?: number | null): string {
  if (num === null || num === undefined) return ''
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K'
  return num.toLocaleString()
}

/**
 * Generate a production-ready, publication-quality vector SVG
 * representing the current layout of the architectural graph
 */
export async function graphToSvg(g: ArchGraph): Promise<string> {
  const layout = await layoutGraph(g)
  const pad = 48
  const headerHeight = 70

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const n of layout.nodes) {
    minX = Math.min(minX, n.x)
    minY = Math.min(minY, n.y)
    maxX = Math.max(maxX, n.x + n.width)
    maxY = Math.max(maxY, n.y + n.height)
  }

  if (layout.nodes.length === 0) {
    minX = 0; minY = 0; maxX = 800; maxY = 400
  }

  const contentW = maxX - minX
  const contentH = maxY - minY
  const totalW = Math.max(900, Math.ceil(contentW + pad * 2))
  const totalH = Math.ceil(contentH + pad * 2 + headerHeight)

  // Center offset if totalW > contentW
  const offsetX = pad - minX + (totalW - contentW - pad * 2) / 2
  const offsetY = pad + headerHeight - minY

  const posMap = new Map<string, { cx: number; cy: number; topY: number; bottomY: number; leftX: number }>()
  for (const n of layout.nodes) {
    const nx = n.x + offsetX
    const ny = n.y + offsetY
    posMap.set(n.id, {
      cx: nx + n.width / 2,
      cy: ny + n.height / 2,
      topY: ny,
      bottomY: ny + n.height,
      leftX: nx,
    })
  }

  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${totalH}" viewBox="0 0 ${totalW} ${totalH}" role="img">`,
    '<style>',
    '  .bg { fill: #ffffff; }',
    '  .grid-pattern { stroke: #f1f5f9; stroke-width: 1; }',
    '  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }',
    '  .title { font-size: 18px; font-weight: 700; fill: #0f172a; }',
    '  .subtitle { font-size: 13px; fill: #64748b; }',
    '  .node-title { font-size: 13px; font-weight: 600; fill: #0f172a; }',
    '  .node-sub { font-size: 11px; fill: #64748b; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
    '  .badge-text { font-size: 10px; font-weight: 700; text-transform: uppercase; }',
    '  .edge-data { stroke: #94a3b8; stroke-width: 2; fill: none; }',
    '  .edge-res { stroke: #f97316; stroke-width: 2; stroke-dasharray: 5,4; fill: none; }',
    '  .edge-ctrl { stroke: #8b5cf6; stroke-width: 2; stroke-dasharray: 4,4; fill: none; }',
    '</style>',
    '<defs>',
    '  <marker id="arrow-data" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#94a3b8"/></marker>',
    '  <marker id="arrow-res" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#f97316"/></marker>',
    '  <marker id="arrow-ctrl" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#8b5cf6"/></marker>',
    '</defs>',
    `<rect class="bg" width="${totalW}" height="${totalH}"/>`,
  ]

  // Header
  const levelNames = ['Level 0 · Whole Model', 'Level 1 · Layer Overview', 'Level 2 · Module Internals', 'Level 3 · Operators & Tensors']
  const levelText = levelNames[g.level] || `Level ${g.level}`
  const layerText = g.level > 0 && g.active_layer !== null ? ` · Layer ${g.active_layer}` : ''
  parts.push(
    `<text class="title" x="${pad}" y="${pad - 12}">${escapeHtml(g.model_id)}</text>`,
    `<text class="subtitle" x="${pad}" y="${pad + 12}">${escapeHtml(levelText)}${escapeHtml(layerText)} · ${escapeHtml(g.architecture)} · ${g.verified_params ? fmtParams(g.verified_params) + ' verified params' : fmtParams(g.total_params) + ' est. params'}</text>`
  )

  // Edges
  for (const e of layout.edges) {
    const srcPos = posMap.get(e.src)
    const dstPos = posMap.get(e.dst)
    if (!srcPos || !dstPos) continue

    const isRes = e.kind === 'residual'
    const isCtrl = e.kind === 'control'
    const marker = isRes ? 'url(#arrow-res)' : isCtrl ? 'url(#arrow-ctrl)' : 'url(#arrow-data)'
    const edgeClass = isRes ? 'edge-res' : isCtrl ? 'edge-ctrl' : 'edge-data'

    if (isRes) {
      // Curve out to left for residual skip connection
      const curveX = Math.min(srcPos.leftX, dstPos.leftX) - 36
      const d = `M ${srcPos.cx} ${srcPos.bottomY} C ${curveX} ${srcPos.bottomY + 20}, ${curveX} ${dstPos.cy - 10}, ${dstPos.leftX} ${dstPos.cy}`
      parts.push(`<path class="${edgeClass}" d="${d}" marker-end="${marker}"/>`)
    } else {
      // Orthogonal or cubic bezier from bottom of src to top of dst
      const midY = (srcPos.bottomY + dstPos.topY) / 2
      const d = `M ${srcPos.cx} ${srcPos.bottomY} C ${srcPos.cx} ${midY}, ${dstPos.cx} ${midY}, ${dstPos.cx} ${dstPos.topY}`
      parts.push(`<path class="${edgeClass}" d="${d}" marker-end="${marker}"/>`)
    }
  }

  // Nodes
  for (const n of layout.nodes) {
    const nx = n.x + offsetX
    const ny = n.y + offsetY
    const col = KIND_COLORS[n.kind] || KIND_COLORS.linear
    const rx = n.kind === 'residual' ? 24 : 10

    parts.push(
      `<g transform="translate(${nx}, ${ny})">`,
      `  <rect width="${n.width}" height="${n.height}" rx="${rx}" fill="${col.fill}" stroke="${col.stroke}" stroke-width="1.5"/>`
    )

    if (n.kind === 'residual') {
      parts.push(
        `  <text x="24" y="${n.height / 2 + 5}" font-size="16" fill="${col.text}">⊕</text>`,
        `  <text class="node-title" x="48" y="${n.height / 2 + 5}">${escapeHtml(n.label.replace('⊕ ', ''))}</text>`
      )
    } else {
      // Badge
      parts.push(
        `  <rect x="12" y="10" width="56" height="18" rx="4" fill="${col.badge}"/>`,
        `  <text class="badge-text" x="18" y="23" fill="${col.text}">${n.kind.slice(0, 7).toUpperCase()}</text>`
      )
      // Title
      parts.push(
        `  <text class="node-title" x="12" y="44">${escapeHtml(n.label.length > 28 ? n.label.slice(0, 26) + '…' : n.label)}</text>`
      )
      // Meta (shape or params)
      const meta = n.shape_str || (n.params ? `${fmtParams(n.params)} params` : '')
      if (meta) {
        parts.push(
          `  <text class="node-sub" x="12" y="64">${escapeHtml(meta)}</text>`
        )
      }
    }

    parts.push('</g>')
  }

  parts.push('</svg>')
  return parts.join('\n')
}
