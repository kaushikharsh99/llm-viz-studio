import type { ArchGraph } from './types'
import { escapeHtml } from '../engine/render/svg'

const KIND_FILL: Record<string, string> = {
  embedding: '#dbeafe',
  norm: '#fef3c7',
  attention: '#ede9fe',
  mlp: '#dcfce7',
  moe: '#fee2e2',
  block: '#f3f4f6',
  head: '#ffedd5',
  op: '#ffffff',
  unknown: '#fecaca',
}

export function graphToSvg(g: ArchGraph, maxNodes = 60): string {
  const nodes = g.nodes.filter(n => n.level === 1 || n.level === 2).slice(0, maxNodes)
  const W = 900, rowH = 64, pad = 20
  const H = pad * 2 + rowH * nodes.length + 60
  const pos: Record<string, [number, number]> = {}
  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img">`,
    '<style>text{font-family:Arial,sans-serif;font-size:15px} .t{font-size:12px;fill:#555}</style>',
    `<text x="${pad}" y="30" font-size="20" font-weight="bold">${escapeHtml(g.model_id)} — architecture graph (${escapeHtml(g.confidence)})</text>`,
  ]
  let y = 60
  for (const n of nodes) {
    const x = pad + (n.level - 1) * 360
    pos[n.id] = [x + 260, y + Math.floor(rowH / 2)]
    const fill = KIND_FILL[n.kind] || '#f3f4f6'
    parts.push(`<rect x="${x}" y="${y}" width="260" height="${rowH - 10}" rx="8" fill="${fill}" stroke="#999"/>`)
    parts.push(`<text x="${x + 10}" y="${y + 24}">${escapeHtml(n.label)}</text>`)
    let sub = `${n.kind} · ${n.params.toLocaleString()} params · ${n.confidence}`
    if (n.collapsed_count) {
      sub += ` · ×${n.collapsed_count}`
    }
    parts.push(`<text class="t" x="${x + 10}" y="${y + 44}">${escapeHtml(sub)}</text>`)
    y += rowH
  }
  for (const e of g.edges) {
    if (pos[e.src] && pos[e.dst]) {
      const [x1, y1] = pos[e.src]
      const [x2, y2] = pos[e.dst]
      parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>`)
    }
  }
  parts.push('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#666"/></marker></defs>')
  parts.push('</svg>')
  return parts.join('\n')
}
