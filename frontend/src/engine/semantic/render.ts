import { renderTensorScene } from './tensorScene'
import type { DiagramCamera } from './orbit'
import { SVG, escapeHtml, fmtCount } from '../render/svg'
import { accentFor, scopeId, stylesheet, blockTint } from '../render/tokens'
import { textWidth } from '../render/widths'
import type { ArchSpec } from '../spec'
import type { DiagramLayout, LayoutedNode, WorkspaceState } from './types'

function attnLines(s: ArchSpec): [string, string][][] {
  const k = s.attention.kind
  if (k === 'MLA') {
    return [[['Multi-head ', 'white'], ['Latent', 'accent-dark']], [['Attention', 'white']]]
  }
  if (k === 'GQA') {
    return [[['Masked ', 'white'], ['grouped-query', 'accent-dark']], [['attention', 'white']]]
  }
  if (k === 'MQA') {
    return [[['Masked ', 'white'], ['multi-query', 'accent-dark']], [['attention', 'white']]]
  }
  return [[['Masked multi-head', 'white']], [['attention', 'white']]]
}

function displayName(name: string): string {
  const words = name.trim().split(/[-_ ]+/)
  const out: string[] = []
  for (const w of words) {
    if (/^\d+(\.\d+)?[bm]$/i.test(w)) {
      out.push(`(${w.toUpperCase()})`)
    } else if (/^(gpt|gpt2|oss|glm|lfm|moe|ai|xl)$/i.test(w)) {
      out.push(w.toUpperCase())
    } else if (w === w.toLowerCase()) {
      out.push(w.charAt(0).toUpperCase() + w.slice(1))
    } else {
      out.push(w)
    }
  }
  return out.join(' ')
}

export interface RenderOptions {
  interactive?: boolean
  camera?: DiagramCamera
  viewportScale?: number
  selectedNodeId?: string | null
  focusedNodeId?: string | null
}

/**
 * Render dynamic DiagramLayout into publication-grade vector SVG
 */
export function renderSemanticDiagram(
  layout: DiagramLayout,
  spec: ArchSpec,
  state: WorkspaceState,
  options: RenderOptions = {}
): string {
  const accent = accentFor(spec.model_type)
  const cx = 470
  const isInteractive = options.interactive ?? true
  const selectedId = options.selectedNodeId || state.selectedNodeId
  if (options.camera) return renderTensorScene(layout, options.camera, selectedId, options.viewportScale ?? 1)
  const g = new SVG()

  // Focus Mode Title or Model Title
  if (state.focusedNodeId) {
    const focusNode = layout.rootNodes[0]
    const title = focusNode?.label ?? displayName(spec.name)
    const titleSize = Math.min(38, (layout.width - 80) / textWidth(title, 1, true))
    g.text(450, 60, title, titleSize, 'accent', 'middle', 800)
    g.text(450, 92, `Isolated Component View · ${spec.name}`, 16, '', 'middle', 500)
  } else {
    const title = displayName(spec.name.slice(spec.name.lastIndexOf('/') + 1) || spec.name)
    const titleSize = Math.min(46, (2 * cx - 80) / textWidth(title, 1, true))
    g.text(cx, 74, title, titleSize, 'accent', 'middle', 800)
  }

  // Draw Outer Container (if present)
  if (layout.outerContainer) {
    const oc = layout.outerContainer
    g.rect(oc.x, oc.y, oc.w, oc.h, 40, 'outer')
  }

  // Draw Repeated Block Container (if present)
  if (layout.repeatedBlockContainer) {
    const rbc = layout.repeatedBlockContainer
    g.rect(rbc.x, rbc.y, rbc.w, rbc.h, 34, 'blockc')

    g.text(rbc.x + 18, rbc.y + 30, `${rbc.numLayers} × Decoder blocks`, 16, 'accent', 'start', 700)
  }

  // Helper for rendering interactive attributes
  function nodeAttrs(n: LayoutedNode): string {
    if (!isInteractive) return ''
    const isSelected = selectedId === n.id
    return `data-node-id="${escapeHtml(n.id)}" data-expandable="${n.is_expandable}" data-expanded="${n.isExpanded}" role="button" tabindex="0" aria-label="${escapeHtml(n.label)}" aria-pressed="${isSelected}" ${n.is_expandable ? `aria-expanded="${n.isExpanded}"` : ''} class="interactive-node node-kind-${n.kind} ${isSelected ? 'node-selected' : ''}"`
  }

  // Render Residual Rails
  for (const rail of layout.residualRails) {
    g.path(`M ${rail.startCx} ${rail.startY} L ${rail.railX} ${rail.startY} L ${rail.railX} ${rail.endY}`)
    g.arrow(rail.railX, rail.endY, rail.endCx, rail.endY)
  }

  // Render Nodes
  for (const n of layout.rootNodes) {
    const isFocusGroup = Boolean(state.focusedNodeId && n.children?.length)
    g.raw(isFocusGroup ? '<g>' : `<g ${nodeAttrs(n)}>`)
    renderNode(g, n, spec, state, nodeAttrs, isInteractive)
    g.raw('</g>')
  }

  // Render Side Magnified Insets (Breakout detail with projection rays)
  if (layout.magnifiedInsets && layout.magnifiedInsets.length > 0) {
    for (const inset of layout.magnifiedInsets) {
      // 1. Translucent magnifying projection cone from central tower to side breakout box
      g.raw(
        `<polygon points="${inset.sourceX},${inset.sourceTopY} ${inset.x},${inset.y} ${inset.x},${inset.y + inset.height} ${inset.sourceX},${inset.sourceBottomY}" fill="rgba(15,23,42,0.035)" stroke="none" />`
      )
      // 2. Dotted leader projection rays
      g.leader(inset.sourceX, inset.sourceTopY, inset.x, inset.y)
      g.leader(inset.sourceX, inset.sourceBottomY, inset.x, inset.y + inset.height)

      // 3. Magnified Inset Container Box
      g.rect(inset.x, inset.y, inset.width, inset.height, 22, 'inset-accent')

      // 4. Inset Header Banner
      g.text(inset.x + inset.width / 2, inset.y + 26, inset.title, 16, 'accent', 'middle', 700)

      // 5. Child projection pills inside side inset
      for (const ch of inset.children) {
        renderChildPill(g, ch, nodeAttrs)
      }

      drawDetailEdges(inset.edges)
    }
  }

  if (state.focusedNodeId) drawDetailEdges(layout.internalEdges ?? [])

  function drawDetailEdges(edges: NonNullable<DiagramLayout['internalEdges']>) {
    for (const edge of edges) {
      if (Math.abs(edge.fromX - edge.toX) < 8) {
        g.arrow(edge.fromX, edge.fromY, edge.toX, edge.toY, 'flow', 8)
      } else {
        const midY = (edge.fromY + edge.toY) / 2
        g.path(`M ${edge.fromX} ${edge.fromY} C ${edge.fromX} ${midY}, ${edge.toX} ${midY}, ${edge.toX} ${edge.toY + 8}`)
        g.arrow(edge.toX, edge.toY + 8, edge.toX, edge.toY, 'flow', 8)
      }

      if (edge.label) {
        const midX = (edge.fromX + edge.toX) / 2 + (edge.fromX !== edge.toX ? 8 : 12)
        const midY = (edge.fromY + edge.toY) / 2
        g.text(midX, midY, edge.label, 11, 'mono', 'start', 600)
      }
    }
  }

  // Vertical Connecting Arrows along Central Spine
  const nodeMap = layout.allNodes
  const head = nodeMap.get('lm_head')
  const finNorm = nodeMap.get('final_norm')
  const res2 = nodeMap.get(`layer_${state.selectedLayer}_res2`)
  const ffn = nodeMap.get(`layer_${state.selectedLayer}_ffn`)
  const norm2 = nodeMap.get(`layer_${state.selectedLayer}_norm2`)
  const res1 = nodeMap.get(`layer_${state.selectedLayer}_res1`)
  const attn = nodeMap.get(`layer_${state.selectedLayer}_attention`)
  const norm1 = nodeMap.get(`layer_${state.selectedLayer}_norm1`)
  const emb = nodeMap.get('embed_tokens')
  const tok = nodeMap.get('tokens_io')

  // Top spine arrows
  if (head) {
    g.arrow(head.cx, head.y, head.cx, 150)
    g.text(head.cx, 132, 'Output logits', 22, 'mono', 'middle')
  }
  if (finNorm && head) {
    g.arrow(cx, finNorm.topY, cx, head.bottomY + 3)
  }
  if (res2 && finNorm) {
    g.arrow(cx, res2.topY - 14, cx, finNorm.bottomY + 3)
  }
  if (ffn && res2) {
    g.arrow(cx, ffn.topY, cx, res2.bottomY + 14)
  }
  if (norm2 && ffn) {
    g.arrow(cx, norm2.topY, cx, ffn.bottomY + 3)
  }
  if (res1 && norm2) {
    g.arrow(cx, res1.topY - 14, cx, norm2.bottomY + 3)
  }
  if (attn && res1) {
    g.arrow(cx, attn.topY, cx, res1.bottomY + 14)
  }
  if (norm1 && attn) {
    g.arrow(cx, norm1.topY, cx, attn.bottomY + 3)
  }
  if (emb && norm1) {
    g.arrow(cx, emb.topY, cx, norm1.bottomY + 3)
  }
  if (tok && emb) {
    g.arrow(cx, tok.topY, cx, emb.bottomY + 4)
    g.arrow(cx, tok.bottomY + 62, cx, tok.bottomY + 4)
    g.text(cx, tok.bottomY + 100, 'Sample input text', 24, 'mono', 'middle')
  }

  // Draw RoPE pill on the left side (if applicable and not focus mode)
  if (spec.positional === 'rope' && attn && !state.focusedNodeId) {
    const ropeCx = 145
    const ropeY = attn.y + attn.height / 2 - 26
    g.rect(ropeCx - 105, ropeY, 210, 52, 14, 'pill')
    g.text(ropeCx, ropeY + 34, 'RoPE', 22, '', 'middle')
    g.arrow(ropeCx + 105, ropeY + 26, attn.leftX - 2, ropeY + 26)
  }

  // Draw MoE Inset on the right (if MoE, not expanded, and not focus mode)
  if (spec.moe && ffn && !ffn.isExpanded && !state.focusedNodeId) {
    drawMoeInset(g, spec, ffn.rightX, ffn.y + ffn.height / 2)
  }

  // Render Callouts & Dotted Leaders
  for (const c of layout.callouts) {
    for (let i = 0; i < c.lines.length; i++) {
      g.richText(c.x, c.y + i * 30, c.lines[i], 21, 'start')
    }
    g.leader(c.x - 10, c.y + 10, c.targetX, c.targetY)
  }

  const id = scopeId(accent)
  const css = stylesheet(accent) + (isInteractive ? workspaceStyles(id) : '')
  return g.document(layout.width, layout.height, css, id, isInteractive)
}

/**
 * Render individual semantic node according to its kind and expansion state
 */
function renderNode(
  g: SVG,
  n: LayoutedNode,
  spec: ArchSpec,
  state: WorkspaceState,
  nodeAttrs: (n: LayoutedNode) => string,
  isInteractive: boolean
): void {
  // 1. Attention Block
  if (n.kind === 'attention') {
    if (state.focusedNodeId && n.children && n.children.length > 0) {
      // Focus Mode isolated container
      g.rect(n.x, n.y, n.width, n.height, 20, 'inset-accent')
      g.text(n.cx, n.y + 26, `▲ Self-Attention Mechanism (Expanded)`, 16, 'accent', 'middle', 700)
      for (const ch of n.children) {
        renderChildPill(g, ch, nodeAttrs)
      }
      return
    }

    // Central Tower Attention Box (always compact in tower)
    g.rect(n.x, n.y, n.width, n.height, 16, 'attnbox')
    const al = attnLines(spec)
    for (let i = 0; i < al.length; i++) {
      g.richText(n.cx, n.y + 40 + i * 30, al[i], 23, 'middle', 400)
    }

    if (n.isExpanded) {
      g.rect(n.rightX - 116, n.y + 12, 102, 22, 5, 'pill')
      g.text(n.rightX - 65, n.y + 26, 'MAGNIFIED ▶', 10, 'accent', 'middle', 700)
      if (isInteractive) {
        g.text(n.rightX - 128, n.y + 28, '−', 20, 'white', 'middle', 700)
      }
    } else if (n.is_expandable && isInteractive) {
      g.text(n.rightX - 20, n.y + 24, '+', 20, 'white', 'middle', 700)
    }
    return
  }

  // 2. FFN / MoE Block
  if (n.kind === 'ffn' || n.kind === 'moe') {
    if (state.focusedNodeId && n.children && n.children.length > 0) {
      // Focus Mode isolated container
      g.rect(n.x, n.y, n.width, n.height, 20, 'inset-accent')
      g.text(
        n.cx,
        n.y + 26,
        `▲ ${n.kind === 'moe' ? 'Mixture-of-Experts' : 'Feed-Forward Network'} (Expanded)`,
        16,
        'accent',
        'middle',
        700
      )
      for (const ch of n.children) {
        renderChildPill(g, ch, nodeAttrs)
      }
      return
    }

    // Central Tower FFN / MoE Pill (always compact in tower)
    g.rect(n.x, n.y, n.width, n.height, 14, 'moepill')
    g.text(n.cx, n.y + 31, spec.moe ? 'MoE' : 'Feed forward', 22, '', 'middle')

    if (n.isExpanded) {
      g.rect(n.rightX - 116, n.y + 13, 102, 22, 5, 'pill')
      g.text(n.rightX - 65, n.y + 27, 'MAGNIFIED ▶', 10, 'accent', 'middle', 700)
      if (isInteractive) {
        g.text(n.rightX - 128, n.y + 28, '−', 18, 'black', 'middle', 700)
      }
    } else if (n.is_expandable && isInteractive) {
      g.text(n.rightX - 16, n.y + 20, '+', 16, 'accent', 'middle', 700)
    }
    return
  }

  // 3. Residual Node (⊕)
  if (n.kind === 'residual_add_1' || n.kind === 'residual_add_2') {
    g.plusNode(n.cx, n.topY + 15)
    return
  }

  if (state.focusedNodeId && n.children?.length) {
    g.rect(n.x, n.y, n.width, n.height, 20, 'inset-accent')
    for (const child of n.children) {
      renderChildPill(g, child, nodeAttrs)
    }
    return
  }

  // 4. Standard Pills (Norm, Output, Embeddings)
  g.rect(n.x, n.y, n.width, n.height, 14, 'pill')
  g.text(n.cx, n.y + n.height / 2 + 8, n.label, 22, '', 'middle')
  if (n.is_expandable && isInteractive) {
    g.text(n.rightX - 16, n.y + 18, '+', 14, 'accent', 'middle', 700)
  }
}

/**
 * Render child pills inside an expanded module with spacious layout and zero overlap
 */
function renderChildPill(
  g: SVG,
  ch: LayoutedNode,
  nodeAttrs: (n: LayoutedNode) => string
): void {
  g.raw(`<g ${nodeAttrs(ch)} id="${escapeHtml(ch.id)}">`)
  const pillClass = 'pill'

  g.rect(ch.x, ch.y, ch.width, ch.height, 10, pillClass)

  const badge = ch.badge
  const badgeW = badge ? Math.max(48, badge.length * 7 + 10) : 0
  const badgeH = 18
  const badgeX = ch.rightX - badgeW - 8
  const badgeY = ch.y + (ch.height - badgeH) / 2

  // Text bounding coordinates to guarantee zero badge collision
  const textLeft = ch.leftX + 10
  const textRight = badge ? badgeX - 8 : ch.rightX - 10
  const textCenterX = (textLeft + textRight) / 2

  // Shorten "(q_proj)" suffix in narrow 3-column projection pill so "Query Projection" fits with 25px margin!
  const displayLabel = ch.width < 220
    ? ch.label.replace(/\s*\([a-z0-9_]+\)$/i, '')
    : ch.label
  const hasSub = Boolean(ch.sublabel || ch.shape_str)
  const maxLabelSize = (textRight - textLeft) / textWidth(displayLabel, 1, true)

  if (hasSub) {
    const labelFontSize = ch.width < 220 ? 11.5 : 13.5
    const subFontSize = ch.width < 220 ? 10 : 11
    const labelY = ch.y + 19
    const subY = ch.y + 36

    const summary = ch.shape_str || ch.sublabel || ''
    const combined = ch.shape_str && ch.sublabel ? `${ch.shape_str} · ${ch.sublabel}` : summary
    const maxChars = Math.floor((textRight - textLeft) / (subFontSize * 0.6))
    const sub = combined.length <= maxChars ? combined : summary
    const visibleSub = sub.length <= maxChars ? sub : `${sub.slice(0, maxChars - 1)}…`
    g.raw(`<title>${escapeHtml(`${ch.label} — ${combined}`)}</title>`)

    if (ch.width < 220) {
      g.text(textLeft, labelY, displayLabel, Math.min(labelFontSize, maxLabelSize), '', 'start', 600)
      g.text(textLeft, subY, visibleSub, subFontSize, 'mono', 'start', 400)
    } else {
      g.text(textCenterX, labelY, displayLabel, Math.min(labelFontSize, maxLabelSize), '', 'middle', 600)
      g.text(textCenterX, subY, visibleSub, subFontSize, 'mono', 'middle', 400)
    }
  } else {
    const labelY = ch.y + ch.height / 2 + 5
    const labelFontSize = ch.width < 220 ? 12 : 14
    if (ch.width < 220) {
      g.text(textLeft, labelY, displayLabel, Math.min(labelFontSize, maxLabelSize), '', 'start', 600)
    } else {
      g.text(textCenterX, labelY, displayLabel, Math.min(labelFontSize, maxLabelSize), '', 'middle', 600)
    }
  }

  // Badge cleanly anchored to the right
  if (badge) {
    g.rect(badgeX, badgeY, badgeW, badgeH, 4, 'pill')
    g.text(badgeX + badgeW / 2, badgeY + 13, badge, 9, 'accent', 'middle', 700)
  }

  g.raw(`</g>`)
}

/**
 * Draw MoE Inset on the right side of the tower in Preset 1 overview
 */
function drawMoeInset(g: SVG, s: ArchSpec, startX: number, startY: number): void {
  const moe = s.moe!
  const ix = 890
  const iy = startY - 140
  const iw = 570
  const ih = 340

  // Inset container box
  g.rect(ix, iy, iw, ih, 24, 'inset')

  // Connecting dotted leader line from FFN pill to MoE Inset
  g.leader(startX + 10, startY, ix, startY)

  // Inset Title
  g.text(ix + iw / 2, iy + 36, `MoE: ${moe.num_experts} Experts (Top-${moe.experts_per_tok})`, 22, 'accent', 'middle', 700)

  // Router box
  const rW = 240, rH = 44
  g.rect(ix + iw / 2 - rW / 2, iy + 60, rW, rH, 12, 'router')
  g.text(ix + iw / 2, iy + 88, 'MoE Gating Router', 18, 'white', 'middle', 600)

  // Routed experts sample pills
  const expY = iy + 140
  const expW = 100, expH = 40
  const numShow = Math.min(4, moe.num_experts)
  const totalW = numShow * expW + (numShow - 1) * 16
  const startExpX = ix + iw / 2 - totalW / 2

  for (let i = 0; i < numShow; i++) {
    const ex = startExpX + i * (expW + 16)
    g.rect(ex, expY, expW, expH, 8, 'moepill')
    g.text(ex + expW / 2, expY + 25, `Expert ${i + 1}`, 14, '', 'middle', 500)
    g.arrow(ix + iw / 2, iy + 104, ex + expW / 2, expY)
  }

  // Dots indicating remaining experts
  if (moe.num_experts > 4) {
    g.text(ix + iw / 2, expY + 70, `… and ${moe.num_experts - 4} more routed experts …`, 15, 'mono', 'middle')
  }

  // Shared experts if present
  if (moe.shared_experts && moe.shared_experts > 0) {
    const sW = 280, sH = 42
    g.rect(ix + iw / 2 - sW / 2, iy + ih - 60, sW, sH, 10, 'pill')
    g.text(ix + iw / 2, iy + ih - 34, `+ ${moe.shared_experts} Shared Expert (Always Active)`, 15, 'accent', 'middle', 600)
  }
}

function workspaceStyles(id: string): string {
  return `
    #${id} .surface { fill: transparent; }
    #${id} .outer { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }
    #${id} .blockc { fill: #eef2f7; stroke: #cbd5e1; stroke-width: 1.5; }
    #${id} .pill, #${id} .plusc { stroke: #94a3b8; stroke-width: 1.5; }
    #${id} .attnbox { fill: #164e63; stroke: #164e63; }
    #${id} .moepill { fill: #e2e8f0; stroke: #94a3b8; stroke-width: 1.5; }
    #${id} .router { fill: #164e63; stroke: #164e63; }
    #${id} .inset, #${id} .inset-accent { fill: #fff; stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: none; }
    #${id} .flow { stroke: #475569; pointer-events: none; }
    #${id} .flowhead { fill: #475569; pointer-events: none; }
    #${id} .leader { stroke: #64748b; stroke-width: 1.5; pointer-events: none; }
    #${id} .interactive-node { cursor: pointer; }
    #${id} .interactive-node:focus-visible { outline: none; }
    #${id} .interactive-node:is(:hover, :focus-visible, .node-selected) > :is(rect, circle) {
      stroke: #0e7490; stroke-width: 4;
    }
  `
}
