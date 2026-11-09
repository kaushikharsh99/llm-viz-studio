import { SVG, escapeHtml, fmtCount } from '../render/svg'
import { accentFor, scopeId, stylesheet, blockTint } from '../render/tokens'
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
  const g = new SVG()
  const cx = 470
  const isInteractive = options.interactive ?? true
  const selectedId = options.selectedNodeId || state.selectedNodeId

  // Focus Mode Title or Model Title
  if (state.focusedNodeId) {
    const focusNode = layout.rootNodes[0]
    g.text(450, 60, focusNode ? focusNode.label : displayName(spec.name), 38, 'accent', 'middle', 800)
    g.text(450, 92, `Isolated Component View · ${spec.name}`, 16, '', 'middle', 500)
  } else {
    g.text(cx, 74, displayName(spec.name), 46, 'accent', 'middle', 800)
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

    // Layer brace on the left
    g.text(rbc.x - 26, rbc.y + rbc.h + 22, '{', 140, '', 'middle', 300)
    g.richText(rbc.x - 158, rbc.y + rbc.h - 12, [[`${rbc.numLayers} ×`, 'accent']], 34)
  }

  // Helper for rendering interactive attributes
  function nodeAttrs(n: LayoutedNode): string {
    if (!isInteractive) return ''
    const isSelected = selectedId === n.id
    const cursor = n.is_expandable ? 'cursor: pointer;' : 'cursor: default;'
    return `data-node-id="${escapeHtml(n.id)}" data-expandable="${n.is_expandable}" data-expanded="${n.isExpanded}" class="interactive-node ${isSelected ? 'node-selected' : ''}" style="${cursor}"`
  }

  // Render Residual Rails
  for (const rail of layout.residualRails) {
    g.path(`M ${rail.startCx} ${rail.startY} L ${rail.railX} ${rail.startY} L ${rail.railX} ${rail.endY}`)
    g.arrow(rail.railX, rail.endY, rail.endCx, rail.endY)
  }

  // Render Nodes
  for (const n of layout.rootNodes) {
    renderNode(g, n, spec, state, nodeAttrs, isInteractive, selectedId)
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
    g.arrow(head.cx, head.y, head.cx, 128)
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

  const css = stylesheet(accent)
  const id = scopeId(accent)
  return g.document(layout.width, layout.height, css, id)
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
  isInteractive: boolean,
  selectedId: string | null
): void {
  const isSelected = selectedId === n.id

  // 1. Attention Block
  if (n.kind === 'attention') {
    if (n.isExpanded && n.children && n.children.length > 0) {
      // Expanded Attention Container
      g.rect(n.x, n.y, n.width, n.height, 20, 'inset-accent')
      // Container Title Banner
      g.text(n.cx, n.y + 24, `▲ Self-Attention Mechanism (Expanded)`, 16, 'accent', 'middle', 700)

      // Render internal projection pills
      for (const ch of n.children) {
        renderChildPill(g, ch, spec, nodeAttrs)
      }
    } else {
      // Collapsed Iconic Dark Attention Box
      g.rect(n.x, n.y, n.width, n.height, 16, 'attnbox')
      const al = attnLines(spec)
      for (let i = 0; i < al.length; i++) {
        g.richText(n.cx, n.y + 40 + i * 30, al[i], 23, 'middle', 400)
      }
      if (n.is_expandable && isInteractive) {
        g.text(n.rightX - 20, n.y + 24, '+', 20, 'white', 'middle', 700)
      }
    }
    return
  }

  // 2. FFN / MoE Block
  if (n.kind === 'ffn' || n.kind === 'moe') {
    if (n.isExpanded && n.children && n.children.length > 0) {
      // Expanded FFN Container
      g.rect(n.x, n.y, n.width, n.height, 20, 'inset-accent')
      g.text(
        n.cx,
        n.y + 24,
        `▲ ${n.kind === 'moe' ? 'Mixture-of-Experts' : 'SwiGLU Feed-Forward'} (Expanded)`,
        16,
        'accent',
        'middle',
        700
      )

      for (const ch of n.children) {
        renderChildPill(g, ch, spec, nodeAttrs)
      }
    } else {
      // Collapsed MoE / FFN Pill
      g.rect(n.x, n.y, n.width, n.height, 14, 'moepill')
      g.text(n.cx, n.y + 31, spec.moe ? 'MoE' : 'Feed forward', 22, '', 'middle')
      if (n.is_expandable && isInteractive) {
        g.text(n.rightX - 16, n.y + 20, '+', 16, 'white', 'middle', 700)
      }
    }
    return
  }

  // 3. Residual Node (⊕)
  if (n.kind === 'residual_add_1' || n.kind === 'residual_add_2') {
    g.plusNode(n.cx, n.topY + 15)
    return
  }

  // 4. Standard Pills (Norm, Output, Embeddings)
  g.rect(n.x, n.y, n.width, n.height, 14, 'pill')
  g.text(n.cx, n.y + n.height / 2 + 8, n.label, 22, '', 'middle')
  if (n.is_expandable && isInteractive) {
    g.text(n.rightX - 16, n.y + 18, '+', 14, 'white', 'middle', 700)
  }
}

/**
 * Render child pills inside an expanded module
 */
function renderChildPill(
  g: SVG,
  ch: LayoutedNode,
  spec: ArchSpec,
  nodeAttrs: (n: LayoutedNode) => string
): void {
  const isTensor = ch.kind.includes('proj') || ch.kind.includes('weight')
  const isMathOp = ch.kind.includes('op') || ch.kind.includes('scores')
  const pillClass = isMathOp ? 'pill' : isTensor ? 'pill' : 'pill'

  g.rect(ch.x, ch.y, ch.width, ch.height, 10, pillClass)

  // Label & Dimension Sublabel
  g.text(ch.cx - 30, ch.y + 22, ch.label, 15, '', 'middle', 600)
  if (ch.sublabel || ch.shape_str) {
    const sub = ch.shape_str ? `${ch.shape_str} · ${ch.sublabel || ''}` : ch.sublabel || ''
    g.text(ch.cx - 30, ch.y + 38, sub.slice(0, 42), 12, 'mono', 'middle')
  }

  // Right-aligned badge
  if (ch.badge) {
    g.rect(ch.rightX - 70, ch.y + 12, 58, 20, 4, 'pill')
    g.text(ch.rightX - 41, ch.y + 26, ch.badge, 10, 'accent', 'middle', 700)
  }
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
