import { SVG, fmtCount, fmtParams } from './svg'
import { accentFor, scopeId, stylesheet, ANIM_CSS } from './tokens'
import type { ArchSpec } from '../spec'

const TCX = 470
const OUTER_X = 150, OUTER_W = 640
const BLK_X = 215, BLK_W = 510
const RAIL_X = BLK_X + BLK_W - 60
const INSET_X = 900, INSET_W = 590

function pill(g: SVG, cx: number, y: number, w: number, h: number, label: string, size = 22): number {
  g.rect(cx - w / 2, y, w, h, 14, 'pill')
  g.text(cx, y + h / 2 + size * 0.36, label, size, '', 'middle')
  return y + h
}

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

const ACRONYMS = new Set(['gpt', 'gpt2', 'oss', 'glm', 'lfm', 'moe', 'ai', 'xl'])

function displayName(name: string): string {
  const words = name.trim().split(/[-_ ]+/)
  const out: string[] = []
  for (const w of words) {
    if (/^\d+(\.\d+)?[bm]$/i.test(w)) {
      out.push(`(${w.toUpperCase()})`)
    } else if (ACRONYMS.has(w.toLowerCase())) {
      out.push(w.toUpperCase())
    } else if (w === w.toLowerCase()) {
      out.push(w.charAt(0).toUpperCase() + w.slice(1))
    } else {
      out.push(w)
    }
  }
  return out.join(' ')
}

function vocabStr(n: number): string {
  return `${Math.round(n / 1000)}k`
}

function ctxStr(n: number): string {
  if (n >= 16384 && n % 1024 === 0) {
    return `${Math.floor(n / 1024)}k`
  }
  return n.toLocaleString()
}

function callout(g: SVG, x: number, y: number, lines: [string, string][][], anchor = 'start'): void {
  for (let i = 0; i < lines.length; i++) {
    g.richText(x, y + i * 30, lines[i], 21, anchor)
  }
}

function drawTower(g: SVG, s: ArchSpec, accent: string, xOff = 0, title = true): [number, { ffnPill: [number, number]; headsBottom: number }] {
  const cx = TCX + xOff
  const outerX = OUTER_X + xOff
  const blkX = BLK_X + xOff
  const railX = RAIL_X + xOff
  const outerIdx = g.mark()

  if (title) {
    g.text(cx, 74, displayName(s.name), 46, 'accent', 'middle', 800)
  }

  const outerTop = 165
  g.arrow(cx, 261, cx, 128)
  let y = pill(g, cx, 261, 310, 52, 'Linear output layer')
  const loBot = y
  y = pill(g, cx, y + 88, 260, 50, s.norm_type === 'rmsnorm' ? 'Final RMSNorm' : 'Final LayerNorm')
  const fnBot = y
  const fnTop = y - 50

  const norm = s.norm_type === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm'
  const sandwich = s.norm_mode === 'sandwich'
  const post = s.norm_mode === 'post'
  const blkTop = y + 92
  const blkIdx = g.mark()
  y = blkTop + 34

  const plus2 = y + 16
  g.plusNode(cx, plus2)
  y += 52
  let post2Bot: number | null = null
  let post2Top: number | null = null
  if (sandwich || post) {
    post2Top = y
    y = pill(g, cx, y, 240, 44, post ? `${norm} 2` : `Post-FFN ${norm}`, 19)
    post2Bot = y
    y += 34
  }
  const pillTop = y
  g.rect(cx - 105, y, 210, 48, 14, 'moepill')
  g.text(cx, y + 31, s.moe ? 'MoE' : 'Feed forward', 22, '', 'middle')
  const ffnPill: [number, number] = [cx + 105, y + 24]
  const pillBot = y + 48
  y += 48

  let rms2Top: number | null = null
  let rms2Bot: number | null = null
  if (!post) {
    rms2Top = y + 40
    y = pill(g, cx, y + 40, 240, 48, `${norm} 2`)
    rms2Bot = y
  } else {
    y += 8
    rms2Bot = y
  }
  const plus1 = y + 60
  g.plusNode(cx, plus1)
  y += 92

  let post1Bot: number | null = null
  let post1Top: number | null = null
  if (sandwich || post) {
    post1Top = y - 8
    y = pill(g, cx, y - 8, 240, 44, post ? `${norm} 1` : `Post-attn ${norm}`, 19)
    post1Bot = y
    y += 34
  }

  const attnH = 96
  g.rect(cx - 155, y, 310, attnH, 16, 'attnbox')
  const al = attnLines(s)
  for (let i = 0; i < al.length; i++) {
    g.richText(cx, y + 40 + i * 30, al[i], 23, 'middle', 400)
  }
  const attnY = y
  const attnBot = y + attnH
  y += attnH

  let rms1Top: number | null = null
  let rms1Bot: number | null = null
  if (!post) {
    rms1Top = y + 42
    y = pill(g, cx, y + 42, 240, 48, `${norm} 1`)
    rms1Bot = y
  } else {
    rms1Bot = y + 12
    y += 12
  }
  const blkBot = y + 36

  const PR = 15
  g.arrow(cx, fnTop, cx, loBot + 3)
  g.arrow(cx, plus2 - PR, cx, fnBot + 3)
  if (sandwich || post) {
    g.arrow(cx, post2Top!, cx, plus2 + PR + 2)
    g.arrow(cx, pillTop, cx, post2Bot! + 3)
    g.arrow(cx, post1Top!, cx, plus1 + PR + 2)
    g.arrow(cx, attnY, cx, post1Bot! + 3)
  } else {
    g.arrow(cx, pillTop, cx, plus2 + PR + 2)
    g.arrow(cx, attnY, cx, plus1 + PR + 2)
  }
  if (!post) {
    g.arrow(cx, rms2Top!, cx, pillBot + 3)
    g.arrow(cx, rms1Top!, cx, attnBot + 3)
    g.arrow(cx, plus1 - PR, cx, rms2Bot! + 3)
  } else {
    g.arrow(cx, plus1 - PR, cx, pillBot + 3)
  }

  const rails: [number, number][] = [[rms1Bot! + 20, plus1], [rms2Bot! + 22, plus2]]
  for (const [branchY, plusY] of rails) {
    g.path(`M ${cx} ${branchY} L ${railX} ${branchY} L ${railX} ${plusY}`)
    g.arrow(railX, plusY, cx + PR + 2, plusY)
  }

  g.rectAt(blkIdx, blkX, blkTop, BLK_W, blkBot - blkTop, 34, 'blockc')

  if (s.positional === 'rope') {
    const ropeCx = outerX - 5
    g.rect(ropeCx - 105, attnY + attnH / 2 - 26, 210, 52, 14, 'pill')
    g.text(ropeCx, attnY + attnH / 2 + 8, 'RoPE', 22, '', 'middle')
    g.arrow(ropeCx + 105, attnY + attnH / 2, cx - 157, attnY + attnH / 2)
  }

  g.text(blkX - 26, blkBot + 22, '{', 140, '', 'middle', 300)
  g.richText(blkX - 158, blkBot - 12, [[`${s.num_layers} ×`, 'accent']], 34)

  y = blkBot
  g.arrow(cx, y + 76, cx, rms1Bot! + 3)
  const embTop = y + 76
  if (s.positional === 'learned') {
    y = pill(g, cx, embTop, 360, 52, 'Positional embedding layer')
    const tokTop = y + 78
    g.arrow(cx, tokTop, cx, y + 4)
    y = pill(g, cx, tokTop, 340, 52, 'Token embedding layer')
  } else {
    y = pill(g, cx, embTop, 340, 52, 'Token embedding layer')
  }
  const embBot = y
  const outerBot = y + 44

  g.rectAt(outerIdx, outerX, outerTop, OUTER_W, outerBot - outerTop, 40, 'outer')

  g.arrow(cx, outerBot + 66, cx, embBot + 4)
  y = pill(g, cx, outerBot + 66, 250, 50, 'Tokenized text')
  g.arrow(cx, y + 62, cx, y + 4)
  g.text(cx, y + 100, 'Sample input text', 24, 'mono', 'middle')
  const bottom = y + 118

  if (s.vocab_size) {
    g.richText(cx + 190, 148, [['Vocabulary size of ', ''], [vocabStr(s.vocab_size), 'accent']], 21)
    g.leader(cx + 212, 162, cx + 140, 258)
  }

  if (s.context_length) {
    callout(g, xOff + 18, blkBot + 66, [
      [['Supported', '']],
      [['context length', '']],
      [['of ', ''], [ctxStr(s.context_length), 'accent'], [' tokens', '']],
    ])
    if (s.positional === 'rope') {
      g.leader(xOff + 44, blkBot + 40, xOff + 58, attnY + attnH / 2 + 32)
    } else if (s.positional === 'learned') {
      g.leader(xOff + 100, blkBot + 140, cx - 184, embTop + 30)
    }
  }

  const a = s.attention
  const headLines: [string, string][][] = [[[`${a.num_heads} `, 'accent'], ['heads', '']]]
  if (a.kind === 'GQA' || a.kind === 'MQA') {
    headLines.push([[`${a.num_kv_heads} `, 'accent'], ['KV groups', '']])
  }
  if (a.sliding_window) {
    headLines.push([['sliding window of ', ''], [fmtCount(a.sliding_window), 'accent'], [' tokens', '']])
  }
  if (a.qk_norm) {
    headLines.push([['QK-Norm on queries & keys', '']])
  }
  if (s.positional === 'alibi') {
    headLines.push([['ALiBi positional bias', '']])
  }
  if (s.parallel_block) {
    headLines.push([['parallel attention + FFN block', '']])
  }
  if (s.hybrid_note) {
    const runs: [string, string][] = []
    const parts = s.hybrid_note.split(/(\d+)/)
    for (const part of parts) {
      if (part) runs.push([part, /^\d+$/.test(part) ? 'accent' : ''])
    }
    headLines.push(runs)
  }
  callout(g, cx + 330, embTop + 44, headLines)
  g.leader(cx + 320, embTop + 20, cx + 152, attnBot - 8)
  const headsBottom = embTop + 44 + headLines.length * 30

  callout(g, cx + 210, embBot + 96, [
    [['Embedding', '']],
    [['dimension of ', ''], [s.hidden_size.toLocaleString(), 'accent']],
  ])
  g.leader(cx + 245, embBot + 66, cx + 176, embBot - 20)

  if (s.moe && s.moe.dense_layers) {
    callout(g, xOff + 14, outerBot + 58, [
      [
        [s.moe.dense_layers > 1 ? 'First ' : 'The first ', ''],
        [s.moe.dense_layers > 1 ? String(s.moe.dense_layers) : '', 'accent'],
        [s.moe.dense_layers > 1 ? ' blocks use' : 'block uses', ''],
      ],
      [['dense FFN with', '']],
      [['hidden size ', ''], [s.intermediate_size.toLocaleString(), 'accent']],
      [['instead of MoE', '']],
    ])
    g.leader(xOff + 252, outerBot + 44, blkX + 66, blkBot - 24)
  } else if (s.tied_embeddings) {
    callout(g, xOff + 14, outerBot + 58, [
      [['Output layer reuses the', '']],
      [['token embedding weights', '']],
    ])
    g.leader(xOff + 296, outerBot + 40, cx - 176, embBot - 20)
  }

  return [bottom, { ffnPill, headsBottom }]
}

function insetFfn(g: SVG, s: ArchSpec, y0: number): number {
  const gated = s.activation === 'silu' || s.activation === 'swiglu' || s.activation.includes('tanh')
  const actMap: Record<string, string> = { silu: 'SiLU', gelu_pytorch_tanh: 'GELU', gelu_new: 'GELU', gelu: 'GELU' }
  const act = actMap[s.activation] || s.activation
  const kind = s.activation === 'silu' ? 'SwiGLU' : (gated ? 'GeGLU' : act)
  const x = INSET_X, w = INSET_W, h = 330
  g.text(x + w / 2, y0 - 18, `FeedForward (${kind}) module`, 27, '', 'middle', 700)
  g.rect(x, y0, w, h, 34, 'inset')
  const cxl = x + w * 0.36
  pill(g, cxl, y0 + h - 88, 200, 52, 'Linear layer')
  g.arrow(cxl, y0 + h - 88, cxl, y0 + h - 142)

  if (gated) {
    pill(g, cxl, y0 + h - 194, 210, 52, `${act} activation`)
    const cxr = x + w * 0.79
    pill(g, cxr, y0 + h - 194, 180, 52, 'Linear layer')
    g.arrow(cxl, y0 + h - 194, cxl, y0 + 94)
    const hy = y0 + 112
    const ox = cxl + 116
    g.path(`M ${cxr} ${y0 + h - 194} L ${cxr} ${hy} L ${ox + 13} ${hy}`)
    g.otimesNode(ox, hy)
    g.line(ox - 13, hy, cxl + 1.5, hy)
    pill(g, cxl, y0 + 42, 200, 52, 'Linear layer')
  } else {
    pill(g, cxl, y0 + h - 194, 210, 52, `${act} activation`)
    g.arrow(cxl, y0 + h - 194, cxl, y0 + 94)
    pill(g, cxl, y0 + 42, 200, 52, 'Linear layer')
  }

  const dim = s.moe ? s.moe.moe_intermediate_size : s.intermediate_size
  const lines: [string, string][][] = s.moe
    ? [
        [['Input expert size: ', ''], [s.hidden_size.toLocaleString(), 'accent']],
        [['Intermediate projection size: ', ''], [dim.toLocaleString(), 'accent']],
      ]
    : [
        [['Intermediate hidden', '']],
        [['layer dimension of ', ''], [dim.toLocaleString(), 'accent']],
      ]
  callout(g, x + 76, y0 + h + 52, lines)
  g.leader(cxl + 66, y0 + h - 62, x + w * 0.42, y0 + h + 22)
  return y0 + h + 52 + lines.length * 30
}

function insetMoe(g: SVG, s: ArchSpec, y0: number): number {
  const m = s.moe!
  const x = INSET_X, w = INSET_W, h = 470
  g.rect(x, y0, w, h, 40, 'inset-accent')
  g.text(x + w - 46, y0 + 70, 'MoE layer', 27, '', 'end', 700)
  const cx = x + w / 2
  const pcx = cx, pcy = y0 + 104, prad = 18
  g.plusNode(pcx, pcy, prad)
  g.arrow(cx, pcy - prad - 2, cx, y0 + 36)

  const lw = 200, eh = 50
  const lx = x + 56, rx = x + w - 56 - lw
  const ey = y0 + 196
  const tags: [number, string, string][] = [[lx, '1', 'tagdark'], [rx, String(m.num_experts), 'tagaccent']]
  for (const [px, tag, tagCls] of tags) {
    g.rect(px, ey, lw, eh, 12, 'moepill')
    g.text(px + lw / 2, ey + 32, 'Feed forward', 21, '', 'middle')
    const tw = 36 + 12 * (tag.length - 1)
    const tx = px + lw - tw + 10
    g.rect(tx, ey + 34, tw, 36, 8, tagCls)
    g.text(tx + tw / 2, ey + 59, tag, 19, 'white', 'middle', 700)
  }
  g.text(cx, ey + 42, '· · ·', 30, '', 'middle', 700)

  const ry = y0 + h - 96
  g.rect(cx - 78, ry, 156, 56, 16, 'router')
  g.text(cx, ry + 36, 'Router', 22, '', 'middle')
  g.arrow(cx, ry + 56 + 34, cx, ry + 56 + 3)

  function edgeArrow(x1: number, y1: number, ccx: number, ccy: number, rr: number): void {
    const ang = Math.atan2(ccy - y1, ccx - x1)
    g.arrow(x1, y1, ccx - (rr + 2) * Math.cos(ang), ccy - (rr + 2) * Math.sin(ang))
  }

  g.arrow(cx - 40, ry, lx + lw * 0.62, ey + eh + 3)
  g.arrow(cx + 40, ry, rx + lw * 0.38, ey + eh + 3)
  g.arrow(cx, ry - 3, cx, ey + eh + 14)
  edgeArrow(lx + lw * 0.62, ey - 1, pcx, pcy, prad)
  edgeArrow(rx + lw * 0.38, ey - 1, pcx, pcy, prad)
  return y0 + h
}

function bulletsMoe(g: SVG, s: ArchSpec, y0: number): number {
  const m = s.moe!
  const x = INSET_X + 44
  g.richText(x, y0, [['Resource savings:', 'accent']], 23)
  const experts = (m.shared_experts ? `${m.shared_experts} (shared) + ` : '') + `${m.experts_per_tok} experts`
  const rows: [string, string][][] = [
    [['Model size is ', ''], [fmtParams(s.total_params), 'accent']],
    [['but only ', ''], [experts, 'accent'], [' active per token', '']],
    [['only ', ''], [fmtParams(s.active_params), 'accent'], [' parameters are', '']],
    [['active per inference step', '']],
  ]
  let y = y0 + 40
  for (let i = 0; i < rows.length; i++) {
    if (i !== 3) g.text(x + 4, y, '•', 22, '', 'start', 700)
    g.richText(x + 30, y, rows[i], 21, 'start', 400)
    y += 33
  }
  return y
}

function kvTable(g: SVG, s: ArchSpec, y0: number): number {
  if (!s.kv_cache_per_token_bytes || !s.context_length) return y0
  const x = INSET_X + 44
  g.richText(x, y0, [['KV cache at ', ''], [ctxStr(s.context_length), 'accent'], [' context', '']], 23)
  const perTok = s.kv_cache_per_token_bytes
  const full = perTok * s.context_length
  const rows: [string, string][] = [
    ['per token', perTok >= 1024 ? `${(perTok / 1024).toFixed(1)} KB` : `${perTok} B`],
    ['fp16 / bf16', `${(full / 2 ** 30).toFixed(1)} GB`],
    ['fp8', `${(full / 2 / 2 ** 30).toFixed(1)} GB`],
  ]
  let y = y0 + 38
  for (const [label, val] of rows) {
    g.text(x + 6, y, label, 20)
    g.text(x + 330, y, val, 20, '', 'end', 700)
    y += 32
  }
  if (s.attention.sliding_window) {
    g.text(x + 6, y, `(sliding-window layers cap at ${fmtCount(s.attention.sliding_window)} tokens)`, 16)
    y += 26
  }
  return y
}

export function renderModel(s: ArchSpec, options?: { accent?: string; animate?: boolean }): string {
  const accent = options?.accent || accentFor(s.model_type)
  const g = new SVG()
  const [bottom, coords] = drawTower(g, s, accent)
  const [fx, fy] = coords.ffnPill

  const ffnTop = 196
  let y = insetFfn(g, s, ffnTop)
  if (s.moe) {
    const moeTop = y + 60
    const y2 = insetMoe(g, s, moeTop)
    g.line(fx + 4, fy, INSET_X + 26, moeTop + 150, 'leader-accent', 4.5)
    g.leader(INSET_X + 156, moeTop + 196, INSET_X + 140, ffnTop + 330 - 26)
    y = bulletsMoe(g, s, Math.max(y2 + 56, coords.headsBottom + 40))
  } else {
    g.leader(fx + 4, fy, INSET_X - 26, ffnTop + 120)
  }
  y = kvTable(g, s, y + 64)

  const width = INSET_X + INSET_W + 60
  const height = Math.floor(Math.max(bottom + 30, y + 40))
  let css = stylesheet(accent)
  if (options?.animate) {
    g.animate()
    css += ANIM_CSS
  }
  return g.document(width, height, css, scopeId(accent))
}
