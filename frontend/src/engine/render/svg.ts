import { textWidth } from './widths'

export function escapeHtml(str: string): string {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

export class SVG {
  parts: string[] = []

  rect(x: number, y: number, w: number, h: number, r = 6, cls = '', extra = ''): void {
    this.parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}" ${extra}/>`)
  }

  text(x: number, y: number, s: string | number, size = 13, cls = '', anchor = 'start', weight?: number | string): void {
    const w = weight ? ` font-weight="${weight}"` : ''
    this.parts.push(`<text x="${x}" y="${y}" font-size="${size}" class="${cls}" text-anchor="${anchor}"${w}>${escapeHtml(String(s))}</text>`)
  }

  richText(x: number, y: number, runs: [string, string][], size = 19, anchor = 'start', weight = 700): void {
    const bold = Boolean(weight) && Number(weight) >= 600
    const widths: number[] = []
    for (let i = 0; i < runs.length; i++) {
      const [t] = runs[i]
      const pad = (t.endsWith(' ') || (i + 1 < runs.length && runs[i + 1][0].startsWith(' '))) ? 0.14 * size : 0
      widths.push(textWidth(t, size, bold) + pad)
    }
    const total = widths.reduce((a, b) => a + b, 0)
    let cur = anchor === 'middle' ? x - total / 2 : (anchor === 'end' ? x - total : x)
    for (let i = 0; i < runs.length; i++) {
      const [t, cls] = runs[i]
      const tw = widths[i]
      this.parts.push(
        `<text x="${cur}" y="${y}" font-size="${size}" font-weight="${weight}" class="${cls}" xml:space="preserve">${escapeHtml(t)}</text>`
      )
      cur += tw
    }
  }

  line(x1: number, y1: number, x2: number, y2: number, cls = 'flow', width?: number): void {
    const w = width ? ` stroke-width="${width}"` : ''
    this.parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"${w}/>`)
  }

  path(d: string, cls = 'flow', extra = ''): void {
    this.parts.push(`<path d="${d}" class="${cls}" ${extra}/>`)
  }

  arrow(x1: number, y1: number, x2: number, y2: number, cls = 'flow', head = 17): void {
    const ang = Math.atan2(y2 - y1, x2 - x1)
    const bx = x2 - head * Math.cos(ang)
    const by = y2 - head * Math.sin(ang)
    const px = head * 0.38 * -Math.sin(ang)
    const py = head * 0.38 * Math.cos(ang)
    this.line(x1, y1, bx, by, cls)
    this.parts.push(
      `<path d="M ${bx + px} ${by + py} L ${x2} ${y2} L ${bx - px} ${by - py} Z" class="flowhead"/>`
    )
  }

  leader(x1: number, y1: number, x2: number, y2: number): void {
    this.line(x1, y1, x2, y2, 'leader')
  }

  circle(cx: number, cy: number, r: number, cls = '', extra = ''): void {
    this.parts.push(`<circle cx="${cx}" cy="${cy}" r="${r}" class="${cls}" ${extra}/>`)
  }

  plusNode(cx: number, cy: number, r = 15): void {
    this.circle(cx, cy, r, 'plusc')
    this.line(cx - r * 0.85, cy, cx + r * 0.85, cy)
    this.line(cx, cy - r * 0.85, cx, cy + r * 0.85)
  }

  otimesNode(cx: number, cy: number, r = 13): void {
    this.circle(cx, cy, r, 'plusc')
    const d = r * 0.4
    this.line(cx - d, cy - d, cx + d, cy + d)
    this.line(cx - d, cy + d, cx + d, cy - d)
  }

  raw(s: string): void {
    this.parts.push(s)
  }

  mark(): number {
    return this.parts.length
  }

  rectAt(idx: number, x: number, y: number, w: number, h: number, r = 6, cls = '', extra = ''): void {
    this.parts.splice(idx, 0, `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}" ${extra}/>`)
  }

  animate(step = 0.012, dur = 0.5): void {
    const out: string[] = []
    let i = 0
    for (let part of this.parts) {
      if (part.startsWith('<rect') || part.startsWith('<text') || part.startsWith('<line') || part.startsWith('<path') || part.startsWith('<circle')) {
        const style = `style="opacity:0; animation: llmviz-studio-in ${dur}s ease-out forwards ${(i * step).toFixed(3)}s"`
        part = part.replace(' ', ` ${style} `)
        i++
      }
      out.push(part)
    }
    this.parts = out
  }

  document(width: number, height: number, css: string, svgId = ''): string {
    const body = this.parts.join('\n')
    const idattr = svgId ? ` id="${svgId}"` : ''
    return `<svg xmlns="http://www.w3.org/2000/svg"${idattr} width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">\n<style>${css}</style>\n<rect width="${width}" height="${height}" class="surface"/>\n${body}\n</svg>`
  }
}

export function fmtParams(n: number): string {
  if (n >= 1e12) {
    const v = (n / 1e12).toFixed(1).replace(/\.?0+$/, '')
    return v + 'T'
  }
  if (n >= 1e9) {
    const v = (n / 1e9).toFixed(1).replace(/\.?0+$/, '')
    return v + 'B'
  }
  if (n >= 1e6) {
    return (n / 1e6).toFixed(0) + 'M'
  }
  return n.toLocaleString()
}

export function fmtCount(n: number): string {
  if (n >= 10000) {
    const k = n % 1024 === 0 ? n / 1024 : n / 1000
    return k.toFixed(1).replace(/\.?0+$/, '') + 'k'
  }
  return n.toLocaleString()
}

export function fmtTheta(theta: number): string {
  const t = Math.floor(theta)
  if (t >= 1000000 && t % 1000000 === 0) return `${t / 1000000}M`
  if (t >= 1000 && t % 1000 === 0) return `${t / 1000}k`
  return t.toLocaleString()
}
