export const CANVAS = '#ffffff'
export const OUTER_GRAY = '#d5d5d5'
export const MID_GRAY = '#929292'
export const ATTN_DARK = '#5e5e5e'
export const INK = '#000000'
export const PILL = '#ffffff'
export const INSET_GRAY = '#cccccc'

export const ACCENTS: Record<string, string> = {
  black: '#000000',
  coral: '#ff664f',
  green: '#1e9e63',
  violet: '#8b5cf6',
  amber: '#e8961e',
  teal: '#0f9b9b',
}
const ACCENT_ORDER = Object.values(ACCENTS)

const FAMILY_ACCENT: Record<string, string> = {
  gpt_oss: '#000000',
  deepseek_v3: ACCENTS.coral,
  llama: '#000000',
  gpt2: '#000000',
  qwen3: ACCENTS.violet,
  qwen3_moe: ACCENTS.violet,
  gemma3: ACCENTS.teal,
  gemma3_text: ACCENTS.teal,
  mistral: ACCENTS.amber,
}

export function accentFor(modelType: string): string {
  const mt = (modelType || '').toLowerCase()
  for (const [fam, col] of Object.entries(FAMILY_ACCENT)) {
    if (mt.includes(fam)) return col
  }
  let sum = 0
  for (let i = 0; i < mt.length; i++) sum += mt.charCodeAt(i)
  return ACCENT_ORDER[sum % ACCENT_ORDER.length]
}

export function blockTint(accent: string): string {
  const a = accent.replace('#', '').toLowerCase()
  if (a === '000000' || a === '0f172a' || a === '1e293b') {
    return '#f1f5f9'
  }
  const ch = [0, 2, 4].map(i => Math.round(parseInt(a.slice(i, i + 2), 16) * 0.67 + 255 * 0.33))
  return '#' + ch.map(c => c.toString(16).padStart(2, '0')).join('')
}

export function accentOnDark(accent: string): string {
  const a = accent.replace('#', '').toLowerCase()
  if (a === '000000' || a === '0f172a' || a === '1e293b') {
    return '#ffffff'
  }
  const ch = [0, 2, 4].map(i => Math.round(parseInt(a.slice(i, i + 2), 16) * 0.55 + 255 * 0.45))
  return '#' + ch.map(c => c.toString(16).padStart(2, '0')).join('')
}

export const FONT = 'Arial, "Helvetica Neue", "Liberation Sans", "DejaVu Sans", sans-serif'
export const MONO = '"Courier New", "Liberation Mono", "DejaVu Sans Mono", monospace'

export const ANIM_CSS = '@keyframes llmviz-studio-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }'

export function scopeId(accent: string): string {
  return `llmviz-studio-${accent.replace('#', '')}`
}

function rules(accent: string): string {
  return `
  text { font-family: ${FONT}; fill: ${INK}; }
  .mono { font-family: ${MONO}; }
  .accent { fill: ${accent}; }
  .white { fill: #ffffff; }
  .surface { fill: ${CANVAS}; }
  .pill { fill: ${PILL}; stroke: ${INK}; stroke-width: 2.5; }
  .outer { fill: ${OUTER_GRAY}; stroke: ${INK}; stroke-width: 2.5; }
  .blockc { fill: ${blockTint(accent)}; stroke: ${INK}; stroke-width: 2.5; }
  .attnbox { fill: ${ATTN_DARK}; stroke: ${INK}; stroke-width: 2.5; }
  .inset { fill: ${INSET_GRAY}; stroke: ${INK}; stroke-width: 3; stroke-dasharray: 2 7; stroke-linecap: round; }
  .inset-accent { fill: ${CANVAS}; stroke: ${accent}; stroke-width: 3.5; stroke-dasharray: 2 8; stroke-linecap: round; }
  .moepill { fill: ${INSET_GRAY}; stroke: ${INK}; stroke-width: 2.5; }
  .router { fill: ${PILL}; stroke: ${accent}; stroke-width: 3.5; }
  .tagdark { fill: ${INK}; }
  .tagaccent { fill: ${accent}; }
  .flow { stroke: ${INK}; stroke-width: 2.5; fill: none; }
  .flowhead { fill: ${INK}; }
  .leader { stroke: ${INK}; stroke-width: 3; stroke-dasharray: 2 7; fill: none; stroke-linecap: round; }
  .leader-accent { stroke: ${accent}; stroke-width: 4; stroke-dasharray: 2 8; fill: none; stroke-linecap: round; }
  .plusc { fill: ${PILL}; stroke: ${INK}; stroke-width: 2.5; }
  .accent-dark { fill: ${accentOnDark(accent)}; }
`
}

export function stylesheet(accent: string): string {
  const sc = `#${scopeId(accent)}`
  const css = rules(accent)
  return css
    .split('\n')
    .map(line => {
      const trimmed = line.trim()
      if (trimmed && trimmed.includes('{') && !trimmed.startsWith('}') && !trimmed.startsWith('stroke') && !trimmed.startsWith('fill')) {
        return `  ${sc} ${trimmed}`
      }
      return line
    })
    .join('\n')
}
