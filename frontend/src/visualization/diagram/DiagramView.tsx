// DiagramView — INDEPENDENT module. Owns its own fetching, zoom/pan state,
// rendering (backend SVG) and downloads. Must not import anything from ../graph.
import { useEffect, useState, useRef, useCallback } from 'react'
import { fetchJSON, fetchText, modelQuery, downloadURL } from '../../shared/api/api'
import { fmtParams, fmtNum, type ArchSpec } from '../../shared/model-metadata/types'

function sanitizeSvg(raw: string) {
  try {
    const doc = new DOMParser().parseFromString(raw, 'image/svg+xml')
    doc.querySelectorAll('script,foreignObject').forEach(e => e.remove())
    doc.querySelectorAll('*').forEach(e => { [...e.attributes].forEach(a => { if (a.name.startsWith('on') || a.value.trim().toLowerCase().startsWith('javascript:')) e.removeAttribute(a.name) }) })
    return new XMLSerializer().serializeToString(doc.documentElement)
  } catch { return raw.replace(/<script[\s\S]*?<\/script>/gi, '') }
}

export default function DiagramView({ model }: { model: string }) {
  const [spec, setSpec] = useState<ArchSpec | null>(null)
  const [svg, setSvg] = useState('')
  const [err, setErr] = useState('')
  const [showInfo, setShowInfo] = useState(true)
  const [scale, setScale] = useState(1); const [tx, setTx] = useState(0); const [ty, setTy] = useState(0)
  const drag = useRef<any>(null); const stage = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const c = new AbortController()
    ;(async () => {
      try {
        setSpec(await fetchJSON(`/api/v1/models/architecture?${modelQuery(model)}`, c.signal))
        setSvg(sanitizeSvg(await fetchText(`/api/v1/models/diagram?${modelQuery(model)}`, c.signal)))
      } catch (e: any) { if (e.name !== 'AbortError') setErr(e.message || 'Failed to load') }
    })()
    return () => c.abort()
  }, [model])

  const fit = useCallback(() => {
    const el = stage.current; const svgEl = el?.querySelector('svg')
    if (!el || !svgEl) { setScale(1); setTx(0); setTy(0); return }
    const vb = (svgEl.getAttribute('viewBox') || '').split(/\s+/).map(Number)
    const w = vb[2] || svgEl.clientWidth || 1000, h = vb[3] || svgEl.clientHeight || 1000
    setScale(Math.min(el.clientWidth / w, el.clientHeight / h, 1)); setTx(0); setTy(0)
  }, [svg])
  useEffect(() => { if (svg) fit() }, [svg, fit])
  useEffect(() => {
    const el = stage.current; if (!el) return
    const fn = (e: WheelEvent) => { e.preventDefault(); setScale(s => Math.min(8, Math.max(0.05, s * (e.deltaY < 0 ? 1.12 : 0.89)))) }
    el.addEventListener('wheel', fn, { passive: false }); return () => el.removeEventListener('wheel', fn)
  }, [])

  const onDown = (e: any) => { drag.current = { x: e.clientX, y: e.clientY, tx, ty } }
  const onMove = (e: any) => { if (drag.current) { setTx(drag.current.tx + e.clientX - drag.current.x); setTy(drag.current.ty + e.clientY - drag.current.y) } }

  if (err) return <main className="error"><h2>Architecture unavailable</h2><p>{err}</p><p>We couldn't generate a diagram for this model because its configuration is missing or unsupported.</p><button onClick={() => location.reload()}>Retry</button> · <a href={`https://huggingface.co/${model}`} target="_blank" rel="noreferrer">View on Hugging Face</a> · <a href="/">New model</a></main>
  if (!svg) return <main className="loading" role="status">Generating architecture diagram…</main>
  const rows = spec ? [['Model', model], ['Architecture', spec.architecture], ['Model type', spec.model_type], ['Total params (est.)', fmtParams(spec.total_params)], ['Active params (est.)', fmtParams(spec.active_params)], ['Layers', fmtNum(spec.num_layers)], ['Hidden size', fmtNum(spec.hidden_size)], ['Attention', spec.attention?.kind || '—'], ['Q heads', fmtNum(spec.attention?.num_heads)], ['KV heads', fmtNum(spec.attention?.num_kv_heads)], ['FFN intermediate', fmtNum(spec.intermediate_size)], ['MoE experts', fmtNum(spec.moe?.num_experts)], ['Experts/token', fmtNum(spec.moe?.experts_per_tok)], ['Context', spec.context_length ? fmtNum(spec.context_length) + ' tokens' : '—'], ['Vocab', fmtNum(spec.vocab_size)], ['Norm', spec.norm_type || '—'], ['Position', spec.positional || '—']] : []
  return <>
    <div className="toolbar diagram-toolbar"><b>{model}</b><a href="/">＋ New model</a><button onClick={fit}>Reset view</button><button onClick={fit}>Fit to screen</button><button onClick={() => setScale(s => Math.min(8, s * 1.2))}>Zoom in</button><button onClick={() => setScale(s => Math.max(0.05, s / 1.2))}>Zoom out</button><a href={downloadURL(model, 'svg')} download>Download SVG</a><a href={downloadURL(model, 'png')} download>Download PNG</a><button onClick={() => setShowInfo(!showInfo)}>{showInfo ? 'Hide' : 'Show'} info</button></div>
    <div className="stage" ref={stage} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={() => drag.current = null} onMouseLeave={() => drag.current = null} onTouchStart={(e: any) => { const t = e.touches[0]; drag.current = { x: t.clientX, y: t.clientY, tx, ty } }} onTouchMove={(e: any) => { const t = e.touches[0]; if (drag.current) { setTx(drag.current.tx + t.clientX - drag.current.x); setTy(drag.current.ty + t.clientY - drag.current.y) }}} style={{ cursor: 'grab' }}>
      <div style={{ transform: `translate(${tx}px,${ty}px) scale(${scale})`, transformOrigin: 'top left' }} dangerouslySetInnerHTML={{ __html: svg }} />
    </div>
    {showInfo && <aside className="panel"><h3>Architecture</h3><table><tbody>{rows.map(([k, v]: any) => <tr key={k}><th>{k}</th><td>{v ?? '—'}</td></tr>)}</tbody></table><p className="note">Parameter counts are estimates from config. Unknown fields are shown as —.</p><a href={`https://huggingface.co/${model}`} target="_blank" rel="noreferrer">Open on Hugging Face ↗</a></aside>}
  </>
}
