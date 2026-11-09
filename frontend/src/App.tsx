import { useEffect, useState, useRef } from 'react'
import DiagramView from './visualization/diagram/DiagramView'
import { searchModels } from './shared/api/api'

const EXAMPLES = [
  { label: 'Qwen 2.5 (Dense)', id: 'Qwen/Qwen2.5-1.5B-Instruct' },
  { label: 'SmolLM2 (Llama-style)', id: 'HuggingFaceTB/SmolLM2-1.7B' },
  { label: 'DeepSeek-V3 (MLA + MoE)', id: 'deepseek-ai/DeepSeek-V3' },
  { label: 'Qwen 3 MoE', id: 'Qwen/Qwen3-30B-A3B' },
]

export function cleanModelId(raw: string): string | null {
  let s = raw.trim()
  if (!s) return null
  s = s.replace(/^https?:\/\/huggingface\.co\//i, '')
  s = s.replace(/^huggingface\.co\//i, '')
  s = s.replace(/\/(?:tree|blob)\/[^/]+(?:\/.*)?$/i, '')
  s = s.replace(/\/+$/, '')
  const parts = s.split('/')
  if (parts.length === 2 && parts[0] && parts[1]) {
    return `${parts[0]}/${parts[1]}`
  }
  if (parts.length === 1 && parts[0] && !parts[0].includes(' ')) {
    return parts[0]
  }
  return null
}

export default function App() {
  const [model, setModel] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search)
    return params.get('model')
  })

  useEffect(() => {
    const onPop = () => {
      const params = new URLSearchParams(window.location.search)
      setModel(params.get('model'))
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const selectModel = (id: string | null) => {
    setModel(id)
    const url = new URL(window.location.href)
    if (id) {
      url.searchParams.set('model', id)
    } else {
      url.searchParams.delete('model')
    }
    window.history.pushState(null, '', url.toString())
  }

  return (
    <div className="app-shell">
      <header className="nav">
        <div className="nav-left">
          <a
            href="#"
            className="brand"
            onClick={e => {
              e.preventDefault()
              selectModel(null)
            }}
          >
            <span className="brand-mark" aria-hidden="true">◈</span>
            <span className="brand-name">LLMViz Studio</span>
          </a>
        </div>

        <nav className="nav-links">
          <a
            href="#"
            onClick={e => {
              e.preventDefault()
              selectModel(null)
            }}
          >
            Explore
          </a>
          <a
            href="https://github.com/kaushikharsh99/llm-viz-studio"
            target="_blank"
            rel="noreferrer"
            className="github-nav-link"
          >
            GitHub ↗
          </a>
        </nav>
      </header>

      {model ? (
        <Viewer model={model} onSelectModel={selectModel} onHome={() => selectModel(null)} />
      ) : (
        <Home onSelect={selectModel} />
      )}

      <footer className="app-footer">
        <span>LLMViz Studio — Interactive LLM Architecture Diagram Explorer.</span>
        <span>100% Client-side serverless execution. Open source under Apache-2.0.</span>
      </footer>
    </div>
  )
}

function Home({ onSelect }: { onSelect: (id: string) => void }) {
  const [q, setQ] = useState('')
  const [sug, setSug] = useState<any[]>([])
  const [idx, setIdx] = useState(-1)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const cleaned = cleanModelId(q)
    if (!q.trim() || (cleaned && cleaned.includes('/'))) {
      setSug([])
      return
    }
    const c = new AbortController()
    setLoading(true)
    const t = setTimeout(() => {
      searchModels(q.trim(), c.signal)
        .then(setSug)
        .catch(() => {})
        .finally(() => setLoading(false))
    }, 280)
    return () => {
      clearTimeout(t)
      c.abort()
    }
  }, [q])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const chosen = idx >= 0 && sug[idx] ? sug[idx].id : q.trim()
    const cleaned = cleanModelId(chosen)
    if (!cleaned) {
      setErr('Enter a valid Hugging Face model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct) or full HF URL.')
      return
    }
    onSelect(cleaned)
  }

  return (
    <main className="home-page">
      <section className="hero">
        <div className="eyebrow-badge">INTERACTIVE ARCHITECTURE EXPLORER</div>
        <h1>Interactive Neural Network Architecture Diagrams</h1>
        <p className="hero-copy">
          Explore large language model architectures with progressive disclosure. Seamlessly navigate from high-level publication overviews down to individual Q/K/V projections, non-linear activations, and verified SafeTensors shapes.
        </p>

        <form className="search" onSubmit={submit} role="search">
          <div className="search-input-wrap">
            <input
              aria-label="Model search"
              placeholder="Paste Hugging Face URL or enter model ID (e.g. meta-llama/Llama-3.2-1B)…"
              value={q}
              onChange={e => {
                setQ(e.target.value)
                setIdx(-1)
                setErr('')
              }}
              onKeyDown={e => {
                if (e.key === 'ArrowDown') {
                  e.preventDefault()
                  setIdx(Math.min(idx + 1, sug.length - 1))
                }
                if (e.key === 'ArrowUp') {
                  e.preventDefault()
                  setIdx(Math.max(idx - 1, -1))
                }
                if (e.key === 'Escape') {
                  setSug([])
                }
              }}
            />
            {q && (
              <button type="button" className="clear-input-btn" onClick={() => setQ('')} aria-label="Clear search input">
                ×
              </button>
            )}
          </div>
          <button type="submit" className="search-submit-btn">
            {loading ? 'Searching…' : 'Explore'}
          </button>

          {sug.length > 0 && (
            <ul className="sug" role="listbox">
              {sug.map((s, i) => (
                <li
                  key={s.id}
                  role="option"
                  aria-selected={i === idx}
                  className={i === idx ? 'sel' : ''}
                  onMouseDown={() => onSelect(s.id)}
                >
                  <div className="sug-title">
                    <b>{s.id}</b>
                    {s.downloads && <span className="sug-downloads">↓ {s.downloads.toLocaleString()}</span>}
                  </div>
                  <div className="sug-meta">
                    <small>{s.author || ''} · {s.pipeline_tag || 'text-generation'}</small>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </form>

        {err && <p className="err" role="alert">{err}</p>}

        <div className="chips-container">
          <span className="chips-label">Popular models:</span>
          <div className="chips">
            {EXAMPLES.map(x => (
              <button key={x.id} type="button" onClick={() => onSelect(x.id)}>
                <span className="chip-tag">{x.label}</span>
                <code>{x.id}</code>
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="engine-showcase-cards">
        <article className="feature-card card-diagram">
          <div className="card-badge">DETAIL SLIDER</div>
          <h3>3 Presets &amp; Manual Expansion</h3>
          <p>
            Switch instantly between Overview (Preset 1), Module Detail (Preset 2), and Deep Operations (Preset 3). Click any section directly in the diagram to expand or collapse it on demand.
          </p>
          <ul className="feature-list">
            <li>✓ Progressive disclosure directly inside the diagram</li>
            <li>✓ Dynamic layout reflow with zero overlapping</li>
            <li>✓ Preserves publication-quality Raschka design tokens</li>
          </ul>
        </article>

        <article className="feature-card card-graph">
          <div className="card-badge">ISOLATED INSPECTION</div>
          <h3>Focus Mode &amp; Inspector</h3>
          <p>
            Study complex mechanisms like Multi-Head Attention, DeepSeek MLA, or MoE Routers in complete isolation with breadcrumb navigation back to the whole model architecture.
          </p>
          <ul className="feature-list">
            <li>✓ Focus individual subsystems in isolated diagrams</li>
            <li>✓ Inspect dimensions, formulas, and SafeTensors weights</li>
            <li>✓ Layer selector to examine any specific decoder block</li>
          </ul>
        </article>

        <article className="feature-card card-serverless">
          <div className="card-badge">PUBLICATION EXPORT</div>
          <h3>4-Mode Vector Export</h3>
          <p>
            Export clean vector SVG or high-resolution PNG for papers, slides, and documentation. Export any preset or your exact custom arrangement of expanded sections.
          </p>
          <ul className="feature-list">
            <li>✓ Export Preset 1, Preset 2, Preset 3, or Current Workspace</li>
            <li>✓ Standalone SVG with embedded typography</li>
            <li>✓ 100% Client-side serverless execution on Vercel</li>
          </ul>
        </article>
      </section>
    </main>
  )
}

function Viewer({
  model,
  onSelectModel,
  onHome,
}: {
  model: string
  onSelectModel: (id: string) => void
  onHome: () => void
}) {
  const [quickSearch, setQuickSearch] = useState('')
  const [quickSug, setQuickSug] = useState<any[]>([])
  const [showSearchDropdown, setShowSearchDropdown] = useState(false)
  const quickSearchRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!quickSearch.trim()) {
      setQuickSug([])
      return
    }
    const c = new AbortController()
    const t = setTimeout(() => {
      searchModels(quickSearch.trim(), c.signal)
        .then(setQuickSug)
        .catch(() => {})
    }, 250)
    return () => {
      clearTimeout(t)
      c.abort()
    }
  }, [quickSearch])

  return (
    <main className="viewer-layout">
      {/* Viewer Sub-Header */}
      <div className="viewer-top-bar">
        <div className="viewer-model-meta">
          <button type="button" className="btn-back-home" onClick={onHome} title="Return to home">
            ← Models
          </button>
          <div className="viewer-model-title-wrap">
            <span className="viewer-model-id">{model}</span>
          </div>

          {/* Quick Model Switcher */}
          <div className="quick-switcher-wrap" ref={quickSearchRef}>
            <input
              type="search"
              placeholder="Switch model…"
              value={quickSearch}
              onChange={e => {
                setQuickSearch(e.target.value)
                setShowSearchDropdown(true)
              }}
              onFocus={() => setShowSearchDropdown(true)}
              className="quick-switcher-input"
            />
            {showSearchDropdown && quickSug.length > 0 && (
              <ul className="quick-switcher-dropdown">
                {quickSug.map(s => (
                  <li
                    key={s.id}
                    onMouseDown={() => {
                      onSelectModel(s.id)
                      setQuickSearch('')
                      setShowSearchDropdown(false)
                    }}
                  >
                    <b>{s.id}</b>
                    <small>{s.author || ''}</small>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* Unified Interactive Diagram Viewport */}
      <div className="viewer-content">
        <DiagramView key={model} model={model} />
      </div>
    </main>
  )
}
