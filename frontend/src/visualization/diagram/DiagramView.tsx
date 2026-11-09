import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
import {
  fetchArchSpec,
  fetchModelInventory,
  downloadArchitectureDiagram,
  generateDiagramSvg,
} from '../../shared/api/api'
import { fmtParams, fmtNum, type ArchSpec } from '../../shared/model-metadata/types'
import type { TensorInfo } from '../../graph/types'
import { buildSemanticTree } from '../../engine/semantic/tree'
import { computeDiagramLayout } from '../../engine/semantic/layout'
import { renderSemanticDiagram } from '../../engine/semantic/render'
import type {
  SemanticNode,
  WorkspaceState,
  DetailPreset,
  LayoutedNode,
} from '../../engine/semantic/types'

export default function DiagramView({ model }: { model: string }) {
  const [spec, setSpec] = useState<ArchSpec | null>(null)
  const [tensors, setTensors] = useState<TensorInfo[]>([])
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  // Interactive Workspace State
  const [preset, setPreset] = useState<DetailPreset>(1)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set())
  const [selectedLayer, setSelectedLayer] = useState<number>(0)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null)

  // UI Panels & Canvas Viewport
  const [showInspector, setShowInspector] = useState(true)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [scale, setScale] = useState(1)
  const [tx, setTx] = useState(0)
  const [ty, setTy] = useState(0)
  const drag = useRef<any>(null)
  const stage = useRef<HTMLDivElement>(null)

  // Fetch model specifications and safetensors metadata
  useEffect(() => {
    const c = new AbortController()
    setLoading(true)
    setErr('')
    ;(async () => {
      try {
        const [s, inv] = await Promise.all([
          fetchArchSpec(model, c.signal),
          fetchModelInventory(model, c.signal).catch(() => ({
            tensors: [],
            available: false,
            coverage: undefined as any,
          })),
        ])
        setSpec(s)
        setTensors(inv.tensors || [])
        setLoading(false)
      } catch (e: any) {
        if (e.name !== 'AbortError') {
          setErr(e.message || 'Failed to load model specification')
          setLoading(false)
        }
      }
    })()
    return () => c.abort()
  }, [model])

  // Build semantic tree
  const semanticTree = useMemo(() => {
    if (!spec) return null
    return buildSemanticTree(spec, tensors, selectedLayer)
  }, [spec, tensors, selectedLayer])

  // Current workspace state object
  const workspaceState: WorkspaceState = useMemo(
    () => ({
      preset,
      expandedIds,
      collapsedIds,
      selectedLayer,
      selectedNodeId,
      focusedNodeId,
    }),
    [preset, expandedIds, collapsedIds, selectedLayer, selectedNodeId, focusedNodeId]
  )

  // Compute dynamic diagram layout
  const layout = useMemo(() => {
    if (!semanticTree || !spec) return null
    return computeDiagramLayout(semanticTree, spec, workspaceState)
  }, [semanticTree, spec, workspaceState])

  // Render SVG string
  const svgString = useMemo(() => {
    if (!layout || !spec) return ''
    return renderSemanticDiagram(layout, spec, workspaceState, {
      interactive: true,
      selectedNodeId,
      focusedNodeId,
    })
  }, [layout, spec, workspaceState, selectedNodeId, focusedNodeId])

  // Selected node lookup for inspector
  const selectedNode: LayoutedNode | SemanticNode | null = useMemo(() => {
    if (!selectedNodeId || !layout) return null
    return layout.allNodes.get(selectedNodeId) || null
  }, [selectedNodeId, layout])

  // Focus node lookup for breadcrumbs
  const focusedNode: LayoutedNode | SemanticNode | null = useMemo(() => {
    if (!focusedNodeId || !layout) return null
    return layout.allNodes.get(focusedNodeId) || null
  }, [focusedNodeId, layout])

  // Auto-fit to screen
  const fit = useCallback(() => {
    const el = stage.current
    if (!el || !layout) return
    const pad = 40
    const availW = el.clientWidth - pad * 2
    const availH = el.clientHeight - pad * 2
    const targetScale = Math.min(availW / layout.width, availH / layout.height, 1)
    setScale(Math.max(0.2, targetScale))
    setTx(Math.max(10, (el.clientWidth - layout.width * targetScale) / 2))
    setTy(20)
  }, [layout])

  useEffect(() => {
    if (layout && !drag.current) {
      fit()
    }
  }, [layout?.width, fit])

  // Mouse wheel zoom
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const fn = (e: WheelEvent) => {
      e.preventDefault()
      setScale(s => Math.min(6, Math.max(0.08, s * (e.deltaY < 0 ? 1.12 : 0.89))))
    }
    el.addEventListener('wheel', fn, { passive: false })
    return () => el.removeEventListener('wheel', fn)
  }, [])

  // Drag pan handlers
  const onDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.interactive-node')) {
      // Don't drag if clicking interactive element
    }
    drag.current = { x: e.clientX, y: e.clientY, tx, ty }
  }

  const onMove = (e: React.MouseEvent) => {
    if (drag.current) {
      setTx(drag.current.tx + e.clientX - drag.current.x)
      setTy(drag.current.ty + e.clientY - drag.current.y)
    }
  }

  // Interactive node click (progressive disclosure & selection)
  const handleStageClick = (e: React.MouseEvent) => {
    const target = (e.target as HTMLElement).closest('[data-node-id]') as HTMLElement | null
    if (!target) return

    const nodeId = target.getAttribute('data-node-id')
    if (!nodeId) return

    const isExpandable = target.getAttribute('data-expandable') === 'true'
    const isCurrentlyExpanded = target.getAttribute('data-expanded') === 'true'

    setSelectedNodeId(nodeId)
    setShowInspector(true)

    if (isExpandable) {
      if (isCurrentlyExpanded) {
        // Collapse
        setExpandedIds(prev => {
          const next = new Set(prev)
          next.delete(nodeId)
          return next
        })
        setCollapsedIds(prev => new Set(prev).add(nodeId))
      } else {
        // Expand
        setCollapsedIds(prev => {
          const next = new Set(prev)
          next.delete(nodeId)
          return next
        })
        setExpandedIds(prev => new Set(prev).add(nodeId))
      }
    }
  }

  // Preset switch handler
  const handleSelectPreset = (p: DetailPreset) => {
    setPreset(p)
    // Clear manual overrides to reflect chosen preset
    setExpandedIds(new Set())
    setCollapsedIds(new Set())
  }

  // Reset to default overview
  const handleResetDetail = () => {
    setPreset(1)
    setExpandedIds(new Set())
    setCollapsedIds(new Set())
    setFocusedNodeId(null)
    setSelectedNodeId(null)
    fit()
  }

  const isCustomExpanded = expandedIds.size > 0 || collapsedIds.size > 0

  if (err) {
    return (
      <main className="error" role="alert">
        <h2>Architecture Unavailable</h2>
        <p>{err}</p>
        <p>The configuration for {model} could not be parsed.</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </main>
    )
  }

  if (loading || !spec) {
    return (
      <main className="loading" role="status">
        <div className="spinner" />
        <p>Generating interactive architecture diagram…</p>
      </main>
    )
  }

  const rows = [
    ['Model ID', model],
    ['Architecture', spec.architecture],
    ['Model Type', spec.model_type],
    ['Total Parameters', fmtParams(spec.total_params)],
    ['Active Parameters', fmtParams(spec.active_params)],
    ['Layers', fmtNum(spec.num_layers)],
    ['Hidden Dimension', fmtNum(spec.hidden_size)],
    ['Attention Kind', spec.attention?.kind || '—'],
    ['Attention Heads', fmtNum(spec.attention?.num_heads)],
    ['KV Heads / Groups', fmtNum(spec.attention?.num_kv_heads)],
    ['FFN Intermediate', fmtNum(spec.intermediate_size)],
    ['Normalization', spec.norm_type || '—'],
    ['Position Encoding', spec.positional || '—'],
    ['Context Length', spec.context_length ? `${fmtNum(spec.context_length)} tokens` : '—'],
    ['Vocabulary Size', fmtNum(spec.vocab_size)],
  ]

  return (
    <div className="interactive-diagram-workspace">
      {/* Top Toolbar */}
      <div className="diagram-workspace-toolbar" role="toolbar" aria-label="Diagram Controls">
        {/* Left: Detail Slider (1 - 2 - 3) */}
        <div className="toolbar-left-group">
          <div className="detail-slider-control">
            <span className="control-label">Detail Level:</span>
            <div className="slider-pills" role="radiogroup" aria-label="Detail Preset">
              <button
                type="button"
                role="radio"
                aria-checked={preset === 1 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 1 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(1)}
                title="Preset 1: Publication Overview"
              >
                1 · Overview
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={preset === 2 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 2 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(2)}
                title="Preset 2: Primary Module Projections"
              >
                2 · Modules
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={preset === 3 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 3 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(3)}
                title="Preset 3: Deep Operations & Shapes"
              >
                3 · Deep
              </button>
            </div>
            {isCustomExpanded && <span className="custom-indicator-tag">Custom</span>}
          </div>

          {(isCustomExpanded || focusedNodeId) && (
            <button
              type="button"
              className="toolbar-btn btn-reset-detail"
              onClick={handleResetDetail}
              title="Reset all sections to base preset"
            >
              ↺ Reset
            </button>
          )}

          {/* Layer Selector */}
          {spec.num_layers > 1 && (
            <div className="layer-picker">
              <label htmlFor="diagram-layer-select">Layer:</label>
              <select
                id="diagram-layer-select"
                value={selectedLayer}
                onChange={e => setSelectedLayer(Number(e.target.value))}
              >
                {Array.from({ length: spec.num_layers }, (_, i) => (
                  <option key={i} value={i}>
                    Layer {i}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Right: Viewport Controls & 4-Mode Export Menu */}
        <div className="toolbar-right-group">
          <button type="button" className="toolbar-btn" onClick={fit} title="Fit to Screen">
            ⛶ Fit
          </button>
          <button
            type="button"
            className="toolbar-btn"
            onClick={() => setScale(s => Math.min(6, s * 1.2))}
            title="Zoom In"
          >
            +
          </button>
          <button
            type="button"
            className="toolbar-btn"
            onClick={() => setScale(s => Math.max(0.08, s / 1.2))}
            title="Zoom Out"
          >
            -
          </button>

          {/* Export Menu Dropdown */}
          <div className="export-dropdown-wrap">
            <button
              type="button"
              className="toolbar-btn btn-export-main"
              onClick={() => setShowExportMenu(!showExportMenu)}
              aria-expanded={showExportMenu}
            >
              Export Diagram ▾
            </button>

            {showExportMenu && (
              <div className="export-menu-dropdown" role="menu">
                <div className="export-menu-section-title">Publication Presets</div>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset1', 'svg')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 1 · Overview</b>
                  <small>SVG vector</small>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset1', 'png')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 1 · Overview</b>
                  <small>PNG (High-res)</small>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset2', 'svg')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 2 · Modules</b>
                  <small>SVG vector</small>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset2', 'png')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 2 · Modules</b>
                  <small>PNG (High-res)</small>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset3', 'svg')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 3 · Deep Detail</b>
                  <small>SVG vector</small>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'preset3', 'png')
                    setShowExportMenu(false)
                  }}
                >
                  <b>Preset 3 · Deep Detail</b>
                  <small>PNG (High-res)</small>
                </button>

                <div className="export-menu-divider" />
                <div className="export-menu-section-title">Current Workspace State</div>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'workspace', 'svg', workspaceState)
                    setShowExportMenu(false)
                  }}
                >
                  <b>Current Layout (Custom)</b>
                  <small>Exact expanded sections · SVG</small>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    downloadArchitectureDiagram(model, 'workspace', 'png', workspaceState)
                    setShowExportMenu(false)
                  }}
                >
                  <b>Current Layout (Custom)</b>
                  <small>Exact expanded sections · PNG</small>
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            className={`toolbar-btn ${showInspector ? 'is-active' : ''}`}
            onClick={() => setShowInspector(!showInspector)}
            title="Toggle Inspector"
          >
            ⓘ Inspector
          </button>
        </div>
      </div>

      {/* Focus Mode Breadcrumb Bar (if focused) */}
      {focusedNodeId && (
        <div className="focus-mode-banner">
          <div className="focus-breadcrumb">
            <span className="crumb-root">Model</span>
            <span className="crumb-sep">/</span>
            <span className="crumb-layer">Layer {selectedLayer}</span>
            <span className="crumb-sep">/</span>
            <span className="crumb-active">{focusedNode?.label || 'Component'}</span>
          </div>
          <button
            type="button"
            className="btn-exit-focus"
            onClick={() => {
              setFocusedNodeId(null)
              fit()
            }}
          >
            ← Return to Full Model Architecture
          </button>
        </div>
      )}

      {/* Main Stage & Canvas */}
      <div className="diagram-canvas-row">
        <div
          className="stage interactive-stage"
          ref={stage}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={() => (drag.current = null)}
          onMouseLeave={() => (drag.current = null)}
          onClick={handleStageClick}
          style={{ cursor: 'grab' }}
        >
          <div
            className="svg-viewport-wrapper"
            style={{
              transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
              transformOrigin: 'top left',
            }}
            dangerouslySetInnerHTML={{ __html: svgString }}
          />
        </div>

        {/* Collapsible Inspector Panel */}
        {showInspector && (
          <aside className="panel contextual-inspector-panel">
            <div className="panel-header">
              <h3>{selectedNode ? selectedNode.label : 'Architecture'}</h3>
              <button
                type="button"
                className="btn-close-panel"
                onClick={() => setShowInspector(false)}
                aria-label="Close Inspector"
              >
                ✕
              </button>
            </div>

            {selectedNode ? (
              <div className="inspector-selected-content">
                <div className="inspector-kind-row">
                  <span className="inspector-badge">{selectedNode.badge || selectedNode.kind.toUpperCase()}</span>
                  {selectedNode.is_expandable && (
                    <button
                      type="button"
                      className="btn-inline-toggle"
                      onClick={() => {
                        const id = selectedNode.id
                        if (expandedIds.has(id)) {
                          setExpandedIds(prev => {
                            const n = new Set(prev)
                            n.delete(id)
                            return n
                          })
                          setCollapsedIds(prev => new Set(prev).add(id))
                        } else {
                          setCollapsedIds(prev => {
                            const n = new Set(prev)
                            n.delete(id)
                            return n
                          })
                          setExpandedIds(prev => new Set(prev).add(id))
                        }
                      }}
                    >
                      {expandedIds.has(selectedNode.id) ? 'Collapse ▲' : 'Expand ▼'}
                    </button>
                  )}
                </div>

                {selectedNode.description && (
                  <p className="inspector-desc">{selectedNode.description}</p>
                )}

                <table className="inspector-specs-table">
                  <tbody>
                    {selectedNode.params !== null && selectedNode.params !== undefined && (
                      <tr>
                        <th>Parameters</th>
                        <td>{fmtParams(selectedNode.params)}</td>
                      </tr>
                    )}
                    {selectedNode.shape_str && (
                      <tr>
                        <th>Dimensions</th>
                        <td><code>{selectedNode.shape_str}</code></td>
                      </tr>
                    )}
                    {selectedNode.dtype && (
                      <tr>
                        <th>Dtype</th>
                        <td><code>{selectedNode.dtype}</code></td>
                      </tr>
                    )}
                    {selectedNode.evidence && (
                      <tr>
                        <th>Evidence</th>
                        <td className="evidence-cell">{selectedNode.evidence}</td>
                      </tr>
                    )}
                  </tbody>
                </table>

                {/* Focus Button */}
                {selectedNode.is_expandable && (
                  <button
                    type="button"
                    className="btn-focus-section"
                    onClick={() => {
                      setFocusedNodeId(selectedNode.id)
                      fit()
                    }}
                  >
                    ⛶ Focus Section in Isolation
                  </button>
                )}

                {/* Associated SafeTensors Table */}
                {selectedNode.tensors && selectedNode.tensors.length > 0 && (
                  <div className="inspector-tensors-box">
                    <h4>Associated SafeTensors ({selectedNode.tensors.length})</h4>
                    <div className="tensors-table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Tensor</th>
                            <th>Shape</th>
                            <th>Dtype</th>
                            <th>Params</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedNode.tensors.map(t => (
                            <tr key={t.name}>
                              <td title={t.name}><code>{t.name}</code></td>
                              <td>{t.shape ? `[${t.shape.join('×')}]` : '—'}</td>
                              <td>{t.dtype || '—'}</td>
                              <td>{t.numel ? fmtParams(t.numel) : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="inspector-model-overview">
                <table>
                  <tbody>
                    {rows.map(([k, v]) => (
                      <tr key={k}>
                        <th>{k}</th>
                        <td>{v ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">
                  Click any module directly in the diagram to inspect its internal operations and SafeTensors weights.
                </p>
                <a
                  href={`https://huggingface.co/${model}`}
                  target="_blank"
                  rel="noreferrer"
                  className="hf-external-link"
                >
                  Open on Hugging Face ↗
                </a>
              </div>
            )}
          </aside>
        )}
      </div>
    </div>
  )
}
