import { useEffect, useState, useRef, useCallback, useMemo, lazy, Suspense } from 'react'
import {
  fetchArchSpec,
  fetchModelInventory,
  downloadArchitectureDiagram,
} from '../../shared/api/api'
import { fmtParams, fmtNum, type ArchSpec } from '../../shared/model-metadata/types'
import type { TensorInfo } from '../../graph/types'
import { buildSemanticTree } from '../../engine/semantic/tree'
import { computeDiagramLayout } from '../../engine/semantic/layout'
import { renderSemanticDiagram } from '../../engine/semantic/render'
import { visibleTensorNodes } from '../../engine/semantic/tensorScene'
import { DEFAULT_CAMERA, ROTATION_KEYS, rotateCamera, type DiagramCamera } from '../../engine/semantic/orbit'
import type {
  WorkspaceState,
  DetailPreset,
  LayoutedNode,
} from '../../engine/semantic/types'

const InferenceWalkthrough = lazy(() => import('../inference/InferenceWalkthrough'))

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

  // In-Diagram Component Search
  const [searchQuery, setSearchQuery] = useState('')
  const [showSearchDropdown, setShowSearchDropdown] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  // Share Toast
  const [copiedToast, setCopiedToast] = useState(false)

  // Model Comparison Modal State
  const [showCompareModal, setShowCompareModal] = useState(false)
  const [compareModel, setCompareModel] = useState('HuggingFaceTB/SmolLM2-1.7B')
  const [compareSpec, setCompareSpec] = useState<ArchSpec | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)

  // UI Panels & Canvas Viewport
  const [showInference, setShowInference] = useState(false)
  const [spatial, setSpatial] = useState(() => new URLSearchParams(window.location.search).get('view') === '3d')
  const [camera, setCamera] = useState(DEFAULT_CAMERA)
  const [showInspector, setShowInspector] = useState(true)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [scale, setScale] = useState(1)
  const [tx, setTx] = useState(0)
  const [ty, setTy] = useState(0)
  const drag = useRef<{
    pointerId: number
    x: number
    y: number
    tx: number
    ty: number
    camera: DiagramCamera | null
    moved: boolean
  } | null>(null)
  const skipClick = useRef(false)
  const stage = useRef<HTMLDivElement>(null)
  const keyboardNode = useRef<string | null>(null)

  // Restore state from URL query parameters on initial mount
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search)
      const p = parseInt(params.get('preset') || '', 10)
      if (p === 1 || p === 2 || p === 3) setPreset(p as DetailPreset)
      const l = parseInt(params.get('layer') || '', 10)
      if (!isNaN(l) && l >= 0) setSelectedLayer(l)
      const n = params.get('node')
      if (n) setSelectedNodeId(n)
      const f = params.get('focus')
      if (f) setFocusedNodeId(f)
      const exp = params.get('expanded')
      if (exp) setExpandedIds(new Set(exp.split(',').filter(Boolean)))
      const collapsed = params.get('collapsed')
      if (collapsed) setCollapsedIds(new Set(collapsed.split(',').filter(Boolean)))
    } catch {
      // Ignore URL parsing errors
    }
  }, [])

  // Close search dropdown on click outside
  useEffect(() => {
    const fn = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchDropdown(false)
      }
    }
    document.addEventListener('mousedown', fn)
    return () => document.removeEventListener('mousedown', fn)
  }, [])

  // Fetch comparison model spec when compare modal opens
  useEffect(() => {
    if (!showCompareModal || !compareModel) return
    let active = true
    setCompareLoading(true)
    fetchArchSpec(compareModel)
      .then(s => {
        if (active) {
          setCompareSpec(s)
          setCompareLoading(false)
        }
      })
      .catch(() => {
        if (active) {
          setCompareSpec(null)
          setCompareLoading(false)
        }
      })
    return () => {
      active = false
    }
  }, [showCompareModal, compareModel])

  // Fetch model specifications and safetensors metadata
  useEffect(() => {
    // Metadata requests are shared across consumers, including Strict Mode remounts.
    let active = true
    setLoading(true)
    setErr('')
    ;(async () => {
      try {
        const [s, inv] = await Promise.all([
          fetchArchSpec(model),
          fetchModelInventory(model).catch(() => ({
            tensors: [],
            available: false,
            coverage: undefined as any,
          })),
        ])
        if (!active) return
        setSpec(s)
        setTensors(inv.tensors || [])
        setLoading(false)
      } catch (e: any) {
        if (active) {
          setErr(e.message || 'Failed to load model specification')
          setLoading(false)
        }
      }
    })()
    return () => { active = false }
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
      camera: spatial ? camera : undefined,
      viewportScale: spatial ? scale : undefined,
      selectedNodeId,
      focusedNodeId,
    })
  }, [layout, spec, workspaceState, selectedNodeId, focusedNodeId, spatial, camera, scale])

  useEffect(() => {
    if (!keyboardNode.current) return
    const nodes = stage.current?.querySelectorAll<SVGElement>('[data-node-id]')
    Array.from(nodes ?? []).find(node => node.dataset.nodeId === keyboardNode.current)?.focus()
    keyboardNode.current = null
  }, [svgString])

  const walkthrough = useMemo(() => layout && spatial ? visibleTensorNodes(layout) : [], [layout, spatial])
  const walkthroughIndex = walkthrough.findIndex(node => node.id === selectedNodeId)

  const moveWalkthrough = (offset: number) => {
    const next = walkthrough[walkthroughIndex + offset]
    if (next) setSelectedNodeId(next.id)
  }

  // Selected node lookup for inspector
  const selectedNode: LayoutedNode | null = useMemo(() => {
    if (!selectedNodeId || !layout) return null
    return layout.allNodes.get(selectedNodeId) || null
  }, [selectedNodeId, layout])

  // Focus node lookup for breadcrumbs
  const focusedNode: LayoutedNode | null = useMemo(() => {
    if (!focusedNodeId || !layout) return null
    return layout.allNodes.get(focusedNodeId) || null
  }, [focusedNodeId, layout])

  const diagramWidth = layout?.width
  const diagramHeight = layout?.height
  const fit = useCallback(() => {
    const el = stage.current
    if (!el || !diagramWidth || !diagramHeight || !el.clientWidth || !el.clientHeight) return
    const pad = 24
    const compactScene = spatial && el.clientWidth < 600
    const top = compactScene ? 76 : pad
    const bottom = compactScene ? 50 : pad
    const targetScale = Math.max(0.08, Math.min(
      (el.clientWidth - pad * 2) / diagramWidth,
      (el.clientHeight - top - bottom) / diagramHeight,
      1,
    ))
    setScale(targetScale)
    setTx((el.clientWidth - diagramWidth * targetScale) / 2)
    setTy(top + (el.clientHeight - top - bottom - diagramHeight * targetScale) / 2)
  }, [diagramWidth, diagramHeight, spatial])

  useEffect(() => {
    const el = stage.current
    if (!el) return
    fit()
    const observer = new ResizeObserver(fit)
    observer.observe(el)
    return () => observer.disconnect()
  }, [fit, loading, err, showInference])

  const zoomAt = useCallback((factor: number, x: number, y: number) => {
    const nextScale = Math.min(6, Math.max(0.08, scale * factor))
    const ratio = nextScale / scale
    setScale(nextScale)
    setTx(tx => x - (x - tx) * ratio)
    setTy(ty => y - (y - ty) * ratio)
  }, [scale])

  useEffect(() => {
    const el = stage.current
    if (!el) return
    const fn = (e: WheelEvent) => {
      e.preventDefault()
      const bounds = el.getBoundingClientRect()
      zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - bounds.left, e.clientY - bounds.top)
    }
    el.addEventListener('wheel', fn, { passive: false })
    return () => el.removeEventListener('wheel', fn)
  }, [zoomAt, loading, err, showInference])

  const zoomFromCenter = (factor: number) => {
    const el = stage.current
    if (el) zoomAt(factor, el.clientWidth / 2, el.clientHeight / 2)
  }

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !e.isPrimary || (e.target as Element).closest('button')) return
    skipClick.current = false
    const shouldOrbit = spatial && !e.shiftKey
    if (!shouldOrbit && (e.target as Element).closest('[data-node-id]')) return
    drag.current = {
      pointerId: e.pointerId,
      x: e.clientX,
      y: e.clientY,
      tx,
      ty,
      camera: shouldOrbit ? camera : null,
      moved: false,
    }
  }

  const onMove = (e: React.PointerEvent) => {
    const start = drag.current
    if (!start || start.pointerId !== e.pointerId) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    if (!start.moved) {
      if (Math.hypot(dx, dy) < 5) return
      e.currentTarget.setPointerCapture(e.pointerId)
      start.moved = true
      skipClick.current = true
    }

    if (start.camera) {
      setCamera(rotateCamera(start.camera, dx * 0.28, dy * 0.28))
    } else {
      setTx(start.tx + dx)
      setTy(start.ty + dy)
    }
  }

  const endDrag = (e: React.PointerEvent) => {
    if (drag.current?.pointerId === e.pointerId) drag.current = null
  }

  const toggleNode = (nodeId: string, expanded: boolean) => {
    setExpandedIds(ids => {
      const next = new Set(ids)
      if (expanded) next.delete(nodeId)
      else next.add(nodeId)
      return next
    })
    setCollapsedIds(ids => {
      const next = new Set(ids)
      if (expanded) next.add(nodeId)
      else next.delete(nodeId)
      return next
    })
  }

  const activateNode = (element: EventTarget) => {
    const target = (element as Element).closest('[data-node-id]')
    const nodeId = target?.getAttribute('data-node-id')
    if (!target || !nodeId) return

    const node = layout?.allNodes.get(nodeId)

    setSelectedNodeId(nodeId)
    setShowInspector(true)

    if (node?.is_expandable) toggleNode(nodeId, node.isExpanded)
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

  // Filter matching components for in-diagram search
  const searchMatches = useMemo(() => {
    if (!layout || !searchQuery.trim()) return []
    const q = searchQuery.toLowerCase().trim()
    const matches: LayoutedNode[] = []
    for (const node of layout.allNodes.values()) {
      if (
        node.label.toLowerCase().includes(q) ||
        node.id.toLowerCase().includes(q) ||
        (node.badge && node.badge.toLowerCase().includes(q)) ||
        (node.sublabel && node.sublabel.toLowerCase().includes(q))
      ) {
        matches.push(node)
      }
    }
    return matches.slice(0, 8)
  }, [layout, searchQuery])

  // Zoom to matched component
  const handleZoomToNode = (node: LayoutedNode) => {
    // If inside attention and collapsed, expand attention
    if (
      node.id.includes('q_proj') ||
      node.id.includes('k_proj') ||
      node.id.includes('v_proj') ||
      node.id.includes('o_proj') ||
      node.id.includes('rope') ||
      node.id.includes('attn')
    ) {
      const attnId = `layer_${selectedLayer}_attention`
      setExpandedIds(prev => new Set(prev).add(attnId))
      setCollapsedIds(prev => {
        const next = new Set(prev)
        next.delete(attnId)
        return next
      })
    }
    // If inside FFN and collapsed, expand FFN
    if (
      node.id.includes('gate') ||
      node.id.includes('up') ||
      node.id.includes('down') ||
      node.id.includes('act') ||
      node.id.includes('hadamard') ||
      node.id.includes('expert') ||
      node.id.includes('router')
    ) {
      const ffnId = `layer_${selectedLayer}_ffn`
      setExpandedIds(prev => new Set(prev).add(ffnId))
      setCollapsedIds(prev => {
        const next = new Set(prev)
        next.delete(ffnId)
        return next
      })
    }

    setSelectedNodeId(node.id)
    setShowInspector(true)
    setShowSearchDropdown(false)
    setSearchQuery('')

    const el = stage.current
    if (el) {
      const targetScale = Math.max(scale, 1.15)
      const volume = spatial ? Array.from(el.querySelectorAll<SVGGElement>('.tensor-volume'))
        .find(volume => volume.dataset.nodeId === node.id) : undefined
      const bounds = volume?.getBoundingClientRect()
      const stageBounds = el.getBoundingClientRect()
      const x = bounds ? (bounds.left - stageBounds.left + bounds.width / 2 - tx) / scale : node.cx
      const y = bounds ? (bounds.top - stageBounds.top + bounds.height / 2 - ty) / scale : node.topY + node.height / 2
      setScale(targetScale)
      setTx(el.clientWidth / 2 - x * targetScale)
      setTy(el.clientHeight / 2 - y * targetScale)
    }
  }

  // 1-Click Share: synchronizes exact view state into URL query params and copies to clipboard
  const handleShareView = () => {
    try {
      const params = new URLSearchParams(window.location.search)
      params.set('model', model)
      params.set('preset', String(preset))
      if (spatial) params.set('view', '3d')
      else params.delete('view')
      params.set('layer', String(selectedLayer))
      if (selectedNodeId) params.set('node', selectedNodeId)
      else params.delete('node')
      if (focusedNodeId) params.set('focus', focusedNodeId)
      else params.delete('focus')
      if (expandedIds.size > 0) params.set('expanded', Array.from(expandedIds).join(','))
      else params.delete('expanded')
      if (collapsedIds.size > 0) params.set('collapsed', Array.from(collapsedIds).join(','))
      else params.delete('collapsed')

      const newUrl = window.location.pathname + '?' + params.toString()
      window.history.replaceState(null, '', newUrl)
      navigator.clipboard.writeText(window.location.href)
      setCopiedToast(true)
      setTimeout(() => setCopiedToast(false), 2500)
    } catch {
      // Fallback
    }
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

  if (showInference) {
    return (
      <Suspense fallback={<main className="loading" role="status">Loading inference model…</main>}>
        <InferenceWalkthrough onClose={() => setShowInference(false)} />
      </Suspense>
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
          <div className="slider-pills diagram-view-switch" role="group" aria-label="Diagram view">
            <button type="button" aria-pressed={!spatial} onClick={() => setSpatial(false)}>2D</button>
            <button type="button" aria-pressed={spatial} onClick={() => setSpatial(true)}>3D</button>
          </div>
          <button type="button" className="toolbar-btn" onClick={() => setShowInference(true)}>
            Live inference
          </button>
          <div className="detail-slider-control">
            <div className="slider-pills" role="radiogroup" aria-label="Detail Preset">
              <button
                type="button"
                role="radio"
                aria-checked={preset === 1 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 1 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(1)}
                title="Preset 1: Publication Overview"
              >
                Overview
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={preset === 2 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 2 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(2)}
                title="Preset 2: Primary Module Projections"
              >
                Modules
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={preset === 3 && !isCustomExpanded}
                className={`slider-pill-btn ${preset === 3 ? 'is-active' : ''}`}
                onClick={() => handleSelectPreset(3)}
                title="Preset 3: Deep Operations & Shapes"
              >
                Deep
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

          {/* In-Diagram Component Search */}
          <div className="component-search-wrap" ref={searchRef}>
            <span className="component-search-icon">🔍</span>
            <input
              type="text"
              className="component-search-input"
              placeholder="Search (q_proj, silu, norm)..."
              value={searchQuery}
              onChange={e => {
                setSearchQuery(e.target.value)
                setShowSearchDropdown(true)
              }}
              onFocus={() => setShowSearchDropdown(true)}
            />
            {showSearchDropdown && searchMatches.length > 0 && (
              <div className="component-search-dropdown" role="listbox">
                {searchMatches.map(m => (
                  <button
                    key={m.id}
                    type="button"
                    className="search-match-item"
                    onClick={() => handleZoomToNode(m)}
                  >
                    <span className="search-match-label" title={m.label}>
                      {m.label}
                    </span>
                    <span className="search-match-badge">{m.badge || m.kind}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Viewport Controls, Compare, Share & 4-Mode Export Menu */}
        <div className="toolbar-right-group">
          <button
            type="button"
            className="toolbar-btn btn-compare"
            onClick={() => setShowCompareModal(true)}
            title="Compare with another architecture"
          >
            ⚖ Compare
          </button>
          <button
            type="button"
            className="toolbar-btn btn-share"
            onClick={handleShareView}
            title="Copy shareable link with current view state"
          >
            🔗 {copiedToast ? 'Copied!' : 'Share'}
          </button>
          <button type="button" className="toolbar-btn" onClick={fit} title="Fit to Screen">
            ⛶ Fit
          </button>
          <button
            type="button"
            className="toolbar-btn"
            onClick={() => zoomFromCenter(1.2)}
            title="Zoom In"
          >
            +
          </button>
          <button
            type="button"
            className="toolbar-btn"
            onClick={() => zoomFromCenter(1 / 1.2)}
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
              {spatial ? 'Export 2D' : 'Export Diagram'} ▾
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
          className={`stage interactive-stage${spatial ? ' is-spatial' : ''}`}
          role={spatial ? 'region' : undefined}
          aria-label={spatial ? '3D architecture canvas' : undefined}
          aria-describedby={spatial ? 'diagram-orbit-instructions' : undefined}
          tabIndex={spatial ? 0 : undefined}
          ref={stage}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onLostPointerCapture={endDrag}
          onClick={e => {
            if (!skipClick.current || e.detail === 0) activateNode(e.target)
            skipClick.current = false
          }}
          onKeyDown={e => {
            const rotation = spatial && e.target === e.currentTarget ? ROTATION_KEYS.get(e.key) : undefined
            if (rotation) {
              e.preventDefault()
              setCamera(previous => rotateCamera(previous, rotation[0], rotation[1]))
            }
            if ((e.key === 'Enter' || e.key === ' ') && (e.target as Element).hasAttribute('data-node-id')) {
              e.preventDefault()
              keyboardNode.current = (e.target as SVGElement).dataset.nodeId ?? null
              activateNode(e.target)
            }
          }}
        >
          <div
            className="svg-viewport-wrapper"
            style={{
              transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
              transformOrigin: 'top left',
            }}
            dangerouslySetInnerHTML={{ __html: svgString }}
          />
          {spatial && (
            <div className="tensor-scene-caption">
              <strong>{spec.name}</strong>
              <span>{focusedNodeId ? 'Component detail' : `${fmtNum(spec.num_layers)} decoder layers`}</span>
              <ul className="tensor-scene-key" aria-label="Tensor colors">
                <li><i className="tensor-key-weights" />Weights</li>
                <li><i className="tensor-key-activations" />Activations</li>
                <li><i className="tensor-key-operations" />Operations</li>
              </ul>
            </div>
          )}
          {spatial && (
            <div className="diagram-orbit-controls">
              <span id="diagram-orbit-instructions" title="Shift-drag to pan">Drag or use arrow keys to rotate</span>
              <button type="button" className="toolbar-btn" onClick={() => setCamera(DEFAULT_CAMERA)}>
                Reset angle
              </button>
            </div>
          )}
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

            {spatial && walkthrough.length > 0 && (
              <nav className="tensor-walkthrough" aria-label="Architecture walkthrough">
                <div>
                  <strong>Architecture walkthrough</strong>
                  <span aria-live="polite" className="tensor-step-count">
                    {walkthroughIndex < 0 ? `${walkthrough.length} components` : `${walkthroughIndex + 1} of ${walkthrough.length}`}
                  </span>
                </div>
                <div className="tensor-walkthrough-actions">
                  {walkthroughIndex >= 0 && (
                    <button type="button" className="toolbar-btn" disabled={walkthroughIndex === 0} onClick={() => moveWalkthrough(-1)}>
                      Previous
                    </button>
                  )}
                  <button type="button" className="toolbar-btn" disabled={walkthroughIndex >= walkthrough.length - 1} onClick={() => moveWalkthrough(1)}>
                    {walkthroughIndex < 0 ? 'Start walkthrough' : 'Next'}
                  </button>
                </div>
              </nav>
            )}

            {selectedNode ? (
              <div className="inspector-selected-content">
                <div className="inspector-kind-row">
                  <span className="inspector-badge">{selectedNode.badge || selectedNode.kind.toUpperCase()}</span>
                  {selectedNode.is_expandable && !focusedNodeId && (
                    <button
                      type="button"
                      className="btn-inline-toggle"
                      onClick={() => toggleNode(selectedNode.id, selectedNode.isExpanded)}
                    >
                      {selectedNode.isExpanded ? 'Collapse ▲' : 'Expand ▼'}
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
                {['attention', 'ffn', 'moe'].includes(selectedNode.kind) && focusedNodeId !== selectedNode.id && (
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

      {/* 1-Click Share Toast Notification */}
      {copiedToast && (
        <div className="share-toast" role="status">
          ✓ Link copied to clipboard with exact view state!
        </div>
      )}

      {/* Side-by-Side Model Architecture Comparison Modal */}
      {showCompareModal && (
        <div className="compare-modal-backdrop" onClick={() => setShowCompareModal(false)}>
          <div className="compare-modal" onClick={e => e.stopPropagation()}>
            <div className="compare-modal-header">
              <h3>Side-by-Side Architecture Comparison</h3>
              <button
                type="button"
                className="btn-close-panel"
                onClick={() => setShowCompareModal(false)}
                aria-label="Close Comparison"
              >
                ✕
              </button>
            </div>
            <div className="compare-modal-body">
              <div className="compare-selector-row">
                <label htmlFor="compare-model-input">Compare with Model:</label>
                <input
                  id="compare-model-input"
                  type="text"
                  value={compareModel}
                  onChange={e => setCompareModel(e.target.value)}
                  placeholder="e.g. HuggingFaceTB/SmolLM2-1.7B, Qwen/Qwen2.5-0.5B"
                />
              </div>

              {compareLoading ? (
                <div style={{ textAlign: 'center', padding: '30px' }}>
                  <div className="spinner" />
                  <p>Loading {compareModel} architecture...</p>
                </div>
              ) : compareSpec ? (
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th>Specification Metric</th>
                      <th className="col-model-a">{model}</th>
                      <th className="col-model-b">{compareModel}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Total Parameters</th>
                      <td className="col-model-a">{fmtParams(spec.total_params)}</td>
                      <td className="col-model-b">
                        {fmtParams(compareSpec.total_params)}
                        {spec.total_params !== compareSpec.total_params && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Layers</th>
                      <td className="col-model-a">{spec.num_layers}</td>
                      <td className="col-model-b">
                        {compareSpec.num_layers}
                        {spec.num_layers !== compareSpec.num_layers && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Hidden Dimension</th>
                      <td className="col-model-a">{spec.hidden_size}</td>
                      <td className="col-model-b">
                        {compareSpec.hidden_size}
                        {spec.hidden_size !== compareSpec.hidden_size && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Attention Kind</th>
                      <td className="col-model-a">{spec.attention?.kind || '—'}</td>
                      <td className="col-model-b">
                        {compareSpec.attention?.kind || '—'}
                        {spec.attention?.kind !== compareSpec.attention?.kind && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Attention Heads</th>
                      <td className="col-model-a">{spec.attention?.num_heads}</td>
                      <td className="col-model-b">
                        {compareSpec.attention?.num_heads}
                        {spec.attention?.num_heads !== compareSpec.attention?.num_heads && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>KV Heads / Groups</th>
                      <td className="col-model-a">{spec.attention?.num_kv_heads}</td>
                      <td className="col-model-b">
                        {compareSpec.attention?.num_kv_heads}
                        {spec.attention?.num_kv_heads !== compareSpec.attention?.num_kv_heads && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Intermediate Dimension</th>
                      <td className="col-model-a">{spec.intermediate_size}</td>
                      <td className="col-model-b">
                        {compareSpec.intermediate_size}
                        {spec.intermediate_size !== compareSpec.intermediate_size && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Normalization</th>
                      <td className="col-model-a">{spec.norm_type || '—'}</td>
                      <td className="col-model-b">
                        {compareSpec.norm_type || '—'}
                        {spec.norm_type !== compareSpec.norm_type && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Positional Embedding</th>
                      <td className="col-model-a">{spec.positional || '—'}</td>
                      <td className="col-model-b">
                        {compareSpec.positional || '—'}
                        {spec.positional !== compareSpec.positional && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Context Window</th>
                      <td className="col-model-a">
                        {spec.context_length ? `${fmtNum(spec.context_length)} tokens` : '—'}
                      </td>
                      <td className="col-model-b">
                        {compareSpec.context_length ? `${fmtNum(compareSpec.context_length)} tokens` : '—'}
                        {spec.context_length !== compareSpec.context_length && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Vocab Size</th>
                      <td className="col-model-a">{fmtNum(spec.vocab_size)}</td>
                      <td className="col-model-b">
                        {fmtNum(compareSpec.vocab_size)}
                        {spec.vocab_size !== compareSpec.vocab_size && (
                          <span className="compare-diff-badge">Diff</span>
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                  Could not load architecture configuration for {compareModel}. Ensure the model ID is correct and has a valid config.json.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
