import { parseConfig, type ArchSpec } from '../../engine/spec'
import { fetchInventory, type InventoryResult } from '../../graph/tensors'
import type { TensorInfo } from '../../graph/types'
import { buildSemanticTree } from '../../engine/semantic/tree'
import { computeDiagramLayout } from '../../engine/semantic/layout'
import { renderSemanticDiagram } from '../../engine/semantic/render'
import type { WorkspaceState, DetailPreset } from '../../engine/semantic/types'

// In-memory cache for fast tab-switching and duplicate requests
const cache = {
  specs: new Map<string, Promise<ArchSpec>>(),
  inventories: new Map<string, Promise<InventoryResult>>(),
}

export async function searchModels(query: string, signal?: AbortSignal): Promise<any[]> {
  if (!query.trim()) return []
  const url = `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&limit=8&full=false`
  const res = await fetch(url, { signal })
  if (!res.ok) return []
  const list = await res.json()
  return list.map((m: any) => ({
    id: m.id,
    author: m.author || (m.id.includes('/') ? m.id.split('/')[0] : ''),
    pipeline_tag: m.pipeline_tag,
    downloads: m.downloads,
    likes: m.likes,
  }))
}

export function fetchArchSpec(modelId: string, signal?: AbortSignal): Promise<ArchSpec> {
  if (cache.specs.has(modelId)) return cache.specs.get(modelId)!
  const p = (async () => {
    const url = `https://huggingface.co/${modelId}/raw/main/config.json`
    const res = await fetch(url, { signal })
    if (!res.ok) {
      throw new Error(`Failed to fetch config for ${modelId} (${res.status})`)
    }
    const json = await res.json()
    return parseConfig(json, modelId)
  })()
  cache.specs.set(modelId, p)
  return p
}

export function fetchModelInventory(modelId: string, signal?: AbortSignal): Promise<InventoryResult> {
  if (cache.inventories.has(modelId)) return cache.inventories.get(modelId)!
  const p = fetchInventory(modelId, signal)
  cache.inventories.set(modelId, p)
  return p
}

/**
 * Generate standalone vector SVG for any preset (1, 2, 3) or custom workspace arrangement
 */
export async function generateDiagramSvg(
  modelId: string,
  mode: 'preset1' | 'preset2' | 'preset3' | 'workspace',
  workspaceState?: Partial<WorkspaceState>,
  signal?: AbortSignal
): Promise<string> {
  const [spec, inv] = await Promise.all([
    fetchArchSpec(modelId, signal),
    fetchModelInventory(modelId, signal).catch(() => ({ tensors: [], available: false, coverage: undefined as any })),
  ])

  const tree = buildSemanticTree(spec, inv.tensors, workspaceState?.selectedLayer || 0)

  let state: WorkspaceState
  if (mode === 'workspace' && workspaceState) {
    state = {
      preset: workspaceState.preset || 1,
      expandedIds: workspaceState.expandedIds || new Set(),
      collapsedIds: workspaceState.collapsedIds || new Set(),
      selectedLayer: workspaceState.selectedLayer || 0,
      selectedNodeId: null,
      focusedNodeId: workspaceState.focusedNodeId || null,
    }
  } else {
    const p: DetailPreset = mode === 'preset3' ? 3 : mode === 'preset2' ? 2 : 1
    state = {
      preset: p,
      expandedIds: new Set(),
      collapsedIds: new Set(),
      selectedLayer: workspaceState?.selectedLayer || 0,
      selectedNodeId: null,
      focusedNodeId: null,
    }
  }

  const layout = computeDiagramLayout(tree, spec, state)
  return renderSemanticDiagram(layout, spec, state, { interactive: false })
}

// Client-side SVG -> PNG conversion using HTML5 Canvas
export function svgToPngBlob(svgString: string, scale = 2): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    img.onload = () => {
      URL.revokeObjectURL(url)
      const w = img.naturalWidth || 1400
      const h = img.naturalHeight || 900
      const canvas = document.createElement('canvas')
      canvas.width = w * scale
      canvas.height = h * scale
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('Canvas context not available'))
        return
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      canvas.toBlob(b => {
        if (b) resolve(b)
        else reject(new Error('PNG export failed'))
      }, 'image/png')
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Failed to load SVG into image for PNG rendering'))
    }
    img.src = url
  })
}

export function triggerDownload(content: Blob | string, filename: string): void {
  const url = typeof content === 'string'
    ? 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(content)
    : URL.createObjectURL(content)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  if (typeof content !== 'string') {
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
}

/**
 * Advanced export with 4 export modes:
 * - preset1 (Overview)
 * - preset2 (Module Detail)
 * - preset3 (Deep Detail)
 * - workspace (Current Arrangement)
 */
export async function downloadArchitectureDiagram(
  modelId: string,
  mode: 'preset1' | 'preset2' | 'preset3' | 'workspace',
  format: 'svg' | 'png',
  workspaceState?: Partial<WorkspaceState>,
  scale = 2
): Promise<void> {
  const svg = await generateDiagramSvg(modelId, mode, workspaceState)
  const safeName = modelId.replace('/', '_')
  const filename = `${safeName}-${mode}.${format}`

  if (format === 'svg') {
    triggerDownload(svg, filename)
  } else {
    const blob = await svgToPngBlob(svg, scale)
    triggerDownload(blob, filename)
  }
}

// Backward-compatible wrapper for existing downloads
export async function downloadDiagram(modelId: string, format: 'svg' | 'png', scale = 2): Promise<void> {
  return downloadArchitectureDiagram(modelId, 'preset1', format, undefined, scale)
}

export function fetchDiagramSvg(modelId: string, signal?: AbortSignal): Promise<string> {
  return generateDiagramSvg(modelId, 'preset1', undefined, signal)
}

// Backward-compatibility wrappers
export async function fetchGraph(
  modelId: string,
  layer?: number | null,
  level = 0,
  signal?: AbortSignal
): Promise<any> {
  const spec = await fetchArchSpec(modelId, signal)
  return {
    model_id: modelId,
    architecture: spec.architecture,
    level,
    active_layer: layer,
    nodes: [],
    edges: [],
  }
}

export async function downloadGraphExport(
  modelId: string,
  format: 'svg' | 'png',
  level = 0,
  layer?: number | null
): Promise<void> {
  return downloadArchitectureDiagram(modelId, 'preset1', format)
}
