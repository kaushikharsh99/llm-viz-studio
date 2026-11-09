import { parseConfig, type ArchSpec } from '../../engine/spec'
import { renderModel } from '../../engine/render/block'
import { fetchInventory } from '../../graph/tensors'
import { buildGraph } from '../../graph/topology'
import { graphToSvg } from '../../graph/export'
import type { ArchGraph, TensorInfo } from '../../graph/types'

// In-memory cache for fast tab-switching and duplicate requests
const cache = {
  specs: new Map<string, Promise<ArchSpec>>(),
  inventories: new Map<string, Promise<{ tensors: TensorInfo[]; available: boolean; note?: string | null }>>(),
  diagrams: new Map<string, Promise<string>>(),
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

export function fetchDiagramSvg(modelId: string, signal?: AbortSignal): Promise<string> {
  if (cache.diagrams.has(modelId)) return cache.diagrams.get(modelId)!
  const p = (async () => {
    const spec = await fetchArchSpec(modelId, signal)
    return renderModel(spec)
  })()
  cache.diagrams.set(modelId, p)
  return p
}

export async function fetchGraph(
  modelId: string,
  layer?: number | null,
  signal?: AbortSignal
): Promise<ArchGraph> {
  const spec = await fetchArchSpec(modelId, signal)
  if (!cache.inventories.has(modelId)) {
    cache.inventories.set(modelId, fetchInventory(modelId, signal))
  }
  const inv = await cache.inventories.get(modelId)!
  return buildGraph(modelId, spec, inv.tensors, inv.available, inv.note, layer)
}

export async function fetchTensors(
  modelId: string,
  layer?: number | null,
  limit = 200,
  signal?: AbortSignal
): Promise<{ tensors: TensorInfo[]; total_tensors?: number; available: boolean }> {
  if (!cache.inventories.has(modelId)) {
    cache.inventories.set(modelId, fetchInventory(modelId, signal))
  }
  const inv = await cache.inventories.get(modelId)!
  let filtered = inv.tensors
  if (layer !== undefined && layer !== null) {
    filtered = filtered.filter(t => t.layer === layer)
  }
  return {
    tensors: filtered.slice(0, limit),
    total_tensors: filtered.length,
    available: inv.available,
  }
}

export async function fetchGraphSvg(modelId: string, layer?: number | null): Promise<string> {
  const g = await fetchGraph(modelId, layer)
  return graphToSvg(g)
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

export async function downloadDiagram(modelId: string, format: 'svg' | 'png', scale = 2): Promise<void> {
  const svg = await fetchDiagramSvg(modelId)
  const safeName = modelId.replace('/', '_')
  if (format === 'svg') {
    triggerDownload(svg, `${safeName}-architecture.svg`)
  } else {
    const blob = await svgToPngBlob(svg, scale)
    triggerDownload(blob, `${safeName}-architecture.png`)
  }
}

export async function downloadGraphExport(modelId: string, format: 'svg' | 'png', scale = 2): Promise<void> {
  const svg = await fetchGraphSvg(modelId)
  const safeName = modelId.replace('/', '_')
  if (format === 'svg') {
    triggerDownload(svg, `${safeName}-graph.svg`)
  } else {
    const blob = await svgToPngBlob(svg, scale)
    triggerDownload(blob, `${safeName}-graph.png`)
  }
}

// Backward-compatibility wrappers for legacy code
export const fetchJSON = async (url: string, signal?: AbortSignal) => {
  if (url.includes('/api/v1/models/search')) {
    const q = new URL(url, 'http://localhost').searchParams.get('q') || ''
    return searchModels(q, signal)
  }
  const res = await fetch(url, { signal })
  return res.json()
}
export const fetchText = async (url: string, signal?: AbortSignal) => {
  const res = await fetch(url, { signal })
  return res.text()
}
export const modelQuery = (model: string) => `model=${encodeURIComponent(model)}`
export const downloadURL = (model: string, format: 'svg' | 'png', scale = 2) => '#'
export const graphDownloadURL = (model: string, format: 'svg' | 'png', scale = 2) => '#'
