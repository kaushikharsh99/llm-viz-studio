import type { TensorInfo, MetadataCoverage, ConfidenceLevel } from './types'

export const DTYPE_BYTES: Record<string, number> = {
  BF16: 2,
  F16: 2,
  F32: 4,
  F64: 8,
  I8: 1,
  U8: 1,
  I16: 2,
  U16: 2,
  I32: 4,
  U32: 4,
  I64: 8,
  U64: 8,
  BOOL: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
}

// In-memory cache keyed by modelId@revision
interface CacheEntry {
  tensors: TensorInfo[]
  coverage: MetadataCoverage
  available: boolean
  note?: string | null
}

const INVENTORY_CACHE = new Map<string, CacheEntry>()
const SHARD_HEADER_CACHE = new Map<string, Record<string, any>>()

/**
 * Fetch a safetensors file header safely using HTTP Range requests.
 * Probes the first 128 KB, reads the 8-byte header length,
 * and fetches the remainder of the header if it exceeds 128 KB.
 * Never downloads weights.
 */
async function fetchShardHeader(
  fileUrl: string,
  cacheKey: string,
  signal?: AbortSignal
): Promise<Record<string, any> | null> {
  if (SHARD_HEADER_CACHE.has(cacheKey)) {
    return SHARD_HEADER_CACHE.get(cacheKey)!
  }

  try {
    // 128KB initial probe
    const probeRes = await fetch(fileUrl, {
      headers: { Range: 'bytes=0-131071' },
      signal,
    })

    if (!probeRes.ok && probeRes.status !== 206) {
      return null
    }

    const buf = await probeRes.arrayBuffer()
    if (buf.byteLength < 8) return null

    const view = new DataView(buf)
    const low = view.getUint32(0, true)
    const high = view.getUint32(4, true)
    const headerLength = low + high * 0x100000000

    // Safety checks: header length must be positive and reasonable (< 35 MB)
    if (headerLength <= 0 || headerLength > 35_000_000) {
      return null
    }

    let jsonBuf: ArrayBuffer
    if (headerLength <= buf.byteLength - 8) {
      jsonBuf = buf.slice(8, 8 + headerLength)
    } else {
      // Header is larger than initial 128 KB probe, fetch exact header range
      const endByte = 8 + headerLength - 1
      const restRes = await fetch(fileUrl, {
        headers: { Range: `bytes=8-${endByte}` },
        signal,
      })
      if (!restRes.ok && restRes.status !== 206) {
        return null
      }
      jsonBuf = await restRes.arrayBuffer()
    }

    const text = new TextDecoder('utf-8').decode(jsonBuf)
    const parsed = JSON.parse(text)
    SHARD_HEADER_CACHE.set(cacheKey, parsed)
    return parsed
  } catch {
    return null
  }
}

/**
 * Run tasks with a maximum concurrency limit
 */
async function runConcurrent<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let index = 0

  async function worker() {
    while (index < items.length) {
      const i = index++
      results[i] = await fn(items[i])
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () => worker())
  await Promise.all(workers)
  return results
}

export interface InventoryResult {
  tensors: TensorInfo[]
  available: boolean
  note?: string | null
  coverage: MetadataCoverage
}

/**
 * Fetch tensor inventory and metadata coverage for a model checkpoint.
 * Supports single safetensors files and sharded checkpoints via model.safetensors.index.json.
 * Inspects all shard headers up to a safe limit (default 24 shards) using concurrent bounded HTTP Range requests.
 */
export async function fetchInventory(
  modelId: string,
  signal?: AbortSignal,
  revision = 'main',
  activeLayer?: number | null
): Promise<InventoryResult> {
  const cacheKey = `${modelId}@${revision}`
  if (INVENTORY_CACHE.has(cacheKey)) {
    const cached = INVENTORY_CACHE.get(cacheKey)!
    return {
      tensors: cached.tensors,
      available: cached.available,
      note: cached.note,
      coverage: cached.coverage,
    }
  }

  try {
    // -------------------------------------------------------------
    // Strategy 1: Try single model.safetensors
    // -------------------------------------------------------------
    const singleUrl = `https://huggingface.co/${modelId}/resolve/${revision}/model.safetensors`
    const singleHeader = await fetchShardHeader(singleUrl, `${cacheKey}:single`, signal)

    if (singleHeader) {
      const tensors: TensorInfo[] = []
      for (const [name, meta] of Object.entries<any>(singleHeader)) {
        if (name === '__metadata__') continue
        const hasShape = Array.isArray(meta.shape)
        const shape: number[] | null = hasShape ? meta.shape : null
        const numel: number | null = shape ? shape.reduce((a, b) => a * b, 1) : null
        const dtype: string | null = meta.dtype ? String(meta.dtype).toUpperCase() : null
        const offsets = meta.data_offsets
        const nbytes = offsets && offsets.length === 2
          ? offsets[1] - offsets[0]
          : numel !== null && dtype && DTYPE_BYTES[dtype]
            ? numel * DTYPE_BYTES[dtype]
            : null

        tensors.push({
          name,
          shape,
          dtype,
          numel,
          bytes: nbytes,
          shard: 'model.safetensors',
          module: 'unknown',
          category: 'unknown',
          layer: null,
          expert: null,
          is_verified_shape: hasShape,
        })
      }

      tensors.sort((a, b) => a.name.localeCompare(b.name))

      const verifiedCount = tensors.filter(t => t.is_verified_shape).length
      const coverage: MetadataCoverage = {
        totalNamedTensors: tensors.length,
        tensorsWithVerifiedShapes: verifiedCount,
        tensorsWithoutShapeMetadata: tensors.length - verifiedCount,
        shardsInspected: 1,
        shardsNotInspected: 0,
        confidence: verifiedCount === tensors.length ? 'verified' : 'partial',
        modelRevision: revision,
      }

      const res: InventoryResult = {
        tensors,
        available: true,
        note: `Single-file safetensors checkpoint verified (${tensors.length} tensors)`,
        coverage,
      }
      INVENTORY_CACHE.set(cacheKey, res)
      return res
    }

    // -------------------------------------------------------------
    // Strategy 2: Sharded model via model.safetensors.index.json
    // -------------------------------------------------------------
    const indexUrl = `https://huggingface.co/${modelId}/raw/${revision}/model.safetensors.index.json`
    const indexRes = await fetch(indexUrl, { signal })

    if (indexRes.ok) {
      const indexJson = await indexRes.json()
      const weightMap: Record<string, string> = indexJson.weight_map || {}
      const allShards: string[] = Array.from(new Set(Object.values(weightMap)))

      // Decide which shards to inspect safely
      // Most models have 2-8 shards; larger ones have up to 30.
      // We inspect all shards if <= 24, or prioritize first 16 + active layer shard.
      const MAX_SHARDS_TO_INSPECT = 24
      const shardsToInspect = new Set<string>()

      if (allShards.length <= MAX_SHARDS_TO_INSPECT) {
        allShards.forEach(s => shardsToInspect.add(s))
      } else {
        // Inspect first 12 shards
        for (let i = 0; i < 12; i++) {
          shardsToInspect.add(allShards[i])
        }
        // Also inspect shards containing embedding, lm_head, and active layer
        for (const [name, shard] of Object.entries(weightMap)) {
          if (
            name.includes('embed_tokens') ||
            name.includes('lm_head') ||
            name.includes('model.norm') ||
            (activeLayer !== undefined && activeLayer !== null && name.includes(`layers.${activeLayer}.`))
          ) {
            shardsToInspect.add(shard)
          }
          if (shardsToInspect.size >= MAX_SHARDS_TO_INSPECT) break
        }
      }

      const shardsList = Array.from(shardsToInspect)

      // Fetch headers concurrently (pool of 6)
      const shardHeaderMap = new Map<string, Record<string, any>>()
      await runConcurrent(shardsList, 6, async shardName => {
        const shardUrl = `https://huggingface.co/${modelId}/resolve/${revision}/${shardName}`
        const header = await fetchShardHeader(shardUrl, `${cacheKey}:${shardName}`, signal)
        if (header) {
          shardHeaderMap.set(shardName, header)
        }
      })

      const tensors: TensorInfo[] = []
      let inspectedShardCount = 0
      for (const shard of allShards) {
        if (shardHeaderMap.has(shard)) {
          inspectedShardCount++
        }
      }

      for (const [name, shard] of Object.entries(weightMap)) {
        const shardMeta = shardHeaderMap.get(shard)
        const meta = shardMeta ? shardMeta[name] : null

        if (meta && Array.isArray(meta.shape)) {
          const shape: number[] = meta.shape
          const numel = shape.reduce((a, b) => a * b, 1)
          const dtype = meta.dtype ? String(meta.dtype).toUpperCase() : null
          const offsets = meta.data_offsets
          const nbytes = offsets && offsets.length === 2
            ? offsets[1] - offsets[0]
            : numel !== null && dtype && DTYPE_BYTES[dtype]
              ? numel * DTYPE_BYTES[dtype]
              : null

          tensors.push({
            name,
            shape,
            dtype,
            numel,
            bytes: nbytes,
            shard,
            module: 'unknown',
            category: 'unknown',
            layer: null,
            expert: null,
            is_verified_shape: true,
          })
        } else {
          // Explicit unknown shape, dtype, and numel — never default to BF16 or 1
          tensors.push({
            name,
            shape: null,
            dtype: null,
            numel: null,
            bytes: null,
            shard,
            module: 'unknown',
            category: 'unknown',
            layer: null,
            expert: null,
            is_verified_shape: false,
          })
        }
      }

      tensors.sort((a, b) => a.name.localeCompare(b.name))

      const verifiedCount = tensors.filter(t => t.is_verified_shape).length
      const shardsNotInspected = allShards.length - inspectedShardCount
      const confidence: ConfidenceLevel =
        verifiedCount === tensors.length
          ? 'verified'
          : verifiedCount > 0
            ? 'partial'
            : 'config'

      const coverage: MetadataCoverage = {
        totalNamedTensors: tensors.length,
        tensorsWithVerifiedShapes: verifiedCount,
        tensorsWithoutShapeMetadata: tensors.length - verifiedCount,
        shardsInspected: inspectedShardCount,
        shardsNotInspected,
        confidence,
        modelRevision: revision,
      }

      const note = shardsNotInspected === 0
        ? `All ${allShards.length} shards inspected (${verifiedCount} / ${tensors.length} tensors verified)`
        : `${inspectedShardCount} / ${allShards.length} shards inspected (${verifiedCount} verified, ${tensors.length - verifiedCount} uninspected)`

      const res: InventoryResult = {
        tensors,
        available: true,
        note,
        coverage,
      }
      INVENTORY_CACHE.set(cacheKey, res)
      return res
    }

    // -------------------------------------------------------------
    // Strategy 3: No safetensors found on Hugging Face
    // -------------------------------------------------------------
    const emptyCoverage: MetadataCoverage = {
      totalNamedTensors: 0,
      tensorsWithVerifiedShapes: 0,
      tensorsWithoutShapeMetadata: 0,
      shardsInspected: 0,
      shardsNotInspected: 0,
      confidence: 'unknown',
      modelRevision: revision,
    }

    return {
      tensors: [],
      available: false,
      note: 'No safetensors checkpoint found on Hugging Face (config only)',
      coverage: emptyCoverage,
    }
  } catch (err: any) {
    const errorCoverage: MetadataCoverage = {
      totalNamedTensors: 0,
      tensorsWithVerifiedShapes: 0,
      tensorsWithoutShapeMetadata: 0,
      shardsInspected: 0,
      shardsNotInspected: 0,
      confidence: 'unknown',
      modelRevision: revision,
    }
    return {
      tensors: [],
      available: false,
      note: `Safetensors metadata extraction error: ${err.message}`,
      coverage: errorCoverage,
    }
  }
}
