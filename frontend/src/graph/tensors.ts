import type { TensorInfo } from './types'

export const DTYPE_BYTES: Record<string, number> = {
  BF16: 2, F16: 2, F32: 4, F64: 8, I8: 1, U8: 1, I16: 2, U16: 2,
  I32: 4, U32: 4, I64: 8, U64: 8, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1,
}

export async function fetchInventory(
  modelId: string,
  signal?: AbortSignal
): Promise<{ tensors: TensorInfo[]; available: boolean; note?: string | null }> {
  try {
    // 1. Try single model.safetensors first via range request
    const singleUrl = `https://huggingface.co/${modelId}/resolve/main/model.safetensors`
    try {
      const res = await fetch(singleUrl, {
        headers: { Range: 'bytes=0-199999' },
        signal,
      })
      if (res.ok || res.status === 206) {
        const buf = await res.arrayBuffer()
        if (buf.byteLength >= 8) {
          const view = new DataView(buf)
          const low = view.getUint32(0, true)
          const high = view.getUint32(4, true)
          const headerLength = low + high * 0x100000000
          if (headerLength > 0 && headerLength < 50_000_000) {
            let jsonBuf = buf.slice(8, 8 + headerLength)
            if (jsonBuf.byteLength < headerLength) {
              // fetch the rest of the header
              const res2 = await fetch(singleUrl, {
                headers: { Range: `bytes=8-${8 + headerLength - 1}` },
                signal,
              })
              if (res2.ok || res2.status === 206) {
                jsonBuf = await res2.arrayBuffer()
              }
            }
            if (jsonBuf.byteLength >= headerLength) {
              const text = new TextDecoder('utf-8').decode(jsonBuf)
              const raw = JSON.parse(text)
              const tensors: TensorInfo[] = []
              for (const [name, meta] of Object.entries<any>(raw)) {
                if (name === '__metadata__') continue
                const shape: number[] = Array.isArray(meta.shape) ? meta.shape : []
                const numel = shape.reduce((a, b) => a * b, 1)
                const dtype = String(meta.dtype || '?').toUpperCase()
                const offsets = meta.data_offsets
                const nbytes = offsets && offsets.length === 2 ? offsets[1] - offsets[0] : numel * (DTYPE_BYTES[dtype] || 2)
                tensors.push({
                  name,
                  shape,
                  dtype,
                  numel,
                  bytes: nbytes,
                  shard: 'model.safetensors',
                  module: 'unknown',
                  category: 'unknown',
                })
              }
              tensors.sort((a, b) => a.name.localeCompare(b.name))
              return { tensors, available: true }
            }
          }
        }
      }
    } catch {
      // Continue to index.json
    }

    // 2. Try sharded model.safetensors.index.json
    const indexUrl = `https://huggingface.co/${modelId}/raw/main/model.safetensors.index.json`
    const indexRes = await fetch(indexUrl, { signal })
    if (indexRes.ok) {
      const indexJson = await indexRes.json()
      const weightMap: Record<string, string> = indexJson.weight_map || {}
      const tensors: TensorInfo[] = []
      const shards = Array.from(new Set(Object.values(weightMap)))
      const firstShard = shards[0]

      // Attempt to inspect first shard's header to get accurate shapes for early layers
      let firstShardMap: Record<string, any> = {}
      if (firstShard) {
        try {
          const shardUrl = `https://huggingface.co/${modelId}/resolve/main/${firstShard}`
          const sRes = await fetch(shardUrl, { headers: { Range: 'bytes=0-199999' }, signal })
          if (sRes.ok || sRes.status === 206) {
            const buf = await sRes.arrayBuffer()
            if (buf.byteLength >= 8) {
              const view = new DataView(buf)
              const low = view.getUint32(0, true)
              const high = view.getUint32(4, true)
              const headerLength = low + high * 0x100000000
              if (headerLength > 0 && headerLength <= buf.byteLength - 8) {
                const text = new TextDecoder('utf-8').decode(buf.slice(8, 8 + headerLength))
                firstShardMap = JSON.parse(text)
              }
            }
          }
        } catch {
          // ignore shard header failure
        }
      }

      for (const [name, shard] of Object.entries(weightMap)) {
        const meta = firstShardMap[name]
        const shape: number[] = meta && Array.isArray(meta.shape) ? meta.shape : []
        const numel = shape.reduce((a: number, b: number) => a * b, 1)
        const dtype = meta ? String(meta.dtype || '?').toUpperCase() : 'BF16'
        tensors.push({
          name,
          shape,
          dtype,
          numel,
          bytes: numel * (DTYPE_BYTES[dtype] || 2),
          shard,
          module: 'unknown',
          category: 'unknown',
        })
      }
      tensors.sort((a, b) => a.name.localeCompare(b.name))
      return { tensors, available: true }
    }

    return { tensors: [], available: false, note: 'No safetensors found on Hugging Face' }
  } catch (err: any) {
    return { tensors: [], available: false, note: `safetensors error: ${err.message}` }
  }
}
