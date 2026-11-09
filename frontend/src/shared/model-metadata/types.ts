import type { ArchSpec as EngineArchSpec } from '../../engine/spec'

export type ArchSpec = EngineArchSpec

export const fmtParams = (n: any): string =>
  typeof n !== 'number' || !n
    ? '—'
    : n >= 1e9
      ? `${(n / 1e9).toFixed(2)}B`
      : n >= 1e6
        ? `${(n / 1e6).toFixed(1)}M`
        : n.toLocaleString()

export const fmtNum = (n: any): string =>
  typeof n !== 'number' || n == null ? '—' : n.toLocaleString()
