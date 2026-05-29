/**
 * Shared types for the dehaze module.
 */

export type DehazeMethod = 'dcp' | 'clahe'

export interface DcpOptions {
  /** Haze removal strength. Range: 0.3–1.0. Default: 0.75. */
  omega: number
  /** Dark-channel patch size in pixels (odd). Range: 5–31. Default: 15. */
  patchSize: number
}

export interface ClaheOptions {
  /** Clip limit multiplier (relative to average bin count). Range: 1.0–5.0. Default: 2.0. */
  clipLimit: number
  /** Tile grid subdivisions per axis. Default: 8. */
  tiles: 4 | 8 | 16
}

export type DehazeOptions =
  | { method: 'dcp';   dcp: DcpOptions }
  | { method: 'clahe'; clahe: ClaheOptions }
