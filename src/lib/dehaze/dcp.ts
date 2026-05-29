/**
 * Dark Channel Prior (DCP) dehazing — He et al. 2009.
 *
 * Improvements over the original paper as implemented here:
 *   - Transmission map is refined with a guided filter (He & Sun 2015) instead
 *     of a soft matting / naive box blur, eliminating halo artifacts at edges.
 *   - `omega` (haze removal strength) and `patchSize` are user-configurable.
 *
 * Internal constants (not exposed in UI):
 *   TOP_PCT  0.001  — top 0.1 % brightest dark-channel pixels used to
 *                      estimate atmospheric light A.
 *   T_MIN    0.1    — minimum transmission floor to prevent division blow-up
 *                      in very dense haze regions.
 */

import { guidedFilter } from './guided-filter'
import type { DcpOptions } from './types'

// Internal tuning knobs — not surfaced in the UI.
const TOP_PCT = 0.001  // fraction of pixels used for atmospheric light estimate
const T_MIN   = 0.1   // transmission floor

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** 2-D min-pool with radius r (separable passes, O(N·r) but simple). */
function minPool(src: Float32Array, W: number, H: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length)
  const dst = new Float32Array(src.length)

  // Horizontal pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let mn = Infinity
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r)
      for (let xi = x0; xi <= x1; xi++) mn = Math.min(mn, src[y * W + xi])
      tmp[y * W + x] = mn
    }
  }
  // Vertical pass
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      let mn = Infinity
      const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r)
      for (let yi = y0; yi <= y1; yi++) mn = Math.min(mn, tmp[yi * W + x])
      dst[y * W + x] = mn
    }
  }
  return dst
}

function clamp255(v: number): number {
  return Math.round(Math.min(1, Math.max(0, v)) * 255)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function dehazeDCP(src: ImageData, opts: DcpOptions): ImageData {
  const { width: W, height: H, data } = src
  const N = W * H
  const r = Math.floor(opts.patchSize / 2)

  // --- 1. Extract float channels -------------------------------------------
  const R = new Float32Array(N)
  const G = new Float32Array(N)
  const B = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    R[i] = data[i * 4]     / 255
    G[i] = data[i * 4 + 1] / 255
    B[i] = data[i * 4 + 2] / 255
  }

  // --- 2. Dark channel: per-pixel RGB min then min-pool over patch ----------
  const dark = new Float32Array(N)
  for (let i = 0; i < N; i++) dark[i] = Math.min(R[i], G[i], B[i])
  const darkPooled = minPool(dark, W, H, r)

  // --- 3. Atmospheric light A: mean of top TOP_PCT pixels ------------------
  const numPx    = Math.max(1, Math.round(N * TOP_PCT))
  const sorted   = darkPooled.slice().sort((a, b) => b - a)
  const threshold = sorted[numPx - 1]

  let aR = 0, aG = 0, aB = 0, aCount = 0
  for (let i = 0; i < N; i++) {
    if (darkPooled[i] >= threshold) {
      aR += R[i]; aG += G[i]; aB += B[i]
      aCount++
    }
  }
  aR /= aCount; aG /= aCount; aB /= aCount

  // --- 4. Transmission estimate: t = 1 - omega * darkChannel(I/A) ----------
  const tRaw = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    tRaw[i] = Math.min(
      aR > 0 ? R[i] / aR : 0,
      aG > 0 ? G[i] / aG : 0,
      aB > 0 ? B[i] / aB : 0,
    )
  }
  const tPooled = minPool(tRaw, W, H, r)
  for (let i = 0; i < N; i++) {
    tPooled[i] = Math.max(T_MIN, 1 - opts.omega * tPooled[i])
  }

  // --- 5. Transmission refinement: guided filter (guide = gray image) ------
  const gray = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    gray[i] = 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i]
  }
  const tRefined = guidedFilter(tPooled, gray, W, H, r, 1e-3)

  // --- 6. Scene radiance recovery: J = (I - A) / t + A --------------------
  const out = new Uint8ClampedArray(N * 4)
  for (let i = 0; i < N; i++) {
    const ti = Math.max(T_MIN, tRefined[i])
    out[i * 4]     = clamp255((R[i] - aR) / ti + aR)
    out[i * 4 + 1] = clamp255((G[i] - aG) / ti + aG)
    out[i * 4 + 2] = clamp255((B[i] - aB) / ti + aB)
    out[i * 4 + 3] = data[i * 4 + 3]
  }

  return new ImageData(out, W, H)
}
