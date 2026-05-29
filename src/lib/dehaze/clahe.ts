/**
 * CLAHE — Contrast-Limited Adaptive Histogram Equalization.
 *
 * Key improvement over a naive per-channel CLAHE:
 *   The equalization operates exclusively on the L* (luminance) channel of the
 *   CIE Lab color space.  Chrominance channels (a*, b*) are left untouched and
 *   passed through unchanged.  This prevents the color casts and hue shifts
 *   that result from equalizing R, G, B independently.
 *
 * `clipLimit` and `tiles` are user-configurable via ClaheOptions.
 * BINS (256) is an internal constant.
 */

import { rgbToLab, labToRgb } from './color-convert'
import type { ClaheOptions } from './types'

const BINS = 256

// ---------------------------------------------------------------------------
// Single-channel CLAHE
// ---------------------------------------------------------------------------

/**
 * Apply CLAHE to a single uint8 channel (0–255).
 * Returns a new Uint8Array of the same length.
 */
function claheChannel(
  ch: Uint8Array,
  W: number,
  H: number,
  tiles: number,
  clipLimit: number,
): Uint8Array {
  const tX = tiles
  const tY = tiles
  const tileCols = Math.ceil(W / tX)
  const tileRows = Math.ceil(H / tY)

  // Build per-tile CLUTs (cumulative lookup tables)
  const tileCLUTs: Uint8Array[] = []

  for (let ty = 0; ty < tY; ty++) {
    for (let tx = 0; tx < tX; tx++) {
      const x0 = tx * tileCols,  x1 = Math.min(W, x0 + tileCols)
      const y0 = ty * tileRows,  y1 = Math.min(H, y0 + tileRows)
      const tilePixels = (x1 - x0) * (y1 - y0)

      // Build histogram
      const hist = new Int32Array(BINS)
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++)
          hist[ch[y * W + x]]++

      // Clip and redistribute excess uniformly
      const limit = Math.max(1, Math.round(clipLimit * tilePixels / BINS))
      let excess = 0
      for (let b = 0; b < BINS; b++) {
        if (hist[b] > limit) { excess += hist[b] - limit; hist[b] = limit }
      }
      const base = Math.floor(excess / BINS)
      const rem  = excess % BINS
      for (let b = 0; b < BINS; b++) {
        hist[b] += base + (b < rem ? 1 : 0)
      }

      // Build CDF → CLUT
      const clut = new Uint8Array(BINS)
      let cdf = 0
      for (let b = 0; b < BINS; b++) {
        cdf += hist[b]
        clut[b] = Math.round((cdf / tilePixels) * 255)
      }
      tileCLUTs.push(clut)
    }
  }

  // Bilinear interpolation across tile CLUTs
  const result = new Uint8Array(W * H)

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const val = ch[y * W + x]

      // Fractional tile position (relative to tile centres)
      const fx = (x + 0.5) / tileCols - 0.5
      const fy = (y + 0.5) / tileRows - 0.5

      const tx0 = Math.max(0, Math.min(tX - 2, Math.floor(fx)))
      const ty0 = Math.max(0, Math.min(tY - 2, Math.floor(fy)))
      const tx1 = tx0 + 1
      const ty1 = ty0 + 1

      const wx = Math.max(0, Math.min(1, fx - tx0))
      const wy = Math.max(0, Math.min(1, fy - ty0))

      const v00 = tileCLUTs[ty0 * tX + tx0][val]
      const v10 = tileCLUTs[ty0 * tX + tx1][val]
      const v01 = tileCLUTs[ty1 * tX + tx0][val]
      const v11 = tileCLUTs[ty1 * tX + tx1][val]

      result[y * W + x] = Math.round(
        v00 * (1 - wx) * (1 - wy) +
        v10 *      wx  * (1 - wy) +
        v01 * (1 - wx) *      wy  +
        v11 *      wx  *      wy,
      )
    }
  }

  return result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function dehazeCLAHE(src: ImageData, opts: ClaheOptions): ImageData {
  const { width: W, height: H, data } = src
  const N = W * H

  // Unpack RGBA → separate R/G/B channels
  const R = new Uint8Array(N)
  const G = new Uint8Array(N)
  const B = new Uint8Array(N)
  for (let i = 0; i < N; i++) {
    R[i] = data[i * 4]
    G[i] = data[i * 4 + 1]
    B[i] = data[i * 4 + 2]
  }

  // Convert to Lab — equalize only the L* channel
  const { L, a, b_ } = rgbToLab(R, G, B)

  // Quantise L* (0–100) → uint8 (0–255), apply CLAHE, dequantise back
  const L_uint8 = new Uint8Array(N)
  for (let i = 0; i < N; i++) L_uint8[i] = Math.round(Math.min(100, Math.max(0, L[i])) * 2.55)

  const L_eq = claheChannel(L_uint8, W, H, opts.tiles, opts.clipLimit)

  const L_new = new Float32Array(N)
  for (let i = 0; i < N; i++) L_new[i] = L_eq[i] / 2.55

  // Convert back to RGB
  const { r, g, b } = labToRgb(L_new, a, b_)

  // Pack back into RGBA
  const out = new Uint8ClampedArray(N * 4)
  for (let i = 0; i < N; i++) {
    out[i * 4]     = r[i]
    out[i * 4 + 1] = g[i]
    out[i * 4 + 2] = b[i]
    out[i * 4 + 3] = data[i * 4 + 3]
  }

  return new ImageData(out, W, H)
}
