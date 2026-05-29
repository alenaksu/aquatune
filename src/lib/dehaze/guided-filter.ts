/**
 * Fast Guided Filter (He & Sun 2015) — O(N) via separable box blur.
 *
 * Given a filtering input `p` and a guidance image `I` (both Float32Array,
 * same W×H), returns a filtered output that transfers structure from `I` to
 * smooth `p` while preserving edges.
 *
 * Used in DCP to refine the transmission map along image edges, replacing the
 * naive box-blur approximation that causes halo artifacts.
 *
 * Reference: K. He, J. Sun, "Fast Guided Filter", arXiv 1505.00996, 2015.
 */

// ---------------------------------------------------------------------------
// Separable box blur (integral image style, O(N))
// ---------------------------------------------------------------------------

/**
 * In-place horizontal prefix sum (for use with sliding-window mean).
 * Operates row by row.
 */
function boxBlurH(
  src: Float32Array,
  dst: Float32Array,
  W: number,
  H: number,
  r: number,
): void {
  for (let y = 0; y < H; y++) {
    const rowOff = y * W
    // Running sum over [x-r, x+r]
    let sum = 0
    // Initialise window for x=0
    const initEnd = Math.min(r, W - 1)
    for (let xi = 0; xi <= initEnd; xi++) sum += src[rowOff + xi]
    dst[rowOff] = sum / (initEnd + 1)

    for (let x = 1; x < W; x++) {
      const add = x + r < W ? src[rowOff + x + r] : 0
      const sub = x - r - 1 >= 0 ? src[rowOff + x - r - 1] : 0
      const left  = Math.max(0, x - r)
      const right = Math.min(W - 1, x + r)
      sum += add - sub
      dst[rowOff + x] = sum / (right - left + 1)
    }
  }
}

/**
 * In-place vertical box blur pass (column-wise sliding window).
 */
function boxBlurV(
  src: Float32Array,
  dst: Float32Array,
  W: number,
  H: number,
  r: number,
): void {
  for (let x = 0; x < W; x++) {
    let sum = 0
    const initEnd = Math.min(r, H - 1)
    for (let yi = 0; yi <= initEnd; yi++) sum += src[yi * W + x]
    dst[x] = sum / (initEnd + 1)

    for (let y = 1; y < H; y++) {
      const add = y + r < H ? src[(y + r) * W + x] : 0
      const sub = y - r - 1 >= 0 ? src[(y - r - 1) * W + x] : 0
      const top    = Math.max(0, y - r)
      const bottom = Math.min(H - 1, y + r)
      sum += add - sub
      dst[y * W + x] = sum / (bottom - top + 1)
    }
  }
}

/** Separable box blur: returns a new array (does not mutate src). */
function boxBlur(src: Float32Array, W: number, H: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length)
  const dst = new Float32Array(src.length)
  boxBlurH(src, tmp, W, H, r)
  boxBlurV(tmp, dst, W, H, r)
  return dst
}

// ---------------------------------------------------------------------------
// Guided filter
// ---------------------------------------------------------------------------

/**
 * Apply the fast guided filter.
 *
 * @param p   - Input to filter (e.g. transmission map), Float32Array W×H.
 * @param I   - Guidance image (e.g. gray luminance), Float32Array W×H.
 * @param W   - Image width in pixels.
 * @param H   - Image height in pixels.
 * @param r   - Filter radius (controls smoothing scale).
 * @param eps - Regularisation parameter (prevents over-sharpening; typical: 1e-3 to 1e-6).
 * @returns   Filtered output, Float32Array W×H.
 */
export function guidedFilter(
  p: Float32Array,
  I: Float32Array,
  W: number,
  H: number,
  r: number,
  eps: number,
): Float32Array {
  const N = W * H

  // Compute element-wise products
  const II = new Float32Array(N)
  const Ip = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    II[i] = I[i] * I[i]
    Ip[i] = I[i] * p[i]
  }

  // Box-blur all four inputs
  const meanI  = boxBlur(I,  W, H, r)
  const meanP  = boxBlur(p,  W, H, r)
  const corrI  = boxBlur(II, W, H, r)
  const corrIp = boxBlur(Ip, W, H, r)

  // Compute per-pixel a and b coefficients
  const a = new Float32Array(N)
  const b = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    const varI = corrI[i] - meanI[i] * meanI[i]
    a[i] = (corrIp[i] - meanI[i] * meanP[i]) / (varI + eps)
    b[i] = meanP[i] - a[i] * meanI[i]
  }

  // Smooth the coefficients
  const meanA = boxBlur(a, W, H, r)
  const meanB = boxBlur(b, W, H, r)

  // Reconstruct output: q = mean_a * I + mean_b
  const out = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    out[i] = meanA[i] * I[i] + meanB[i]
  }

  return out
}
