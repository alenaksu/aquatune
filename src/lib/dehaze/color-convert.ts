/**
 * RGB ↔ CIE Lab color space conversion.
 *
 * Pipeline: sRGB (uint8) → linear RGB (float) → XYZ D65 → Lab
 *
 * All arrays are flat, one value per pixel, in the same order as the source
 * ImageData (row-major).  No allocations are made inside the conversion
 * functions beyond the output arrays.
 */

// ---------------------------------------------------------------------------
// sRGB gamma
// ---------------------------------------------------------------------------

/** sRGB → linear (removes gamma). */
function srgbToLinear(v: number): number {
  const n = v / 255
  return n <= 0.04045 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4
}

/** Linear → sRGB (applies gamma), returns 0–255 clamped uint8. */
function linearToSrgb(v: number): number {
  const c = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055
  return Math.round(Math.min(1, Math.max(0, c)) * 255)
}

// ---------------------------------------------------------------------------
// XYZ D65 ↔ Lab
// ---------------------------------------------------------------------------

// D65 white point (CIE 1931 2°)
const D65_X = 0.95047
const D65_Y = 1.00000
const D65_Z = 1.08883

function xyzToLab(t: number): number {
  return t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116
}

function labToXyz(t: number): number {
  return t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface LabChannels {
  L:  Float32Array  // 0–100
  a:  Float32Array  // approx −128 – +127
  b_: Float32Array  // approx −128 – +127
}

export interface RgbChannels {
  r: Uint8Array
  g: Uint8Array
  b: Uint8Array
}

/**
 * Convert separate R/G/B uint8 arrays to CIE Lab float arrays.
 * Input channels must all have the same length N = W × H.
 */
export function rgbToLab(r: Uint8Array, g: Uint8Array, b: Uint8Array): LabChannels {
  const N = r.length
  const L  = new Float32Array(N)
  const a  = new Float32Array(N)
  const b_ = new Float32Array(N)

  for (let i = 0; i < N; i++) {
    // sRGB → linear
    const rl = srgbToLinear(r[i])
    const gl = srgbToLinear(g[i])
    const bl = srgbToLinear(b[i])

    // Linear RGB → XYZ D65 (IEC 61966-2-1 matrix)
    const X = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / D65_X
    const Y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750) / D65_Y
    const Z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / D65_Z

    // XYZ → Lab
    const fx = xyzToLab(X)
    const fy = xyzToLab(Y)
    const fz = xyzToLab(Z)

    L[i]  = 116 * fy - 16
    a[i]  = 500 * (fx - fy)
    b_[i] = 200 * (fy - fz)
  }

  return { L, a, b_ }
}

/**
 * Convert CIE Lab float arrays back to separate R/G/B uint8 arrays.
 * L is in [0, 100]; a and b_ are in approximately [−128, +127].
 */
export function labToRgb(L: Float32Array, a: Float32Array, b_: Float32Array): RgbChannels {
  const N = L.length
  const r = new Uint8Array(N)
  const g = new Uint8Array(N)
  const b = new Uint8Array(N)

  for (let i = 0; i < N; i++) {
    // Lab → XYZ
    const fy = (L[i] + 16) / 116
    const fx = a[i] / 500 + fy
    const fz = fy - b_[i] / 200

    const X = labToXyz(fx) * D65_X
    const Y = labToXyz(fy) * D65_Y
    const Z = labToXyz(fz) * D65_Z

    // XYZ → linear RGB (inverse of IEC 61966-2-1)
    const rl =  X *  3.2404542 + Y * -1.5371385 + Z * -0.4985314
    const gl =  X * -0.9692660 + Y *  1.8760108 + Z *  0.0415560
    const bl =  X *  0.0556434 + Y * -0.2040259 + Z *  1.0572252

    r[i] = linearToSrgb(rl)
    g[i] = linearToSrgb(gl)
    b[i] = linearToSrgb(bl)
  }

  return { r, g, b }
}
