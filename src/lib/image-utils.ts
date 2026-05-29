/**
 * Image utility functions for LU2Net pre/post processing.
 *
 * LU2Net expects:
 *   - Input:  float32 tensor [1, 3, H, W], values in [0, 1]
 *   - Output: float32 tensor [1, 3, H, W], values in [0, 1]  (unclamped from model)
 *
 * H and W must each be divisible by 4 (two 2x downsamplings in the encoder).
 */

/** Pad a value up to the nearest multiple of `n`. */
export function padToMultiple(value: number, n: number): number {
  return Math.ceil(value / n) * n
}

/**
 * Convert an ImageData (RGBA, uint8) to a Float32Array in NCHW layout [1,3,H,W],
 * normalized to [0, 1]. Input is padded to nearest multiple of 4 if needed.
 *
 * Returns the tensor data and the padded dimensions so the padding can be
 * reversed on the output.
 */
export function imageDataToTensor(imageData: ImageData): {
  data: Float32Array
  paddedW: number
  paddedH: number
} {
  const { width: W, height: H, data: rgba } = imageData
  const paddedW = padToMultiple(W, 4)
  const paddedH = padToMultiple(H, 4)

  // [1, 3, paddedH, paddedW]
  const tensor = new Float32Array(paddedH * paddedW * 3)

  const rOffset = 0
  const gOffset = paddedH * paddedW
  const bOffset = paddedH * paddedW * 2

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const srcIdx = (y * W + x) * 4
      const dstIdx = y * paddedW + x
      tensor[rOffset + dstIdx] = rgba[srcIdx]     / 255
      tensor[gOffset + dstIdx] = rgba[srcIdx + 1] / 255
      tensor[bOffset + dstIdx] = rgba[srcIdx + 2] / 255
    }
    // Padded columns: replicate the last valid pixel
    for (let x = W; x < paddedW; x++) {
      const srcIdx = (y * W + (W - 1)) * 4
      const dstIdx = y * paddedW + x
      tensor[rOffset + dstIdx] = rgba[srcIdx]     / 255
      tensor[gOffset + dstIdx] = rgba[srcIdx + 1] / 255
      tensor[bOffset + dstIdx] = rgba[srcIdx + 2] / 255
    }
  }
  // Padded rows: replicate the last valid row
  for (let y = H; y < paddedH; y++) {
    for (let x = 0; x < paddedW; x++) {
      const srcIdx = ((H - 1) * W + Math.min(x, W - 1)) * 4
      const dstIdx = y * paddedW + x
      tensor[rOffset + dstIdx] = rgba[srcIdx]     / 255
      tensor[gOffset + dstIdx] = rgba[srcIdx + 1] / 255
      tensor[bOffset + dstIdx] = rgba[srcIdx + 2] / 255
    }
  }

  return { data: tensor, paddedW, paddedH }
}

/**
 * Convert a Float32Array NCHW output tensor [1, 3, paddedH, paddedW]
 * back to an ImageData of the original (unpadded) dimensions.
 */
export function tensorToImageData(
  tensor: Float32Array,
  originalW: number,
  originalH: number,
  paddedW: number,
  _paddedH: number,
): ImageData {
  const rgba = new Uint8ClampedArray(originalW * originalH * 4)

  const rOffset = 0
  const gOffset = paddedW * _paddedH
  const bOffset = paddedW * _paddedH * 2

  for (let y = 0; y < originalH; y++) {
    for (let x = 0; x < originalW; x++) {
      const srcIdx = y * paddedW + x
      const dstIdx = (y * originalW + x) * 4
      rgba[dstIdx]     = Math.round(Math.min(1, Math.max(0, tensor[rOffset + srcIdx])) * 255)
      rgba[dstIdx + 1] = Math.round(Math.min(1, Math.max(0, tensor[gOffset + srcIdx])) * 255)
      rgba[dstIdx + 2] = Math.round(Math.min(1, Math.max(0, tensor[bOffset + srcIdx])) * 255)
      rgba[dstIdx + 3] = 255
    }
  }

  return new ImageData(rgba, originalW, originalH)
}

/**
 * Extract a padded tile from an ImageData for tiled inference.
 *
 * The tile covers the output region (tx, ty) → (tx+tileW, ty+tileH) in the
 * source image, plus `padding` extra pixels on each side for context.  Pixels
 * outside the image bounds are filled by edge-replication (clamp-to-edge),
 * which is the same strategy used by imageDataToTensor for the pad-to-multiple
 * step, so border tiles produce consistent results.
 *
 * Returns an ImageData of size (tileW + 2*padding) × (tileH + 2*padding).
 */
export function extractTile(
  imageData: ImageData,
  tx: number,
  ty: number,
  tileW: number,
  tileH: number,
  padding: number,
): ImageData {
  const { width: W, height: H, data: src } = imageData
  const outW = tileW + 2 * padding
  const outH = tileH + 2 * padding
  const dst = new Uint8ClampedArray(outW * outH * 4)

  for (let dy = 0; dy < outH; dy++) {
    // Source row — clamped to [0, H-1]
    const py = Math.min(H - 1, Math.max(0, ty - padding + dy))
    for (let dx = 0; dx < outW; dx++) {
      // Source column — clamped to [0, W-1]
      const px = Math.min(W - 1, Math.max(0, tx - padding + dx))
      const srcIdx = (py * W + px) * 4
      const dstIdx = (dy * outW + dx) * 4
      dst[dstIdx]     = src[srcIdx]
      dst[dstIdx + 1] = src[srcIdx + 1]
      dst[dstIdx + 2] = src[srcIdx + 2]
      dst[dstIdx + 3] = src[srcIdx + 3]
    }
  }

  return new ImageData(dst, outW, outH)
}

/**
 * Copy the center region of an inferred tile output back into the full-image
 * output buffer, discarding the context padding on each side.
 *
 * @param outputRGBA  Uint8ClampedArray for the full output image (W×H×4)
 * @param tileOutput  ImageData returned by inference for a padded tile
 * @param tx          X position of this tile in the full image
 * @param ty          Y position of this tile in the full image
 * @param tileW       Width of the usable output region (excluding padding)
 * @param tileH       Height of the usable output region (excluding padding)
 * @param imageW      Width of the full output image
 * @param padding     Number of context pixels that were added on each side
 */
export function pasteTileCenter(
  outputRGBA: Uint8ClampedArray,
  tileOutput: ImageData,
  tx: number,
  ty: number,
  tileW: number,
  tileH: number,
  imageW: number,
  padding: number,
): void {
  const { data: src, width: tileOutW } = tileOutput

  for (let dy = 0; dy < tileH; dy++) {
    const srcRow = dy + padding
    for (let dx = 0; dx < tileW; dx++) {
      const srcIdx = (srcRow * tileOutW + dx + padding) * 4
      const dstIdx = ((ty + dy) * imageW + (tx + dx)) * 4
      outputRGBA[dstIdx]     = src[srcIdx]
      outputRGBA[dstIdx + 1] = src[srcIdx + 1]
      outputRGBA[dstIdx + 2] = src[srcIdx + 2]
      outputRGBA[dstIdx + 3] = 255
    }
  }
}

/**
 * Downscale an ImageData so its longest side is at most `maxDim` pixels,
 * preserving aspect ratio.  Uses OffscreenCanvas / drawImage for
 * GPU-accelerated bilinear filtering.
 *
 * If the image already fits within maxDim × maxDim it is returned unchanged.
 */
export function downscaleImageData(imageData: ImageData, maxDim: number): ImageData {
  const { width: W, height: H } = imageData
  const scale = Math.min(1, maxDim / Math.max(W, H))

  if (scale === 1) return imageData

  const newW = Math.max(1, Math.round(W * scale))
  const newH = Math.max(1, Math.round(H * scale))

  const srcCanvas = new OffscreenCanvas(W, H)
  srcCanvas.getContext('2d')!.putImageData(imageData, 0, 0)

  const dstCanvas = new OffscreenCanvas(newW, newH)
  const dstCtx = dstCanvas.getContext('2d')!
  dstCtx.drawImage(srcCanvas, 0, 0, newW, newH)

  return dstCtx.getImageData(0, 0, newW, newH)
}

/** Draw an ImageData onto an OffscreenCanvas and return an ImageBitmap. */
export async function imageDataToBitmap(imageData: ImageData): Promise<ImageBitmap> {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height)
  const ctx = canvas.getContext('2d')!
  ctx.putImageData(imageData, 0, 0)
  return createImageBitmap(canvas)
}

/** Read a File or Blob into an ImageData using OffscreenCanvas. */
export async function fileToImageData(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file)
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(bitmap, 0, 0)
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height)
}
