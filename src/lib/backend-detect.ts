/** Detect the best available ONNX Runtime execution provider. */

export type ExecutionProvider = 'webgpu' | 'webgl' | 'wasm'

export async function detectBestEP(): Promise<ExecutionProvider> {
  // WebGPU
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter()
      if (adapter) return 'webgpu'
    } catch {
      // fall through
    }
  }

  // WebGL2
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2')
    if (gl) {
      gl.getExtension('WEBGL_lose_context')?.loseContext()
      return 'webgl'
    }
  } catch {
    // fall through
  }

  return 'wasm'
}

export const EP_LABELS: Record<ExecutionProvider, string> = {
  webgpu: 'WebGPU',
  webgl: 'WebGL',
  wasm: 'WASM',
}
