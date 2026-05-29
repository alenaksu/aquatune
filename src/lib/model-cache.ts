/**
 * Model caching via the Cache API.
 * Stores the ONNX model bytes so it doesn't re-download on every visit.
 */

const CACHE_NAME = 'diveye-models-v1'

export async function getCachedModel(url: string): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME)
    const response = await cache.match(url)
    if (!response) return null
    return response.arrayBuffer()
  } catch {
    return null
  }
}

export async function setCachedModel(url: string, data: ArrayBuffer): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME)
    const response = new Response(data, {
      headers: { 'Content-Type': 'application/octet-stream' },
    })
    await cache.put(url, response)
  } catch {
    // Cache write failure is non-fatal
  }
}

export type ProgressCallback = (loaded: number, total: number) => void

/**
 * Fetch a model with progress reporting, using Cache API for persistence.
 * Returns the raw ArrayBuffer of the model file.
 */
export async function fetchModel(
  url: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  // Try cache first
  const cached = await getCachedModel(url)
  if (cached) {
    onProgress?.(1, 1)
    return cached
  }

  // Fetch with streaming progress
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`)

  const contentLength = Number(response.headers.get('Content-Length') ?? 0)
  const reader = response.body!.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress?.(loaded, contentLength)
  }

  // Reassemble
  const buffer = new ArrayBuffer(loaded)
  const view = new Uint8Array(buffer)
  let offset = 0
  for (const chunk of chunks) {
    view.set(chunk, offset)
    offset += chunk.length
  }

  // Cache for next time
  await setCachedModel(url, buffer)
  return buffer
}

/** Purge old model cache versions (call on app init). */
export async function purgeOldCaches(): Promise<void> {
  try {
    const keys = await caches.keys()
    await Promise.all(
      keys
        .filter((k) => k.startsWith('diveye-models-') && k !== CACHE_NAME)
        .map((k) => caches.delete(k)),
    )
  } catch {
    // Non-fatal
  }
}

/** Returns estimated cache storage usage in bytes. */
export async function getCacheSize(): Promise<number> {
  try {
    const estimate = await navigator.storage.estimate()
    return estimate.usage ?? 0
  } catch {
    return 0
  }
}
