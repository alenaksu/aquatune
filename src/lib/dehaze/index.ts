/**
 * Public API for the dehaze module.
 *
 * Single entry point: `dehaze(imageData, options)`.
 * Dispatches to the appropriate algorithm based on `options.method`.
 */

export type { DehazeMethod, DehazeOptions, DcpOptions, ClaheOptions } from './types'

import { dehazeDCP }   from './dcp'
import { dehazeCLAHE } from './clahe'
import type { DehazeOptions } from './types'

export function dehaze(imageData: ImageData, options: DehazeOptions): ImageData {
  switch (options.method) {
    case 'dcp':   return dehazeDCP(imageData, options.dcp)
    case 'clahe': return dehazeCLAHE(imageData, options.clahe)
  }
}
