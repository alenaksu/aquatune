import { EP_LABELS, type ExecutionProvider } from '../lib/backend-detect'

const EP_COLORS: Record<ExecutionProvider, string> = {
  webgpu: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  webgl:  'bg-blue-500/20 text-blue-300 border-blue-500/40',
  wasm:   'bg-amber-500/20 text-amber-300 border-amber-500/40',
}

interface Props {
  ep: ExecutionProvider
}

export function BackendBadge({ ep }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${EP_COLORS[ep]}`}
    >
      <span className="size-1.5 rounded-full bg-current opacity-80" />
      {EP_LABELS[ep]}
    </span>
  )
}
