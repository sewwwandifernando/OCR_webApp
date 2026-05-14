import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'

export default function LogViewer({ runId, onDone, onClose }) {
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState('connecting') // connecting | live | done | error
  const scrollRef = useRef(null)
  const esRef = useRef(null)

  useEffect(() => {
    if (!runId) return

    setLines([])
    setStatus('connecting')

    const es = new EventSource(`/api/training/runs/${runId}/logs`)
    esRef.current = es

    es.onopen = () => setStatus('live')

    es.onmessage = (e) => {
      const line = e.data
      if (line === '[DONE]') {
        setStatus('done')
        es.close()
        onDone?.()
      } else {
        setLines((prev) => [...prev, line])
      }
    }

    es.onerror = () => {
      setStatus('error')
      es.close()
    }

    return () => es.close()
  }, [runId])

  // Auto-scroll to bottom on new lines
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines])

  if (!runId) return null

  const statusLabel =
    status === 'connecting' ? '● Connecting…'
    : status === 'live' ? '● Live'
    : status === 'done' ? '✓ Done'
    : '✗ Error'

  const statusColor =
    status === 'live' ? 'text-green-400'
    : status === 'done' ? 'text-neutral-400'
    : status === 'error' ? 'text-red-400'
    : 'text-neutral-500'

  return (
    <div className="rounded-lg border border-neutral-700 bg-neutral-950 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-700">
        <span className="font-mono text-xs text-neutral-300">{runId}</span>
        <div className="flex items-center gap-3">
          <span className={`font-mono text-xs ${statusColor}`}>{statusLabel}</span>
          {onClose && (
            <Button
              size="icon"
              variant="ghost"
              className="h-5 w-5 text-neutral-400 hover:text-neutral-100"
              onClick={onClose}
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
      <div
        ref={scrollRef}
        className="h-64 overflow-y-auto p-3 font-mono text-xs text-green-400 whitespace-pre-wrap leading-5"
      >
        {lines.length === 0 && status === 'connecting' && (
          <span className="text-neutral-500">Connecting to log stream…</span>
        )}
        {lines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
        {status === 'done' && (
          <div className="text-neutral-500 mt-2">── Training complete ──</div>
        )}
        {status === 'error' && (
          <div className="text-red-400 mt-2">── Stream error. Training may still be running. ──</div>
        )}
      </div>
    </div>
  )
}
