import { useEffect, useRef, useState } from 'react'
import { getTrainingData, getTrainingRuns, getTrainingStatus } from '@/services/api'
import { toast } from 'sonner'
import { Separator } from '@/components/ui/separator'
import ReadinessPanel from './ReadinessPanel'
import StartTrainingForm from './StartTrainingForm'
import LogViewer from './LogViewer'
import TrainingHistoryTable from './TrainingHistoryTable'

export default function TrainingPage() {
  const [readyCount, setReadyCount] = useState(0)
  const [loadingReady, setLoadingReady] = useState(true)
  const [runs, setRuns] = useState([])
  const [isTraining, setIsTraining] = useState(false)
  const [logRunId, setLogRunId] = useState(null)
  const pollRef = useRef(null)

  // Initial data load
  useEffect(() => {
    getTrainingData()
      .then((r) => {
        const items = r.data.items
        const ready = items.filter((i) => i.status_lstmf === 'done').length
        setReadyCount(ready)
      })
      .catch(() => toast.error('Failed to load training data.'))
      .finally(() => setLoadingReady(false))

    fetchRuns()
    fetchStatus()
  }, [])

  // Poll status every 5 s
  useEffect(() => {
    pollRef.current = setInterval(fetchStatus, 5000)
    return () => clearInterval(pollRef.current)
  }, [])

  async function fetchRuns() {
    try {
      const r = await getTrainingRuns()
      setRuns(r.data)
    } catch {
      // silently ignore — table stays stale
    }
  }

  async function fetchStatus() {
    try {
      const r = await getTrainingStatus()
      setIsTraining(r.data.is_training)
      // If a run is active and we have no log viewer open, auto-open it
      if (r.data.is_training && r.data.active_run_id) {
        setLogRunId((prev) => prev ?? r.data.active_run_id)
      }
    } catch {
      // silently ignore
    }
  }

  function handleStarted(runId) {
    setLogRunId(runId)
    setIsTraining(true)
    fetchRuns()
  }

  function handleLogDone() {
    setIsTraining(false)
    fetchRuns()
  }

  function handleActivated() {
    fetchRuns()
  }

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Training</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ReadinessPanel readyCount={readyCount} loading={loadingReady} />
        <StartTrainingForm
          readyCount={readyCount}
          isTraining={isTraining}
          onStarted={handleStarted}
        />
      </div>

      {logRunId && (
        <>
          <Separator />
          <div className="space-y-1">
            <h2 className="text-sm font-medium text-muted-foreground">Log Stream</h2>
            <LogViewer
              runId={logRunId}
              onDone={handleLogDone}
              onClose={() => setLogRunId(null)}
            />
          </div>
        </>
      )}

      <Separator />

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-muted-foreground">History</h2>
        <TrainingHistoryTable
          runs={runs}
          onViewLogs={(id) => setLogRunId(id)}
          onActivated={handleActivated}
        />
      </div>
    </main>
  )
}
