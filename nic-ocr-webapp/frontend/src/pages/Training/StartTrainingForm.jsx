import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { startTraining } from '@/services/api'
import { toast } from 'sonner'

export default function StartTrainingForm({ readyCount, isTraining, onStarted }) {
  const [iterations, setIterations] = useState(400)
  const [starting, setStarting] = useState(false)

  const canStart = !isTraining && readyCount >= 10

  async function handleSubmit(e) {
    e.preventDefault()
    if (!canStart) return
    const iters = parseInt(iterations, 10)
    if (!iters || iters < 1) return toast.error('Iterations must be a positive integer.')

    setStarting(true)
    try {
      const res = await startTraining(iters)
      toast.success(`Run ${res.data.run_id} started.`)
      onStarted(res.data.run_id)
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Failed to start training.')
    } finally {
      setStarting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Start Training</CardTitle>
      </CardHeader>
      <CardContent>
        {isTraining && (
          <p className="text-sm text-amber-600 mb-3">Training is currently in progress.</p>
        )}
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Max Iterations</label>
            <Input
              type="number"
              min={1}
              value={iterations}
              onChange={(e) => setIterations(e.target.value)}
              className="w-32"
              disabled={!canStart || starting}
            />
          </div>
          <Button type="submit" disabled={!canStart || starting}>
            {starting ? 'Starting…' : isTraining ? 'In Progress' : 'Start Training'}
          </Button>
        </form>
        {readyCount < 10 && !isTraining && (
          <p className="text-xs text-muted-foreground mt-2">
            {10 - readyCount} more LSTMF-ready file{10 - readyCount !== 1 ? 's' : ''} needed.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
