import { useEffect, useState } from 'react'
import { getModels, activateModel, getModelDownloadUrl } from '@/services/api'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog'
import { Download, Zap } from 'lucide-react'

function StatusBadge({ status }) {
  if (status === 'completed') return <Badge className="bg-green-600 text-white hover:bg-green-700">completed</Badge>
  if (status === 'running') return <Badge className="bg-blue-600 text-white hover:bg-blue-700">running</Badge>
  if (status === 'failed') return <Badge variant="destructive">failed</Badge>
  return <Badge variant="secondary">{status}</Badge>
}

function fmt(dt) {
  if (!dt) return '—'
  return new Date(dt).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
}

function fmtBytes(n) {
  if (!n) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export default function ModelsPage() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [confirmRun, setConfirmRun] = useState(null)
  const [activating, setActivating] = useState(false)

  useEffect(() => {
    fetchModels()
  }, [])

  async function fetchModels() {
    try {
      const r = await getModels()
      setModels(r.data)
    } catch {
      toast.error('Failed to load models.')
    } finally {
      setLoading(false)
    }
  }

  async function handleConfirmActivate() {
    if (!confirmRun) return
    setActivating(true)
    try {
      await activateModel(confirmRun.id)
      toast.success(`Model ${confirmRun.id} is now active.`)
      setConfirmRun(null)
      fetchModels()
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Failed to activate model.')
    } finally {
      setActivating(false)
    }
  }

  const activeModel = models.find((m) => m.is_active)

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Models</h1>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium text-muted-foreground">Active Model</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : activeModel ? (
            <div className="flex flex-col gap-1">
              <p className="text-lg font-semibold font-mono">{activeModel.id}</p>
              <p className="text-sm text-muted-foreground">
                Trained {fmt(activeModel.completed_at)}
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No active model.</p>
          )}
        </CardContent>
      </Card>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-36">Run ID</TableHead>
              <TableHead className="w-36">Trained At</TableHead>
              <TableHead className="w-24 text-right">Files Used</TableHead>
              <TableHead className="w-28">Status</TableHead>
              <TableHead className="w-20 text-center">Active</TableHead>
              <TableHead className="w-24 text-right">File Size</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  Loading…
                </TableCell>
              </TableRow>
            ) : models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  No training runs yet.
                </TableCell>
              </TableRow>
            ) : (
              models.map((run) => (
                <TableRow key={run.id} className={run.is_active ? 'bg-primary/5' : ''}>
                  <TableCell className="font-mono text-sm">{run.id}</TableCell>
                  <TableCell className="text-sm">{fmt(run.completed_at)}</TableCell>
                  <TableCell className="text-right tabular-nums">{run.file_count ?? '—'}</TableCell>
                  <TableCell><StatusBadge status={run.status} /></TableCell>
                  <TableCell className="text-center">
                    {run.is_active
                      ? <Badge className="bg-primary/90 text-primary-foreground">Active</Badge>
                      : '—'}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-sm">{fmtBytes(run.file_size)}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      {run.status === 'completed' && !run.is_active && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setConfirmRun(run)}
                        >
                          <Zap className="h-4 w-4 mr-1" />
                          Make Active
                        </Button>
                      )}
                      {run.status === 'completed' && run.model_path && (
                        <Button size="sm" variant="outline" asChild>
                          <a href={getModelDownloadUrl(run.id)} download>
                            <Download className="h-4 w-4 mr-1" />
                            Download
                          </a>
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <Dialog open={!!confirmRun} onOpenChange={(open) => { if (!open) setConfirmRun(null) }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Activate model?</DialogTitle>
            <DialogDescription>
              Set <span className="font-mono font-medium">{confirmRun?.id}</span> as the active
              model. The current active model will be deactivated.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmRun(null)} disabled={activating}>
              Cancel
            </Button>
            <Button onClick={handleConfirmActivate} disabled={activating}>
              {activating ? 'Activating…' : 'Activate'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  )
}
