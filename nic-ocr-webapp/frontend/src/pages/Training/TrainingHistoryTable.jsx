import { useState } from 'react'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollText, Download, Zap } from 'lucide-react'
import { activateModel, getModelDownloadUrl } from '@/services/api'
import { toast } from 'sonner'

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

export default function TrainingHistoryTable({ runs, onViewLogs, onActivated }) {
  const [activating, setActivating] = useState(null)

  async function handleActivate(runId) {
    setActivating(runId)
    try {
      await activateModel(runId)
      toast.success(`Model ${runId} set as active.`)
      onActivated()
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Failed to activate model.')
    } finally {
      setActivating(null)
    }
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-28">Run ID</TableHead>
            <TableHead className="w-36">Started</TableHead>
            <TableHead className="w-28">Status</TableHead>
            <TableHead className="w-24 text-right">Iterations</TableHead>
            <TableHead className="w-20 text-right">Files</TableHead>
            <TableHead className="w-20 text-center">Active</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableRow>
              <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                No training runs yet.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => (
              <TableRow key={run.id} className={run.is_active ? 'bg-primary/5' : ''}>
                <TableCell className="font-mono text-sm">{run.id}</TableCell>
                <TableCell className="text-sm">{fmt(run.started_at)}</TableCell>
                <TableCell><StatusBadge status={run.status} /></TableCell>
                <TableCell className="text-right tabular-nums">{run.iterations}</TableCell>
                <TableCell className="text-right tabular-nums">{run.file_count ?? '—'}</TableCell>
                <TableCell className="text-center">
                  {run.is_active ? (
                    <Badge className="bg-primary/90 text-primary-foreground">Active</Badge>
                  ) : '—'}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => onViewLogs(run.id)}
                      title="View logs"
                    >
                      <ScrollText className="h-4 w-4 mr-1" />
                      Logs
                    </Button>
                    {run.status === 'completed' && !run.is_active && (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={activating === run.id}
                        onClick={() => handleActivate(run.id)}
                        title="Make this model active"
                      >
                        <Zap className="h-4 w-4 mr-1" />
                        {activating === run.id ? 'Activating…' : 'Make Active'}
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
  )
}
