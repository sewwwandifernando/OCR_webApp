import { useState } from 'react'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from '@/components/ui/dialog'
import { Eye, Pencil, Trash2, Check, X, ChevronLeft, ChevronRight } from 'lucide-react'
import { getPreviewUrl, updateGroundTruth, deleteTrainingData } from '@/services/api'
import { toast } from 'sonner'

const PAGE_SIZE = 20

function StatusBadge({ status }) {
  if (status === 'done') return <Badge className="bg-green-600 text-white hover:bg-green-700">done</Badge>
  if (status === 'failed') return <Badge variant="destructive">failed</Badge>
  return <Badge variant="secondary">pending</Badge>
}

function PreviewModal({ itemId, open, onClose }) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{itemId}</DialogTitle>
        </DialogHeader>
        {itemId && (
          <img
            src={getPreviewUrl(itemId)}
            alt={itemId}
            className="w-full object-contain max-h-[70vh] rounded"
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

function EditableRow({ item, onUpdated, onDeleted, onPreview }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(item.ground_truth)
  const [saving, setSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  async function handleSave() {
    if (!draft.trim()) return
    setSaving(true)
    try {
      const res = await updateGroundTruth(item.id, draft.trim())
      onUpdated(res.data)
      toast.success('Ground truth updated.')
      setEditing(false)
    } catch {
      toast.error('Failed to update ground truth.')
    } finally {
      setSaving(false)
    }
  }

  function handleCancelEdit() {
    setDraft(item.ground_truth)
    setEditing(false)
  }

  async function handleDelete() {
    try {
      await deleteTrainingData(item.id)
      onDeleted(item.id)
      toast.success(`Deleted ${item.id}.`)
    } catch {
      toast.error('Failed to delete.')
    } finally {
      setConfirmDelete(false)
    }
  }

  const uploadedAt = item.uploaded_at
    ? new Date(item.uploaded_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
    : '—'

  return (
    <>
      <TableRow>
        {/* Thumbnail */}
        <TableCell className="w-16">
          <img
            src={getPreviewUrl(item.id)}
            alt={item.id}
            className="h-10 w-16 object-cover rounded cursor-pointer border"
            onClick={() => onPreview(item.id)}
          />
        </TableCell>

        {/* ID + date */}
        <TableCell className="w-32">
          <p className="font-mono text-sm">{item.id}</p>
          <p className="text-xs text-muted-foreground">{uploadedAt}</p>
        </TableCell>

        {/* Ground truth */}
        <TableCell>
          {editing ? (
            <div className="flex gap-2 items-start">
              <Textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                rows={2}
                className="text-sm min-w-[200px]"
                autoFocus
              />
              <div className="flex flex-col gap-1">
                <Button size="icon" variant="ghost" disabled={saving} onClick={handleSave}>
                  <Check className="h-4 w-4 text-green-600" />
                </Button>
                <Button size="icon" variant="ghost" onClick={handleCancelEdit}>
                  <X className="h-4 w-4 text-muted-foreground" />
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-sm line-clamp-2">{item.ground_truth}</p>
          )}
        </TableCell>

        {/* Status badges */}
        <TableCell className="w-16 text-center"><StatusBadge status={item.status_tif} /></TableCell>
        <TableCell className="w-16 text-center"><StatusBadge status={item.status_box} /></TableCell>
        <TableCell className="w-16 text-center"><StatusBadge status={item.status_lstmf} /></TableCell>

        {/* Actions */}
        <TableCell className="w-28">
          {confirmDelete ? (
            <div className="flex gap-1 items-center">
              <span className="text-xs text-destructive">Sure?</span>
              <Button size="icon" variant="ghost" onClick={handleDelete}>
                <Check className="h-4 w-4 text-destructive" />
              </Button>
              <Button size="icon" variant="ghost" onClick={() => setConfirmDelete(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="flex gap-1">
              <Button size="icon" variant="ghost" onClick={() => onPreview(item.id)} title="Preview">
                <Eye className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" onClick={() => { setEditing(true); setDraft(item.ground_truth) }} title="Edit">
                <Pencil className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" onClick={() => setConfirmDelete(true)} title="Delete">
                <Trash2 className="h-4 w-4 text-destructive" />
              </Button>
            </div>
          )}
        </TableCell>
      </TableRow>
    </>
  )
}

export default function FileTable({ items, loading, onUpdated, onDeleted }) {
  const [page, setPage] = useState(0)
  const [previewId, setPreviewId] = useState(null)

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE))
  const safePage = Math.min(page, totalPages - 1)
  const pageItems = items.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE)

  return (
    <div className="space-y-2">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Preview</TableHead>
              <TableHead className="w-32">ID</TableHead>
              <TableHead>Ground Truth</TableHead>
              <TableHead className="w-16 text-center">TIF</TableHead>
              <TableHead className="w-16 text-center">Box</TableHead>
              <TableHead className="w-16 text-center">LSTMF</TableHead>
              <TableHead className="w-28">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  Loading…
                </TableCell>
              </TableRow>
            ) : pageItems.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  No records found.
                </TableCell>
              </TableRow>
            ) : (
              pageItems.map((item) => (
                <EditableRow
                  key={item.id}
                  item={item}
                  onUpdated={onUpdated}
                  onDeleted={onDeleted}
                  onPreview={setPreviewId}
                />
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-end gap-2 text-sm">
          <Button
            size="icon"
            variant="ghost"
            disabled={safePage === 0}
            onClick={() => setPage(safePage - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-muted-foreground">
            Page {safePage + 1} of {totalPages}
          </span>
          <Button
            size="icon"
            variant="ghost"
            disabled={safePage >= totalPages - 1}
            onClick={() => setPage(safePage + 1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      <PreviewModal
        itemId={previewId}
        open={!!previewId}
        onClose={() => setPreviewId(null)}
      />
    </div>
  )
}
