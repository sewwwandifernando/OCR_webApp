import { useEffect, useState } from 'react'
import { getTrainingData } from '@/services/api'
import { toast } from 'sonner'
import UploadPanel from './UploadPanel'
import SummaryBar from './SummaryBar'
import FileTable from './FileTable'

function computeSummary(items) {
  return {
    total: items.length,
    ready: items.filter(
      (r) => r.status_tif === 'done' && r.status_box === 'done' && r.status_lstmf === 'done'
    ).length,
    errors: items.filter((r) =>
      [r.status_tif, r.status_box, r.status_lstmf].includes('failed')
    ).length,
  }
}

export default function TrainingDataPage() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [errorFilter, setErrorFilter] = useState(false)

  useEffect(() => {
    getTrainingData()
      .then((r) => setItems(r.data.items))
      .catch(() => toast.error('Failed to load training data.'))
      .finally(() => setLoading(false))
  }, [])

  function handleUploaded(newItem) {
    setItems((prev) => [newItem, ...prev])
  }

  function handleUpdated(updatedItem) {
    setItems((prev) => prev.map((item) => (item.id === updatedItem.id ? updatedItem : item)))
  }

  function handleDeleted(id) {
    setItems((prev) => prev.filter((item) => item.id !== id))
  }

  const summary = computeSummary(items)

  const displayItems = errorFilter
    ? items.filter((r) => [r.status_tif, r.status_box, r.status_lstmf].includes('failed'))
    : items

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Training Data</h1>
      <UploadPanel onUploaded={handleUploaded} />
      <SummaryBar
        summary={summary}
        errorFilter={errorFilter}
        onToggleErrorFilter={() => setErrorFilter((f) => !f)}
      />
      <FileTable
        items={displayItems}
        loading={loading}
        onUpdated={handleUpdated}
        onDeleted={handleDeleted}
      />
    </main>
  )
}
