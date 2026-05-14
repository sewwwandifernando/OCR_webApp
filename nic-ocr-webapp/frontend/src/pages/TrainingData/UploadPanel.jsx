import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { UploadCloud } from 'lucide-react'
import { uploadTrainingData } from '@/services/api'
import { toast } from 'sonner'

export default function UploadPanel({ onUploaded }) {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [groundTruth, setGroundTruth] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const [previewUrl, setPreviewUrl] = useState(null)

  function handleFile(f) {
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(f))
    } else {
      toast.error('Please select an image file.')
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) return toast.error('Select an image first.')
    if (!groundTruth.trim()) return toast.error('Ground truth text is required.')

    const fd = new FormData()
    fd.append('file', file)
    fd.append('ground_truth', groundTruth.trim())

    setUploading(true)
    setUploadPct(0)
    setProcessing(false)

    try {
      const res = await uploadTrainingData(fd, (evt) => {
        if (evt.total) {
          const pct = Math.round((evt.loaded / evt.total) * 100)
          setUploadPct(pct)
          if (pct === 100) setProcessing(true)
        }
      })
      onUploaded(res.data)
      toast.success(`Uploaded ${res.data.id} successfully.`)
      setFile(null)
      setGroundTruth('')
      if (fileInputRef.current) fileInputRef.current.value = ''
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Upload failed.')
    } finally {
      setUploading(false)
      setUploadPct(0)
      setProcessing(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Upload NIC Crop</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Dropzone */}
          <div
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
              ${dragOver ? 'border-primary bg-primary/5' : 'border-muted-foreground/30 hover:border-primary/60'}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud className="mx-auto mb-2 h-8 w-8 text-muted-foreground" />
            {file ? (
              <p className="text-sm font-medium">{file.name}</p>
            ) : (
              <p className="text-sm text-muted-foreground">
                Drop an image here or <span className="text-primary">click to browse</span>
              </p>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>

          {/* Preview */}
          {previewUrl && (
            <div className="rounded-md border overflow-hidden bg-muted/30 flex items-center justify-center p-2">
              <img
                src={previewUrl}
                alt="Crop preview"
                className="max-h-32 max-w-full object-contain rounded"
              />
            </div>
          )}

          {/* Ground truth */}
          <Textarea
            placeholder="Ground truth text (Sinhala or Latin)…"
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            rows={2}
            disabled={uploading}
          />

          {/* Progress */}
          {uploading && (
            <div className="space-y-1">
              <Progress value={processing ? 100 : uploadPct} className="h-2" />
              <p className="text-xs text-muted-foreground">
                {processing ? 'Generating TIF / box / lstmf on server…' : `Uploading… ${uploadPct}%`}
              </p>
            </div>
          )}

          <Button type="submit" disabled={uploading || !file || !groundTruth.trim()}>
            {uploading ? 'Uploading…' : 'Upload'}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
