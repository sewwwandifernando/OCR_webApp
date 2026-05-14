import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { UploadCloud, Copy, Check } from 'lucide-react'
import { runOcr } from '@/services/api'
import { toast } from 'sonner'

const NIC_TYPES = [
  { value: 'new', label: 'New NIC (2016+)' },
  { value: 'old', label: 'Old NIC (pre-2016)' },
]

export default function TestingPage() {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [nicType, setNicType] = useState('new')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [copied, setCopied] = useState(false)

  function handleFile(f) {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      toast.error('Please select an image file.')
      return
    }
    setFile(f)
    setResult(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(f))
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function handleRun() {
    if (!file) return toast.error('Select an image first.')
    const fd = new FormData()
    fd.append('file', file)
    fd.append('nic_type', nicType)
    setRunning(true)
    setResult(null)
    try {
      const res = await runOcr(fd)
      setResult(res.data)
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'OCR failed.')
    } finally {
      setRunning(false)
    }
  }

  async function handleCopy() {
    if (!result) return
    const text = result.zones.map((z) => z.text).join('\n')
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const rawText = result ? result.zones.map((z) => z.text).join('\n') : ''

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-2xl font-semibold">OCR Testing</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left: input ── */}
        <div className="space-y-4">
          {/* Dropzone */}
          <Card>
            <CardHeader><CardTitle className="text-base">Upload NIC Image</CardTitle></CardHeader>
            <CardContent className="space-y-4">
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

              {/* NIC type radio */}
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">NIC Type</p>
                <div className="flex gap-4">
                  {NIC_TYPES.map((t) => (
                    <label key={t.value} className="flex items-center gap-2 cursor-pointer text-sm">
                      <input
                        type="radio"
                        name="nic_type"
                        value={t.value}
                        checked={nicType === t.value}
                        onChange={() => setNicType(t.value)}
                        className="accent-primary"
                      />
                      {t.label}
                    </label>
                  ))}
                </div>
              </div>

              <Button onClick={handleRun} disabled={running || !file} className="w-full">
                {running ? 'Running OCR…' : 'Run OCR'}
              </Button>
            </CardContent>
          </Card>

          {/* Image preview */}
          {previewUrl && (
            <Card>
              <CardHeader><CardTitle className="text-base">Preview</CardTitle></CardHeader>
              <CardContent>
                <img
                  src={previewUrl}
                  alt="NIC preview"
                  className="w-full object-contain max-h-64 rounded border"
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* ── Right: results ── */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Zone table */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    OCR Results
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      {result.zones.length} zone{result.zones.length !== 1 ? 's' : ''}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="rounded-b-md border-t overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-12">#</TableHead>
                          <TableHead className="w-36">BBox</TableHead>
                          <TableHead className="w-24">Script</TableHead>
                          <TableHead>Text</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.zones.map((z) => (
                          <TableRow key={z.zone_index}>
                            <TableCell className="tabular-nums">{z.zone_index}</TableCell>
                            <TableCell className="font-mono text-xs text-muted-foreground">
                              {z.bbox.join(', ')}
                            </TableCell>
                            <TableCell>
                              <Badge variant="secondary">{z.script}</Badge>
                            </TableCell>
                            <TableCell className="font-mono text-sm break-all">{z.text}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              {/* Raw text box */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">Raw Text</CardTitle>
                    <Button size="sm" variant="ghost" onClick={handleCopy}>
                      {copied
                        ? <><Check className="h-4 w-4 mr-1 text-green-600" /> Copied</>
                        : <><Copy className="h-4 w-4 mr-1" /> Copy</>}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <pre className="rounded bg-muted p-3 text-sm font-mono whitespace-pre-wrap break-all min-h-[60px]">
                    {rawText}
                  </pre>
                </CardContent>
              </Card>
            </>
          ) : (
            <div className="flex items-center justify-center h-full min-h-[200px] rounded-lg border border-dashed border-muted-foreground/30">
              <p className="text-sm text-muted-foreground">
                {running ? 'Processing…' : 'OCR results will appear here.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
