import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertTriangle, CheckCircle2 } from 'lucide-react'

export default function ReadinessPanel({ readyCount, loading }) {
  const sufficient = readyCount >= 10

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Readiness</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4">
          <span className="text-4xl font-bold tabular-nums">
            {loading ? '…' : readyCount}
          </span>
          <div>
            <p className="text-sm font-medium">LSTMF-ready files</p>
            {!loading && sufficient ? (
              <p className="text-xs text-green-600 flex items-center gap-1 mt-0.5">
                <CheckCircle2 className="h-3 w-3" />
                Ready to train
              </p>
            ) : !loading ? (
              <p className="text-xs text-amber-600 flex items-center gap-1 mt-0.5">
                <AlertTriangle className="h-3 w-3" />
                Need at least 10 to start
              </p>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
