import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

function StatCard({ label, value, active, onClick }) {
  return (
    <Card
      className={cn('cursor-default', onClick && 'cursor-pointer hover:ring-2 hover:ring-primary/50', active && 'ring-2 ring-primary')}
      onClick={onClick}
    >
      <CardContent className="p-4 flex flex-col items-center gap-1">
        <span className="text-3xl font-bold tabular-nums">{value}</span>
        <span className="text-xs text-muted-foreground uppercase tracking-wide">{label}</span>
      </CardContent>
    </Card>
  )
}

export default function SummaryBar({ summary, errorFilter, onToggleErrorFilter }) {
  const { total, ready, errors } = summary
  return (
    <div className="grid grid-cols-3 gap-4">
      <StatCard label="Total" value={total} />
      <StatCard label="Ready" value={ready} />
      <StatCard
        label={errorFilter ? 'Errors (filtered)' : 'Errors'}
        value={errors}
        active={errorFilter}
        onClick={errors > 0 ? onToggleErrorFilter : undefined}
      />
    </div>
  )
}
