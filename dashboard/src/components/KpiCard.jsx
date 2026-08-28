/**
 * Stat tile.
 *
 * A single headline number does not need a chart - it needs to be legible at a
 * glance, with one line of context underneath. The optional sparkline is a
 * shape cue only: no axes, no grid, no labels, because the tile's job is the
 * number and the direction, not precise reading.
 */

import clsx from 'clsx'
import { Area, AreaChart, ResponsiveContainer } from 'recharts'
import { Card } from './ui'

const STATUS_COLOR = {
  good: 'var(--status-good)',
  warning: 'var(--status-warning)',
  serious: 'var(--status-serious)',
  critical: 'var(--status-critical)',
  neutral: 'var(--text-muted)',
}

export function KpiCard({
  label,
  value,
  unit,
  hint,
  status,
  statusLabel,
  accent = 'var(--text-primary)',
  spark,
  sparkColor = 'var(--series-1)',
  loading,
  className,
}) {
  // The gradient is referenced as url(#id), and a raw label like "Feedback
  // analysed" contains a space - which makes the reference fail to resolve and
  // the area render unfilled. Slugify it.
  const gradientId = `spark-${String(label).replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase()}`

  return (
    <Card className={clsx('relative overflow-hidden p-4', className)}>
      <div className="flex items-start justify-between gap-2">
        <p className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
          {label}
        </p>
        {statusLabel && (
          <span
            className="rounded-md px-1.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide"
            style={{
              color: STATUS_COLOR[status] ?? STATUS_COLOR.neutral,
              background: 'var(--surface-2)',
            }}
          >
            {statusLabel}
          </span>
        )}
      </div>

      {loading ? (
        <div className="skeleton mt-2 h-9 w-24 rounded-md" />
      ) : (
        <p className="mt-1.5 flex items-baseline gap-1">
          <span className="text-[30px] leading-none font-semibold" style={{ color: accent }}>
            {value}
          </span>
          {unit && <span className="text-[13px] text-[var(--text-muted)]">{unit}</span>}
        </p>
      )}

      {hint && (
        <p className="mt-2 text-[12px] leading-snug text-[var(--text-muted)]">{hint}</p>
      )}

      {spark?.length > 1 && (
        <div className="pointer-events-none mt-3 h-9" aria-hidden>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={spark} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={sparkColor} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={sparkColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="value"
                stroke={sparkColor}
                strokeWidth={2}
                fill={`url(#${gradientId})`}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}
