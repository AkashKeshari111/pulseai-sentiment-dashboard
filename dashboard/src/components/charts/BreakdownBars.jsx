/**
 * Sentiment split per category (channel, product, …).
 *
 * Horizontal bars because the category names are words, not dates - horizontal
 * labels stay readable without rotation. Stacked by sentiment with a 2px
 * surface gap between segments, and sorted by volume so the biggest channels
 * are at the top where the eye lands first.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { SENTIMENT_META, num, pct, shortNum } from '../../lib/format'
import { useChartColors } from '../../lib/theme'
import { ChartWithTable, TooltipShell, axisProps, cursorBar, gridProps } from './ChartKit'

const SERIES = ['negative', 'neutral', 'positive']

const titleCase = (value = '') =>
  value.replace(/[_-]/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())

export function BreakdownBars({ data = [], keyLabel = 'Channel', height }) {
  const colors = useChartColors()
  const rows = data.map((row) => ({ ...row, label: titleCase(row.key) }))
  const computedHeight = height ?? Math.max(180, rows.length * 42 + 30)

  const renderTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const row = payload[0].payload
    return (
      <TooltipShell
        title={row.label}
        rows={[
          ...SERIES.map((key) => ({
            label: SENTIMENT_META[key].label,
            value: num(row[key]),
            color: colors.sentiment?.[key],
          })),
          { label: 'Total', value: num(row.total) },
        ]}
        footer={`${pct(row.negative_rate)} negative`}
      />
    )
  }

  return (
    <ChartWithTable
      columns={[
        { key: 'label', label: keyLabel },
        { key: 'negative', label: 'Negative', numeric: true },
        { key: 'neutral', label: 'Neutral', numeric: true },
        { key: 'positive', label: 'Positive', numeric: true },
        { key: 'total', label: 'Total', numeric: true },
        { key: 'rate', label: 'Negative %', numeric: true },
      ]}
      rows={rows.map((row) => ({
        label: row.label,
        negative: num(row.negative),
        neutral: num(row.neutral),
        positive: num(row.positive),
        total: num(row.total),
        rate: pct(row.negative_rate),
      }))}
    >
      <ul className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        {SERIES.map((key) => (
          <li key={key} className="flex items-center gap-1.5 text-[12px] text-[var(--text-secondary)]">
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-[3px]"
              style={{ background: colors.sentiment?.[key] }}
            />
            {SENTIMENT_META[key].label}
          </li>
        ))}
      </ul>

      <div style={{ height: computedHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 4 }}>
            <CartesianGrid {...gridProps} vertical horizontal={false} />
            <XAxis type="number" {...axisProps} tickFormatter={shortNum} />
            <YAxis
              type="category"
              dataKey="label"
              {...axisProps}
              width={118}
              tick={{ fontSize: 12, fill: 'var(--text-secondary)' }}
            />
            <Tooltip content={renderTooltip} cursor={cursorBar} />
            {SERIES.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                stackId="sentiment"
                fill={colors.sentiment?.[key]}
                barSize={18}
                isAnimationActive={false}
                /* Only the final segment gets rounded ends, so the stack reads
                   as one bar rather than three pills. */
                radius={index === SERIES.length - 1 ? [0, 4, 4, 0] : 0}
                stroke="var(--surface-1)"
                strokeWidth={2}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartWithTable>
  )
}
