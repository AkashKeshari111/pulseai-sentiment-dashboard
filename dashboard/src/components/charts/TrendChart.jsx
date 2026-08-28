/**
 * Sentiment over time.
 *
 * Two views, never two y-axes. "Volume" stacks the three sentiment counts;
 * "Net score" plots the single NSS line on its own scale. Overlaying a count
 * axis and a -100..100 index axis in one frame would let the visual crossing
 * points imply relationships that are not in the data, so the chart switches
 * instead of doubling up.
 */

import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { SENTIMENT_META, formatPeriod, num, shortNum, signed } from '../../lib/format'
import { useChartColors } from '../../lib/theme'
import { Button } from '../ui'
import { ChartWithTable, TooltipShell, axisProps, cursorLine, gridProps } from './ChartKit'

const SERIES = ['negative', 'neutral', 'positive']

export function TrendChart({ data = [], granularity = 'day', height = 300 }) {
  const colors = useChartColors()
  const [view, setView] = useState('volume')

  const rows = useMemo(
    () => data.map((point) => ({ ...point, label: formatPeriod(point.period, granularity) })),
    [data, granularity],
  )

  const volumeTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const point = payload[0].payload
    return (
      <TooltipShell
        title={label}
        rows={[
          ...SERIES.map((key) => ({
            label: SENTIMENT_META[key].label,
            value: num(point[key]),
            color: colors.sentiment?.[key],
          })),
          { label: 'Total', value: num(point.total) },
        ]}
        footer={`Net sentiment ${signed(point.net_sentiment_score)}`}
      />
    )
  }

  const scoreTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const point = payload[0].payload
    return (
      <TooltipShell
        title={label}
        rows={[
          {
            label: 'Net sentiment',
            value: signed(point.net_sentiment_score),
            color: colors.series1,
          },
          { label: 'Volume', value: num(point.total) },
        ]}
        footer="% positive − % negative"
      />
    )
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        {/* Legend is always present for the stacked view (3 series). */}
        {view === 'volume' ? (
          <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {SERIES.map((key) => (
              <li
                key={key}
                className="flex items-center gap-1.5 text-[12px] text-[var(--text-secondary)]"
              >
                <span
                  aria-hidden
                  className="h-2.5 w-2.5 rounded-[3px]"
                  style={{ background: colors.sentiment?.[key] }}
                />
                {SENTIMENT_META[key].label}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[12px] text-[var(--text-muted)]">
            Net sentiment score · % positive − % negative
          </p>
        )}

        <div className="flex gap-1" role="group" aria-label="Trend view">
          {[
            { key: 'volume', label: 'Volume' },
            { key: 'score', label: 'Net score' },
          ].map((option) => (
            <Button
              key={option.key}
              variant={view === option.key ? 'subtle' : 'ghost'}
              aria-pressed={view === option.key}
              onClick={() => setView(option.key)}
              className="!px-2.5 !py-1 !text-[12px]"
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      <ChartWithTable
        columns={[
          { key: 'label', label: 'Period' },
          { key: 'negative', label: 'Negative', numeric: true },
          { key: 'neutral', label: 'Neutral', numeric: true },
          { key: 'positive', label: 'Positive', numeric: true },
          { key: 'total', label: 'Total', numeric: true },
          { key: 'nss', label: 'Net score', numeric: true },
        ]}
        rows={rows.map((row) => ({
          label: row.label,
          negative: num(row.negative),
          neutral: num(row.neutral),
          positive: num(row.positive),
          total: num(row.total),
          nss: signed(row.net_sentiment_score),
        }))}
      >
        <div style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            {view === 'volume' ? (
              <AreaChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="label" {...axisProps} minTickGap={24} />
                <YAxis {...axisProps} width={48} tickFormatter={shortNum} />
                <Tooltip content={volumeTooltip} cursor={cursorLine} />
                {SERIES.map((key) => (
                  <Area
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stackId="sentiment"
                    stroke={colors.sentiment?.[key]}
                    strokeWidth={2}
                    fill={colors.sentiment?.[key]}
                    fillOpacity={0.22}
                    isAnimationActive={false}
                    activeDot={{
                      r: 4,
                      strokeWidth: 2,
                      stroke: 'var(--surface-1)',
                    }}
                  />
                ))}
              </AreaChart>
            ) : (
              <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="label" {...axisProps} minTickGap={24} />
                <YAxis {...axisProps} width={48} domain={[-100, 100]} />
                {/* Zero is the meaningful crossing: below it, negatives outnumber
                    positives. */}
                <ReferenceLine y={0} stroke="var(--baseline)" strokeWidth={1} />
                <Tooltip content={scoreTooltip} cursor={cursorLine} />
                <Line
                  type="monotone"
                  dataKey="net_sentiment_score"
                  stroke={colors.series1}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </ChartWithTable>
    </div>
  )
}
