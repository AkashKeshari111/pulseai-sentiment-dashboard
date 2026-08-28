/**
 * Sentiment mix.
 *
 * A donut is used rather than a pie because the hole carries the total, which
 * is the number people actually look for first. Three parts is inside the
 * range where part-to-whole reads reliably; the direct percentage labels beside
 * the legend mean nobody has to judge an angle.
 */

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { SENTIMENT_META, num, pct } from '../../lib/format'
import { useChartColors } from '../../lib/theme'
import { ChartWithTable, TooltipShell } from './ChartKit'

export function SentimentDonut({ counts, total, height = 250 }) {
  const colors = useChartColors()

  const data = ['negative', 'neutral', 'positive'].map((key) => ({
    key,
    name: SENTIMENT_META[key].label,
    value: counts?.[key] ?? 0,
    share: total ? ((counts?.[key] ?? 0) / total) * 100 : 0,
    color: colors.sentiment?.[key],
  }))

  const renderTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const item = payload[0].payload
    return (
      <TooltipShell
        title={item.name}
        rows={[
          { label: 'Volume', value: num(item.value), color: item.color },
          { label: 'Share', value: pct(item.share) },
        ]}
      />
    )
  }

  return (
    <ChartWithTable
      columns={[
        { key: 'name', label: 'Sentiment' },
        { key: 'volume', label: 'Volume', numeric: true },
        { key: 'share', label: 'Share', numeric: true },
      ]}
      rows={data.map((item) => ({
        name: item.name,
        volume: num(item.value),
        share: pct(item.share),
      }))}
      /* w-full matters: the parent CardBody is a flex container so this chart is
         a flex item, and without an explicit width it shrinks to its content -
         which for a ResponsiveContainer measuring its parent is zero. */
      className="w-full"
    >
      <div className="flex flex-col items-center gap-4 sm:flex-row">
        <div className="relative w-full max-w-[230px]" style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                innerRadius="63%"
                outerRadius="94%"
                startAngle={90}
                endAngle={-270}
                /* A 2px gap of surface between segments keeps adjacent fills
                   from reading as one shape. */
                paddingAngle={1.4}
                stroke="var(--surface-1)"
                strokeWidth={2}
                isAnimationActive={false}
              >
                {data.map((item) => (
                  <Cell key={item.key} fill={item.color} />
                ))}
              </Pie>
              <Tooltip content={renderTooltip} />
            </PieChart>
          </ResponsiveContainer>

          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-[26px] font-semibold leading-none">{num(total)}</span>
            <span className="mt-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
              responses
            </span>
          </div>
        </div>

        {/* Legend doubles as the direct-label channel: every slice is named and
            given its exact share, so the chart never depends on colour alone. */}
        <ul className="w-full flex-1 space-y-2">
          {data.map((item) => (
            <li key={item.key} className="flex items-center gap-2.5">
              <span
                aria-hidden
                className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
                style={{ background: item.color }}
              />
              <span className="text-[13px] text-[var(--text-secondary)]">{item.name}</span>
              <span className="tabular ml-auto text-[13px] font-medium">{pct(item.share)}</span>
              <span className="tabular w-14 text-right text-[12px] text-[var(--text-muted)]">
                {num(item.value)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </ChartWithTable>
  )
}
