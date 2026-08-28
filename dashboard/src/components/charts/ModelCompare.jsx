/**
 * Baseline vs fine-tuned transformer, on the same held-out test set.
 *
 * Grouped bars, not stacked: these are competing measurements of the same
 * quantity, so they must share a baseline and sit side by side to be compared.
 * The two models are identities, so they take categorical slots 1 and 2.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useChartColors } from '../../lib/theme'
import { ChartWithTable, TooltipShell, axisProps, cursorBar, gridProps } from './ChartKit'

const METRICS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'f1_macro', label: 'F1 (macro)' },
  { key: 'f1_weighted', label: 'F1 (weighted)' },
]

/** Short axis-friendly names. Unknown keys fall back to a tidied key. */
const SHORT_NAMES = {
  baseline: 'TF-IDF baseline',
  distilbert: 'DistilBERT (256 tok)',
  distilbert_seq128: 'DistilBERT (128 tok)',
}

/** Fixed left-to-right order: control first, ablation, then the final model. */
const ORDER = ['baseline', 'distilbert_seq128', 'distilbert']

const shortName = (key) =>
  SHORT_NAMES[key] ?? key.replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

export function ModelCompare({ models = {} }) {
  const colors = useChartColors()

  // Sorted by a fixed list rather than object order, so a series never changes
  // colour just because metrics.json was written in a different sequence.
  const entries = Object.keys(models).sort((a, b) => {
    const rank = (key) => (ORDER.indexOf(key) === -1 ? ORDER.length : ORDER.indexOf(key))
    return rank(a) - rank(b)
  })
  if (entries.length === 0) return null

  const names = entries.map((key) => ({
    key,
    label: models[key].model ?? key,
    short: shortName(key),
  }))

  const data = METRICS.map((metric) => {
    const row = { metric: metric.label }
    for (const { key, short } of names) {
      row[short] = models[key]?.[metric.key] ?? null
    }
    return row
  })

  const palette = [colors.series1, colors.series2, colors.series3]

  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <TooltipShell
        title={label}
        rows={payload.map((entry) => ({
          label: entry.dataKey,
          value: entry.value?.toFixed(4),
          color: entry.fill,
        }))}
      />
    )
  }

  return (
    <ChartWithTable
      columns={[
        { key: 'metric', label: 'Metric' },
        ...names.map(({ short }) => ({ key: short, label: short, numeric: true })),
        { key: 'delta', label: 'Delta', numeric: true },
      ]}
      rows={data.map((row) => {
        const values = names.map(({ short }) => row[short])
        // Delta is always "final model minus baseline", regardless of how many
        // intermediate runs sit between them in the table.
        const first = values[0]
        const last = values[values.length - 1]
        const delta =
          values.length >= 2 && Number.isFinite(first) && Number.isFinite(last)
            ? `${last - first >= 0 ? '+' : ''}${(last - first).toFixed(4)}`
            : '—'
        return {
          metric: row.metric,
          ...Object.fromEntries(
            names.map(({ short }) => [short, row[short]?.toFixed(4) ?? '—']),
          ),
          delta,
        }
      })}
    >
      <ul className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        {names.map(({ short }, index) => (
          <li key={short} className="flex items-center gap-1.5 text-[12px] text-[var(--text-secondary)]">
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-[3px]"
              style={{ background: palette[index] }}
            />
            {short}
          </li>
        ))}
      </ul>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 8, bottom: 0, left: -16 }} barGap={4}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="metric" {...axisProps} />
            <YAxis {...axisProps} domain={[0, 1]} width={46} tickFormatter={(v) => v.toFixed(1)} />
            <Tooltip content={renderTooltip} cursor={cursorBar} />
            {names.map(({ short }, index) => (
              <Bar
                key={short}
                dataKey={short}
                fill={palette[index]}
                barSize={34}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              >
                {/* Three metrics x two models is few enough to label directly,
                    which removes the need to read values off the axis. */}
                <LabelList
                  dataKey={short}
                  position="top"
                  offset={6}
                  formatter={(value) => (Number.isFinite(value) ? value.toFixed(3) : '')}
                  style={{ fontSize: 10.5, fill: 'var(--text-secondary)' }}
                />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartWithTable>
  )
}
