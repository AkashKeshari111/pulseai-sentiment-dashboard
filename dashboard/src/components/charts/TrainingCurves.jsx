/**
 * Fine-tuning history.
 *
 * Loss and macro-F1 live in separate frames rather than on twin axes: loss is
 * unbounded and falling, F1 is 0-1 and rising, and putting them in one frame
 * would make their crossing point look meaningful when it is an artefact of
 * the two scales.
 *
 * Train vs validation loss are two identities, so they take categorical slots
 * 1 and 2 - the gap between them is the overfitting signal the chart exists to
 * show.
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useChartColors } from '../../lib/theme'
import { TooltipShell, axisProps, cursorLine, gridProps } from './ChartKit'

export function TrainingCurves({ history = [] }) {
  const colors = useChartColors()
  if (history.length === 0) return null

  const singleEpoch = history.length === 1

  const lossTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <TooltipShell
        title={`Epoch ${label}`}
        rows={payload.map((entry) => ({
          label: entry.dataKey === 'train_loss' ? 'Training loss' : 'Validation loss',
          value: entry.value?.toFixed(4),
          color: entry.stroke,
        }))}
      />
    )
  }

  const scoreTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const point = payload[0].payload
    return (
      <TooltipShell
        title={`Epoch ${label}`}
        rows={[
          { label: 'Macro F1', value: point.val_f1_macro?.toFixed(4), color: colors.series1 },
          { label: 'Accuracy', value: point.val_accuracy?.toFixed(4), color: colors.series3 },
        ]}
        footer={`${point.seconds}s`}
      />
    )
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <figure className="m-0">
        <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <span className="text-[12.5px] font-medium text-[var(--text-secondary)]">
            Loss per epoch
          </span>
          <span className="flex items-center gap-3">
            {[
              { label: 'Train', color: colors.series1 },
              { label: 'Validation', color: colors.series2 },
            ].map((item) => (
              <span
                key={item.label}
                className="flex items-center gap-1.5 text-[11.5px] text-[var(--text-secondary)]"
              >
                <span
                  aria-hidden
                  className="h-2.5 w-2.5 rounded-[3px]"
                  style={{ background: item.color }}
                />
                {item.label}
              </span>
            ))}
          </span>
        </figcaption>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 10, bottom: 0, left: -16 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epoch" {...axisProps} allowDecimals={false} />
              <YAxis {...axisProps} width={50} tickFormatter={(value) => value.toFixed(2)} />
              <Tooltip content={lossTooltip} cursor={cursorLine} />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke={colors.series1}
                strokeWidth={2}
                dot={singleEpoch ? { r: 4 } : false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke={colors.series2}
                strokeWidth={2}
                dot={singleEpoch ? { r: 4 } : false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <figure className="m-0">
        <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <span className="text-[12.5px] font-medium text-[var(--text-secondary)]">
            Validation score per epoch
          </span>
          <span className="flex items-center gap-3">
            {[
              { label: 'Macro F1', color: colors.series1 },
              { label: 'Accuracy', color: colors.series3 },
            ].map((item) => (
              <span
                key={item.label}
                className="flex items-center gap-1.5 text-[11.5px] text-[var(--text-secondary)]"
              >
                <span
                  aria-hidden
                  className="h-2.5 w-2.5 rounded-[3px]"
                  style={{ background: item.color }}
                />
                {item.label}
              </span>
            ))}
          </span>
        </figcaption>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 6, right: 10, bottom: 0, left: -16 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epoch" {...axisProps} allowDecimals={false} />
              <YAxis {...axisProps} width={50} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip content={scoreTooltip} cursor={cursorLine} />
              <Line
                type="monotone"
                dataKey="val_f1_macro"
                stroke={colors.series1}
                strokeWidth={2}
                dot={singleEpoch ? { r: 4 } : false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                stroke={colors.series3}
                strokeWidth={2}
                dot={singleEpoch ? { r: 4 } : false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>
    </div>
  )
}
