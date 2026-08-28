/**
 * Confusion matrix.
 *
 * Magnitude on a grid, so the encoding is a single-hue sequential ramp
 * (light = few, dark = many) - never a rainbow, which would make "more" and
 * "different" look like the same thing. Each cell prints its own count, so the
 * colour is reinforcement rather than the only channel.
 *
 * Rows are normalised for the colour scale, because what the reader is asking
 * is "of the items that really were negative, where did they go?" - a question
 * about each row's distribution, not about the global maximum.
 */

import { useMemo } from 'react'
import { num, pct } from '../../lib/format'
import { useChartColors } from '../../lib/theme'

export function ConfusionMatrix({ matrix = [], labels = [] }) {
  const colors = useChartColors()

  const rows = useMemo(
    () =>
      matrix.map((row) => {
        const total = row.reduce((sum, value) => sum + value, 0) || 1
        return { counts: row, total, shares: row.map((value) => value / total) }
      }),
    [matrix],
  )

  if (!matrix.length) return null

  const ramp = colors.sequential ?? []
  const colorFor = (share) => {
    if (!ramp.length) return 'transparent'
    const index = Math.min(ramp.length - 1, Math.floor(share * ramp.length))
    return ramp[index]
  }
  // Past the midpoint of the ramp the fill is dark enough that dark ink fails;
  // flip to the surface colour so the number stays readable.
  const inkFor = (share) => (share >= 0.5 ? 'var(--surface-1)' : 'var(--text-primary)')

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-[12.5px]">
        <caption className="mb-3 text-left text-[12px] text-[var(--text-muted)]">
          Rows are the true label, columns the prediction. Cell shading is the share of
          that row, so the diagonal should be darkest.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="px-2 py-1.5 text-left text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              True ╲ Pred
            </th>
            {labels.map((label) => (
              <th
                key={label}
                scope="col"
                className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]"
              >
                {label}
              </th>
            ))}
            <th scope="col" className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Recall
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={labels[rowIndex] ?? rowIndex}>
              <th
                scope="row"
                className="px-2 py-1.5 text-left text-[12px] font-medium text-[var(--text-secondary)]"
              >
                {labels[rowIndex]}
              </th>
              {row.counts.map((count, columnIndex) => {
                const share = row.shares[columnIndex]
                const isDiagonal = rowIndex === columnIndex
                return (
                  <td key={columnIndex} className="p-[2px]">
                    <div
                      title={`True ${labels[rowIndex]} → predicted ${labels[columnIndex]}: ${num(count)} (${pct(share * 100)})`}
                      className="grid h-14 w-[86px] place-items-center rounded-md"
                      style={{
                        background: colorFor(share),
                        color: inkFor(share),
                        // The diagonal is marked with a surface-coloured inset
                        // ring plus an outer ink ring. An accent-coloured ring
                        // would be blue on a blue ramp - invisible exactly
                        // where the cell is darkest.
                        boxShadow: isDiagonal
                          ? 'inset 0 0 0 2px var(--surface-1), inset 0 0 0 3px var(--text-primary)'
                          : 'none',
                      }}
                    >
                      <span className="tabular text-[14px] font-semibold">{num(count)}</span>
                      <span className="tabular text-[10.5px] opacity-80">{pct(share * 100, 0)}</span>
                    </div>
                  </td>
                )
              })}
              <td className="tabular px-3 py-1.5 text-right font-medium">
                {pct(row.shares[rowIndex] * 100)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
