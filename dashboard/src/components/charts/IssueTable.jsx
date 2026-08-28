/**
 * Top complaint drivers.
 *
 * This is the panel an operations team actually acts on, so it is a table with
 * an inline magnitude bar rather than a standalone chart: the ranking, the
 * absolute volume, the negative rate and a real example all have to be visible
 * together. The bar is a single-series magnitude encoding, so it uses one
 * colour and is anchored to a common baseline.
 */

import { AlertTriangle } from 'lucide-react'
import { num, pct, truncate } from '../../lib/format'
import { EmptyState } from '../ui'

export function IssueTable({ data = [], onSelect }) {
  if (!data.length) {
    return (
      <EmptyState
        title="No issue categories detected"
        hint="Categories are assigned from the feedback text. Ingest more feedback, or extend ISSUE_TAXONOMY in src/preprocessing.py."
        icon={AlertTriangle}
      />
    )
  }

  const peak = Math.max(...data.map((row) => row.negative), 1)

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <caption className="sr-only">
          Issue categories ranked by volume of negative feedback
        </caption>
        <thead>
          <tr className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
            <th scope="col" className="py-2 pr-3 text-left font-medium">
              Category
            </th>
            <th scope="col" className="py-2 pr-3 text-left font-medium">
              Negative mentions
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">
              Rate
            </th>
            <th scope="col" className="py-2 text-right font-medium">
              Total
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const isSevere = row.negative_rate >= 60
            return (
              <tr
                key={row.category}
                className="border-t border-[var(--border)] align-middle transition-colors hover:bg-[var(--surface-2)]"
              >
                <th scope="row" className="max-w-[190px] py-2.5 pr-3 text-left font-medium">
                  <button
                    type="button"
                    onClick={() => onSelect?.(row.category)}
                    className="truncate text-left hover:underline"
                    title={row.sample ? `e.g. "${truncate(row.sample, 160)}"` : row.category}
                  >
                    {row.category}
                  </button>
                </th>

                <td className="py-2.5 pr-3">
                  <span className="flex items-center gap-2">
                    <span
                      className="h-2.5 rounded-[4px]"
                      style={{
                        width: `${Math.max((row.negative / peak) * 100, 3)}%`,
                        minWidth: 6,
                        background: 'var(--sentiment-negative)',
                      }}
                      aria-hidden
                    />
                    <span className="tabular w-10 shrink-0 text-[12.5px]">
                      {num(row.negative)}
                    </span>
                  </span>
                </td>

                <td className="tabular py-2.5 pr-3 text-right">
                  <span
                    className="inline-flex items-center gap-1"
                    style={{ color: isSevere ? 'var(--status-critical)' : 'var(--text-secondary)' }}
                  >
                    {/* Icon + label, never colour alone, for the severe case. */}
                    {isSevere && <AlertTriangle className="h-3 w-3" aria-label="High negative rate" />}
                    {pct(row.negative_rate, 0)}
                  </span>
                </td>

                <td className="tabular py-2.5 text-right text-[var(--text-muted)]">
                  {num(row.total)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
