/**
 * Shared chart chrome.
 *
 * Axis, grid and tooltip defaults live here so every chart in the app reads as
 * one system: hairline grid, recessive axes, one tooltip design, and a table
 * view available behind every plot for readers who need the exact numbers (or
 * who cannot use the colours at all).
 */

import { useState } from 'react'
import clsx from 'clsx'
import { Table2 } from 'lucide-react'
import { Button } from '../ui'

/** Recharts prop bundles - spread these instead of restating them per chart. */
export const axisProps = {
  tickLine: false,
  axisLine: false,
  tick: { fontSize: 11, fill: 'var(--text-muted)' },
}

export const gridProps = {
  strokeDasharray: '0',
  stroke: 'var(--gridline)',
  vertical: false,
}

/**
 * Tooltip shell.
 *
 * Rows are passed in already formatted so each chart controls its own wording
 * while the frame, spacing and type stay identical everywhere.
 */
export function TooltipShell({ title, rows, footer }) {
  return (
    <div
      className="min-w-[168px] rounded-lg border px-3 py-2 shadow-lg"
      style={{
        background: 'var(--surface-1)',
        borderColor: 'var(--border-strong)',
      }}
    >
      <p className="text-[12px] font-semibold text-[var(--text-primary)]">{title}</p>
      <dl className="mt-1.5 flex flex-col gap-1">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center gap-3">
            {row.color && (
              <span
                aria-hidden
                className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
                style={{ background: row.color }}
              />
            )}
            <dt className="mr-auto text-[12px] text-[var(--text-secondary)]">{row.label}</dt>
            <dd className="tabular text-[12px] font-medium text-[var(--text-primary)]">
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
      {footer && (
        <p className="mt-1.5 border-t border-[var(--border)] pt-1.5 text-[11px] text-[var(--text-muted)]">
          {footer}
        </p>
      )}
    </div>
  )
}

/** Crosshair styling for line/area charts. */
export const cursorLine = {
  stroke: 'var(--baseline)',
  strokeWidth: 1,
}

/** Hover wash for bar charts - a fill, never a colour change on the mark. */
export const cursorBar = { fill: 'var(--surface-2)', fillOpacity: 0.6 }

/**
 * Wraps a chart with a "Table" toggle.
 *
 * The table is the accessibility fallback the method requires: exact values,
 * no colour dependency, and copy-pasteable.
 */
export function ChartWithTable({ columns, rows, children, className, tableLabel = 'Table' }) {
  const [showTable, setShowTable] = useState(false)

  return (
    <div className={className}>
      <div className="mb-2 flex justify-end">
        <Button
          variant="ghost"
          onClick={() => setShowTable((open) => !open)}
          aria-pressed={showTable}
          className="!px-2 !py-1 !text-[11.5px]"
        >
          <Table2 className="h-3.5 w-3.5" aria-hidden />
          {showTable ? 'Chart' : tableLabel}
        </Button>
      </div>

      {showTable ? (
        <div className="max-h-[320px] overflow-auto rounded-lg border border-[var(--border)]">
          <table className="w-full border-collapse text-[12.5px]">
            <thead className="sticky top-0 bg-[var(--surface-2)]">
              <tr>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    scope="col"
                    className={clsx(
                      'px-3 py-2 font-medium text-[var(--text-secondary)]',
                      column.numeric ? 'text-right' : 'text-left',
                    )}
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-t border-[var(--border)]">
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className={clsx(
                        'px-3 py-1.5',
                        column.numeric
                          ? 'tabular text-right text-[var(--text-primary)]'
                          : 'text-[var(--text-secondary)]',
                      )}
                    >
                      {row[column.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        children
      )}
    </div>
  )
}
