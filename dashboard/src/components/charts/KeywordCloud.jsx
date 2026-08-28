/**
 * Most frequent content words.
 *
 * Deliberately *not* a scattered word cloud: random placement and rotation
 * encode nothing, and area-by-frequency is one of the least accurate visual
 * comparisons there is. Here the words stay in rank order and frequency is
 * carried by type size **and** a printed count, so the ranking is exact while
 * the shape still reads at a glance.
 */

import clsx from 'clsx'
import { num } from '../../lib/format'
import { EmptyState } from '../ui'

const MIN_SIZE = 12
const MAX_SIZE = 27

export function KeywordCloud({ data = [], sentiment = 'all', onSelect }) {
  if (!data.length) {
    return <EmptyState title="Not enough text to extract keywords" hint="Ingest more feedback and try again." />
  }

  const peak = Math.max(...data.map((item) => item.value), 1)
  const floor = Math.min(...data.map((item) => item.value), 0)
  const span = Math.max(peak - floor, 1)

  const tone =
    sentiment === 'negative'
      ? 'var(--sentiment-negative)'
      : sentiment === 'positive'
        ? 'var(--sentiment-positive)'
        : 'var(--text-primary)'

  return (
    <ul className="flex flex-wrap items-baseline gap-x-3 gap-y-2">
      {data.map((item, index) => {
        const weight = (item.value - floor) / span
        return (
          <li key={item.text}>
            <button
              type="button"
              onClick={() => onSelect?.(item.text)}
              title={`${item.text} — ${num(item.value)} mentions (rank ${index + 1})`}
              className={clsx(
                'rounded px-1 leading-tight transition-opacity hover:opacity-70',
                weight > 0.55 ? 'font-semibold' : 'font-normal',
              )}
              style={{
                fontSize: MIN_SIZE + weight * (MAX_SIZE - MIN_SIZE),
                // Frequency also drives opacity so the ranking survives in
                // greyscale, where size alone is hard to compare.
                color: tone,
                opacity: 0.55 + weight * 0.45,
              }}
            >
              {item.text}
              <span className="tabular ml-1 align-super text-[10px] font-normal text-[var(--text-muted)]">
                {item.value}
              </span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
