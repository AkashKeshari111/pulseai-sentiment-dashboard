/**
 * Sentiment label chip.
 *
 * Colour is never the only channel: the chip always carries a glyph and the
 * written label. That keeps it readable under colour-vision deficiency, in
 * greyscale print, and in forced-colours mode.
 */

import clsx from 'clsx'
import { SENTIMENT_META } from '../lib/format'

const SOFT = {
  negative: 'var(--sentiment-negative-soft)',
  neutral: 'var(--sentiment-neutral-soft)',
  positive: 'var(--sentiment-positive-soft)',
}

export function SentimentBadge({ sentiment, confidence, size = 'md', className }) {
  const meta = SENTIMENT_META[sentiment]
  if (!meta) return null

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-md font-medium whitespace-nowrap',
        size === 'sm' ? 'px-1.5 py-0.5 text-[11px]' : 'px-2 py-1 text-[12px]',
        className,
      )}
      style={{ background: SOFT[sentiment], color: `var(${meta.varName})` }}
    >
      <span aria-hidden className="text-[9px] leading-none">
        {meta.glyph}
      </span>
      {meta.label}
      {Number.isFinite(confidence) && (
        <span className="tabular opacity-75">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  )
}

/** Colour swatch + name, used in chart legends. */
export function SentimentLegend({ items = ['negative', 'neutral', 'positive'], className }) {
  return (
    <ul className={clsx('flex flex-wrap items-center gap-x-4 gap-y-1.5', className)}>
      {items.map((key) => {
        const meta = SENTIMENT_META[key]
        return (
          <li key={key} className="flex items-center gap-1.5 text-[12px] text-[var(--text-secondary)]">
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-[3px]"
              style={{ background: `var(${meta.varName})` }}
            />
            {meta.label}
          </li>
        )
      })}
    </ul>
  )
}
