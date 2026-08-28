/**
 * Live feed of newly classified feedback.
 *
 * Fed by the API's SSE endpoint. The connection state is always visible,
 * because a "live" panel that has quietly stopped updating is worse than no
 * live panel at all.
 */

import { Pause, Play, Radio } from 'lucide-react'
import { useState } from 'react'
import { useLiveFeed } from '../hooks/useLiveFeed'
import { relativeTime, truncate } from '../lib/format'
import { SentimentBadge } from './SentimentBadge'
import { Button, EmptyState } from './ui'

const STATUS_META = {
  live: { label: 'Live', color: 'var(--status-good)' },
  connecting: { label: 'Connecting', color: 'var(--status-warning)' },
  reconnecting: { label: 'Reconnecting', color: 'var(--status-warning)' },
  paused: { label: 'Paused', color: 'var(--text-muted)' },
}

export function LiveFeed({ height = 'max-h-[420px]' }) {
  const [enabled, setEnabled] = useState(true)
  const { items, status, transport } = useLiveFeed({ enabled })
  const meta = STATUS_META[status] ?? STATUS_META.connecting

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
          <span
            className={status === 'live' ? 'live-dot' : ''}
            aria-hidden
            style={{
              width: 8,
              height: 8,
              borderRadius: 999,
              background: meta.color,
              display: 'inline-block',
            }}
          />
          {meta.label}
          {transport && (
            <span className="text-[var(--text-muted)]" title="Transport used by the server">
              · {transport === 'change_stream' ? 'change stream' : 'polling'}
            </span>
          )}
        </span>

        <Button
          variant="ghost"
          onClick={() => setEnabled((current) => !current)}
          className="!px-2 !py-1 !text-[11.5px]"
        >
          {enabled ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {enabled ? 'Pause' : 'Resume'}
        </Button>
      </div>

      {items.length === 0 ? (
        <EmptyState
          icon={Radio}
          title="Waiting for new feedback"
          hint="Submit something on the Analyze page, or POST to /api/feedback — it will appear here within a second."
        />
      ) : (
        <ul className={`space-y-2 overflow-y-auto pr-1 ${height}`}>
          {items.map((item) => (
            <li
              key={item.id}
              className="slide-in rounded-lg border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2.5"
            >
              <div className="flex items-start justify-between gap-2">
                <SentimentBadge
                  sentiment={item.sentiment}
                  confidence={item.confidence}
                  size="sm"
                />
                <span className="shrink-0 text-[11px] text-[var(--text-muted)]">
                  {relativeTime(item.analyzed_at ?? item.created_at)}
                </span>
              </div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-[var(--text-secondary)]">
                {truncate(item.text, 160)}
              </p>
              {item.categories?.length > 0 && (
                <p className="mt-1.5 flex flex-wrap gap-1">
                  {item.categories.map((category) => (
                    <span
                      key={category}
                      className="rounded bg-[var(--surface-1)] px-1.5 py-0.5 text-[10.5px] text-[var(--text-muted)]"
                    >
                      {category}
                    </span>
                  ))}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
