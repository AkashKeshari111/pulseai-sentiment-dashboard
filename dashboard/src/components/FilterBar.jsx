/**
 * The single filter row that sits above the charts on every analytics page.
 *
 * Filters compose: date window, channel, issue category, sentiment and free
 * text all narrow the same query. Because every panel reads the same state,
 * one change updates the whole page consistently.
 */

import { useEffect, useState } from 'react'
import { RotateCcw, Search } from 'lucide-react'
import { DATE_RANGES, useFilters } from '../lib/filters'
import { useFilterOptions } from '../hooks/useAnalytics'
import { Button, Select } from './ui'

const SENTIMENT_OPTIONS = [
  { value: 'all', label: 'All sentiment' },
  { value: 'negative', label: 'Negative' },
  { value: 'neutral', label: 'Neutral' },
  { value: 'positive', label: 'Positive' },
]

const titleCase = (value) =>
  value.replace(/[_-]/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())

export function FilterBar({ showSearch = true }) {
  const { filters, setFilter, reset, activeCount } = useFilters()
  const { data: options } = useFilterOptions()
  const [draftSearch, setDraftSearch] = useState(filters.search)

  // Keep the input in sync when the filter changes elsewhere (reset, back).
  useEffect(() => setDraftSearch(filters.search), [filters.search])

  // Debounced so typing does not fire a request per keystroke.
  useEffect(() => {
    if (draftSearch === filters.search) return undefined
    const timer = setTimeout(() => setFilter('search', draftSearch), 350)
    return () => clearTimeout(timer)
  }, [draftSearch, filters.search, setFilter])

  const sourceOptions = [
    { value: 'all', label: 'All channels' },
    ...(options?.sources ?? []).map((value) => ({ value, label: titleCase(value) })),
  ]
  const categoryOptions = [
    { value: 'all', label: 'All categories' },
    ...(options?.categories ?? []).map((value) => ({ value, label: value })),
  ]

  return (
    <div className="flex flex-wrap items-end gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface-1)] px-4 py-3">
      <Select
        label="Period"
        value={filters.days}
        onChange={(value) => setFilter('days', value)}
        options={DATE_RANGES}
      />
      <Select
        label="Channel"
        value={filters.source}
        onChange={(value) => setFilter('source', value)}
        options={sourceOptions}
      />
      <Select
        label="Category"
        value={filters.category}
        onChange={(value) => setFilter('category', value)}
        options={categoryOptions}
      />
      <Select
        label="Sentiment"
        value={filters.sentiment}
        onChange={(value) => setFilter('sentiment', value)}
        options={SENTIMENT_OPTIONS}
      />

      {showSearch && (
        <label className="flex min-w-[200px] flex-1 flex-col gap-1">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
            Search
          </span>
          <span className="relative">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text-muted)]"
              aria-hidden
            />
            <input
              value={draftSearch}
              onChange={(event) => setDraftSearch(event.target.value)}
              placeholder="Search feedback text…"
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--surface-1)] py-1.5 pl-8 pr-2.5 text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] hover:border-[var(--border-strong)]"
            />
          </span>
        </label>
      )}

      {activeCount > 0 && (
        <Button variant="ghost" onClick={reset} className="mb-[1px]">
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
          Reset ({activeCount})
        </Button>
      )}
    </div>
  )
}
