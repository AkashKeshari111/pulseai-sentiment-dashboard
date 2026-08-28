/**
 * Explorer - the audit surface.
 *
 * Every aggregate on the other pages is a claim; this page is where that claim
 * can be checked against the individual records behind it. It shares the same
 * filter state, so "the Overview says 42 negative" and "the table lists 42
 * negative" are guaranteed to be the same query.
 */

import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Download, Trash2 } from 'lucide-react'
import { FilterBar } from '../components/FilterBar'
import { SentimentBadge } from '../components/SentimentBadge'
import { Button, Card, CardBody, CardHeader, EmptyState, ErrorState, LoadingBlock, Select } from '../components/ui'
import { useDeleteFeedback, useFeedbackList } from '../hooks/useAnalytics'
import { downloadCsv, formatDateTime, num, ratio, truncate } from '../lib/format'

const PAGE_SIZES = [
  { value: '25', label: '25 per page' },
  { value: '50', label: '50 per page' },
  { value: '100', label: '100 per page' },
]

const SORTS = [
  { value: '-created_at', label: 'Newest first' },
  { value: 'created_at', label: 'Oldest first' },
  { value: '-confidence', label: 'Most confident' },
  { value: 'confidence', label: 'Least confident' },
]

const titleCase = (value = '') =>
  value.replace(/[_-]/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())

export function Explorer() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [pageSize, setPageSize] = useState('25')
  const [sort, setSort] = useState('-created_at')
  const [expanded, setExpanded] = useState(null)

  const page = Number(searchParams.get('page') ?? 1)
  const setPage = (next) =>
    setSearchParams(
      (previous) => {
        const params = new URLSearchParams(previous)
        if (next <= 1) params.delete('page')
        else params.set('page', String(next))
        return params
      },
      { replace: true },
    )

  const query = useFeedbackList({ page, pageSize: Number(pageSize), sort })
  const remove = useDeleteFeedback()

  const items = query.data?.items ?? []
  const total = query.data?.total ?? 0
  const pages = query.data?.pages ?? 1

  return (
    <div className="space-y-4">
      <FilterBar />

      <Card>
        <CardHeader
          title="Feedback explorer"
          subtitle={
            query.isLoading
              ? 'Loading…'
              : `${num(total)} records match the current filters · page ${page} of ${pages}`
          }
          actions={
            <div className="flex flex-wrap items-end gap-2">
              <Select
                label="Sort"
                value={sort}
                onChange={(value) => {
                  setSort(value)
                  setPage(1)
                }}
                options={SORTS}
              />
              <Select
                label="Size"
                value={pageSize}
                onChange={(value) => {
                  setPageSize(value)
                  setPage(1)
                }}
                options={PAGE_SIZES}
              />
              <Button
                variant="ghost"
                disabled={!items.length}
                onClick={() =>
                  downloadCsv(
                    items.map((item) => ({
                      id: item.id,
                      created_at: item.created_at,
                      source: item.source,
                      product: item.product ?? '',
                      rating: item.rating ?? '',
                      sentiment: item.sentiment,
                      confidence: item.confidence,
                      categories: (item.categories ?? []).join(' | '),
                      text: item.text,
                    })),
                    `pulseai-feedback-page-${page}.csv`,
                  )
                }
                className="mb-[1px]"
              >
                <Download className="h-3.5 w-3.5" aria-hidden />
                Export page
              </Button>
            </div>
          }
        />

        <CardBody className="!px-0">
          {query.isLoading ? (
            <LoadingBlock height="h-72" label="Loading feedback" />
          ) : query.isError ? (
            <ErrorState error={query.error} onRetry={query.refetch} />
          ) : items.length === 0 ? (
            <EmptyState
              title="No feedback matches these filters"
              hint="Widen the date range or clear the search box. If the database is empty, seed it with: python -m api.seed --count 400"
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] border-collapse text-[13px]">
                <thead>
                  <tr className="border-y border-[var(--border)] text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                    <th scope="col" className="px-5 py-2.5 text-left font-medium">Feedback</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-medium">Sentiment</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-medium">Channel</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-medium">Categories</th>
                    <th scope="col" className="px-3 py-2.5 text-right font-medium">Received</th>
                    <th scope="col" className="px-5 py-2.5 text-right font-medium">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => {
                    const isOpen = expanded === item.id
                    return (
                      <tr
                        key={item.id}
                        className="border-b border-[var(--border)] align-top transition-colors hover:bg-[var(--surface-2)]"
                      >
                        <td className="max-w-[420px] px-5 py-3">
                          <button
                            type="button"
                            onClick={() => setExpanded(isOpen ? null : item.id)}
                            className="text-left leading-relaxed text-[var(--text-primary)]"
                            aria-expanded={isOpen}
                          >
                            {isOpen ? item.text : truncate(item.text, 150)}
                          </button>
                          {isOpen && (
                            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11.5px] text-[var(--text-muted)] sm:grid-cols-3">
                              <div>
                                <dt className="inline">Confidence: </dt>
                                <dd className="tabular inline">{ratio(item.confidence)}</dd>
                              </div>
                              <div>
                                <dt className="inline">Negative: </dt>
                                <dd className="tabular inline">{ratio(item.scores?.negative)}</dd>
                              </div>
                              <div>
                                <dt className="inline">Neutral: </dt>
                                <dd className="tabular inline">{ratio(item.scores?.neutral)}</dd>
                              </div>
                              <div>
                                <dt className="inline">Positive: </dt>
                                <dd className="tabular inline">{ratio(item.scores?.positive)}</dd>
                              </div>
                              <div className="col-span-2">
                                <dt className="inline">Model: </dt>
                                <dd className="inline">{item.model}</dd>
                              </div>
                            </dl>
                          )}
                        </td>

                        <td className="px-3 py-3">
                          <SentimentBadge
                            sentiment={item.sentiment}
                            confidence={item.confidence}
                            size="sm"
                          />
                        </td>

                        <td className="px-3 py-3 text-[12.5px] text-[var(--text-secondary)]">
                          {titleCase(item.source)}
                          {item.product && (
                            <span className="block text-[11px] text-[var(--text-muted)]">
                              {item.product}
                            </span>
                          )}
                        </td>

                        <td className="px-3 py-3">
                          <span className="flex flex-wrap gap-1">
                            {(item.categories ?? []).slice(0, 2).map((category) => (
                              <span
                                key={category}
                                className="rounded bg-[var(--surface-2)] px-1.5 py-0.5 text-[10.5px] text-[var(--text-muted)]"
                              >
                                {category}
                              </span>
                            ))}
                            {(item.categories?.length ?? 0) > 2 && (
                              <span className="text-[10.5px] text-[var(--text-muted)]">
                                +{item.categories.length - 2}
                              </span>
                            )}
                          </span>
                        </td>

                        <td className="tabular whitespace-nowrap px-3 py-3 text-right text-[12px] text-[var(--text-muted)]">
                          {formatDateTime(item.created_at)}
                        </td>

                        <td className="px-5 py-3 text-right">
                          <button
                            type="button"
                            onClick={() => remove.mutate(item.id)}
                            disabled={remove.isPending}
                            aria-label="Delete this feedback"
                            className="rounded-md p-1.5 text-[var(--text-muted)] transition-colors hover:bg-[var(--sentiment-negative-soft)] hover:text-[var(--status-critical)] disabled:opacity-40"
                          >
                            <Trash2 className="h-3.5 w-3.5" aria-hidden />
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>

        {pages > 1 && (
          <footer className="flex items-center justify-between gap-3 border-t border-[var(--border)] px-5 py-3">
            <p className="text-[12px] text-[var(--text-muted)]">
              Showing {(page - 1) * Number(pageSize) + 1}–
              {Math.min(page * Number(pageSize), total)} of {num(total)}
            </p>
            <div className="flex items-center gap-2">
              <Button variant="ghost" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                <ChevronLeft className="h-3.5 w-3.5" aria-hidden /> Previous
              </Button>
              <Button variant="ghost" disabled={page >= pages} onClick={() => setPage(page + 1)}>
                Next <ChevronRight className="h-3.5 w-3.5" aria-hidden />
              </Button>
            </div>
          </footer>
        )}
      </Card>
    </div>
  )
}
