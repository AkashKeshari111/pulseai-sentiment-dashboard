/**
 * Insights - "what are customers actually talking about".
 *
 * The Overview answers *how much*; this page answers *about what*. Issue
 * categories give the structured view, the keyword panel gives the unstructured
 * one, and both are clickable so a finding turns straight into a filtered
 * query rather than a note to investigate later.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FilterBar } from '../components/FilterBar'
import { BreakdownBars } from '../components/charts/BreakdownBars'
import { IssueTable } from '../components/charts/IssueTable'
import { KeywordCloud } from '../components/charts/KeywordCloud'
import { Button, Card, CardBody, CardHeader, QueryBoundary } from '../components/ui'
import { useIssues, useKeywords, useProducts, useSummary } from '../hooks/useAnalytics'
import { useFilters } from '../lib/filters'
import { num, pct } from '../lib/format'

const KEYWORD_VIEWS = [
  { key: 'negative', label: 'Complaints' },
  { key: 'positive', label: 'Praise' },
  { key: 'all', label: 'Everything' },
]

export function Insights() {
  const { setFilter } = useFilters()
  const navigate = useNavigate()
  const [keywordView, setKeywordView] = useState('negative')

  const issues = useIssues(12)
  const keywords = useKeywords(keywordView, 50)
  const products = useProducts()
  const summary = useSummary()

  const topIssue = issues.data?.[0]

  return (
    <div className="space-y-4">
      <FilterBar />

      {topIssue && (
        <Card className="border-l-4" style={{ borderLeftColor: 'var(--status-serious)' }}>
          <CardBody className="!py-4">
            <p className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Headline
            </p>
            <p className="mt-1.5 text-[15px] leading-relaxed">
              <strong>{topIssue.category}</strong> is the largest source of negative feedback:{' '}
              <strong className="tabular">{num(topIssue.negative)}</strong> negative mentions,{' '}
              <strong className="tabular">{pct(topIssue.negative_rate, 0)}</strong> of everything
              said about it. That is {pct(
                summary.data?.total ? (topIssue.negative / summary.data.total) * 100 : 0,
                1,
              )}{' '}
              of all feedback in this period.
            </p>
            {topIssue.sample && (
              <blockquote className="mt-2.5 border-l-2 border-[var(--border-strong)] pl-3 text-[12.5px] italic leading-relaxed text-[var(--text-muted)]">
                “{topIssue.sample}”
              </blockquote>
            )}
          </CardBody>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Issue categories"
            subtitle="Ranked by absolute negative volume — a 100% negative category with two mentions is noise, not a problem. Click a category to filter the whole dashboard."
          />
          <CardBody>
            <QueryBoundary query={issues} isEmpty={(rows) => !rows?.length} height="h-72">
              {(rows) => (
                <IssueTable
                  data={rows}
                  onSelect={(category) => {
                    setFilter('category', category)
                    navigate('/explorer')
                  }}
                />
              )}
            </QueryBoundary>
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Language"
            subtitle="Most frequent content words, ranked. Click a word to search for it."
            actions={
              <div className="flex gap-1" role="group" aria-label="Keyword source">
                {KEYWORD_VIEWS.map((view) => (
                  <Button
                    key={view.key}
                    variant={keywordView === view.key ? 'subtle' : 'ghost'}
                    aria-pressed={keywordView === view.key}
                    onClick={() => setKeywordView(view.key)}
                    className="!px-2 !py-1 !text-[11.5px]"
                  >
                    {view.label}
                  </Button>
                ))}
              </div>
            }
          />
          <CardBody>
            <QueryBoundary query={keywords} isEmpty={(rows) => !rows?.length} height="h-72">
              {(rows) => (
                <KeywordCloud
                  data={rows}
                  sentiment={keywordView}
                  onSelect={(word) => {
                    setFilter('search', word)
                    navigate('/explorer')
                  }}
                />
              )}
            </QueryBoundary>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="By product"
          subtitle="Sentiment split for each product or service mentioned in the feedback"
        />
        <CardBody>
          <QueryBoundary
            query={products}
            isEmpty={(rows) => !rows?.filter((row) => row.key !== 'unknown').length}
            height="h-56"
          >
            {(rows) => (
              <BreakdownBars
                data={rows.filter((row) => row.key && row.key !== 'unknown')}
                keyLabel="Product"
              />
            )}
          </QueryBoundary>
        </CardBody>
      </Card>
    </div>
  )
}
