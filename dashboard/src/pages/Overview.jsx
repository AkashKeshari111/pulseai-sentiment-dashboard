/**
 * Overview - the "how are we doing right now" page.
 *
 * Reading order is deliberate: four headline numbers, then the trend that
 * explains them, then the composition, then the live evidence. Nothing here
 * needs a click to be useful.
 */

import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, ArrowRight, MessageSquare } from 'lucide-react'
import { FilterBar } from '../components/FilterBar'
import { KpiCard } from '../components/KpiCard'
import { LiveFeed } from '../components/LiveFeed'
import { BreakdownBars } from '../components/charts/BreakdownBars'
import { SentimentDonut } from '../components/charts/SentimentDonut'
import { TrendChart } from '../components/charts/TrendChart'
import { IssueTable } from '../components/charts/IssueTable'
import { Card, CardBody, CardHeader, EmptyState, QueryBoundary } from '../components/ui'
import { useFilters } from '../lib/filters'
import { useIssues, useSources, useSummary, useTrends } from '../hooks/useAnalytics'
import { num, nssBand, pct, ratio, signed } from '../lib/format'

/** Bucket size that keeps the trend chart readable for the selected window. */
function granularityFor(days) {
  const window = Number(days)
  if (!Number.isFinite(window)) return 'week'
  if (window <= 2) return 'hour'
  if (window <= 45) return 'day'
  if (window <= 200) return 'week'
  return 'month'
}

export function Overview() {
  const { filters, setFilter } = useFilters()
  const granularity = granularityFor(filters.days)

  const summary = useSummary()
  const trends = useTrends(granularity)
  const sources = useSources()
  const issues = useIssues(8)

  const data = summary.data
  const total = data?.total ?? 0
  const band = nssBand(data?.net_sentiment_score)

  // Sparklines share the trend query - no extra request, and they cannot
  // disagree with the chart below them.
  const sparks = useMemo(() => {
    const points = trends.data ?? []
    return {
      volume: points.map((point) => ({ value: point.total })),
      negative: points.map((point) => ({ value: point.negative })),
      score: points.map((point) => ({ value: point.net_sentiment_score })),
    }
  }, [trends.data])

  const negativeShare = data?.distribution?.negative ?? 0

  return (
    <div className="space-y-4">
      <FilterBar />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Feedback analysed"
          value={num(total)}
          hint={`Across ${sources.data?.length ?? 0} channels in the selected period`}
          spark={sparks.volume}
          sparkColor="var(--series-1)"
          loading={summary.isLoading}
        />
        <KpiCard
          label="Net sentiment score"
          value={signed(data?.net_sentiment_score ?? 0)}
          hint="% positive minus % negative, from -100 to +100"
          status={band.role}
          statusLabel={band.label}
          accent={
            (data?.net_sentiment_score ?? 0) >= 0
              ? 'var(--sentiment-positive)'
              : 'var(--sentiment-negative)'
          }
          spark={sparks.score}
          sparkColor="var(--series-1)"
          loading={summary.isLoading}
        />
        <KpiCard
          label="Negative share"
          value={pct(negativeShare)}
          hint={`${num(data?.counts?.negative ?? 0)} items need attention`}
          accent="var(--sentiment-negative)"
          status={negativeShare > 35 ? 'critical' : negativeShare > 20 ? 'warning' : 'good'}
          statusLabel={negativeShare > 35 ? 'High' : negativeShare > 20 ? 'Watch' : 'Low'}
          spark={sparks.negative}
          sparkColor="var(--sentiment-negative)"
          loading={summary.isLoading}
        />
        <KpiCard
          label="Model confidence"
          value={ratio(data?.avg_confidence ?? 0)}
          hint="Mean softmax probability of the predicted class"
          loading={summary.isLoading}
        />
      </div>

      <Card>
        <CardHeader
          title="Sentiment trend"
          subtitle={`Bucketed by ${granularity}. Direction matters more than level: a steady negative rate is business as usual, a rising one is an incident.`}
        />
        <CardBody>
          <QueryBoundary query={trends} isEmpty={(rows) => !rows?.length} height="h-72">
            {(rows) => <TrendChart data={rows} granularity={granularity} />}
          </QueryBoundary>
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Sentiment mix" subtitle="Share of all feedback in the period" />
          <CardBody className="flex items-center">
            <QueryBoundary query={summary} isEmpty={(value) => !value?.total} height="h-64">
              {(value) => <SentimentDonut counts={value.counts} total={value.total} />}
            </QueryBoundary>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="By channel"
            subtitle="Where feedback arrives, and how it splits"
          />
          <CardBody>
            <QueryBoundary query={sources} isEmpty={(rows) => !rows?.length} height="h-64">
              {(rows) => <BreakdownBars data={rows} keyLabel="Channel" />}
            </QueryBoundary>
          </CardBody>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Top issue drivers"
            subtitle="Categories ranked by negative volume"
            actions={
              <Link
                to="/insights"
                className="inline-flex items-center gap-1 text-[12px] text-[var(--accent)] hover:underline"
              >
                All insights <ArrowRight className="h-3.5 w-3.5" aria-hidden />
              </Link>
            }
          />
          <CardBody>
            <QueryBoundary
              query={issues}
              isEmpty={(rows) => !rows?.length}
              height="h-56"
              empty={
                <EmptyState
                  icon={AlertTriangle}
                  title="No categorised issues yet"
                  hint="Issue tags are derived from the feedback text as it is ingested."
                />
              }
            >
              {(rows) => (
                <IssueTable data={rows} onSelect={(category) => setFilter('category', category)} />
              )}
            </QueryBoundary>
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Live feed"
            subtitle="Streamed from the API as feedback is classified"
            actions={<MessageSquare className="h-4 w-4 text-[var(--text-muted)]" aria-hidden />}
          />
          <CardBody>
            <LiveFeed />
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
