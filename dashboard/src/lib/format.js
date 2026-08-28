/** Formatting and labelling helpers shared by every view. */

import { format, formatDistanceToNowStrict, parseISO } from 'date-fns'

export const SENTIMENTS = ['negative', 'neutral', 'positive']

/** Display metadata per sentiment. The glyph is the secondary encoding that
 *  keeps meaning available when colour is not (CVD, print, forced colours). */
export const SENTIMENT_META = {
  negative: { label: 'Negative', glyph: '▼', varName: '--sentiment-negative' },
  neutral: { label: 'Neutral', glyph: '■', varName: '--sentiment-neutral' },
  positive: { label: 'Positive', glyph: '▲', varName: '--sentiment-positive' },
}

export const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })
export const plain = new Intl.NumberFormat('en')

export const num = (value) => (Number.isFinite(value) ? plain.format(value) : '—')
export const shortNum = (value) =>
  Number.isFinite(value) ? (Math.abs(value) >= 10000 ? compact.format(value) : plain.format(value)) : '—'

export const pct = (value, digits = 1) =>
  Number.isFinite(value) ? `${value.toFixed(digits)}%` : '—'

export const ratio = (value, digits = 1) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'

export const signed = (value, digits = 1) =>
  Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(digits)}` : '—'

/** Net Sentiment Score band -> the status role used to annotate it. */
export function nssBand(score) {
  if (!Number.isFinite(score)) return { label: 'No data', role: 'neutral' }
  if (score >= 40) return { label: 'Excellent', role: 'good' }
  if (score >= 10) return { label: 'Healthy', role: 'good' }
  if (score >= -10) return { label: 'Mixed', role: 'warning' }
  if (score >= -40) return { label: 'Poor', role: 'serious' }
  return { label: 'Critical', role: 'critical' }
}

export function safeDate(value) {
  if (!value) return null
  try {
    const parsed = typeof value === 'string' ? parseISO(value) : new Date(value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  } catch {
    return null
  }
}

export function formatDate(value, pattern = 'd MMM yyyy') {
  const parsed = safeDate(value)
  return parsed ? format(parsed, pattern) : '—'
}

export function formatDateTime(value) {
  return formatDate(value, 'd MMM yyyy, HH:mm')
}

export function relativeTime(value) {
  const parsed = safeDate(value)
  if (!parsed) return '—'
  return `${formatDistanceToNowStrict(parsed)} ago`
}

/** Axis tick label for a trend bucket, adapted to the granularity. */
export function formatPeriod(period, granularity = 'day') {
  if (!period) return ''
  if (granularity === 'week') return period.replace('-W', ' W')
  if (granularity === 'month') {
    const parsed = safeDate(`${period}-01`)
    return parsed ? format(parsed, 'MMM yy') : period
  }
  if (granularity === 'hour') {
    const parsed = safeDate(`${period}:00`)
    return parsed ? format(parsed, 'd MMM HH:mm') : period
  }
  const parsed = safeDate(period)
  return parsed ? format(parsed, 'd MMM') : period
}

export function truncate(text, length = 140) {
  if (!text) return ''
  return text.length > length ? `${text.slice(0, length - 1).trimEnd()}…` : text
}

/** Turn any array of flat objects into a CSV blob download. */
export function downloadCsv(rows, filename) {
  if (!rows?.length) return
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))]
  const escape = (value) => {
    if (value === null || value === undefined) return ''
    const text = typeof value === 'object' ? JSON.stringify(value) : String(value)
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  const csv = [
    columns.join(','),
    ...rows.map((row) => columns.map((column) => escape(row[column])).join(',')),
  ].join('\n')

  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
