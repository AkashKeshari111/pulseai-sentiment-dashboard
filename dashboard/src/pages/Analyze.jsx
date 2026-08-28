/**
 * Analyze - the live inference playground.
 *
 * This is the page a demo actually runs on: type a sentence, watch the model
 * classify it in real time, see *why* it decided that, and optionally push the
 * result into the database so it flows through to every other page.
 */

import { useRef, useState } from 'react'
import { Loader2, Send, Sparkles, Upload } from 'lucide-react'
import { SentimentBadge } from '../components/SentimentBadge'
import { Button, Card, CardBody, CardHeader, ErrorState, Select, TextInput } from '../components/ui'
import { useCreateFeedback, usePredict, useUploadCsv } from '../hooks/useAnalytics'
import { SENTIMENT_META, num, ratio } from '../lib/format'

const SAMPLES = [
  'The delivery was three days late and the box arrived completely crushed. Support never replied to my emails.',
  'Absolutely brilliant service — the team resolved my billing issue in under five minutes and followed up the next day.',
  'It works as described. Nothing special about the app, but the checkout was fine and the price is reasonable.',
  'I was excited at first, but the battery drains within four hours and the replacement process is a nightmare.',
]

const SOURCES = [
  { value: 'web', label: 'Website' },
  { value: 'mobile_app', label: 'Mobile app' },
  { value: 'email', label: 'Email' },
  { value: 'twitter', label: 'Twitter / X' },
  { value: 'play_store', label: 'Play Store' },
  { value: 'survey', label: 'Survey' },
  { value: 'call_center', label: 'Call centre' },
]

/** Horizontal probability bars - one row per class, common baseline. */
function ScoreBars({ scores, predicted }) {
  return (
    <ul className="space-y-2">
      {['negative', 'neutral', 'positive'].map((key) => {
        const value = scores?.[key] ?? 0
        const isPredicted = key === predicted
        return (
          <li key={key} className="flex items-center gap-3">
            <span className="w-16 shrink-0 text-[12px] text-[var(--text-secondary)]">
              {SENTIMENT_META[key].label}
            </span>
            <span className="h-2.5 flex-1 overflow-hidden rounded-[4px] bg-[var(--surface-2)]">
              <span
                className="block h-full rounded-[4px] transition-[width] duration-300"
                style={{
                  width: `${Math.max(value * 100, 1)}%`,
                  background: `var(${SENTIMENT_META[key].varName})`,
                  opacity: isPredicted ? 1 : 0.45,
                }}
              />
            </span>
            <span className="tabular w-12 shrink-0 text-right text-[12px] font-medium">
              {ratio(value)}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

/**
 * Word-level attribution.
 *
 * Each word's shade is how much removing it would drop the predicted class
 * probability, so the highlight answers "which words drove this call?".
 */
function Explanation({ tokens, label }) {
  if (!tokens?.length) return null
  const tone = `var(${SENTIMENT_META[label].varName})`

  return (
    <p className="flex flex-wrap gap-x-1 gap-y-1.5 text-[14px] leading-relaxed">
      {tokens.map((token, index) => {
        const weight = Math.max(token.normalised ?? 0, 0)
        return (
          <span
            key={`${token.token}-${index}`}
            title={`contribution ${token.weight >= 0 ? '+' : ''}${token.weight}`}
            className="rounded px-1 py-0.5"
            style={{
              // Only supporting words are tinted; words that argue against the
              // prediction stay plain rather than getting a second colour scale.
              background: weight > 0.08 ? tone : 'transparent',
              opacity: weight > 0.08 ? 0.25 + weight * 0.75 : 1,
              color: weight > 0.55 ? 'var(--surface-1)' : 'var(--text-primary)',
            }}
          >
            {token.token}
          </span>
        )
      })}
    </p>
  )
}

export function Analyze() {
  const [text, setText] = useState(SAMPLES[0])
  const [explain, setExplain] = useState(true)
  const [source, setSource] = useState('web')
  const [product, setProduct] = useState('')
  const [saved, setSaved] = useState(null)
  const fileInput = useRef(null)

  const predict = usePredict()
  const create = useCreateFeedback()
  const upload = useUploadCsv()

  const result = predict.data

  const runPrediction = () => {
    setSaved(null)
    if (text.trim()) predict.mutate({ text: text.trim(), explain })
  }

  const saveToDatabase = () => {
    create.mutate(
      { text: text.trim(), source, product: product.trim() || null },
      { onSuccess: (response) => setSaved(response.ids?.[0] ?? 'saved') },
    )
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader
          title="Classify feedback"
          subtitle="Runs against the live model behind POST /api/predict"
        />
        <CardBody className="space-y-3">
          <label className="block">
            <span className="sr-only">Feedback text</span>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') runPrediction()
              }}
              rows={6}
              maxLength={5000}
              placeholder="Paste a customer review, support ticket or survey comment…"
              className="w-full resize-y rounded-lg border border-[var(--border)] bg-[var(--surface-1)] p-3 text-[14px] leading-relaxed text-[var(--text-primary)] placeholder:text-[var(--text-muted)] hover:border-[var(--border-strong)]"
            />
          </label>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11.5px] text-[var(--text-muted)]">
              {text.length}/5000 · Ctrl+Enter to analyse
            </span>
            <label className="flex items-center gap-1.5 text-[12px] text-[var(--text-secondary)]">
              <input
                type="checkbox"
                checked={explain}
                onChange={(event) => setExplain(event.target.checked)}
                className="accent-[var(--accent)]"
              />
              Explain the prediction
            </label>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              variant="primary"
              onClick={runPrediction}
              disabled={!text.trim() || predict.isPending}
            >
              {predict.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              ) : (
                <Sparkles className="h-3.5 w-3.5" aria-hidden />
              )}
              Analyse
            </Button>

            {SAMPLES.map((sample, index) => (
              <Button
                key={index}
                variant="ghost"
                onClick={() => {
                  setText(sample)
                  predict.reset()
                  setSaved(null)
                }}
                className="!px-2 !py-1 !text-[11.5px]"
              >
                Sample {index + 1}
              </Button>
            ))}
          </div>

          <div className="border-t border-[var(--border)] pt-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Store it
            </p>
            <div className="flex flex-wrap items-end gap-2">
              <Select label="Channel" value={source} onChange={setSource} options={SOURCES} />
              <TextInput
                label="Product"
                value={product}
                onChange={(event) => setProduct(event.target.value)}
                placeholder="optional"
                className="w-40"
              />
              <Button
                variant="subtle"
                onClick={saveToDatabase}
                disabled={!text.trim() || create.isPending}
                className="mb-[1px]"
              >
                {create.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <Send className="h-3.5 w-3.5" aria-hidden />
                )}
                Save to database
              </Button>
            </div>
            {saved && (
              <p className="mt-2 text-[12px]" style={{ color: 'var(--status-good)' }}>
                Stored. It is already in the analytics and the live feed.
              </p>
            )}
            {create.isError && (
              <p className="mt-2 text-[12px]" style={{ color: 'var(--status-critical)' }}>
                {create.error.message}
              </p>
            )}
          </div>

          <div className="border-t border-[var(--border)] pt-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Bulk import
            </p>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) upload.mutate({ file, source })
                event.target.value = ''
              }}
            />
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="ghost" onClick={() => fileInput.current?.click()} disabled={upload.isPending}>
                {upload.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <Upload className="h-3.5 w-3.5" aria-hidden />
                )}
                Upload CSV
              </Button>
              <span className="text-[11.5px] text-[var(--text-muted)]">
                Needs a <code>text</code> column. Optional: source, product, rating, created_at.
              </span>
            </div>
            {upload.isSuccess && (
              <p className="mt-2 text-[12px]" style={{ color: 'var(--status-good)' }}>
                Imported {num(upload.data.inserted)} rows
                {upload.data.skipped ? ` (${num(upload.data.skipped)} skipped)` : ''}.
              </p>
            )}
            {upload.isError && (
              <p className="mt-2 text-[12px]" style={{ color: 'var(--status-critical)' }}>
                {upload.error.message}
              </p>
            )}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Result"
          subtitle={result ? `${result.model} · ${result.latency_ms} ms` : 'Nothing analysed yet'}
        />
        <CardBody className="flex flex-col justify-center">
          {predict.isError ? (
            <ErrorState error={predict.error} onRetry={runPrediction} />
          ) : !result ? (
            <p className="py-10 text-center text-[13px] text-[var(--text-muted)]">
              Enter some feedback and press <strong>Analyse</strong>.
            </p>
          ) : (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-3">
                <SentimentBadge sentiment={result.label} confidence={result.confidence} />
                {result.categories?.map((category) => (
                  <span
                    key={category}
                    className="rounded-md bg-[var(--surface-2)] px-2 py-1 text-[11.5px] text-[var(--text-secondary)]"
                  >
                    {category}
                  </span>
                ))}
              </div>

              <div>
                <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Class probabilities
                </p>
                <ScoreBars scores={result.scores} predicted={result.label} />
              </div>

              {result.explanation && (
                <div>
                  <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                    Why this prediction
                  </p>
                  <Explanation tokens={result.explanation} label={result.label} />
                  <p className="mt-2 text-[11.5px] leading-relaxed text-[var(--text-muted)]">
                    Shading is leave-one-out attribution: each word is removed in turn and the
                    drop in the predicted class probability is its contribution. Darker means the
                    word mattered more.
                  </p>
                </div>
              )}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}
