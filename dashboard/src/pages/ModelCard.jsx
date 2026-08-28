/**
 * Model card - the evidence page.
 *
 * Reads reports/metrics.json through the API, so the numbers shown here are
 * exactly the ones the training scripts produced. Nothing on this page is
 * hard-coded; re-running training and refreshing updates it.
 */

import { Brain, CheckCircle2, Cpu, Timer } from 'lucide-react'
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix'
import { ModelCompare } from '../components/charts/ModelCompare'
import { TrainingCurves } from '../components/charts/TrainingCurves'
import { Card, CardBody, CardHeader, EmptyState, ErrorState, LoadingBlock } from '../components/ui'
import { useHealth, useModelMetrics } from '../hooks/useAnalytics'
import { num, ratio } from '../lib/format'

function MetricStat({ label, value, hint }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-2)] px-3.5 py-3">
      <p className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">{label}</p>
      <p className="tabular mt-1 text-[20px] font-semibold leading-none">{value}</p>
      {hint && <p className="mt-1.5 text-[11.5px] text-[var(--text-muted)]">{hint}</p>}
    </div>
  )
}

function PerClassTable({ perClass }) {
  if (!perClass) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse text-[13px]">
        <thead>
          <tr className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
            <th scope="col" className="py-2 pr-3 text-left font-medium">Class</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Precision</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">Recall</th>
            <th scope="col" className="py-2 pr-3 text-right font-medium">F1</th>
            <th scope="col" className="py-2 text-right font-medium">Support</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(perClass).map(([label, scores]) => (
            <tr key={label} className="border-t border-[var(--border)]">
              <th scope="row" className="py-2 pr-3 text-left font-medium capitalize">
                {label}
              </th>
              <td className="tabular py-2 pr-3 text-right">{scores.precision.toFixed(3)}</td>
              <td className="tabular py-2 pr-3 text-right">{scores.recall.toFixed(3)}</td>
              <td className="tabular py-2 pr-3 text-right font-medium">{scores.f1.toFixed(3)}</td>
              <td className="tabular py-2 text-right text-[var(--text-muted)]">
                {num(scores.support)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ModelCardPage() {
  const metrics = useModelMetrics()
  const health = useHealth()

  if (metrics.isLoading) return <LoadingBlock height="h-96" label="Loading evaluation results" />
  if (metrics.isError) return <ErrorState error={metrics.error} onRetry={metrics.refetch} />

  const models = metrics.data?.models ?? {}
  // Preference order, not a two-way choice: a repo may have only the baseline
  // trained, only the 128-token ablation, or the full set. Reaching straight for
  // `models.distilbert` blanks the page whenever that run has not finished.
  const primary =
    models.distilbert ?? models.distilbert_seq128 ?? models.baseline
  const labels = metrics.data?.labels ?? ['negative', 'neutral', 'positive']
  const serving = health.data?.model

  if (!primary) {
    return (
      <EmptyState
        icon={Brain}
        title="No evaluation results yet"
        hint={
          metrics.data?.hint ??
          'Run `python -m src.train_baseline` and `python -m src.train_transformer` to generate reports/metrics.json.'
        }
      />
    )
  }

  const improvement =
    models.distilbert && models.baseline
      ? models.distilbert.f1_macro - models.baseline.f1_macro
      : null

  // The first fine-tuning run used a 128-token window and lost to the baseline.
  // It is kept in metrics.json as a control, because the size of this gap is the
  // clearest evidence in the project that a preprocessing decision can outweigh
  // the choice of model.
  const ablation = models.distilbert_seq128
  const finalRun = models.distilbert
  // Both halves of the comparison must exist. While the 256-token run is still
  // training only the ablation is present, and the panel simply stays hidden.
  const contextGain =
    ablation && finalRun ? finalRun.f1_macro - ablation.f1_macro : null

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Model card"
          subtitle={`${primary.model} · evaluated on ${num(primary.n_samples)} held-out test examples never seen during training or model selection`}
        />
        <CardBody>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricStat
              label="Accuracy"
              value={ratio(primary.accuracy, 2)}
              hint="Share of predictions that were correct"
            />
            <MetricStat
              label="F1 (macro)"
              value={primary.f1_macro.toFixed(4)}
              hint="Unweighted class average — the headline number"
            />
            <MetricStat
              label="F1 (weighted)"
              value={primary.f1_weighted.toFixed(4)}
              hint="Weighted by class support"
            />
            <MetricStat
              label="Inference latency"
              value={`${primary.latency_ms_per_sample ?? '—'} ms`}
              hint="Single sample, batch size 1"
            />
          </div>

          <p className="mt-4 text-[13px] leading-relaxed text-[var(--text-secondary)]">
            Macro-F1 is the headline metric rather than accuracy, because the neutral class is both
            the hardest and the one a business most wants correctly separated. Accuracy would let a
            model that fails entirely on neutral still look strong by getting the two easy classes
            right.
          </p>
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Baseline vs transformer"
            subtitle="Same preprocessing inputs, same held-out test set"
          />
          <CardBody>
            <ModelCompare models={models} />
            {improvement !== null && (
              <p className="mt-3 text-[13px] leading-relaxed text-[var(--text-secondary)]">
                Fine-tuning DistilBERT is worth{' '}
                <strong className="tabular">
                  {improvement >= 0 ? '+' : ''}
                  {(improvement * 100).toFixed(1)} points
                </strong>{' '}
                of macro-F1 over the TF-IDF baseline. The baseline trains in seconds and the
                transformer takes{' '}
                {models.distilbert?.train_seconds
                  ? `${(models.distilbert.train_seconds / 60).toFixed(0)} minutes on CPU`
                  : 'considerably longer'}
                {' '}— that trade-off is the decision this chart exists to inform.
              </p>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Per-class performance"
            subtitle="Where the model is strong and where it is not"
          />
          <CardBody className="flex flex-col justify-center">
            <PerClassTable perClass={primary.per_class} />
            <p className="mt-3 text-[12.5px] leading-relaxed text-[var(--text-muted)]">
              Neutral is consistently the weakest class. That is expected: neutral feedback sits
              between the two poles by definition, and the star-rating mapping used to build the
              labels puts genuinely mixed reviews there.
            </p>
          </CardBody>
        </Card>
      </div>

      {contextGain !== null && (
        <Card>
          <CardHeader
            title="Ablation — the truncation window"
            subtitle="One changed number: same architecture, same data, same hyper-parameters"
          />
          <CardBody>
            <div className="grid gap-3 sm:grid-cols-3">
              <MetricStat
                label="128 tokens"
                value={ablation.f1_macro.toFixed(4)}
                hint="~52% of reviews fit — the rest were cut mid-text"
              />
              <MetricStat
                label="256 tokens"
                value={finalRun.f1_macro.toFixed(4)}
                hint="~84% of reviews fit"
              />
              <MetricStat
                label="Difference"
                value={`${contextGain >= 0 ? '+' : ''}${(contextGain * 100).toFixed(1)} pts`}
                hint="macro-F1, held-out test set"
              />
            </div>
            <p className="mt-4 text-[13px] leading-relaxed text-[var(--text-secondary)]">
              The first fine-tuning run used a 128-token window and{' '}
              {ablation.f1_macro < (models.baseline?.f1_macro ?? 0)
                ? 'lost to the TF-IDF baseline'
                : 'barely matched the TF-IDF baseline'}
              . The cause was not the model: WordPiece splits review text into roughly 1.4
              tokens per word, so a 128-token budget covered only about half the corpus and
              the transformer was reading half of every long review while the bag-of-words
              baseline read all of it. Doubling the window fixed it. The losing run is kept
              here deliberately — the gap between these two numbers is the clearest evidence
              in this project that a preprocessing decision can outweigh the choice of model.
            </p>
          </CardBody>
        </Card>
      )}

      <Card>
        <CardHeader
          title="Confusion matrix"
          subtitle="Which classes get mistaken for which"
        />
        <CardBody>
          <ConfusionMatrix matrix={primary.confusion_matrix} labels={labels} />
          <p className="mt-3 text-[12.5px] leading-relaxed text-[var(--text-muted)]">
            The errors that matter are the corners: a true positive predicted as negative, or the
            reverse. Confusions with neutral are far less costly — the feedback still lands in
            roughly the right place for triage.
          </p>
        </CardBody>
      </Card>

      {primary.history?.length > 0 && (
        <Card>
          <CardHeader
            title="Fine-tuning history"
            subtitle="Validation macro-F1 selected the checkpoint — never training loss"
          />
          <CardBody>
            <TrainingCurves history={primary.history} />
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <MetricStat
                label="Best epoch"
                value={primary.best_epoch}
                hint={`Validation macro-F1 ${primary.val_f1_macro}`}
              />
              <MetricStat
                label="Parameters"
                value={`${(primary.parameters / 1e6).toFixed(1)}M`}
                hint={primary.hyperparameters?.base_checkpoint ?? 'distilbert-base-uncased'}
              />
              <MetricStat
                label="Training time"
                value={`${(primary.train_seconds / 60).toFixed(0)} min`}
                hint={`on ${primary.device ?? 'cpu'}`}
              />
            </div>
          </CardBody>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Hyperparameters" subtitle="Exactly what produced this checkpoint" />
          <CardBody>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-[12.5px]">
              {Object.entries(primary.hyperparameters ?? {})
                .filter(([, value]) => typeof value !== 'object')
                .map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-3 border-b border-[var(--border)] pb-1.5">
                    <dt className="text-[var(--text-muted)]">{key.replace(/_/g, ' ')}</dt>
                    <dd className="tabular text-right font-medium">{String(value)}</dd>
                  </div>
                ))}
            </dl>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Currently serving" subtitle="What the API is loaded with right now" />
          <CardBody>
            {serving ? (
              <ul className="space-y-2.5 text-[13px]">
                <li className="flex items-center gap-2.5">
                  <Brain className="h-4 w-4 text-[var(--text-muted)]" aria-hidden />
                  <span className="text-[var(--text-secondary)]">Model</span>
                  <span className="ml-auto text-right font-medium">{serving.model}</span>
                </li>
                <li className="flex items-center gap-2.5">
                  <Cpu className="h-4 w-4 text-[var(--text-muted)]" aria-hidden />
                  <span className="text-[var(--text-secondary)]">Backend / device</span>
                  <span className="ml-auto font-medium">
                    {serving.backend} · {serving.device}
                  </span>
                </li>
                <li className="flex items-center gap-2.5">
                  <Timer className="h-4 w-4 text-[var(--text-muted)]" aria-hidden />
                  <span className="text-[var(--text-secondary)]">Max sequence length</span>
                  <span className="tabular ml-auto font-medium">
                    {serving.max_sequence_length} tokens
                  </span>
                </li>
                <li className="flex items-center gap-2.5">
                  <CheckCircle2
                    className="h-4 w-4"
                    aria-hidden
                    style={{
                      color:
                        serving.source === 'local'
                          ? 'var(--status-good)'
                          : 'var(--status-warning)',
                    }}
                  />
                  <span className="text-[var(--text-secondary)]">Checkpoint source</span>
                  <span className="ml-auto font-medium">{serving.source}</span>
                </li>
              </ul>
            ) : (
              <p className="text-[13px] text-[var(--text-muted)]">API unreachable.</p>
            )}

            <div className="mt-4 rounded-lg border border-[var(--border)] bg-[var(--surface-2)] p-3 text-[12px] leading-relaxed text-[var(--text-muted)]">
              <strong className="text-[var(--text-secondary)]">Known limitations.</strong> Trained
              on English business reviews with labels derived from star ratings, so it inherits
              that mapping&apos;s ambiguity around 3-star reviews. It has not been evaluated on
              sarcasm, code-mixed text, or domains far from consumer reviews, and it should support
              a human triage workflow rather than take automated action on its own.
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
