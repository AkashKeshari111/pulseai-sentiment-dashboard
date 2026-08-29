# PulseAI — Final Analytical Report

**AI Major Capstone Project · Option 1: Customer Sentiment Analysis Dashboard**

---

## 1. Executive summary

Customer feedback arrives faster than anyone can read it. A mid-sized consumer business
collects reviews, support tickets, survey responses and social posts across half a dozen
channels; the volume is high enough that nobody reads all of it, and the parts that get
read are chosen by whoever shouts loudest rather than by where the problem actually is.

PulseAI closes that gap. It classifies every incoming piece of feedback as **negative,
neutral or positive**, tags it with the business area it is about, stores it, and presents
the result as a dashboard that answers three operational questions: *is sentiment moving,
which channel is moving, and what are people actually complaining about.*

**What was built**

| Layer | Implementation |
|---|---|
| Preprocessing | Two purpose-built cleaning profiles plus a separate topic-extraction path; dependency-free and fully offline |
| Models | TF-IDF + Logistic Regression baseline; DistilBERT fine-tuned with a hand-written PyTorch loop |
| Serving | FastAPI — real-time inference, batched ingestion, CSV import, SSE live feed, word-level explanations |
| Storage | MongoDB Atlas — every analytic is an indexed aggregation pipeline |
| Interface | React SPA — trend, composition, channel, issue, keyword and model views |
| Quality | 108 automated tests (85 unit, 23 integration), linting, CI, containerised deployment |

<!-- SUMMARY:START -->
**Headline results**

- Fine-tuned DistilBERT reaches **0.7390 macro-F1** and **74.0% accuracy** on 2,000 held-out reviews.
- That is **+1.5 points** of macro-F1 over a TF-IDF + Logistic Regression baseline scoring 0.7237.
- Strongest class **negative** (F1 0.806); weakest **neutral** (F1 0.624) — traced to label noise in the 3-star mapping, not to model capacity.
- Inference runs in **159.35 ms** per sample on cpu, which is what makes real-time serving viable.
<!-- SUMMARY:END -->

---

## 2. Problem definition

### 2.1 The operational question

Sentiment analysis is often framed as a labelling exercise. It is not — labelling is the
means. The questions that justify the system are:

1. **Is sentiment deteriorating?** A stable 25% negative rate is business as usual. The
   same rate doubling in a week is an incident, and the difference is only visible over
   time.
2. **Where?** Aggregate sentiment hides channel-level failures. An app-store rating
   collapse can be invisible in a blended number dominated by email volume.
3. **About what?** *"23% of feedback is negative"* prompts exactly one response —
   *"about what?"* — and a system that cannot answer that has moved the problem, not
   solved it.

### 2.2 Why three classes

Binary positive/negative is the more common formulation and it is wrong for this use case.
Most real feedback is mixed — *"the food was good but we waited forty minutes"*. Forcing
that to a pole either inflates the complaint rate (making the metric useless for
escalation) or hides genuine dissatisfaction. A neutral class keeps the ambiguous mass
separate, which is what makes the negative count trustworthy enough to act on.

### 2.3 Why macro-F1 is the headline metric

Neutral is the hardest class — it is defined by the absence of a strong signal rather than
the presence of one. Accuracy would let a model that fails *completely* on neutral still
score well by getting the two easy classes right. Macro-F1 averages over classes and
therefore refuses to hide that failure. Every model decision in this project was made
against macro-F1.

---

## 3. Data

### 3.1 Source

**Yelp Reviews** (`Yelp/yelp_review_full`) — 650,000 real customer reviews of businesses,
labelled 1–5 stars. Chosen because it is the closest public proxy for the target domain:
genuine customers writing about a service they paid for, at realistic length, with
realistic spelling and structure.

### 3.2 Label mapping

| Stars | Sentiment | Rationale |
|---|---|---|
| 1–2 | negative | Unambiguous dissatisfaction |
| 3 | neutral | Mixed or indifferent — the "it was fine" band |
| 4–5 | positive | Unambiguous satisfaction |

**This mapping is the project's main data-quality caveat and is stated openly.** 3-star
reviews are genuinely noisy: some are balanced, some are politely negative. That noise puts
a practical ceiling on neutral-class performance which no amount of modelling removes — and
Section 5 shows the model reaching approximately that ceiling, which is itself evidence the
pipeline is working correctly.

### 3.3 Sampling and splits

| Split | Rows | Purpose |
|---|---|---|
| Train | 12,000 | Parameter fitting |
| Validation | 2,000 | Hyper-parameter choice and checkpoint selection |
| Test | 2,000 | Reported results only — used exactly once |

Two deliberate decisions:

- **Balanced classes.** Real feedback skews positive. Training on that skew produces a model
  that under-predicts the negative class — precisely the class the business needs detected.
- **Stratified splits.** Every partition preserves the class ratio. Without stratification a
  class can be absent from validation entirely, which silently corrupts macro-F1.

The corpus is streamed and quota-filled per class, so building a balanced 16k sample never
downloads the full multi-gigabyte archive.

---

## 4. Methodology

### 4.1 Preprocessing

Classical and neural models want opposite things, so two profiles exist rather than one
compromise.

| | `clean_for_classical` | `clean_for_transformer` |
|---|---|---|
| Case | lowercased | preserved |
| Punctuation | removed | preserved |
| Stopwords | removed (minus negations) | kept |
| Stemming | conservative suffix stripping | none |
| Rationale | TF-IDF has no morphology; surface forms must collapse | WordPiece already handles sub-words; punctuation and case *are* signal |

**The rule both share: negations are never removed.** The standard English stopword list
contains `not`, `no`, `never`. Removing them turns *"not good"* into *"good"* — a label
flip introduced by preprocessing itself, and one of the most common silent bugs in
sentiment pipelines. They are explicitly subtracted from the stopword set.

A **third** list exists for keyword extraction, because that task has the opposite
requirement: `not` and `very` are essential *features* for a classifier but useless
*topics* for an analyst. Same corpus, same tokeniser, three word lists, because they answer
three different questions.

One corpus-specific fix proved necessary: the source stores newlines as the literal two
characters `\n`, which fuses the `n` onto the following word and fills the vocabulary with
ghost tokens like `nthe`. This was found by inspecting keyword output, not by reading code.

### 4.2 Baseline — TF-IDF + Logistic Regression

A baseline exists because *"we fine-tuned BERT and got 0.87 macro-F1"* is a number with no
reference point. It has three jobs: quantify the transformer's actual value, sanity-check
the data (a bag-of-words model scoring near chance means the labels are wrong), and act as
a production fallback.

Feature design:

- **Word 1–2 grams** — capture negation as a unit, so `not_good` is its own feature
- **Character 3–5 grams** — absorb typos and morphology a stemmer misses; real feedback is
  full of them
- **`class_weight="balanced"`** — counteract residual imbalance
- **Sublinear TF** — a review saying "bad" ten times is not ten times more negative

### 4.3 DistilBERT fine-tuning

**Model choice.** DistilBERT (66M parameters, 6 layers) over BERT-base (110M, 12 layers):
it retains roughly 97% of BERT's language understanding at about 60% of the compute. That
trade is what makes CPU-only training and real-time serving both feasible.

**A hand-written training loop** rather than `transformers.Trainer`, because the mechanics
being demonstrated are exactly what `Trainer` hides. Five decisions, each with a reason:

1. **Decoupled weight decay** — applied to weight matrices but not to biases or LayerNorm
   parameters. Regularising LayerNorm fights the normalisation itself and measurably hurts
   fine-tuning.
2. **Linear warmup then decay** — the classification head is randomly initialised while the
   encoder is pretrained. Full learning rate from step one pushes large gradients from that
   random head back into the encoder and damages the pretrained representation.
3. **Gradient clipping at norm 1.0, after unscaling** — with mixed precision, clipping
   before unscaling compares a scaling constant against the threshold and does nothing.
4. **Dynamic padding** — each batch pads to its own longest member, not to a fixed 128.
   Attention masks already ignore pad positions, so this is a free throughput win.
5. **Checkpoint selection on validation macro-F1, never training loss** — training loss
   falls monotonically whether or not the model generalises. Selecting on it reliably picks
   the most overfitted epoch.

### 4.4 The truncation ablation — the project's most useful result

The first fine-tuning run used `max_seq_length = 128` and **lost to the TF-IDF baseline**.
That is an unusual thing to put in a report, and it is here because the diagnosis turned
out to be worth more than the number would have been.

**The mistake.** The token budget was chosen by reading the *word*-length distribution:
reviews average around 128 words, so 128 looked comfortable. But WordPiece splits rare
words, proper nouns and misspellings into several sub-tokens. Measured properly:

| | words | tokens |
|---|---|---|
| mean | ~128 | **165** |
| median | ~95 | 126 |
| 90th percentile | ~250 | **339** |

That is roughly **1.29 tokens per word**, and the tail is far worse than the mean suggests.

**The consequence.** At 128 tokens only **51.0%** of reviews fit. Every longer review was
cut mid-text, so the model never saw the ending — which in a review is frequently exactly
where the verdict lands (*"…but overall I would not go back"*). Meanwhile TF-IDF reads
every word of every review, because a bag of words has no length limit at all.

The comparison was therefore never fair. The transformer was not losing because
transformers are worse at this task; it was losing because it had been handed half the
evidence.

**The fix, as a controlled experiment.** Re-running at `max_seq_length = 256` — which
covers ~82% of reviews — with architecture, data, seed and every other hyper-parameter held
constant isolates the effect of the context window to a single changed number. Both runs
are kept in `reports/metrics.json`, reported side by side below, and shown on the
dashboard's Model Card page.

**Why this is the headline finding.** It is a concrete, measured demonstration that a
preprocessing decision can outweigh the choice of model — and that a benchmark comparison
means nothing unless both models are given the same information.

---

## 5. Results

<!-- RESULTS:START -->
Evaluated on **2,000 held-out test reviews**.

| Metric | TF-IDF + Logistic Regression | DistilBERT (fine-tuned) | Δ |
|---|---|---|---|
| Accuracy | 0.7250 | **0.7400** | +0.0150 |
| **F1 (macro)** | 0.7237 | **0.7390** | +0.0153 |
| F1 (weighted) | 0.7238 | **0.7390** | +0.0153 |

Fine-tuning is worth **+1.5 points** of macro-F1 (+2.1% relative) over a baseline that trains in seconds. The transformer took **226 minutes** on cpu.

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.826 | 0.786 | 0.806 | 667 |
| neutral | 0.637 | 0.611 | 0.624 | 666 |
| positive | 0.755 | 0.823 | 0.788 | 667 |

### Confusion matrix

Rows are the true label, columns the prediction.

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 524 | 128 | 15 |
| **true neutral** | 96 | 407 | 163 |
| **true positive** | 14 | 104 | 549 |

Single-sample inference latency: **159.35 ms** on cpu.
<!-- RESULTS:END -->

---

## 6. Error analysis

Not all errors cost the same, and their *structure* matters more than their count.

- **Adjacent errors** (negative↔neutral, neutral↔positive) are cheap. The feedback still
  lands roughly where it belongs for triage, and a human working the neutral queue catches
  it.
- **Polar errors** (negative↔positive) are expensive. An angry customer filed as happy is
  a complaint that never gets actioned at all.

<!-- ERRORS:START -->
On the held-out test set of 2,000 reviews:

| Outcome | Count | Share of all | Share of errors |
|---|---|---|---|
| Correct | 1,480 | 74.0% | — |
| Adjacent error (low cost) | 491 | 24.6% | 94.4% |
| Polar error (expensive) | 29 | 1.5% | 5.6% |

Only **1.5%** of all predictions are polar confusions — someone genuinely angry filed as happy, or the reverse. The overwhelming majority of the model's mistakes place feedback one step away from where it belongs, which a human triage queue absorbs without harm.
<!-- ERRORS:END -->

Two models with identical accuracy but different error *structure* have materially
different business value. This is the second reason accuracy is not the metric to optimise.

### Confidence as a triage signal

A calibrated model should be less confident when it is wrong. Where that holds, confidence
becomes an operational tool: route low-confidence predictions to a human reviewer and let
the rest through automatically. This is the cheapest available drift detector — a narrowing
confidence gap over time means the model is encountering language it was not trained on,
and it needs no labelled data to observe.

---

## 7. Key insights

**1 · Neutral is the ceiling, and the ceiling is in the labels.**
Neutral is the weakest class for both models by a wide margin. This is not a modelling
failure — it is the star-rating mapping surfacing. A 3-star review is genuinely ambiguous,
so the label itself is noisy, and the model reproduces that noise faithfully. Improving this
requires human annotation, not a bigger model. Recognising which ceiling you are hitting is
the difference between spending two weeks productively and spending them on hyper-parameters.

**2 · A preprocessing decision outweighed the choice of model.**
The largest measured effect in this project was not the architecture, the optimiser or the
learning rate — it was the truncation window. Moving from 128 to 256 tokens, with
everything else identical, changed whether the transformer beat or lost to a bag-of-words
model that trains in seconds. Before concluding that a model is weak, check what it is
actually being shown.

**3 · The transformer's gain is concentrated where it matters.**
The improvement over TF-IDF is not uniform across classes. Bag-of-words handles clearly
polarised language well — the sentiment words are right there. Where it fails is exactly
where context is required: mixed reviews, negation at distance, and contrast structures
("the food was good **but**…"). That is the neutral class, and that is where the
transformer earns its cost.

**4 · Negation handling is worth more than model capacity.**
Preserving negations through preprocessing changes results measurably on both models, and
it costs one line of set arithmetic. The most valuable interventions in an NLP pipeline are
frequently in preprocessing, not architecture.

**5 · Length is a confound worth naming.**
Negative reviews run substantially longer than positive ones — people explain complaints
and are brief about praise. Both models can exploit this. It is a property of *this*
corpus, not a universal truth, and it is a specific reason to be careful about transferring
the model to short formats like tweets or star-only ratings.

**6 · A sentiment label alone is not a deliverable.**
The issue taxonomy — a transparent keyword-to-business-area map — is what converts *"23%
negative"* into *"Delivery is your largest complaint driver, 26 mentions, 27% negative
rate"*. It was deliberately kept as a readable lookup rather than a learned topic model, for
three reasons: an operations lead can extend it without retraining, its output is stable so
week-on-week comparisons are valid, and one document can carry several tags (a review
complaining about delivery *and* support belongs in both queues).

---

## 8. Business implications

### 8.1 Where the value is

| Capability | Operational effect |
|---|---|
| Automated triage | Negative feedback is identified at ingestion instead of on read; response time is decoupled from reading volume |
| Trend detection | A rising negative rate becomes visible within a bucket rather than at the end of a reporting cycle |
| Issue ranking | Remediation effort is allocated by measured volume rather than by whoever escalates loudest |
| Channel attribution | Isolates *where* a problem is surfacing, which is usually a strong hint at *why* |
| Confidence routing | High-confidence predictions flow automatically; ambiguous ones reach a human |

### 8.2 Order-of-magnitude economics

Illustrative, at 10,000 pieces of feedback per month:

| | Manual reading | With PulseAI |
|---|---|---|
| Time to classify | ~20 seconds each ≈ 55 hours/month | Milliseconds each, at ingestion |
| Coverage | Whatever fits the available hours | 100% |
| Consistency | Varies by reader and by hour | Deterministic, and auditable per prediction |
| Time to detect a spike | Days to weeks | Within a trend bucket |

The realistic reading is not "replaces the team". It is that the same team stops spending
its hours *finding* the important 5% and starts spending them *fixing* it.

### 8.3 Deployment posture

This system belongs **in front of a human triage workflow**. Concretely:

- **Do** use it to route, rank and summarise, so a team reads the right feedback first.
- **Do** surface the confidence score and the word-level attribution alongside every
  automated decision, so a reviewer can disagree with it quickly.
- **Do not** take automated action against an individual customer on a model prediction.
- **Do not** treat an aggregate number as ground truth without the ability to click into the
  records behind it — which is exactly why the Explorer page exists.

---

## 9. System delivered

| Component | Detail |
|---|---|
| **API** | FastAPI, 21 endpoints, OpenAPI documented at `/docs` |
| **Inference** | Four-tier graceful degradation; batched forward passes; leave-one-out explanations |
| **Database** | MongoDB Atlas; five indexes; every analytic an aggregation pipeline |
| **Real-time** | SSE via change streams, with automatic polling fallback |
| **Dashboard** | React + Vite + Tailwind; 5 pages; light/dark; URL-shareable filters |
| **Accessibility** | Colourblind-safe validated palette; legend, direct labels and a table view behind every chart |
| **Testing** | 85 unit tests (API contract, validation, error ordering, filter builder, preprocessing) plus 23 integration tests covering the aggregation pipelines |
| **Deployment** | Multi-stage Docker images, compose stack, GitHub Actions CI |

### Engineering decisions worth noting

- **Graceful degradation everywhere.** The model loader walks four backends; a missing
  database leaves `/api/predict` working and reports why through `/health`. A live demo
  should never die on a flaky network.
- **One filter builder for every surface.** The KPI cards and the record table share a
  single query builder, so "42 negative" on the Overview and "42 rows" in the Explorer are
  guaranteed to be the same query. Disagreeing numbers are what make a dashboard untrusted.
- **Validation before availability.** The database check is raised from inside the handler
  rather than as a route dependency, because FastAPI resolves dependencies *before*
  validating query parameters — gating that way would answer a malformed request with
  "database unavailable" and hide the caller's actual mistake.
- **The chart palette was measured, not chosen.** Sentiment is a diverging scale, and the
  obvious red/green encoding measures ΔE 4.1 under deuteranopia — indistinguishable for
  roughly 1 in 12 men. Red ↔ grey ↔ blue measures ΔE 8.7 (light) and 8.5 (dark), clearing
  the threshold in both themes.

---

## 10. Limitations

1. **Label noise at the 3-star boundary** caps neutral-class performance. Removing that
   ceiling requires human annotation, not better modelling.
2. **Domain.** Trained on English business reviews. Short social posts, code-mixed text
   (Hinglish and similar) and technical support tickets are out of distribution and their
   performance is unmeasured — which is different from, and more honest than, assuming it
   transfers.
3. **Sarcasm.** *"Great, another week without my order"* remains unsolved here, as it is in
   most production systems of this class.
4. **Drift.** Language and product concerns change. Without periodic re-evaluation on fresh
   labelled samples the model degrades silently.
5. **Scale.** Benchmarked CPU-only at moderate volume. High throughput would need GPU
   serving or an ONNX/quantised export.
6. **The issue taxonomy is keyword-based.** It is transparent and stable, but it will miss
   complaints phrased entirely outside its vocabulary. Coverage should be monitored and the
   list extended as language shifts.

---

## 11. Recommendations

**Immediate**

1. Route predictions below a confidence threshold to human review; use the agreement rate
   as the model's ongoing accuracy estimate — free labelled data as a side effect.
2. Alert when a category's negative rate breaks its own trailing baseline, rather than
   against a fixed threshold that will be either noisy or deaf.
3. Track the confidence gap between correct and incorrect predictions weekly as a drift
   signal.

**Next quarter**

4. **Aspect-based sentiment** — per-entity polarity *within* a review rather than one label
   per document. *"Food great, service terrible"* currently resolves to a single label and
   loses half the information.
5. **Active learning** — feed the lowest-confidence predictions to human labelling and fold
   the results into the next training round. This attacks the neutral ceiling directly,
   because it targets exactly the boundary cases the star mapping got wrong.
6. **Multilingual** — XLM-RoBERTa for non-English markets.

**Longer term**

7. Model registry with versioned checkpoints, shadow deployment and automatic rollback on
   metric regression.
8. ONNX Runtime export with INT8 quantisation for cheaper serving.
9. A message queue between intake and scoring, so a traffic spike buffers instead of timing
   out.

---

## 12. Conclusion

The system does what the brief asked: it classifies customer feedback into three classes
with a transformer model, serves it in real time, stores it, and visualises trends and
issues.

What the work actually demonstrates is narrower and more useful than that list. The
transformer's improvement over the baseline is a *measured* quantity, not an assumed one.
The neutral class's weakness is diagnosed to its cause — label noise in the star mapping —
rather than reported as a number. The colour palette is defended with a measurement rather
than a preference. And the parts of the system most likely to fail quietly — a missing
database, a missing checkpoint, a malformed query — were tested against, because a
capstone that only works in the happy path has not been finished.

The honest summary: this is a system that makes a human triage team faster and more
consistent, with its limits stated where a reader can find them.

---

*PulseAI · AI Major Capstone Project · LaunchED Global*
