# Learning path — how to actually own this project

You are going to be asked to explain this. Not to recite it — to explain *why* each
decision was made, and what would happen if it had been made differently. This document
is the route from "the code exists" to "I can defend every line of it."

**The honest bar:** if someone points at any file and asks *"why is this here and what
breaks without it?"*, you should have an answer. That is a real bar, and it is reachable
in about a week of focused evenings.

---

## How to use this

Work through the six stages **in order**. Each stage has three parts:

| Part | What it means |
|---|---|
| **Read** | Open the file and read the comments, not just the code. The comments explain the *reasoning*. |
| **Run** | Execute it yourself and watch the output. Reading code you have never run is not learning. |
| **Break it** | Change one thing, re-run, and watch what fails. **This is the part that actually teaches.** |

Do not skip "Break it". Understanding comes from seeing what a decision *prevents*, and
the only way to see that is to undo it and watch the damage.

Keep a notebook (paper is fine). For every stage write down, in your own words:
*what this does, why it is done this way, what breaks without it.* If you cannot write
those three sentences, you have not finished that stage.

---

## Stage 1 · The problem and the data (2 hours)

**Read**
- `README.md` — the top section, and design decisions 1–4
- `src/config.py` — the whole file. It is short, and it is where every other file gets
  its settings from.
- `src/dataset.py` — focus on `_yelp_stars_to_sentiment` and `stratified_split`

**Run**
```bash
python -m src.dataset --prepare --synthetic --train-size 300 --val-size 60 --test-size 60
```
Then open `data/processed/train.csv` in Excel and just *look at it*. Real text, real
labels. Get a feel for what the model is being asked to do.

**Break it**

In `src/dataset.py`, find `stratified_split` and delete `stratify=frame["label"]` from the
first `train_test_split` call. Re-run the prepare command. Now count the classes:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/processed/val.csv').label_name.value_counts())"
```

The classes will no longer be evenly split. Now imagine that happening with a rare class —
it can vanish from validation entirely, and macro-F1 silently averages over a class with
no data. **Put the line back.**

**Answer in your notebook**
1. Why are 3-star reviews labelled *neutral* and not dropped entirely?
2. Why does the project deliberately sample a *balanced* corpus when real feedback is
   mostly positive?
3. What does "stratified" mean and what breaks without it?

---

## Stage 2 · Preprocessing — the highest-value hour in this project (2 hours)

This is where the project's most quotable decisions live. If your time is limited, spend
it here.

**Read**
- `src/preprocessing.py` — the whole file. It is the most commented file in the repo.
- Pay special attention to `_SENTIMENT_CRITICAL`, `KEYWORD_STOPWORDS`, and `_ESCAPE_RE`.

**Run**
```python
python
>>> from src.preprocessing import clean_for_classical, clean_for_transformer
>>> clean_for_classical("I don't think the DELIVERY was GOOOOOD!!!")
>>> clean_for_transformer("I don't think the DELIVERY was GOOOOOD!!!")
```
Try ten of your own sentences. Watch how the two profiles differ.

**Break it — the important one**

In `src/preprocessing.py`, change:
```python
STOPWORDS = _BASE_STOPWORDS - _SENTIMENT_CRITICAL
```
to:
```python
STOPWORDS = _BASE_STOPWORDS          # the "normal" way everyone does it
```
Then:
```python
>>> clean_for_classical("This is not good at all")
```
It now returns `"good"`. The preprocessing has **flipped the label** before the model ever
sees the text. Retrain the baseline (`python -m src.train_baseline`) and watch the score
drop.

**Put it back.** You now understand, from evidence rather than from being told, why one
line of set arithmetic matters more than most model tuning.

**Answer in your notebook**
1. Why does the classical profile strip punctuation but the transformer profile keep it?
2. Why are there *three* different word lists rather than one?
3. What is `_ESCAPE_RE` fixing, and how was that bug found?

---

## Stage 3 · The two models (3 hours)

**Read**
- `src/train_baseline.py` — start with `build_pipeline`
- `src/train_transformer.py` — read `train()` slowly. Every non-obvious line has a comment
  above it saying why.
- `src/metrics.py` — the docstring on `compute_metrics` explains the macro-F1 choice

**Run**
```bash
python -m src.train_baseline                        # ~2 minutes
python -m src.train_transformer --limit 64 --epochs 1 --output-dir models/_practice
```
The second is a deliberately tiny run. It will produce a bad model — that is fine. The
point is to watch the loop print steps, learning rate, and validation score so the
mechanics stop being abstract.

**Break it**

In `src/train_transformer.py`, find the checkpoint-selection block:
```python
if val_metrics["f1_macro"] > best_f1:
```
Change it to select on training loss instead:
```python
if epoch_record["train_loss"] < best_f1:   # deliberately wrong
```
Run 2 epochs. Training loss always falls, so this always picks the last epoch — even when
the model has started overfitting. **Put it back**, and you can now explain checkpoint
selection from experience.

**Answer in your notebook**
1. Why DistilBERT and not BERT-base? Give the numbers.
2. Why is weight decay *not* applied to LayerNorm and bias parameters?
3. What is warmup, and what goes wrong without it?
4. Why is macro-F1 the headline metric instead of accuracy?
5. What is dynamic padding and why is it free performance?

---

## Stage 4 · The headline finding — reproduce it yourself (1 hour)

This is the part you will be asked about most, so do not take it on trust. Measure it.

**Run**
```python
python
>>> import pandas as pd, numpy as np
>>> from transformers import AutoTokenizer
>>> from src.preprocessing import clean_for_transformer
>>> tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
>>> df = pd.read_csv("data/processed/test.csv").head(500)
>>> lens = np.array([len(tok(clean_for_transformer(t))["input_ids"]) for t in df.text])
>>> print("mean tokens:", lens.mean(), " mean words:", df.text.str.split().str.len().mean())
>>> for cap in (128, 256): print(cap, f"{(lens <= cap).mean():.1%} of reviews fit")
```

You will see it with your own eyes: ~1.29 tokens per word, and 128 tokens covering only
just over half the corpus. Then open the **Model card** page in the dashboard and look at the
two runs side by side.

**Answer in your notebook**
1. Why did the first BERT run lose to a bag-of-words model?
2. Why is comparing them at 128 tokens an *unfair* comparison?
3. Why is changing only one variable called a controlled experiment, and why does that
   matter?

> This single story — hypothesis, measurement, diagnosis, fix, re-measurement — is worth
> more in an interview than any accuracy number. Make sure you can tell it without notes.

---

## Stage 5 · The API (3 hours)

**Read**
- `api/main.py` — small, and it shows how everything is wired together
- `api/inference.py` — the four-tier loader and `predict_batch`
- `api/db.py` — pick **one** aggregation pipeline (`summary` is the easiest) and trace it
  line by line
- `api/routers/predict.py` — short, and shows the request path end to end

**Run**
```bash
uvicorn api.main:app --reload --port 8020
```
Open <http://localhost:8020/docs> and click **Try it out** on `POST /api/predict`. Send
your own sentences. This page is auto-generated from the Pydantic models — the same
definitions that validate incoming requests.

**Break it**
1. Stop the API and open the dashboard. Watch the status pill turn red and the pages show
   a clear error instead of a blank screen. That is the graceful-degradation design.
2. Comment out `MONGODB_URI` in `.env` and restart. `/api/predict` still works;
   `/api/analytics/summary` returns a 503 that tells you exactly what to fix.

**Answer in your notebook**
1. What are the four model backends and why does a fallback chain exist at all?
2. Why does the forward pass run in a worker thread rather than directly?
3. Why is the database check inside the handler instead of a route dependency?
   (Hint: read the comment above `database_unavailable_handler` in `api/main.py`.)

---

## Stage 6 · The dashboard (3 hours)

**Read**
- `dashboard/src/index.css` — the design-token block at the top, and the comment
  explaining the colour choice
- `dashboard/src/lib/filters.jsx` — why filter state lives in the URL
- `dashboard/src/pages/Overview.jsx` — how a page composes hooks and charts
- `dashboard/src/components/charts/TrendChart.jsx` — the two-view toggle and why it is not
  a dual axis

**Run**
```bash
cd dashboard && npm run dev
```
Open the browser dev tools **Network** tab and click around. Watch the actual API calls
fire. Then change a filter and watch which requests re-fire and which come from cache.

**Break it**

In `dashboard/src/index.css`, change `--sentiment-positive` to a green
(`#0ca30c`). The dashboard now uses the "obvious" red/green. Then look up a
colourblindness simulator online, paste a screenshot in, and see the two become
indistinguishable. **Put it back.**

**Answer in your notebook**
1. Why is sentiment a *diverging* colour scale rather than three arbitrary colours?
2. Why does the trend chart switch between views instead of drawing two y-axes?
3. Why does every chart have a "Table" button?

---

## The final test — can you actually explain it?

Do this **before** you record the video.

1. **Close the laptop.** Explain the whole project out loud for five minutes to a wall, a
   friend, or your phone's recorder. No notes.
2. **Play it back.** Every place you hesitated is a place you do not understand yet. Go
   back to that stage.
3. **Answer these five cold:**
   - Why three classes instead of two?
   - Why macro-F1 instead of accuracy?
   - Why did your first transformer lose to the baseline, and how did you find out?
   - Which class is weakest and *why* — is it the model's fault?
   - Where should this system *not* be used?

If those five come out fluently, you are ready. They are also the five most likely
questions.

---

## What to do if you are short on time

Priority order, highest value first:

1. **Stage 4** — the truncation finding. This is your differentiator.
2. **Stage 2** — preprocessing, especially the negation decision.
3. **Stage 3, questions only** — you must know why DistilBERT and why macro-F1.
4. Stages 1, 5, 6 — good to have, less likely to be probed deeply.

---

## Honest note

An AI assistant wrote most of this code with you. That is normal, and it is how a large
amount of professional software gets written now. It stops being a problem the moment you
can explain, defend and modify it — and it stays a problem if you cannot.

The difference between those two states is roughly one week of the work above. It is
worth doing, and not only for the certificate: the truncation story in Stage 4 is a
genuinely good engineering story, and it will only be *yours* to tell once you have
measured it yourself.
