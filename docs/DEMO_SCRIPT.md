# Project demonstration video — script and shot list

A 5–10 minute walkthrough of PulseAI, structured to the LaunchED submission checklist.

**Before you record**

```bash
python -m api.seed --count 600 --reset      # fresh, realistic data
uvicorn api.main:app --reload --port 8020   # terminal 1
cd dashboard && npm run dev                 # terminal 2
```

Then open these tabs in order so you never hunt for one on camera:

1. `http://localhost:5173` — the dashboard
2. `http://localhost:8020/docs` — the API documentation
3. VS Code with `src/train_transformer.py` open
4. `notebooks/01_sentiment_analysis_capstone.ipynb`

Record at 1080p or higher, hide bookmarks and notifications, and close anything
personal. Two takes of a good five minutes beat one rambling nine.

---

## 0:00 – 0:45 · Introduction *(English — mandatory)*

> "Hello, my name is **[FULL NAME]**, currently pursuing **[COURSE & BRANCH]** at
> **[COLLEGE NAME]**. I'm in my **[YEAR]** year, and I've completed my internship with
> **LaunchED** in the **Artificial Intelligence** domain.
>
> During the internship I worked through the machine learning and NLP track, and this
> capstone is where I applied all of it end to end.
>
> In this video I'll be demonstrating my capstone project, **PulseAI — a Customer
> Sentiment Intelligence Platform**: an AI system that automatically classifies customer
> feedback as positive, negative or neutral using a fine-tuned transformer model, and
> visualises sentiment trends and complaint drivers on a live dashboard."

*On screen: the dashboard Overview page, or a title slide.*

> *(Everything after this point may be in English or Hindi — whichever you're more fluent
> in. Explain it as if to a technical panel that has never seen the project.)*

---

## 0:45 – 1:45 · Problem statement

*On screen: the Overview page.*

Points to make, in your own words:

- A business collects feedback across many channels — app reviews, support email, surveys,
  social. **Nobody reads all of it.**
- The three questions that go unanswered: *is sentiment getting worse? on which channel?
  and about what?*
- **Why three classes and not two.** Most real feedback is mixed: *"the food was good but
  we waited forty minutes."* Forcing that to positive or negative either inflates the
  complaint rate or hides it. Keeping neutral separate is what makes the negative count
  trustworthy enough to escalate on.
- **Why macro-F1 and not accuracy.** Neutral is the hardest class. Accuracy would let a
  model that fails completely on neutral still look good by getting the two easy classes
  right.

> This is the single strongest minute in the video. It shows you chose the problem
> framing deliberately rather than following a tutorial.

---

## 1:45 – 2:45 · Tools, technologies and architecture

*On screen: the architecture diagram in `README.md`.*

| Layer | Say this |
|---|---|
| Model | DistilBERT fine-tuned in PyTorch; TF-IDF + Logistic Regression as the baseline |
| Backend | FastAPI, 21 endpoints, async |
| Database | MongoDB Atlas — every analytic is an aggregation pipeline |
| Frontend | React, Vite, Tailwind, Recharts |
| Real-time | Server-Sent Events over MongoDB change streams |
| Quality | 85 pytest tests, ruff, GitHub Actions CI, Docker |

Then trace one request out loud:

> "A piece of feedback arrives at `POST /api/feedback`. It's validated, cleaned,
> tokenised, and passed through DistilBERT. The prediction is enriched with issue
> categories, stored in Atlas with its full score vector — so it stays auditable after a
> model upgrade — and a MongoDB change stream pushes it to the dashboard's live feed
> within about a second."

**Why DistilBERT rather than BERT-base:** 40% fewer parameters, about 97% of the language
understanding, which is what makes CPU training and real-time inference both realistic.

---

## 2:45 – 4:15 · Live demonstration

This is what the panel remembers. Move briskly and keep talking.

### Overview page (~40s)
- Four KPI tiles: volume, Net Sentiment Score, negative share, model confidence.
- **Explain NSS**: percentage positive minus percentage negative, −100 to +100.
- Point at the trend chart: *"direction matters more than level — a steady negative rate
  is business as usual, a rising one is an incident."*
- Toggle **Volume → Net score**. Say why it's a toggle and not two y-axes: *"two different
  scales in one frame make the crossing point look meaningful when it isn't."*
- Change the period filter and let the whole page update together.

### Analyze page (~40s) — **the moment that lands**
- Paste: *"The delivery was three days late and the box arrived damaged. Support never
  replied to my emails."*
- Click **Analyse**. Show the label, the confidence, and the class probability bars.
- Point at **Why this prediction** — the highlighted words.

  > "This is leave-one-out attribution: each word is removed in turn, and the drop in the
  > predicted class probability is that word's contribution. A dashboard that just says
  > 'negative, 96% confident' won't be trusted by an operations team — this shows them
  > which words drove the decision."

- Click **Save to database**, go back to **Overview**, and show it appear in the live feed.
  *That single move demonstrates the model, the API, the database and the real-time
  pipeline in one gesture.*

### Insights page (~20s)
- Read the headline sentence aloud.
- Explain the issue table: *"ranked by absolute negative volume, not by rate — a category
  that's 100% negative across two mentions is noise; 55% of four hundred is a real
  problem."*
- Click a category and show the whole dashboard filter to it.

### Explorer page (~15s)
- *"Every aggregate on the other pages is a claim; this is where you check it against the
  actual records."*
- Expand a row to show the stored score vector and the model that produced it.

### Model card page (~15s)
- Baseline vs DistilBERT comparison, confusion matrix, training curves.
- *"These numbers are read live from the training artefacts — nothing on this page is
  hard-coded."*

### API docs (~10s)
- Switch to `localhost:8020/docs` and run `POST /api/predict` from the Swagger UI.

---

## 4:15 – 5:30 · Technical depth

*On screen: `src/train_transformer.py`.*

Pick **three** of these — depth on three beats a list of eight:

1. **A hand-written training loop, not `Trainer`.** Warmup schedule, decoupled weight decay,
   gradient clipping, best-checkpoint selection. *"`Trainer` hides exactly the mechanics I
   wanted to demonstrate."*
2. **Checkpoint selection on validation macro-F1, never training loss.** Training loss falls
   whether or not the model generalises; selecting on it picks the most overfitted epoch.
3. **Weight decay is not applied to LayerNorm or biases.** Those parameters calibrate
   activation scale — regularising them fights the normalisation itself.
4. **Negations are never removed in preprocessing.** The standard stopword list contains
   "not" and "never"; removing them turns *"not good"* into *"good"*. One line of set
   arithmetic, and it changes results on both models.
5. **Dynamic padding.** Each batch pads to its own longest member rather than to a fixed
   128 — a free throughput win.

---

## 5:30 – 6:30 · Challenges faced and how you solved them

Real problems from this build. Pick two or three and tell them as short stories — a
challenge with a diagnosis is far more convincing than a challenge with a workaround.

**1 · A silent train/serve mismatch.**
The corpus stores newlines as the literal characters `\n`, which fused onto the next word
and produced vocabulary ghosts like `nthe`. I found it by *looking at the keyword output*,
not by reading code. Fixing it meant discarding a training run that was already 40% done
and restarting — because a model trained on text the API would never see again is a bug
that hides until production.

**2 · Error ordering in the API.**
FastAPI resolves route dependencies *before* it validates query parameters. With the
database check as a dependency, a request with a malformed `days=0` came back "database
unavailable" — hiding the caller's real mistake behind an unrelated one. Moving the check
inside the handler put validation first, so bad input gets a 422 and only valid requests
ever see a 503.

**3 · Colour that would have excluded readers.**
Red-for-negative and green-for-positive is the obvious choice. Measured, that pair sits at
ΔE 4.1 under deuteranopia — indistinguishable for roughly 1 in 12 men. I tested candidate
palettes and switched to red ↔ grey ↔ blue, which measures 8.7. Every chart also carries a
legend, direct labels and a table view, so meaning never depends on colour alone.

**4 · CPU-only training.**
No GPU available. Handled with DistilBERT over BERT-base, dynamic padding, a 128-token
budget justified by the length distribution, and a balanced 12k subsample — a two-hour run
instead of an impossible one.

---

## 6:30 – 7:30 · Results and conclusion

*On screen: the Model card page, or the results table in `README.md`.*

- Quote the real numbers: baseline macro-F1, DistilBERT macro-F1, and the gain.
- **Be honest about the weakest class.** Neutral is hardest, and say *why*: the labels come
  from a star mapping and 3-star reviews are genuinely ambiguous. *"That's a ceiling in the
  labels, not in the model — fixing it needs human annotation, not a bigger network."*

  > Panels notice when a candidate can diagnose their own weakest result instead of
  > glossing over it. This is worth more than a higher score would be.

- **Error structure**: most mistakes are adjacent (neutral↔negative), very few are polar
  (positive↔negative). Adjacent errors still land in roughly the right triage queue.
- **Where it should be used**: in front of a human triage workflow — routing and ranking so
  a team reads the right 5% first. Not for automated action on an individual customer.
- **What's next**: aspect-based sentiment, active learning on low-confidence predictions,
  multilingual support.

Close:

> "That's PulseAI — an end-to-end customer sentiment intelligence platform, from data
> preparation and model training through to a deployed API and a live dashboard. Thank you
> for watching."

---

## Delivery checklist

- [ ] Introduction recorded **in English** with full name, college, course, year, domain,
      LaunchED mention, project title and one-line overview
- [ ] 5–10 minutes total
- [ ] Quiet room, clear audio, no notifications on screen
- [ ] The live demo actually runs — API and dashboard both started beforehand
- [ ] Real numbers quoted, not placeholders
- [ ] Saved as **MP4**
- [ ] Uploaded to Google Drive with "Anyone with the link can view"
- [ ] Posted to LinkedIn tagging **LaunchED Global** and your HOD

## Things that quietly cost marks

- Reading this script word for word. Use it as a route, not a transcript.
- Silence while something loads — start every service *before* you hit record.
- Scrolling code without saying why it matters.
- Claiming the model is perfect. Stating its limits is what makes the rest credible.
