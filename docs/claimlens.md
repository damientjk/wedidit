# ClaimLens: watch the model read a claim

Demo UI (owner: Da Feng). A claims officer uploads a claim PDF. The reader goes
through it sentence by sentence, left to right, and every fact it finds
re-scores the claim with two pre-trained models. When it finishes, the officer
can step back through the exact sentences that moved the score.

## Two modes (tabs in the header)

| Mode | Model | What it does |
|---|---|---|
| **Claim form** (default) | Marcus's final XGBoost (`POST /score`, `src/explain.py`) | 19 Kaggle fields as dropdowns (options from `GET /model/final`). Three real test-set claims as examples. Shows the risk **ranking score** (uncalibrated, never a %), the tier, the recommendation, and a TreeSHAP waterfall in log-odds: base value + contributions = log-odds, then score = sigmoid(log-odds). The ‹ › trail steps through the drivers, and clicking a field's number switches it off for a what-if. |
| **PDF walkthrough** (`#pdf`) | Demo model on synthetic data (`POST /demo/score`, `/analyze`) | The reader cursor, yellow chunks and live LogReg + tree view described below. |

The demo model's metrics live in `models/demo_metrics.json`. `models/metrics.json` belongs to Marcus's real model.

## What you see in the PDF walkthrough

| Area | What it shows |
|---|---|
| Document (left) | The PDF text. A cursor moves word by word, line by line. Sentences that contain a fact are highlighted yellow, with a badge showing how many points they moved the risk (red = up, green = down). |
| Decision bar | Risk score, tier (`fast_track` / `review` / `investigate`) and a ‹ › **decision trail**. After scoring, ‹ steps back through each highlighted chunk and shows the score before and after it. The keyboard arrow keys also work. |
| What if | Untick a fact to re-score the claim without it (calls `POST /score` with `exclude`). |
| Logistic regression | Sigmoid curve with a numbered dot for each fact, plus every feature's term in z. |
| XGBoost | The claim's path through one tree, highlighted. **Auto** shows the tree whose leaf changed most after the latest fact, and T1 to T6 pin a tree. |
| Reader reasoning | A typed log of what each sentence contributed and how both models moved. |

## How it works

```
PDF -> pypdf text -> sentence splitter -> rule-based fact extractor (5 features)
    -> after each sentence: LogReg + XGBoost re-score (risk = average of both)
    -> UI replays the steps
```

- `src/pdf_pipeline.py`: text extraction, sentence splitting, and regex fact extraction for
  `claims_last_12m`, `days_policy_to_incident`, `vehicle_age`, `claim_amount`, `police_report`.
  Unread features start at the training median, so the score starts neutral.
- `src/demo_server.py`: FastAPI with `/health`, `/model/info`, `/score` (the team contract) and `/analyze` (PDF upload). It also serves `ui/` and `samples/`.
- `src/train_demo.py`: trains both models and saves them as JSON in `models/`.
  The committed `models/` were trained **on LinuxONE (s390x)** inside the container.
- `samples/`: three fictional claims (safe, borderline, suspicious) and the script that makes them.

## Honest limits (say these in the video)

- **The training data is synthetic** (`src/synth_data.py`). It is a placeholder until Yu Heng's
  processed data lands in `data/processed/`. Swap it in `train_demo.py`; the five feature names must match.
- **The "reader" is rule-based, not an LLM.** It runs offline on LinuxONE with no API key. The
  reasoning panel is generated from what the rules matched. An LLM extractor could replace
  `find_facts()` without changing the UI.
- The server analyses the whole PDF in one call, and the UI replays the steps with animation.
- Text PDFs only. Scanned image PDFs need OCR, which isn't included.

## Run it

**LinuxONE (Docker).** From the repo root on the server:
```bash
docker build -f deploy/claimlens.Dockerfile -t claimlens:latest .
docker run -d --name claimlens --restart unless-stopped -p 127.0.0.1:8000:8000 claimlens:latest
```
Then from your laptop: `ssh -i datathon.pem -L 8000:localhost:8000 linux1@<SERVER_IP>` and open http://localhost:8000.

**Laptop:**
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-demo.txt   # Mac: brew install libomp
.venv/bin/python src/train_demo.py        # optional, models/ is committed
.venv/bin/uvicorn demo_server:app --app-dir src --port 8000
```

**For the video:** http://localhost:8000/?sample=claim_suspicious.pdf opens the PDF mode and autoplays a sample (add `&fast=1` to skip the animation).
