# ClaimLens: watch the model read a claim

Demo UI (owner: Da Feng). A claims officer uploads a claim PDF. The reader goes
through it sentence by sentence, left to right, and every fact it finds
re-scores the claim with two pre-trained models. When it finishes, the officer
can step back through the exact sentences that moved the score.

## Two screens, one model

Both tabs score with Marcus's final XGBoost model (`xgboost_09`, `POST /score`, `src/explain.py`). The score is an uncalibrated **ranking score** (for example 0.43), never a % chance of fraud. Contributions are exact TreeSHAP values in log-odds.

| Screen | What it does |
|---|---|
| **Claim form** (default) | The agent enters the 19 fields by hand. Policy coverage is required. The other fields are grouped Recommended / Optional / No effect (see `docs/missing_fields.md`). Blank recommended fields show their best and worst case (`POST /score/range`). |
| **PDF walkthrough** (`#pdf`) | Upload or drop a claim PDF, or pick a sample, then press **Start**. The reader moves a cursor word by word and fills in the 19 fields from the text. After every sentence that adds a field, the claim is re-scored. Sentences that changed something get a yellow highlight and a score badge. **Stop / Resume / Restart / Skip** control the playback. When the reader finishes, ‹ › steps back through the sentences that moved the score, unticking a field re-scores without it (what-if), and fields the PDF never mentions are checked for whether they could change the tier. |

## Why there is no score until Policy coverage is known

If nothing is known about a claim, the model fills every field with its training average and scores that "average claim". Most claims are not fraud (about 94%), so the average claim looks fairly safe: the old demo gave an empty PDF 12% and *Fast track*. That is the worst failure for a triage tool, because an unread claim gets waved through. Absence of evidence is not evidence of a safe claim.

So both screens follow one rule:
- **No score until Policy coverage is known.** It is the field that decides the score: left blank, 37% of claims land in the wrong tier.
- **Instead of a number, the screen shows the range:** for example, "depending on Policy coverage this claim would score 0.04 (Liability) to 0.29 (All Perils)", with a prompt to ask for the policy type.
- **Never fast-track on missing information.** An empty or unreadable PDF ends as *Cannot score*, so it goes to manual review (and gets checked for being a scanned image).
- **Once Policy coverage is known**, any other missing fields are listed and the score is labelled provisional. Blank recommended fields that could still change the tier are named.

## How the PDF reader works (`src/pdf_pipeline.py`)

- **Text:** pypdf extracts it, then a splitter breaks it into sentences (headings and initials like "J. Tan" are handled).
- **Fields:** rule-based patterns per sentence. No LLM, no API key, so it runs offline on LinuxONE.
- **Dates:**
  - Accident date gives Month, DayOfWeek and WeekOfMonth.
  - Claim date gives MonthClaimed, DayOfWeekClaimed and WeekOfMonthClaimed.
  - Policy start date plus those dates gives the policy-duration bands.
- **Week of month** = ceil(day / 7), capped at 5. This matched 94–96% of the Kaggle rows, better than calendar weeks (87–90%). To be confirmed with Yu Heng and Marcus.
- **Numbers are mapped to the model's bands:** vehicle price, vehicle age, days, prior claims and number of cars. Values the model never saw (a $450 deductible, a "1 to 7 days" policy-to-claim band) are snapped to the nearest known one, with a note in the reasoning panel.
- **Scoring:** the claim starts empty and is re-scored with `explain.score()` after each sentence. Fields never mentioned stay blank, and the model flags them as missing.

**Samples** (`samples/make_samples.py`) are built from real test claims #6180 (low, 0.03), #7223 (medium, 0.29) and #2857 (high, 0.43, labelled fraud). All 19 fields are stated in ordinary sentences, and the PDF scores match the real claims exactly.

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
