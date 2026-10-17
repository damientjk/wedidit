# ETHAN.md: Task brief for Claude Code (Ethan's role)

Read CLAUDE.md first for project context. This file is only Ethan's part.
Role: mandatory labs, feature engineering, API testing, README, final submission checklist.
Deadline: everything done by 19:00 for the first checklist pass, final pass at 22:30, submit 23:00 SGT Sat 17 Oct 2026.

## Files Ethan owns (do not edit anyone else's files, ask the owner)
- `src/features.py`
- `tests/test_api.py` (new, test script)
- `README.md`
- `docs/evidence/labs/lab1/` and `docs/evidence/labs/lab2/` (screenshots)
- Submission checklist walk-through

## What Claude Code can do vs what Ethan must do by hand
| Task | Claude Code | Ethan manually |
|---|---|---|
| Write `features.py`, test script, README | Yes | Review and understand every line |
| Run the two mandatory labs | No (browser notebooks) | Yes, in Jupyter Lab |
| Screenshots of labs | No | Yes |
| Ask Discord which notebooks are the labs | No | Yes |
Ethan must be able to explain his own code in the video, so read everything Claude writes.

## Task 1 (10:00): confirm the mandatory labs, HIGHEST STAKES
Hard requirement, unverified. Two labs:
1. Predictive AI hybrid AI scenarios (multi-model)
2. Predictive AI + Generative AI
Ethan to ask in Discord #ibm-z-sme-helpdesk which notebooks count, and check the 16 Oct Zoom recording (YouTube link in the chat).
Best guesses from the `jupyter-lab-ml` repo (NOT confirmed):
- Lab 1: `Fraud_LSTM_Keras_TF` and `Churn_Scikit_PyTorch`
- Lab 2: `Loan_Approval_Demo`
Do not start screenshotting until the helpdesk confirms, or the work may be wasted.

## Task 2 (11:00 to 13:30): run the labs
1. Open the tunnel (see SETUP.md section 4), then http://localhost:38888 and enter the Jupyter token from Damien.
2. Lab 1: screenshot each major step into `docs/evidence/labs/lab1/` (use numbered names like `01_open.png`).
3. Lab 2: same into `docs/evidence/labs/lab2/`.
4. At 13:30 post a 3 line summary of each lab in the group chat, plus anything reusable for Marcus (model) or Da Feng (UI).
Screenshots must show the LinuxONE/Jupyter environment so judges can see the labs ran on IBM Z.

## Task 3 (14:00): feature candidates
After Yu Heng posts the schema (`data/README.md` and the column list), list candidate engineered features. Ideas to check against real columns:
- claim frequency (`claims_last_12m`)
- time from policy start to incident (`days_policy_to_incident`)
- vehicle age
- claimant history
- claim amount relative to vehicle value or policy limit
Only use columns that exist. If a column is missing, say so, do not invent it.

## Task 4 (14:30): `src/features.py`
Requirements:
- One function, e.g. `build_features(df) -> df`, used by BOTH `train.py` and `api.py`.
- Must work on a single claim (one-row DataFrame) as well as the full file.
- No leakage: never use the fraud label or anything computed from the full dataset (e.g. dataset-wide means) unless fit on train only and saved.
- Handle missing values and unseen categories the same way in training and serving.
- Keep feature names identical to the names in the `/score` `contributions` array.
- Pure functions, no file I/O, so Damien can import it in the container.
RISK: Marcus trains the first model on raw columns, then switches to this for the final XGBoost run. A train/serve mismatch here silently breaks the demo. Post the final feature list in the group chat as soon as it is locked, ideally before 15:00.

## Task 5 (15:30): `tests/test_api.py`
- Read 20 claims from `data/processed/val.csv`.
- POST each to `http://<SERVER_IP>:<PORT>/score` (take the URL from an env var `API_URL`, never hardcode the IP).
- Assert for every response: keys `risk_score`, `tier`, `contributions`, `excluded` exist; `risk_score` is between 0 and 1; `tier` is one of `fast_track`, `review`, `investigate`; each contribution has `feature`, `value`, `impact`.
- Also test the what-if: send one claim with `"exclude": ["claims_last_12m"]` and assert it appears in `excluded` and the score changes.
- Also test `GET /health`.
- Print a clear pass/fail summary. Use `requests` and plain asserts or pytest.
Run it against the dummy API at 12:30 shape if available, then against the deployed API at 16:30. Report bugs to the file owner (Damien for api.py, Marcus for model/explain).

## Task 6 (17:30): `README.md`
Sections:
1. Problem and approach (Marcus drafts the problem statement, paste it in)
2. Architecture (link to `docs/architecture.md` and `docs/flowchart.png`)
3. How to run (clone, install `requirements.txt`, run training, start API, open UI, Docker commands from Damien)
4. IBM Z / LinuxONE usage (what ran where, why LinuxONE fits: data stays on the platform, low latency, security)
5. Datasets with links and licences (from Yu Heng's `data/README.md`)
6. Open-source libraries with links: XGBoost, scikit-learn, SHAP, FastAPI, Uvicorn, pandas, Docker, plus anything else in `requirements.txt`
7. Results (from Yizhuo's `docs/evaluation.md`, only real numbers)
8. Limitations: scores are ranking scores unless calibrated, synthetic columns are documented, PDF intake is future work
9. Team
Never put the SSH key, tokens or the server IP in the README. The repo is public.

## Task 7 (19:00 and 22:30): submission checklist
Walk through every item and tick only what is verified:
- [ ] Project title contains team name: `we did it_IBM Z Datathon 2026 Project`
- [ ] Video of 3 minutes or less covering problem, data, solution, IBM Z
- [ ] Video shows the LinuxONE instance and the model running
- [ ] GitHub link works for someone not on the team (test in a private/incognito window)
- [ ] README lists all datasets and open-source libraries with links
- [ ] `docs/evidence/` screenshots included
- [ ] Optional flowchart attached
- [ ] Both mandatory labs completed and screenshotted
- [ ] No secrets in the repo (search for `.pem`, tokens, the IP)
- [ ] Feedback survey by 23 Oct

## Rules reminder
- All code written during the event. No reused old projects.
- Pull before push, commit small, only touch your own files.
- Style: direct, concise, no em dashes, challenge flawed assumptions.

## Order of work for Claude Code
1. Wait for schema, then do Task 3 and 4.
2. Write the test script (Task 5) early against `ui/mock_score.json` shape so it is ready when the API goes live.
3. Draft the README skeleton (Task 6) with TODO markers, fill real numbers at the end.
4. Do not invent metrics, dataset names, or results. Leave a clearly marked TODO instead.
