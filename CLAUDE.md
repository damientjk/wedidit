# CLAUDE.md: "we did it" | IBM Z Datathon 2026

Last updated: Sat 17 Oct 2026. Repo: https://github.com/damientjk/wedidit (PUBLIC, never commit keys, tokens or the server IP).

## Project
Explainable AI model that scores insurance claims for fraud and risk. Low-risk claims are fast-tracked, suspicious ones go to a human. The model advises, the claims officer decides.
Theme 1: Real-Time AI for Critical Decisions. Submission title: `we did it_IBM Z Datathon 2026 Project`.
Deadline: submit Sat 17 Oct 2026 by 23:00 SGT. Sun 18 Oct 10:30am SGT is buffer only.

## Pipeline
claims CSV -> `src/prepare_data.py` (clean, merge, split 9k/3k/3k stratified on fraud label) -> `src/train.py` (LogReg baseline + XGBoost with `scale_pos_weight`) -> `models/model.json` + `models/metrics.json` -> `src/api.py` (FastAPI in Docker on LinuxONE) -> UI (risk score, per-feature contributions, what-if toggle).

## API contract (everyone codes against this)
`POST /score` with a claim as JSON. Returns:
```json
{
  "risk_score": 0.82,
  "tier": "investigate",
  "contributions": [
    {"feature": "claims_last_12m", "value": 4, "impact": 0.21},
    {"feature": "days_policy_to_incident", "value": 6, "impact": 0.14}
  ],
  "excluded": []
}
```
- `tier` is one of `fast_track`, `review`, `investigate`.
- Request field `"exclude": ["claims_last_12m"]` re-scores with that factor neutralised (what-if feature).
- Also expose `GET /health`.
- UI uses `ui/mock_score.json` (same shape) until the real API exists.
- `explain.py` exposes `score(claim, exclude=[])`, which `api.py` calls.

## Repo ownership (one owner per file, ask the owner before editing; pull before push, commit small to main)
| Path | Owner |
|---|---|
| README.md, src/features.py, docs/evidence/labs/ | Ethan |
| requirements.txt, deploy/, src/api.py, docs/evidence/ | Damien |
| data/raw/ (gitignored), data/README.md, src/prepare_data.py, data/processed/, video/final.mp4 | Yu Heng |
| src/train.py, src/explain.py, models/, video/script.md, video/shotlist.md | Marcus |
| ui/ | Da Feng |
| docs/architecture.md, docs/flowchart.png, docs/evaluation.md, docs/figures/ | Yizhuo |

`features.py` is shared by training and the API. Lock the feature list early to avoid train/serve mismatch.

## Timeline (SGT, Sat 17 Oct)
- 12:30 data ready, dummy `/score` live
- 14:30 baseline trained, metrics.json exists
- 16:30 real model behind API, UI wired end to end
- 18:00 FEATURE FREEZE (bug fixes and polish only)
- 18:30 evidence screenshots done
- 20:00 test-set numbers frozen
- 21:00 video recorded (3 min max)
- 22:30 video edited, final checklist
- 23:00 submit on Hack2skill
If 90+ min behind at 16:30: cut AI explanation text first, then jump-to-next-risk navigation. Policy-document (PDF) intake is out of scope.

## Infrastructure
- LinuxONE server (Ubuntu, s390x architecture, NOT x86) created by Damien. Shared login `linux1`. Event code SSF2026.
- Connection details (IP, key, Jupyter token) are shared privately by Damien. See SETUP.md for how to connect.
- Jupyter Lab ML container (`linuxone-community-cloud/jupyter-lab-ml`) runs on port 38888 on the server, reached through an SSH tunnel.
- Everyone shares ONE Linux account. Work in your own folder on the server (`~/yourname/`) and use the git repo as the only shared space.
- s390x warning: test `pip install xgboost shap fastapi uvicorn` early. Some wheels may need building from source.
- Fallback if Docker fights back: run `uvicorn api:app` directly. Judges need the model running on LinuxONE, a container is the neat way, not the only way.

## Key concepts (agreed explanations)
- Logistic regression = baseline, weighted sum squashed to 0..1, linear only, explainable.
- XGBoost = eXtreme Gradient Boosting. Many small trees, each fixing the last one's errors. Uses gradient g = p - y and hessian h = p(1-p), split gain, leaf weight w = -G/(H+lambda), updates scaled by learning rate eta. Regularization (lambda, gamma) stops overfitting.
- Gain = how much a split improves fit. Gain minus gamma below zero means no split.
- SHAP = Shapley values, per-claim feature contributions. Shows what the model relied on, not what caused fraud.
- Scores are only true probabilities if calibrated. `scale_pos_weight` inflates scores, so treat risk_score as a ranking score or calibrate it. Do not say "82 percent chance of fraud" in the video.
- XGBoost cannot read PDFs. A PDF needs extract -> structure into model columns -> score. Future extension only.
- API = the waiter between UI and model. Docker = sealed box holding the program and everything it needs.
- Why LinuxONE: it is the event requirement, and the pitch is scoring claims where sensitive data already lives (security, low latency). Give it a 10 second line in the video.

## Rules
- All code written during the event window. No reusing old projects. AI assistants allowed, but each person must be able to explain their own part in the video.
- Open-source libraries must be linked in the README.
- Document any synthetic data column clearly.
- Report precision and recall at the chosen threshold, not just AUC. Do not overclaim.
- Never commit `*.pem`, `.env`, tokens, the server IP, or `data/raw/`.

## Submission checklist
Title with team name, video of 3 min or less (problem, data, solution, IBM Z), video shows LinuxONE instance and model running, public GitHub link works for outsiders, README lists all datasets and libraries with links, docs/evidence screenshots, both mandatory labs done and screenshotted (confirm which notebooks via Discord #ibm-z-sme-helpdesk), optional flowchart, feedback survey by 23 Oct.

## Style for Claude Code
Be direct and concise. No em dashes. Challenge flawed assumptions. Explain things simply, the user is a student.
