## 3. Architecture

The final model is tuned XGBoost (`xgboost_09`) with its own fitted preprocessing. The claim form passes 19 source fields to `/score`. The PDF reader uses pypdf and rule-based patterns to extract those same fields, then calls the same scorer through `/analyze`. Both intake screens display the risk ranking, tier and native TreeSHAP explanations for a claims officer to review.

The service runs in FastAPI and Docker on IBM LinuxONE (Ubuntu 24.04, s390x). The UI withholds the score until the policy-coverage category (`BasePolicy`) is provided; other missing fields are flagged and scores are provisional. Reading a coverage category from a claim does not verify the policy itself.

![Claim! architecture](docs/flowchart.png)

See [the architecture and verification status](docs/architecture.md), [editable SVG diagram](docs/flowchart.svg), and [evaluation](docs/evaluation.md). The historical live checks and latest PDF source review are documented separately; the held-out model metrics do not measure PDF extraction accuracy.
