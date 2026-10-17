"""
tests/test_api.py (owner: Ethan)

Checks that the scoring API returns the agreed response shape.

Usage:
    export API_URL=http://<SERVER_IP>:8000      # never hardcode the IP in git
    python tests/test_api.py                     # runs against val.csv if present
    python tests/test_api.py --n 20 --label FraudFound_P --csv data/processed/val.csv
    python tests/test_api.py --format demo       # against src/demo_server.py (ClaimLens)

Windows PowerShell:
    $env:API_URL = "http://<SERVER_IP>:8000"
    python tests/test_api.py

If the CSV is missing, one built-in sample claim is used so the response
shape can still be checked before Yu Heng's data lands.
"""
import argparse
import json
import os
import sys

import requests

TIERS = {"fast_track", "review", "investigate"}
TIMEOUT = 15

# Fallback only (used when val.csv is missing). Matches the raw columns in data/README.md.
FALLBACK_CLAIM = {
    "Month": "Dec", "WeekOfMonth": 5, "DayOfWeek": "Wednesday", "Make": "Honda",
    "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": 1, "Sex": "Female", "MaritalStatus": "Single", "Age": 21,
    "Fault": "Policy Holder", "VehicleCategory": "Sport", "VehiclePrice": "more than 69000",
    "Deductible": 300, "DriverRating": 1, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years", "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
    "WitnessPresent": "No", "AgentType": "External", "NumberOfSuppliments": "none",
    "AddressChange_Claim": "1 year", "NumberOfCars": "3 to 4", "BasePolicy": "All Perils",
    "age_was_missing": 0,
}


def load_claims(csv_path, label_col, n):
    """Return a list of claim dicts (label column removed)."""
    if not os.path.exists(csv_path):
        print(f"[warn] {csv_path} not found, using 1 fallback claim")
        return [FALLBACK_CLAIM]
    import pandas as pd

    df = pd.read_csv(csv_path)
    # drop the label and bookkeeping/synthetic columns so only real claim fields are sent
    drop = [c for c in df.columns if c == label_col or c in ("split", "claim_id") or c.endswith("(synth)")]
    df = df.drop(columns=drop)
    df = df.head(n).where(df.head(n).notna(), None)
    return json.loads(df.to_json(orient="records"))


# Da Feng's interim ClaimLens server (src/demo_server.py) does not take the raw
# columns. Its /score reads five demo fields and silently fills any missing one
# with the training median, so raw claims all get the same score. --format demo
# converts raw claims to those fields using the same band midpoints as features.py.
def to_demo(claim):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from features import DAYS_POLICY_ACCIDENT, PAST_CLAIMS, VEHICLE_AGE_YEARS

    out = {
        "claims_last_12m": PAST_CLAIMS.get(claim.get("PastNumberOfClaims")),
        "days_policy_to_incident": DAYS_POLICY_ACCIDENT.get(claim.get("Days_Policy_Accident")),
        "vehicle_age": VEHICLE_AGE_YEARS.get(claim.get("AgeOfVehicle")),
        "police_report": {"Yes": 1, "No": 0}.get(claim.get("PoliceReportFiled")),
    }
    # claim_amount only exists as a synthetic column, which load_claims drops, so it stays at the median.
    return {k: v for k, v in out.items() if v is not None}


def check_shape(resp_json):
    """Return a list of problems (empty list means the shape is fine)."""
    problems = []
    for key in ("risk_score", "tier", "contributions", "excluded"):
        if key not in resp_json:
            problems.append(f"missing key: {key}")
    if problems:
        return problems

    score = resp_json["risk_score"]
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        problems.append(f"risk_score not in [0,1]: {score}")
    if resp_json["tier"] not in TIERS:
        problems.append(f"bad tier: {resp_json['tier']}")
    if not isinstance(resp_json["contributions"], list):
        problems.append("contributions is not a list")
    else:
        for i, c in enumerate(resp_json["contributions"]):
            for k in ("feature", "value", "impact"):
                if k not in c:
                    problems.append(f"contributions[{i}] missing {k}")
    if not isinstance(resp_json["excluded"], list):
        problems.append("excluded is not a list")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("API_URL"))
    ap.add_argument("--csv", default="data/processed/val.csv")
    ap.add_argument("--label", default="FraudFound_P")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--format", choices=["raw", "demo"], default="raw",
                    help="raw = agreed contract (raw columns); demo = src/demo_server.py fields")
    args = ap.parse_args()

    if not args.url:
        sys.exit("Set API_URL (e.g. http://<SERVER_IP>:8000) or pass --url")
    base = args.url.rstrip("/")

    passed, failed = 0, 0

    def record(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"[PASS] {name}")
        else:
            failed += 1
            print(f"[FAIL] {name} {detail}")

    # 1. health
    try:
        r = requests.get(f"{base}/health", timeout=TIMEOUT)
        record("GET /health returns 200", r.status_code == 200, f"(got {r.status_code})")
    except requests.RequestException as e:
        record("GET /health reachable", False, f"({e})")
        sys.exit("Server unreachable. Check the SSH tunnel, port, and firewall.")

    # 2. score many claims
    claims = load_claims(args.csv, args.label, args.n)
    if args.format == "demo":
        claims = [to_demo(c) for c in claims]
    last_ok = None
    for i, claim in enumerate(claims):
        try:
            r = requests.post(f"{base}/score", json=claim, timeout=TIMEOUT)
        except requests.RequestException as e:
            record(f"claim {i}: request", False, f"({e})")
            continue
        if r.status_code != 200:
            record(f"claim {i}: status 200", False, f"(got {r.status_code}: {r.text[:120]})")
            continue
        problems = check_shape(r.json())
        record(f"claim {i}: response shape", not problems, "; ".join(problems))
        if not problems:
            last_ok = (claim, r.json())

    # 3. what-if exclusion
    if last_ok and last_ok[1]["contributions"]:
        claim, base_resp = last_ok
        feat = base_resp["contributions"][0]["feature"]
        r = requests.post(f"{base}/score", json={**claim, "exclude": [feat]}, timeout=TIMEOUT)
        if r.status_code == 200:
            body = r.json()
            record("what-if: feature listed in excluded", feat in body.get("excluded", []))
            record(
                "what-if: score changed",
                body.get("risk_score") != base_resp["risk_score"],
                "(unchanged, check exclusion logic)",
            )
        else:
            record("what-if: status 200", False, f"(got {r.status_code})")
    else:
        print("[skip] what-if test (no valid scored claim with contributions)")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
