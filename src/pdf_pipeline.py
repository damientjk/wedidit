"""PDF claim -> sentences -> facts -> risk after every sentence.

The reader is rule-based (regular expressions over each sentence), so it runs
offline on LinuxONE with no API key. Each sentence can set one or more of the
model's five features. After every sentence the claim is re-scored, so the
change in risk is attributed to the sentence that caused it.
"""
import io
import re

from pypdf import PdfReader

WORDS = {"no": 0, "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
         "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
         "thirty": 30, "forty": 40, "sixty": 60, "ninety": 90}
NUM = r"(\d+|" + "|".join(WORDS) + r")"
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
CURRENT_YEAR = 2026

LABELS = {
    "claims_last_12m": "Claims in last 12 months",
    "days_policy_to_incident": "Days from policy start to incident",
    "vehicle_age": "Vehicle age (years)",
    "claim_amount": "Claim amount ($)",
    "police_report": "Police report filed",
}


def num(tok):
    tok = tok.lower()
    return int(tok) if tok.isdigit() else WORDS[tok]


def extract_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|No|St|Rd|Ave|Ref|Pte|Ltd|approx|e\.g|i\.e))\.$")


def split_sentences(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)           # re-join hyphenated breaks
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    longest = max((len(l) for l in lines), default=0)
    # A short line with no closing punctuation is a heading: end it with a break.
    blocks = []
    for i, l in enumerate(lines):
        is_heading = (not re.search(r"[.!?:;,]$", l) and len(l) < 0.6 * longest
                      and (i + 1 == len(lines) or lines[i + 1][:1].isupper()))
        blocks.append(l + (" " if is_heading else " "))
    text = re.sub(r"[ \t]{2,}", " ", "".join(blocks)).strip()

    parts = []
    for chunk in text.split(" "):
        # End of sentence: . ! ? followed by space and a capital or digit.
        for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", chunk.strip()):
            if parts and ABBREV.search(parts[-1]) and not parts[-1].endswith(" "):
                parts[-1] += " " + p                    # "J. Tan" is not two sentences
            else:
                parts.append(p)
        if parts:
            parts[-1] += " "
    return [p.rstrip(" ").strip() for p in parts if len(p.strip("  ")) > 1]


def find_facts(sentence):
    """Return [(feature, value, evidence_span)] found in one sentence."""
    s = sentence
    low = s.lower()
    facts = []

    # Prior claims in the last 12 months.
    if re.search(r"12 months|twelve months|past year|last year|previous year", low):
        m = re.search(NUM + r"\s+(?:prior |previous |earlier |other )?claims?\b", low)
        if m:
            facts.append(("claims_last_12m", num(m.group(1)), m.group(0)))
    if re.search(r"\b(first claim|no (?:prior|previous) claims|never (?:made|filed) a claim)", low):
        facts.append(("claims_last_12m", 0, re.search(r"first claim|no (?:prior|previous) claims|never (?:made|filed) a claim", low).group(0)))

    # Time from policy start to incident.
    m = re.search(NUM + r"\s+(day|week|month|year)s?\s+(?:before|prior to)\s+the\s+(?:reported\s+)?(?:loss|incident|accident|collision|theft)", low)
    if m and "polic" in low:
        facts.append(("days_policy_to_incident", num(m.group(1)) * UNIT_DAYS[m.group(2)], m.group(0)))
    m = re.search(NUM + r"\s+(day|week|month|year)s?\s+after\s+(?:the\s+)?(?:policy|cover|inception)", low)
    if m:
        facts.append(("days_policy_to_incident", num(m.group(1)) * UNIT_DAYS[m.group(2)], m.group(0)))
    m = re.search(r"(?:held|had) (?:the |this )?polic(?:y|ies) for\s+" + NUM + r"\s+(day|week|month|year)s?", low)
    if m:
        facts.append(("days_policy_to_incident", num(m.group(1)) * UNIT_DAYS[m.group(2)], m.group(0)))

    # Vehicle age.
    if re.search(r"vehicle|car|sedan|hatchback|van|truck|suv|motorcycle", low):
        m = re.search(NUM + r"[- ]years?[- ]old", low)
        if m:
            facts.append(("vehicle_age", num(m.group(1)), m.group(0)))
        m = re.search(r"(?:manufactured|built|first registered|made) in ((?:19|20)\d\d)", low)
        if m:
            facts.append(("vehicle_age", CURRENT_YEAR - int(m.group(1)), m.group(0)))

    # Claim amount: first dollar figure in a sentence about the claim / repair.
    if re.search(r"claim|repair|estimate|total|damage|cost|quote|invoice", low):
        m = re.search(r"(?:s?\$|sgd\s?)\s?([\d,]+(?:\.\d+)?)", low)
        if m:
            facts.append(("claim_amount", float(m.group(1).replace(",", "")), m.group(0)))

    # Police report.
    neg = re.search(r"no police report|(?:not|never) (?:been )?reported to (?:the )?police|did not (?:file|make|lodge) a police report|without a police report|no report (?:was )?(?:made|filed) (?:with|to) (?:the )?police", low)
    pos = re.search(r"police report (?:was |has been )?(?:filed|lodged|made|obtained|attached)|reported (?:the (?:incident|accident|theft) |it )?to (?:the )?police|attached police report|police report (?:number|no\.|ref)", low)
    if neg:
        facts.append(("police_report", 0, neg.group(0)))
    elif pos:
        facts.append(("police_report", 1, pos.group(0)))

    # One value per feature per sentence (the last match wins).
    seen = {}
    for f, v, span in facts:
        seen[f] = (f, v, span)
    return list(seen.values())


def fmt(feature, value):
    if feature == "claim_amount":
        return f"${value:,.0f}"
    if feature == "police_report":
        return "yes" if value else "no"
    return f"{value:g}"


def analyse(pdf_bytes, scorer, baseline):
    """Score the claim after each sentence. scorer(features) -> dict."""
    sentences = split_sentences(extract_text(pdf_bytes))
    state = dict(baseline)
    prev = scorer(state)
    start = prev
    steps = []
    for i, sent in enumerate(sentences):
        facts = find_facts(sent)
        thinking = [f"Reading sentence {i + 1} of {len(sentences)}."]
        changed = []
        for f, v, span in facts:
            old = state[f]
            state[f] = v
            changed.append({"feature": f, "label": LABELS[f], "old": old, "value": v, "span": span})
            thinking.append(f"Found “{span}”, so {LABELS[f].lower()} = {fmt(f, v)} (was {fmt(f, old)}).")
        cur = scorer(state) if facts else prev
        delta = cur["risk_score"] - prev["risk_score"]
        if facts:
            thinking.append(
                f"Logistic regression z {prev['z_lr']:+.2f} → {cur['z_lr']:+.2f} (p {prev['p_lr']:.2f} → {cur['p_lr']:.2f}). "
                f"XGBoost p {prev['p_xgb']:.2f} → {cur['p_xgb']:.2f}.")
            thinking.append(f"Risk {prev['risk_score']:.0%} → {cur['risk_score']:.0%} ({delta * 100:+.0f} pts).")
        else:
            thinking.append("No risk factor in this sentence. Score unchanged.")
        steps.append({"index": i, "text": sent, "facts": changed, "delta": round(delta, 4),
                      "thinking": thinking, "features": dict(state), **cur})
        prev = cur
    return {"sentences": len(sentences), "start": {"features": dict(baseline), **start},
            "steps": steps, "final": {"features": dict(state), **prev}}
