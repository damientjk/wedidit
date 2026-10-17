"""PDF claim -> sentences -> the final model's 19 fields -> score after every sentence.

The reader is rule-based (regular expressions over each sentence), so it runs offline on
LinuxONE with no API key. Each sentence can fill in one or more of the 19 fields that
Marcus's final XGBoost model reads. The claim starts empty; after every sentence that
adds a field it is re-scored with explain.score(), so the change in score is attributed
to the sentence that caused it. Fields the PDF never mentions stay blank, and
explain.score() flags them as missing.

Every value is mapped onto the exact categories the model was trained on
(GET /model/final "options"): numbers become bands, dates become month / weekday /
week of month.

Week of month = ceil(day / 7), capped at 5 (days 1-7 are week 1, 8-14 week 2, ...).
This matched 94-96% of rows in the Kaggle data (Year + Month + DayOfWeek + WeekOfMonth),
better than calendar weeks (87-90%). To be confirmed with the data owners.
"""
import calendar
import io
import re
from datetime import date

from pypdf import PdfReader

import explain

LABELS = explain.LABELS
# No score until these are known: left blank, BasePolicy puts 37% of claims in the wrong tier
# (docs/missing_fields.md). Same rule as the claim form.
REQUIRED = ["BasePolicy"]
DEDUCTIBLES = [300.0, 400.0, 500.0, 700.0]  # the only deductibles in the training data
CURRENT_YEAR = 2026

WORDS = {"no": 0, "zero": 0, "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3,
         "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "eighteen": 18, "twenty": 20, "thirty": 30}
NUM = r"(\d+|" + "|".join(WORDS) + r")"
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
DUR = NUM + r"\s+(day|week|month|year)s?"
MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_NAMES.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTH_RE = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Band order per field, used to snap a value to the nearest band the model knows.
BAND_ORDER = {
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000",
                     "60000 to 69000", "more than 69000"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}
MAKE_ALIASES = {"mercedes-benz": "Mercedes", "mercedes benz": "Mercedes", "volkswagen": "VW",
                "chevy": "Chevrolet", "accura": "Acura", "porche": "Porsche", "nisson": "Nissan"}


def allowed_values():
    """The model's categories per field (exact strings / ints), plus the observed deductibles."""
    _, _, _, _, encoder, categorical, _ = explain._load()
    opts = {f: [c.item() if hasattr(c, "item") else c for c in cats]
            for f, cats in zip(categorical, encoder.categories_)}
    opts["Deductible"] = DEDUCTIBLES
    return opts


# ---------------------------------------------------------------------------
# Numbers -> bands
# ---------------------------------------------------------------------------
def num(tok):
    tok = tok.lower()
    return int(tok) if tok.isdigit() else WORDS[tok]


def money(tok, suffix=""):
    v = float(tok.replace(",", ""))
    return v * 1000 if suffix.lower() in ("k", "000") else v


def band_vehicle_price(v):
    for limit, band in [(20000, "less than 20000"), (30000, "20000 to 29000"), (40000, "30000 to 39000"),
                        (60000, "40000 to 59000"), (70000, "60000 to 69000")]:
        if v < limit:
            return band
    return "more than 69000"


def band_vehicle_age(years):
    years = int(years)
    return "new" if years < 2 else f"{years} years" if years <= 7 else "more than 7"


def band_days(days):
    return ("none" if days <= 0 else "1 to 7" if days <= 7 else "8 to 15" if days <= 15
            else "15 to 30" if days <= 30 else "more than 30")


def band_past_claims(n):
    return "none" if n == 0 else "1" if n == 1 else "2 to 4" if n <= 4 else "more than 4"


def band_cars(n):
    return "1 vehicle" if n <= 1 else "2 vehicles" if n == 2 else "3 to 4" if n <= 4 else "5 to 8" if n <= 8 else "more than 8"


def week_of_month(d):
    return min(5, (d.day - 1) // 7 + 1)


def snap(field, value, opts):
    """Map a value onto a category the model knows. Returns (value, note or None)."""
    allowed = opts[field]
    if value in allowed:
        return value, None
    if field == "Deductible":
        near = min(allowed, key=lambda a: abs(a - value))
        return near, f"${value:,.0f} is not a deductible the model has seen, using the nearest (${near:,.0f})"
    if field in BAND_ORDER:
        order = [b for b in BAND_ORDER[field] if b in allowed]
        i = BAND_ORDER[field].index(value)
        near = min(order, key=lambda b: abs(BAND_ORDER[field].index(b) - i))
        return near, f"the model has no '{value}' band, using the closest ('{near}')"
    if all(isinstance(a, int) for a in allowed):
        near = min(allowed, key=lambda a: abs(a - int(value)))
        return near, f"{value} is outside the model's range, using {near}"
    return None, f"'{value}' is not a value the model knows, left blank"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dates: which event does each date belong to?
# ---------------------------------------------------------------------------
DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + MONTH_RE + r")\.?,?\s+(\d{4})\b", re.I), "dmy"),
    (re.compile(r"\b(" + MONTH_RE + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "numeric"),   # DD/MM/YYYY (Singapore)
]
EVENT_WORDS = {
    "policy_start": r"polic(?:y|ies) (?:start(?:ed)?|began|commenced|took effect|was issued|incepted)|"
                    r"(?:cover|coverage) (?:start(?:ed)?|began|commenced)|inception|start date of the policy|policy start",
    "claim": r"\bclaim\b[^.]*?\b(?:submitted|filed|lodged|received|reported|made)\b|"
             r"\b(?:submitted|filed|lodged|reported|made)\b (?:the |this |a |their |his |her )?claim|notified",
    "accident": r"accident|incident|collision|crash|happened|occurred|struck|was hit|damaged|stolen|theft|loss",
}


def find_dates(sentence):
    """[(event, date, span)] for each date in the sentence."""
    found = []
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(sentence):
            try:
                if kind == "dmy":
                    d = date(int(m.group(3)), MONTH_NAMES[m.group(2).lower()], int(m.group(1)))
                elif kind == "mdy":
                    d = date(int(m.group(3)), MONTH_NAMES[m.group(1).lower()], int(m.group(2)))
                else:
                    d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                continue
            if not any(s <= m.start() < e for s, e, *_ in found):
                found.append((m.start(), m.end(), d, m.group(0)))
    found.sort()
    out, prev_end = [], 0
    low = sentence.lower()
    for start, end, d, span in found:
        # The event is the keyword closest before the date (else anywhere in the sentence).
        before = low[prev_end:start]
        best, best_pos = None, -1
        for event, pat in EVENT_WORDS.items():
            for m in re.finditer(pat, before):
                if m.end() > best_pos:
                    best, best_pos = event, m.end()
        if best is None:
            for event, pat in EVENT_WORDS.items():
                if re.search(pat, low):
                    best = event
                    break
        if best:
            out.append((best, d, span))
        prev_end = end
    return out


# ---------------------------------------------------------------------------
# One sentence -> facts
# ---------------------------------------------------------------------------
def find_facts(sentence):
    """Return [(feature, raw value, evidence span)] found in one sentence, plus dates.

    Raw values are numbers or band strings; analyse() snaps them to the model's categories.
    """
    s, low = sentence, sentence.lower()
    facts = []

    def add(f, v, m):
        # Quote the original text (lower() keeps positions for these ASCII patterns).
        facts.append((f, v, sentence[m.start():m.end()] if hasattr(m, "group") else m))

    is_claim = bool(re.search(EVENT_WORDS["claim"], low))
    is_accident = bool(re.search(EVENT_WORDS["accident"], low))

    # Policy type.
    m = re.search(r"all[- ]perils|comprehensive (?:cover|coverage|policy|plan|insurance)", low)
    if m:
        add("BasePolicy", "All Perils", m)
    else:
        m = re.search(r"collision(?:[- ]only)? (?:cover|coverage|policy|plan|insurance)", low)
        if m:
            add("BasePolicy", "Collision", m)
        else:
            m = re.search(r"liability(?:[- ]only)? (?:cover|coverage|policy|plan|insurance)|third[- ]party(?: only)? (?:cover|policy|insurance)", low)
            if m:
                add("BasePolicy", "Liability", m)

    # Deductible / excess.
    m = re.search(r"(?:deductible|excess)[^.$]*?(?:s?\$|sgd\s?)\s?([\d,]+)|(?:s?\$|sgd\s?)\s?([\d,]+)\s+(?:deductible|excess)", low)
    if m:
        add("Deductible", money(m.group(1) or m.group(2)), m)

    # Agent.
    m = re.search(r"(?:internal|in-house|company|direct) (?:agent|sales channel)", low)
    if m:
        add("AgentType", "Internal", m)
    else:
        m = re.search(r"(?:external|independent) (?:agent|broker)|through a broker|via a broker", low)
        if m:
            add("AgentType", "External", m)

    # Driver rating 1-4.
    m = re.search(r"driver rating (?:on file )?(?:of |is |was |: )?([1-4])\b|rated ([1-4]) (?:out of (?:4|four)|on (?:a|the) (?:1|one)[- ]to[- ](?:4|four) scale)", low)
    if m:
        add("DriverRating", int(m.group(1) or m.group(2)), m)

    # Make.
    makes = {mk.lower(): mk for mk in allowed_values()["Make"]}
    makes.update(MAKE_ALIASES)
    m = re.search(r"\b(" + "|".join(re.escape(k) for k in sorted(makes, key=len, reverse=True)) + r")\b", low)
    if m:
        add("Make", makes[m.group(1)], m)

    # Vehicle category.
    m = re.search(r"\bsedan\b|\bsaloon\b", low)
    if m:
        add("VehicleCategory", "Sedan", m)
    else:
        m = re.search(r"sports? car|\bcoupe\b|convertible|roadster", low)
        if m:
            add("VehicleCategory", "Sport", m)
        else:
            m = re.search(r"\bsuv\b|utility vehicle|pick-?up|\btruck\b|\bvan\b|4x4|four-wheel drive", low)
            if m:
                add("VehicleCategory", "Utility", m)

    # Accident area.
    m = re.search(r"\burban\b|city cent(?:re|er)|downtown|town cent(?:re|er)|\bcbd\b", low)
    if m:
        add("AccidentArea", "Urban", m)
    else:
        m = re.search(r"\brural\b|countryside|country road|farmland|\bvillage\b", low)
        if m:
            add("AccidentArea", "Rural", m)

    # Vehicle price (value of the car, not the repair bill).
    if not re.search(r"repair|estimate|invoice|quote", low):
        # The amount right after the value word, so "$450 excess ... worth $35k" gives $35k.
        m = re.search(r"(?:valued|worth|market value|purchase price|priced|bought (?:it )?for|price)[^$]{0,40}?"
                      r"(?:s?\$|sgd\s?)\s?([\d,]+(?:\.\d+)?)\s?(k\b)?", low)
        if m:
            add("VehiclePrice", band_vehicle_price(money(m.group(1), m.group(2) or "")), m)

    # Vehicle age.
    if re.search(r"vehicle|car\b|sedan|coupe|suv|van|truck|honda|toyota|pontiac|ford", low):
        m = re.search(r"brand[- ]new|\bnew (?:car|vehicle)\b|less than (?:a|one) year old", low)
        if m:
            add("AgeOfVehicle", "new", m)
        else:
            m = re.search(NUM + r"[- ]years?[- ]old", low)
            if m:
                add("AgeOfVehicle", band_vehicle_age(num(m.group(1))), m)
            else:
                m = re.search(r"(?:manufactured|built|first registered|made) in ((?:19|20)\d\d)", low)
                if m:
                    add("_vehicle_year", int(m.group(1)), m)

    # Policy duration before the accident / claim.
    dur_pats = [
        r"(?:policy|cover|coverage) had been (?:in force|active|running|held) for " + DUR,
        r"(?:held|had) (?:the |this |their |his |her )?(?:policy|cover) for " + DUR,
        DUR + r" after (?:the )?(?:policy|cover|coverage) (?:started|began|took effect|was issued|commenced)",
        r"(?:policy|cover) (?:was issued|started|began|took effect|commenced) " + DUR + r" (?:before|prior to)",
    ]
    for pat in dur_pats:
        m = re.search(pat, low)
        if m:
            days = num(m.group(1)) * UNIT_DAYS[m.group(2)]
            target = "Days_Policy_Claim" if is_claim and not is_accident else "Days_Policy_Accident"
            add(target, band_days(days), m)
            break
    m = re.search(r"(?:on )?the (?:same )?day the (?:policy|cover) (?:started|began|took effect)", low)
    if m:
        add("Days_Policy_Claim" if is_claim and not is_accident else "Days_Policy_Accident", "none", m)

    # Past claims (all prior claims, as in the Kaggle data).
    m = re.search(r"\bno (?:prior|previous|past|earlier|other) claims|first claim|never (?:made|filed) a claim", low)
    if m:
        add("PastNumberOfClaims", "none", m)
    else:
        m = re.search(NUM + r"\s+(?:prior|previous|past|earlier|other)\s+claims?\b", low)
        if m:
            add("PastNumberOfClaims", band_past_claims(num(m.group(1))), m)

    # Number of cars on the policy.
    m = re.search(r"(?:only|just) (?:one|1|a single) (?:vehicle|car) (?:is |was )?(?:insured|covered|on the policy)", low)
    if m:
        add("NumberOfCars", "1 vehicle", m)
    else:
        m = re.search(NUM + r"\s+(?:vehicles|cars)\s+(?:are |were )?(?:insured|covered|on (?:the|this|their) policy|registered)|insures " + NUM + r"\s+(?:vehicles|cars)", low)
        if m:
            add("NumberOfCars", band_cars(num(m.group(1) or m.group(2))), m)

    # One value per feature per sentence (the last match wins).
    seen = {}
    for f, v, span in facts:
        seen[f] = (f, v, span)
    return list(seen.values()), find_dates(sentence)


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------
KEEP = ("risk_score", "tier", "contributions", "raw_margin", "base_value", "missing_features",
        "unknown_categories", "recommendation", "model_id")


def _slim(result, claim):
    out = {k: result[k] for k in KEEP}
    out["claim"] = dict(claim)
    return out


def _fmt(f, v):
    return f"${v:,.0f}" if f == "Deductible" else str(v)


def analyse(pdf_bytes, scorer=explain.score):
    """Score the claim after each sentence, starting from an empty claim."""
    sentences = split_sentences(extract_text(pdf_bytes))
    opts = allowed_values()
    claim, dates, from_text = {}, {}, set()
    prev = scorer({})
    start = _slim(prev, claim)
    steps = []
    for i, sent in enumerate(sentences):
        raw, found_dates = find_facts(sent)
        facts = []

        def put(f, value, span, how=None):
            v, note = snap(f, value, opts)
            if v is None:
                facts.append({"feature": f, "label": LABELS[f], "value": value, "old": claim.get(f),
                              "span": span, "note": note, "used": False})
                return
            facts.append({"feature": f, "label": LABELS[f], "value": v, "old": claim.get(f),
                          "span": span, "note": note or how, "used": True})
            claim[f] = v

        for event, d, span in found_dates:
            dates[event] = d
            if event == "accident":
                put("Month", calendar.month_abbr[d.month], span, f"from {d:%d %b %Y}")
                put("DayOfWeek", WEEKDAYS[d.weekday()], span, f"{d:%d %b %Y} was a {WEEKDAYS[d.weekday()]}")
                put("WeekOfMonth", week_of_month(d), span, f"day {d.day} falls in week {week_of_month(d)}, counting 7-day blocks")
            elif event == "claim":
                put("MonthClaimed", calendar.month_abbr[d.month], span, f"from {d:%d %b %Y}")
                put("DayOfWeekClaimed", WEEKDAYS[d.weekday()], span, f"{d:%d %b %Y} was a {WEEKDAYS[d.weekday()]}")
                put("WeekOfMonthClaimed", week_of_month(d), span, f"day {d.day} falls in week {week_of_month(d)}, counting 7-day blocks")
        for f, v, span in raw:
            if f == "_vehicle_year":
                ref = dates.get("accident", date(CURRENT_YEAR, 1, 1)).year
                put("AgeOfVehicle", band_vehicle_age(ref - v), span, f"built in {v}, {ref - v} years before the accident")
                continue
            from_text.add(f)
            put(f, v, span)
        # Durations from dates, unless the text stated them.
        if "policy_start" in dates:
            for f, event in (("Days_Policy_Accident", "accident"), ("Days_Policy_Claim", "claim")):
                if event in dates and f not in from_text and f not in claim:
                    days = (dates[event] - dates["policy_start"]).days
                    put(f, band_days(days), None, f"{days} days from policy start to the {event}")

        used = [x for x in facts if x["used"] and x["value"] != x["old"]]
        cur = scorer(claim) if used else prev
        delta = cur["risk_score"] - prev["risk_score"]
        thinking = [f"Reading sentence {i + 1} of {len(sentences)}."]
        for x in facts:
            line = f"Found “{x['span']}”, so " if x["span"] else "Worked out "
            line += f"{x['label'].lower()} = {_fmt(x['feature'], x['value'])}" if x["used"] else f"{x['label'].lower()}: {x['note']}"
            if x["used"] and x["note"]:
                line += f" ({x['note']})"
            thinking.append(line + ".")
        if used:
            impact = {c["feature"]: c["impact"] for c in cur["contributions"]}
            parts = [f"{x['label']} {impact[x['feature']]:+.2f}" for x in used]
            thinking.append("Contribution in log-odds: " + ", ".join(parts) + ".")
            tier_note = "" if cur["tier"] == prev["tier"] else f" Tier {prev['tier'].replace('_', ' ')} → {cur['tier'].replace('_', ' ')}."
            change = "no visible change" if abs(delta) < 0.005 else f"{delta:+.2f}"
            if all(f in claim for f in REQUIRED):
                thinking.append(f"Score {prev['risk_score']:.2f} → {cur['risk_score']:.2f} ({change}).{tier_note}")
            else:
                thinking.append("Policy coverage not found yet, so there is no score: without it the score is not reliable.")
        else:
            thinking.append("No claim field in this sentence. Score unchanged.")
        steps.append({"index": i, "text": sent, "facts": facts, "delta": round(delta, 4),
                      "thinking": thinking, **_slim(cur, claim)})
        prev = cur
    return {"sentences": len(sentences), "required": REQUIRED, "start": start, "steps": steps, "final": _slim(prev, claim)}
