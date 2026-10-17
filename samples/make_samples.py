"""Generate the three demo claim PDFs from real test-set claims.

Each PDF states all 19 fields the final model reads, in ordinary sentences, using the
field values of a real claim in data/processed/test.csv:
  claim_safe.pdf        claim_id 6180  (low priority, legitimate)
  claim_borderline.pdf  claim_id 7223  (medium priority, legitimate)
  claim_suspicious.pdf  claim_id 2857  (high priority, labelled fraud)
People, places and reference numbers are made up. Dates are chosen so that month, weekday
and week of month (ceil(day / 7)) match the claim exactly.

Run:  python samples/make_samples.py
"""
import calendar
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def find_date(month, weekday, week, after=None, years=(2026, 2025, 2027)):
    """First date (on or after `after`) with this month abbreviation, weekday and week of month."""
    m = list(calendar.month_abbr).index(month)
    for y in years:  # prefer 2026
        for d in range(1, calendar.monthrange(y, m)[1] + 1):
            dt = date(y, m, d)
            if WEEKDAYS[dt.weekday()] == weekday and min(5, (d - 1) // 7 + 1) == week and (after is None or dt >= after):
                return dt
    raise ValueError(f"no {weekday} in week {week} of {month}")


def fmt(d):
    return f"{WEEKDAYS[d.weekday()]}, {d.day} {calendar.month_name[d.month]} {d.year}"


def claim_safe():
    acc = find_date("Apr", "Wednesday", 2)
    clm = find_date("Apr", "Wednesday", 3, after=acc)
    return "Motor Vehicle Accident Claim MV-6180", [
        f"Claim reference MV-6180 was submitted on {fmt(clm)}.",
        f"The accident happened on {fmt(acc)} at a junction in the city centre, an urban area.",
        "The insured vehicle is a brand new Honda sports car.",
        "The car is valued at $72,000.",
        "The policyholder holds a liability-only policy with a $400 deductible.",
        "The policy was sold through an independent broker.",
        "The policy had been in force for 14 months when the accident happened.",
        "By the time the claim was filed, the policy had been in force for 14 months.",
        "The policyholder has made 3 previous claims.",
        "Only one car is insured on the policy.",
        "The driver rating on file is 1.",
        "The other driver admitted fault at the scene and photographs of the damage are attached.",
    ]


def claim_borderline():
    acc = find_date("May", "Thursday", 1)
    clm = find_date("May", "Thursday", 4, after=acc)
    start = acc - timedelta(days=790)
    return "Motor Vehicle Accident Claim MV-7223", [
        f"Claim reference MV-7223 was submitted on {fmt(clm)}.",
        f"The policy started on {start.day} {calendar.month_name[start.month]} {start.year}.",
        f"The accident happened on {fmt(acc)} in an urban area near a shopping mall.",
        "The insured vehicle is a 7-year-old Pontiac sedan.",
        "The vehicle is valued at $26,000.",
        "The policyholder holds an all-perils policy with a $400 deductible.",
        "The policy was sold through an independent broker.",
        "The policyholder has made 3 previous claims.",
        "Only one car is insured on the policy.",
        "The driver rating on file is 2.",
        "The vehicle was scraped along the left side while parked and no witnesses were present.",
    ]


def claim_suspicious():
    acc = find_date("Apr", "Monday", 2)
    clm = find_date("Apr", "Friday", 3, after=acc)
    return "Motor Vehicle Accident Claim MV-2857", [
        f"Claim reference MV-2857 was submitted on {fmt(clm)}.",
        f"The accident happened on {fmt(acc)} in an urban area, on a quiet street at night.",
        "The insured vehicle is a 7-year-old Pontiac sedan.",
        "The car is valued at $24,500.",
        "The policyholder holds an all-perils policy with a $500 deductible.",
        "The policy was sold through an independent broker.",
        "The policy had been in force for 2 years when the accident happened.",
        "By the time the claim was filed, the policy had been in force for 2 years.",
        "This is the policyholder's first claim.",
        "Two cars are insured on the policy.",
        "The driver rating on file is 2.",
        "The claimant states there were no witnesses and the car was repaired before an adjuster could inspect it.",
    ]


def main():
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize, body.leading = 11, 17
    for name, make in [("claim_safe.pdf", claim_safe), ("claim_borderline.pdf", claim_borderline),
                       ("claim_suspicious.pdf", claim_suspicious)]:
        title, sentences = make()
        doc = SimpleDocTemplate(str(HERE / name), pagesize=A4, title=title)
        story = [Paragraph(title, styles["Title"]), Spacer(1, 8),
                 Paragraph("Description of claim", styles["Heading3"]),
                 Paragraph(" ".join(sentences), body),
                 Spacer(1, 12),
                 Paragraph("Fictional claim document built from a real test-set claim, for the IBM Z Datathon demo.", styles["Italic"])]
        doc.build(story)
        print("wrote", name)


if __name__ == "__main__":
    main()
