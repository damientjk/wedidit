"""Generate the three demo claim PDFs (fictional claims, made-up people).

Run:  python samples/make_samples.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent

CLAIMS = {
    "claim_safe.pdf": ("Motor Vehicle Accident Claim MV-1042", [
        "Claim reference MV-1042 was submitted by the policyholder on 2 October 2026.",
        "The insured vehicle is a 4-year-old Toyota Corolla used for commuting.",
        "The policyholder has held the policy for 3 years and this is the first claim on record.",
        "On 28 September 2026 the vehicle was rear-ended at a traffic light on Clementi Road.",
        "The other driver admitted fault at the scene and two witnesses gave statements.",
        "A police report was filed the same evening under reference SPF-2026-55120.",
        "The workshop estimate for the rear bumper and boot lid totals $1,850.",
        "Photographs of the damage are attached and match the workshop estimate.",
    ]),
    "claim_borderline.pdf": ("Motor Vehicle Accident Claim MV-2210", [
        "Claim reference MV-2210 was submitted online on 9 October 2026.",
        "The insured vehicle is a 9-year-old Honda Civic.",
        "The claimant reports one prior claim in the last 12 months for a cracked windscreen.",
        "The policy started four months before the incident.",
        "The vehicle was scraped along the left side while parked at a multi-storey car park.",
        "No witnesses were present and there is no CCTV footage available.",
        "No police report was made because the claimant considered the damage minor.",
        "The repair quote from the claimant's chosen workshop is $6,900.",
    ]),
    "claim_suspicious.pdf": ("Motor Vehicle Accident Claim MV-7734", [
        "Claim reference MV-7734 was submitted by J. Tan on 8 August 2026.",
        "The policy was issued six days before the reported loss.",
        "The insured vehicle is a 13-year-old BMW 5 Series bought shortly before cover began.",
        "The vehicle was reportedly struck while parked outside a shopping centre at night.",
        "The claimant states there were no witnesses and the car was repaired before an adjuster could inspect it.",
        "The claimant has filed 3 prior claims in the past 12 months with other insurers.",
        "The incident was not reported to the police.",
        "The repair invoice totals $18,400 against a market assessment of $6,400 for comparable damage.",
    ]),
}


def main():
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize, body.leading = 11, 17
    for name, (title, sentences) in CLAIMS.items():
        doc = SimpleDocTemplate(str(HERE / name), pagesize=A4, title=title)
        story = [Paragraph(title, styles["Title"]), Spacer(1, 8),
                 Paragraph("Description of claim", styles["Heading3"]),
                 Paragraph(" ".join(sentences), body),
                 Spacer(1, 12),
                 Paragraph("Fictional claim generated for the IBM Z Datathon demo.", styles["Italic"])]
        doc.build(story)
        print("wrote", name)


if __name__ == "__main__":
    main()
