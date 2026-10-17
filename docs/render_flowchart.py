"""Render Claim!'s documented architecture without loading data or models.

Run from the repository root: python docs/render_flowchart.py
Requires matplotlib. Both the PNG and editable-text SVG are written next to this file.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, title, body, color="#192b43"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                   facecolor=color, edgecolor="#54708e", linewidth=1.1))
        ax.text(x+w/2, y+h*.72, title, ha="center", va="center", fontsize=13,
                color="white", weight="bold")
        ax.text(x+w/2, y+h*.30, body, ha="center", va="center", fontsize=11,
                color="#c6d6e6", linespacing=1.5)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                    color="#79ddce", linewidth=1.6))

    ax.text(.55, 11.45, "Claim! architecture", fontsize=28, color="white", weight="bold")
    ax.text(.55, 10.95, "One final model for the claim form and rule-based PDF intake",
            fontsize=16, color="#79ddce")

    box(.65, 8.9, 4.35, 1.35, "Historical claims / fixed splits",
        "Train 9,251 / val 3,084 / test 3,084\n19 inputs; synthetic fields excluded")
    box(6.25, 8.9, 4.35, 1.35, "Train-only fitting and selection",
        "Logistic baseline and candidate models\nValidation analysis remains exploratory")
    box(12.0, 8.9, 5.25, 1.35, "Locked tuned XGBoost",
        "xgboost_09 with fitted preprocessing\nFinal test evaluated after model freeze")
    arrow((5.15, 9.575), (6.1, 9.575))
    arrow((10.75, 9.575), (11.85, 9.575))

    box(.65, 6.4, 4.35, 1.5, "Claim! intake",
        "Form: enter the 19 source fields\nPDF: pypdf text + rule-based field extraction")
    box(6.25, 6.4, 4.35, 1.5, "FastAPI / Docker / LinuxONE",
        "POST /score and POST /analyze\nPDF facts feed the same final-model scorer")
    box(12.0, 6.4, 5.25, 1.5, "explain.score()",
        "Fitted preprocessing + XGBoost inference\nNative TreeSHAP in original-field log-odds")
    arrow((5.15, 7.15), (6.1, 7.15))
    arrow((10.75, 7.15), (11.85, 7.15))
    arrow((14.625, 8.75), (14.625, 8.05))

    box(.65, 3.95, 4.35, 1.35, "Human decision",
        "Officer reviews the claim and evidence\nNo automatic approval or fraud finding", "#173c3b")
    box(6.25, 3.95, 4.35, 1.35, "Score, tier and reasons",
        "Relative ranking; hypothetical what-if\nProvisional scores and quality warnings")
    box(12.0, 3.95, 5.25, 1.35, "Saved evaluation and figures",
        "Frozen test: ROC / PR / confusion / TreeSHAP\nLater routing reanalysis stays separate")
    arrow((6.1, 4.625), (5.15, 4.625))
    ax.plot([12.8, 12.8, 8.425], [6.25, 5.8, 5.8], color="#79ddce", linewidth=1.5)
    arrow((8.425, 5.8), (8.425, 5.45))
    arrow((14.625, 6.25), (14.625, 5.45))

    box(.65, 1.55, 16.6, 1.25, "PDF and missing-field safeguards",
        "UI withholds the score until BasePolicy is known. Other missing inputs are flagged; scores remain provisional.\n"
        "Rule-based extraction is not policy verification. Scanned PDFs need manual handling; OCR is not implemented.",
        "#17313b")
    ax.text(.65, .75, "Review: 0.1754    Investigate: 0.31254968    XGBoost alone; scores are not calibrated probabilities",
            color="#bdcbdc", fontsize=12)
    ax.text(.65, .30, "Application source reviewed at 1cea778. Latest PDF path: source-reviewed; no new live-service check performed.",
            color="#bdcbdc", fontsize=11)

    folder = Path(__file__).resolve().parent
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"flowchart.{suffix}", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    svg = folder / "flowchart.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
