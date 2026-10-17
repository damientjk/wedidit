# Yizhuo's final documentation and video handoff

Prepared 17 October 2026 against application source `1cea778`. The frozen model, original evaluation artifacts and numerical thresholds are unchanged.

## Ethan: README integration

Ethan owns the README. The following are replacement sections for him to review and paste into the latest root README; their links are intentionally relative to the repository root.

- [Section 3: architecture](README-section-3-architecture.md)
- [Section 8: results](README-section-8-results.md)

Use the two snippets together with [the architecture](../architecture.md), [flowchart PNG](../flowchart.png), [editable flowchart SVG](../flowchart.svg) and [evaluation](../evaluation.md). Avoid replacing the whole README while other owners are editing it.

Other README statements to reconcile during Ethan's pass:

- Use Claim! consistently as the project name while retaining the required team submission title.
- Section 4 should distinguish the LinuxONE inference deployment from model development/evaluation on other platforms. Do not claim training ran on LinuxONE without evidence.
- Section 5 needs the actual service entry point and final Docker recipe from Damien. The current source serves the UI from FastAPI; it does not require opening a local HTML file with a hand-edited API URL.
- Section 9 should describe the current rule-based PDF intake, with OCR and policy verification as limitations. PDF intake is no longer entirely future work.
- Results must keep final-test model metrics separate from validation routing and the later cached-test policy reanalysis.

Ready-to-send note:

> Hey Ethan, the docs PR includes ready-to-paste README sections 3 and 8 under docs/handoff, plus the final architecture diagram and evaluation links. Please integrate those into your latest README. I also noted the run-command, training-location and PDF-description statements to reconcile during your pass. The test metrics and thresholds are unchanged.

## Video editor and Marcus: timing and narration

Use [Yizhuo's final recording script](yizhuo-recording-script.md). The opening is 30 seconds and the closing is 10 seconds. The original opening was 25 seconds, so the middle needs five seconds removed; the script includes a proposed 180-second schedule for approval. No other speaker's source script has been overwritten.

Ready-to-send note:

> My final intro uses the title and current Problem Statement slides and lasts 30 seconds; the closing lasts 10 seconds. This adds five seconds to the earlier intro allocation. Can we recover those five seconds in the middle? The handoff proposes 35 seconds for model/results, but please confirm the final allocation with Marcus before editing. I will provide the opening and closing as separate clips.

## Damien: deployment evidence

The [architecture evidence section](../architecture.md#linuxone-environment-and-evidence) links the committed LinuxONE screenshots. Historical claim-form/API parity checks remain documented. Source at `1cea778` connects the PDF reader to the final scorer, but no fresh live PDF upload/parity check was performed for this documentation update. Match the source revision, deployed service and screenshot capture before treating the latest PDF flow as verified. The held-out CSV results do not validate arbitrary document extraction.

## Remaining human handoffs

- Yizhuo records the 30-second opening and 10-second closing in their own voice and supplies the two clips.
- Ethan reviews and integrates the two README sections with his remaining README work.
- The video editor and Marcus confirm where to recover five seconds and check that the completed video is at most three minutes, including transitions.
- Damien confirms the final deployed revision and evidence capture. No additional live uploads are authorised by this handoff.

The messages above are drafts; they have not been sent to team members.
