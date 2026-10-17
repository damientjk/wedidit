// Claim! UI: shared helpers + the PDF walkthrough. Talks to src/demo_server.py (same origin).
// POST /analyze reads the whole PDF with the rule-based reader and scores the claim with
// the final XGBoost model after every sentence. This file replays that result sentence by
// sentence so you can watch the reader fill in the 19 fields and the score move.
// Scores are uncalibrated ranking scores (0.43), never "% chance"; contributions are log-odds.

const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const TIER_LABEL = {fast_track: "Fast track", review: "Review", investigate: "Investigate"};
const MOVE = 0.005; // a chunk must move the score by at least 0.005

let A = null;             // current analysis (POST /analyze)
let chunks = [];          // steps that moved the score
let pointer = 0;          // index into chunks; chunks.length = final view
let token = 0;            // cancels an older playback
let fast = false;         // Skip pressed
let speed = 1;
let autoFast = false;     // ?fast=1
let autoStart = false;    // ?sample=... starts reading without pressing Start
let pdfState = "empty";   // empty | ready | loading | playing | paused | done
let playIdx = 0;          // sentence to resume from after Stop
let lastPdf = null;
let pdfChip = "Final model";
let pRanges = {};
const excluded = new Set();
// "form" (final.js) or "pdf" (this file). ?sample=... or #pdf opens the PDF mode.
let mode = new URLSearchParams(location.search).get("sample") || location.hash === "#pdf" ? "pdf" : "form";

// No score until the required fields are found (same rule as the claim form).
// With nothing known, the model would score an "average claim", which looks safe only because
// most claims are not fraud. Absence of evidence is not evidence of a safe claim.
const scored = (s) => ((A && A.required) || ["BasePolicy"]).every((f) => s.claim && f in s.claim);

/* ---------- helpers (also used by final.js and layout.js) ---------- */
const sc = (v) => v.toFixed(2);
const dsc = (d) => Math.abs(d) < MOVE ? "0.00" : (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(2);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const svg = (tag, attrs, text) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
const sleep = (ms) => fast ? Promise.resolve() : new Promise((r) => setTimeout(r, ms / speed));
function status(text, kind) { $("#statusText").textContent = text; $("#statusDot").className = "dot " + (kind || ""); }

/* ---------- boot ---------- */
async function boot() {
  try {
    await loadFinal(); // final.js: GET /model/final
  } catch (e) {
    status("API offline", "err");
    $("#modelChip").textContent = "Start src/demo_server.py to load the model";
    think("!", "Cannot reach the scoring API. Run: uvicorn demo_server:app --app-dir src", "err");
    return;
  }
  pdfChip = `Final model ${fin.model_id} · XGBoost · ${fin.platform}`;
  if (mode === "pdf") $("#modelChip").textContent = pdfChip;
  status(`Ready on ${fin.platform}`, "on");
  const span = 0.6; // meter shows 0 to 0.6, where all the action is
  $("#pMarkR").style.left = (fin.tiers.review / span) * 100 + "%";
  $("#pMarkI").style.left = (fin.tiers.investigate / span) * 100 + "%";
  think("i", "Final model loaded. Upload a claim PDF or pick a sample, then press Start.", "quiet");
  // ?sample=claim_suspicious.pdf autoplays a sample (handy for the video); &fast=1 skips the animation.
  const q = new URLSearchParams(location.search);
  const b = q.get("sample") && document.querySelector(`[data-sample="${CSS.escape(q.get("sample"))}"]`);
  if (b) { autoFast = !!q.get("fast"); autoStart = true; b.click(); }
}

/* ---------- loading a PDF ---------- */
// Loading a PDF (upload, drop or sample) only prepares it. Start begins reading.
function loadPdf(blob, name) {
  if (!fin) return;
  ++token;
  lastPdf = {blob, name};
  A = null;
  fast = false;
  pdfState = "ready";
  resetTrail();
  const paper = $("#paper");
  const cursor = $("#cursor");
  cursor.hidden = true;
  paper.textContent = "";
  paper.appendChild(cursor);
  const empty = el("div", "empty");
  empty.append(el("p", "", ""), el("p", "muted", "Press Start to let the reader go through it. Stop pauses, Restart begins again."));
  empty.firstChild.appendChild(el("strong", "", `Ready: ${name}`));
  paper.appendChild(empty);
  $("#docTitle").textContent = name;
  $("#docSub").textContent = "Loaded. Press Start to read it.";
  $("#thinking").textContent = "";
  think("i", `${name} loaded. Press Start to begin.`, "quiet");
  status("Ready to read", "on");
  pdfButtons();
  if (autoStart) { autoStart = false; startPdf(); }
}

function pdfButtons() {
  $("#start").disabled = !["ready", "paused"].includes(pdfState);
  $("#start").textContent = pdfState === "paused" ? "Resume" : "Start";
  $("#stop").disabled = pdfState !== "playing";
  $("#restart").disabled = !A || !["playing", "paused", "done"].includes(pdfState);
  $("#skip").disabled = !["playing", "paused"].includes(pdfState);
}

function startPdf() {
  if (pdfState === "paused") { pdfState = "playing"; pdfButtons(); play(++token, playIdx); return; }
  if (pdfState === "ready") analysePdf(lastPdf.blob, lastPdf.name);
}
function stopPdf() {
  if (pdfState !== "playing") return;
  ++token;
  pdfState = "paused";
  fast = false;
  status("Paused", "");
  think("ll", "Paused. Press Resume to continue, or Restart to begin again.", "quiet");
  pdfButtons();
}
function restartPdf() {
  if (!A) return;
  fast = false;
  $("#thinking").textContent = "";
  play(++token, 0);
}

async function analysePdf(blob, name) {
  const my = ++token;
  fast = false;
  pdfState = "loading";
  pdfButtons();
  $("#thinking").textContent = "";
  status("Extracting text", "on");
  think("i", `Opening ${name} and extracting text with pypdf...`, "quiet");
  const fd = new FormData();
  fd.append("file", blob, name);
  let res;
  try {
    const r = await fetch("analyze", {method: "POST", body: fd});
    res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.status);
  } catch (e) {
    think("!", "Could not analyse this file: " + e.message, "err");
    status("Error", "err");
    pdfState = "ready";
    pdfButtons();
    return;
  }
  if (my !== token) return;
  A = res;
  play(my);
}

function renderDoc() {
  const paper = $("#paper");
  const cursor = $("#cursor");
  paper.textContent = "";
  paper.appendChild(cursor);
  let headings = 0;
  A.steps.forEach((s) => {
    const isHeading = !/[.!?]["')]?$/.test(s.text) && s.text.length < 90;
    const box = el(isHeading ? "div" : "span", isHeading ? "sent heading" + (headings++ === 0 ? " first" : "") : "sent");
    const low = s.text.toLowerCase();
    const ranges = s.facts.filter((f) => f.span).map((f) => {
      const a = low.indexOf(f.span.toLowerCase());
      return a < 0 ? null : [a, a + f.span.length];
    }).filter(Boolean);
    s.words = [];
    let pos = 0;
    s.text.split(/(\s+)/).forEach((tok) => {
      if (!tok) return;
      if (/^\s+$/.test(tok)) { box.appendChild(document.createTextNode(" ")); pos += tok.length; return; }
      const w = el("span", "w", tok);
      const a = pos, b = pos + tok.length;
      if (ranges.some(([x, y]) => a < y && b > x)) w.classList.add("ev");
      box.appendChild(w);
      s.words.push(w);
      pos = b;
    });
    s.el = box;
    paper.appendChild(box);
    if (!isHeading) paper.appendChild(document.createTextNode(" "));
  });
  $("#docTitle").textContent = A.filename;
  $("#docSub").textContent = `${A.sentences} sentences · reading left to right, line by line`;
}

/* ---------- playback ---------- */
async function play(my, from = 0) {
  pdfState = "playing";
  pdfButtons();
  if (from === 0) {
    renderDoc();
    resetTrail();
    showState(A.start);
    $("#paperWrap").scrollTop = 0;
    think("i", `${A.sentences} sentences found. Nothing known yet, so no score until Policy coverage is found.`, "quiet");
  } else {
    think("▶", `Resuming at sentence ${from + 1}.`, "quiet");
  }
  status("Reading", "on");
  const cursor = $("#cursor");
  cursor.hidden = false;
  if (autoFast) fast = true;
  for (let i = from; i < A.steps.length; i++) {
    if (my !== token) return;
    playIdx = i;
    await readSentence(A.steps[i], my);
    if (my !== token) return;
    await applyStep(i, my);
    setRead((i + 1) / A.steps.length);
  }
  if (my !== token) return;
  cursor.hidden = true;
  finish();
}

async function readSentence(s, my) {
  s.el.classList.add("reading");
  const per = s.el.classList.contains("heading") ? 35 : 75;
  for (const w of s.words) {
    if (my !== token) return;
    moveCursor(w);
    w.classList.add("read");
    await sleep(per);
  }
  s.el.classList.remove("reading");
}

function moveCursor(w) {
  const cursor = $("#cursor");
  const x = w.offsetLeft + w.offsetWidth - 6;
  const y = w.offsetTop + w.offsetHeight - 12;
  cursor.style.transform = `translate(${x}px, ${y}px)`;
  const wrap = $("#paperWrap");
  const top = w.offsetTop + $("#paper").offsetTop;
  if (top > wrap.scrollTop + wrap.clientHeight - 90 || top < wrap.scrollTop) wrap.scrollTop = top - wrap.clientHeight / 3;
}

function kind(s) {
  if (!s.facts.some((f) => f.used)) return null;
  return s.delta > MOVE ? "up" : s.delta < -MOVE ? "down" : "noted";
}

async function applyStep(i, my) {
  const s = A.steps[i];
  const k = kind(s);
  const tag = "S" + (i + 1);
  if (!k) {
    await typeLine(tag, s.thinking[s.thinking.length - 1], "quiet", my);
    return;
  }
  const kk = scored(s) ? k : "noted";
  s.el.classList.add("fact", kk);
  s.el.querySelectorAll(".badge").forEach((b) => b.remove()); // resumed after Stop mid-step
  // Before the required field is known there is no score, so the badge just says "read".
  const badge = el("button", "badge " + kk, scored(s) ? dsc(s.delta) : "read");
  badge.setAttribute("aria-label", `Sentence ${i + 1}: score ${dsc(s.delta)}`);
  badge.addEventListener("click", () => { const c = chunks.indexOf(s); if (c >= 0) focusChunk(c); });
  s.el.appendChild(badge);
  showState(s);
  for (let j = 1; j < s.thinking.length; j++) {
    if (my !== token) return;
    const last = j === s.thinking.length - 1;
    await typeLine(j === 1 ? tag : "", s.thinking[j], last ? "res" : "", my);
  }
  await sleep(250);
}

function finish() {
  pdfState = "done";
  pdfButtons();
  fast = false;
  chunks = A.steps.filter((s) => kind(s) && scored(s) && Math.abs(s.delta) >= MOVE);
  const f = A.final;
  const found = Object.keys(f.claim).length;
  if (!scored(f)) {
    think("✗", found
      ? `Done. ${found} of ${fin.features.length} fields found, but not Policy coverage, so this claim cannot be scored. Send it for manual review and ask for the policy type.`
      : "Done. No claim fields found in this PDF, so there is nothing to score. Send it for manual review (and check it is not a scanned image).", "err");
  } else {
    think("✓", `Done. ${found} of ${fin.features.length} fields found. Final score ${sc(f.risk_score)}, ${TIER_LABEL[f.tier].toUpperCase()}. ${chunkWord(chunks.length)} moved the score. Use ‹ to trace back through them.`, "res");
    if (f.missing_features.length) {
      think("missing", `Not in the PDF, so typical values were used: ${f.missing_features.map((x) => fin.labels[x]).join(", ")}.`, "err");
    }
  }
  status(scored(f) ? "Decision reached" : "Cannot score", scored(f) ? "on" : "err");
  $("#prev").disabled = chunks.length === 0;
  excluded.clear();
  focusChunk(chunks.length);
  if (scored(f)) rescoreWhatIf();
  checkRanges();
}
const chunkWord = (n) => n === 1 ? "1 chunk" : n + " chunks";

/* ---------- decision trail (< >) ---------- */
function focusChunk(k) {
  if (!A || pdfState !== "done") return; // only after the score is done
  pointer = Math.max(0, Math.min(chunks.length, k));
  A.steps.forEach((s) => s.el.classList.remove("active"));
  $("#prev").disabled = pointer === 0;
  $("#next").disabled = pointer >= chunks.length;
  if (pointer === chunks.length) {
    $("#decLabel").textContent = "FINAL RISK RANKING SCORE";
    $("#stepText").textContent = chunks.length
      ? `Final decision. ${chunkWord(chunks.length)} moved the score. Press ‹ to step back through them.`
      : "Final decision. No sentence moved the score.";
    $("#whatif").hidden = !scored(A.final);
    showState(whatIf || A.final);
    renderRangeNote();
    return;
  }
  $("#whatif").hidden = true;
  $("#pRange").hidden = true;
  const s = chunks[pointer];
  const prev = s.index ? A.steps[s.index - 1] : A.start;
  s.el.classList.add("active");
  s.el.scrollIntoView({block: "center", behavior: "smooth"});
  $("#decLabel").textContent = `AFTER CHUNK ${pointer + 1} OF ${chunks.length}`;
  const why = s.facts.filter((f) => f.used).map((f) => `${f.label}: ${f.value}`).join(", ");
  $("#stepText").textContent = `${dsc(s.delta)} (${sc(prev.risk_score)} → ${sc(s.risk_score)}) because ${why}.`;
  showState(s);
  think("↩", `Trace chunk ${pointer + 1}: “${s.text}” ${dsc(s.delta)}.`, "quiet");
}

// Score card + waterfall for one state of the claim; the fields this sentence added are highlighted.
function showState(s) {
  setRisk(s);
  if (!scored(s)) {
    const g = $("#pWaterfall");
    g.textContent = "";
    g.setAttribute("viewBox", "0 0 640 120");
    g.appendChild(svg("text", {x: 320, y: 56, "text-anchor": "middle", "font-size": 14, fill: "#5B6472"}, "Waiting for Policy coverage before scoring."));
    g.appendChild(svg("text", {x: 320, y: 80, "text-anchor": "middle", "font-size": 12, fill: "#8D97A5"}, `${Object.keys(s.claim || {}).length} of ${fin.features.length} fields found so far.`));
    return;
  }
  drawWaterfallInto($("#pWaterfall"), s); // final.js
  const added = new Set((s.facts || []).filter((f) => f.used).map((f) => f.feature));
  document.querySelectorAll("#pWaterfall .wf-row").forEach((r) => r.classList.toggle("hot", added.has(r.dataset.f)));
}

/* ---------- what-if (final model, POST /score with exclude) ---------- */
let whatIf = null;

async function rescoreWhatIf() {
  const body = {...A.final.claim, exclude: [...excluded]};
  let res;
  try {
    const r = await fetch("score", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.status);
  } catch (e) {
    think("!", "What-if failed: " + e.message, "err");
    return;
  }
  whatIf = excluded.size ? {...res, claim: A.final.claim} : null;
  renderChips(res);
  if (pointer === chunks.length) showState(whatIf || A.final);
}

function renderChips(res) {
  const box = $("#chips");
  box.textContent = "";
  const impact = Object.fromEntries(res.contributions.map((c) => [c.feature, c.impact]));
  Object.keys(A.final.claim).sort((a, b) => Math.abs(impact[b] || 0) - Math.abs(impact[a] || 0)).forEach((f) => {
    const off = excluded.has(f);
    const lab = el("label", off ? "off" : "");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !off;
    cb.addEventListener("change", () => { cb.checked ? excluded.delete(f) : excluded.add(f); rescoreWhatIf(); });
    lab.append(cb, el("span", "", `${fin.labels[f]}: ${A.final.claim[f]}`));
    if (!off) {
      const d = impact[f] || 0;
      lab.append(el("b", d > MOVE ? "up" : d < -MOVE ? "down" : "", fmtImp(d))); // final.js, log-odds
    }
    box.appendChild(lab);
  });
  if (res.sensitivity_analysis) {
    const s = res.sensitivity_analysis;
    think("what-if", `Without ${[...excluded].map((f) => fin.labels[f]).join(", ")}: ${sc(s.original_risk_score)} → ${sc(res.risk_score)} (${dsc(s.score_change)}). Hypothetical only.`, "quiet");
  }
}

/* ---------- blanks the PDF never mentioned: could they change the tier? ---------- */
async function checkRanges() {
  pRanges = {};
  // Scored: which blank recommended fields could change the tier. Not scored: range over Policy coverage.
  const pool = scored(A.final) ? F_RECOMMENDED : (A.required || ["BasePolicy"]); // final.js
  const blank = pool.filter((f) => fin.features.includes(f) && !(f in A.final.claim));
  if (!blank.length) { renderRangeNote(); return; }
  try {
    const r = await fetch("score/range", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({claim: A.final.claim, fields: blank})});
    const res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.status);
    pRanges = res.ranges;
  } catch (e) {
    return;
  }
  renderRangeNote();
  const risky = Object.entries(pRanges).filter(([, x]) => x.crosses_tier);
  if (risky.length) think("check", `Ask for: ${risky.map(([f, x]) => `${fin.labels[f]} (score could be ${sc(x.min)} to ${sc(x.max)})`).join(", ")}. ${risky.length > 1 ? "Each could" : "It could"} change the tier.`, "err");
}

function renderRangeNote() {
  const risky = Object.entries(pRanges).filter(([, x]) => x.crosses_tier);
  const box = $("#pRange");
  if (A && !scored(A.final) && pointer === chunks.length) {
    const x = pRanges.BasePolicy;
    box.hidden = false;
    box.textContent = "Policy coverage is not in the PDF. " + (x
      ? `Depending on it, this claim would score anywhere from ${sc(x.min)} (${x.min_value}) to ${sc(x.max)} (${x.max_value})${x.crosses_tier ? ", across different tiers" : ""}. Ask for the policy type before deciding.`
      : "Ask for the policy type before deciding.");
    return;
  }
  if (!risky.length || pointer !== chunks.length) { box.hidden = true; return; }
  const lo = Math.min(...risky.map(([, x]) => x.min)), hi = Math.max(...risky.map(([, x]) => x.max));
  box.hidden = false;
  box.textContent = `The PDF does not mention ${risky.map(([f]) => fin.labels[f]).join(", ")}. ` +
    `Each one alone could move the score between ${sc(lo)} and ${sc(hi)} and change the tier. Ask the claimant.`;
}

/* ---------- risk readout ---------- */
function setRisk(s) {
  if (!scored(s)) {
    const found = Object.keys(s.claim || {}).length;
    $("#riskNum").textContent = "-";
    $("#tierPill").textContent = "Needs info";
    $("#advice").textContent = found
      ? `${found} of ${fin.features.length} fields found. No score until Policy coverage is found: it decides the score.`
      : "No claim fields found yet. No score: an empty claim would only look safe because most claims are not fraud.";
    $("#advice").title = "";
    $("#decision").className = "decision needinfo";
    $("#meterFill").style.transform = "scaleX(0)";
    $("#pScore").textContent = $("#pMargin").textContent = "-";
    return;
  }
  $("#riskNum").textContent = sc(s.risk_score);
  $("#tierPill").textContent = TIER_LABEL[s.tier];
  const missing = (s.missing_features || []).length;
  $("#advice").textContent = missing ? `Provisional: ${missing} of ${fin.features.length} fields not found yet.` : (s.recommendation || "");
  $("#advice").title = s.recommendation || "";
  $("#decision").className = "decision " + s.tier;
  $("#meterFill").style.transform = `scaleX(${Math.min(1, s.risk_score / 0.6)})`;
  $("#pScore").textContent = sc(s.risk_score);
  $("#pMargin").textContent = s.raw_margin != null ? s.raw_margin.toFixed(2) : "-";
}
function setRead(f) {
  $("#readPct").textContent = "Read " + Math.round(f * 100) + "%";
  $("#readBar").style.transform = `scaleX(${f})`;
}
function resetTrail() {
  chunks = []; pointer = 0; whatIf = null; pRanges = {}; excluded.clear();
  $("#prev").disabled = $("#next").disabled = true;
  $("#stepText").textContent = "Available once the score is done";
  $("#decLabel").textContent = "RISK RANKING SCORE";
  $("#whatif").hidden = true;
  $("#pRange").hidden = true;
  setRead(0);
}

/* ---------- reasoning stream ---------- */
async function typeLine(tag, text, cls, my) {
  const box = $("#thinking");
  const ln = el("div", "ln " + (cls || ""));
  if (tag) ln.appendChild(el("span", "tag", tag));
  const body = el("span");
  const caret = el("span", "caret");
  ln.append(body, caret);
  box.appendChild(ln);
  if (fast) body.textContent = text;
  else {
    for (let i = 0; i < text.length; i += 3) {
      if (my !== undefined && my !== token) break;
      body.textContent = text.slice(0, i + 3);
      box.scrollTop = box.scrollHeight;
      await sleep(14);
    }
    body.textContent = text;
  }
  caret.remove();
  box.scrollTop = box.scrollHeight;
  while (box.children.length > 300) box.firstChild.remove();
}
function think(tag, text, cls) { typeLine(tag, text, cls); }

/* ---------- wiring ---------- */
$("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) loadPdf(f, f.name); e.target.value = ""; });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", async () => {
  document.querySelectorAll("[data-sample]").forEach((x) => x.classList.toggle("on", x === b));
  const r = await fetch("samples/" + b.dataset.sample);
  if (!r.ok) { think("!", "Sample not found on the server.", "err"); return; }
  loadPdf(await r.blob(), b.dataset.sample);
}));
$("#start").addEventListener("click", startPdf);
$("#stop").addEventListener("click", stopPdf);
$("#restart").addEventListener("click", restartPdf);
$("#skip").addEventListener("click", () => {
  if (pdfState === "paused") { fast = true; startPdf(); } else fast = true;
});
$("#speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
$("#prev").addEventListener("click", () => focusChunk(pointer - 1));
$("#next").addEventListener("click", () => focusChunk(pointer + 1));
document.addEventListener("keydown", (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (mode !== "pdf") return; // final.js handles the form mode
  if (e.key === "ArrowLeft" && !$("#prev").disabled) focusChunk(pointer - 1);
  if (e.key === "ArrowRight" && !$("#next").disabled) focusChunk(pointer + 1);
});
const wrap = $("#paperWrap");
["dragenter", "dragover"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.remove("drag"); }));
wrap.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadPdf(f, f.name); });

// final.js and layout.js load after this file, so start once everything is parsed.
window.addEventListener("DOMContentLoaded", boot);
