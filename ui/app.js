// ClaimLens demo UI. Talks to src/demo_server.py (same origin).
// The server analyses the whole PDF in one call; this file replays the result
// sentence by sentence so you can watch the reader and both models update.

const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const SHORT = {
  claims_last_12m: "claims 12m", days_policy_to_incident: "policy days",
  vehicle_age: "vehicle age", claim_amount: "amount", police_report: "police report",
};
const TIER_LABEL = {fast_track: "Fast track", review: "Review", investigate: "Investigate"};
const ADVICE = {
  fast_track: "Low risk. Safe to pay out quickly.",
  review: "Warning signs. A claims officer should check it.",
  investigate: "High risk. Refer to an investigator before paying.",
};
const MOVE = 0.005; // a chunk must move the score by at least 0.5 pts

let info = null;          // GET /model/info
let A = null;             // current analysis (POST /analyze)
let chunks = [];          // steps that moved the score
let pointer = 0;          // index into chunks; chunks.length = final view
let token = 0;            // cancels an older playback
let fast = false;         // Skip pressed
let speed = 1;
let autoFast = false;   // ?fast=1
let pinnedTree = null;    // null = auto-pick the tree that changed most
let shownTree = 0;
let lastPdf = null;
const excluded = new Set();

/* ---------- helpers ---------- */
const pct = (p) => Math.round(p * 100) + "%";
const pts = (d) => Math.round(Math.abs(d * 100)) === 0 ? "0 pts" : (d >= 0 ? "+" : "−") + Math.abs(d * 100).toFixed(0) + " pts";
const chunkWord = (n) => n === 1 ? "1 chunk" : n + " chunks";
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const svg = (tag, attrs, text) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
const sleep = (ms) => fast ? Promise.resolve() : new Promise((r) => setTimeout(r, ms / speed));
function fmtVal(f, v) {
  if (f === "claim_amount") return "$" + Math.round(v).toLocaleString();
  if (f === "police_report") return v ? "yes" : "no";
  return String(Math.round(v * 10) / 10);
}
function status(text, kind) { $("#statusText").textContent = text; $("#statusDot").className = "dot " + (kind || ""); }

/* ---------- boot ---------- */
async function boot() {
  try {
    const r = await fetch("model/info");
    if (!r.ok) throw new Error(r.status);
    info = await r.json();
  } catch (e) {
    status("API offline", "err");
    $("#modelChip").textContent = "Start src/demo_server.py to load the model";
    think("!", "Cannot reach the scoring API. Run: uvicorn demo_server:app --app-dir src", "err");
    return;
  }
  const t = info.metrics.test;
  $("#modelChip").textContent = `Model pre-trained on ${info.metrics.platform} · test AUC XGBoost ${t.xgboost.auc.toFixed(2)} · LogReg ${t.logistic_regression.auc.toFixed(2)}`;
  status(`Reader ready on ${info.platform}`, "on");
  buildTabs();
  drawSigmoid([], null);
  drawTree(info.baseline, info.baseline);
  renderTerms(null, null);
  think("i", "Model loaded. Upload a claim PDF or pick a sample to start reading.", "quiet");
  // ?sample=claim_suspicious.pdf autoplays a sample (handy for the video); &fast=1 skips the animation.
  const q = new URLSearchParams(location.search);
  const b = q.get("sample") && document.querySelector(`[data-sample="${CSS.escape(q.get("sample"))}"]`);
  if (b) { autoFast = !!q.get("fast"); b.click(); }
}

/* ---------- loading a PDF ---------- */
async function analysePdf(blob, name) {
  if (!info) return;
  lastPdf = {blob, name};
  const my = ++token;
  fast = false;
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
    const ranges = s.facts.map((f) => { const a = low.indexOf(f.span.toLowerCase()); return a < 0 ? null : [a, a + f.span.length]; }).filter(Boolean);
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
async function play(my) {
  renderDoc();
  resetTrail();
  setRisk(A.start);
  drawSigmoid([], A.start);
  drawTree(A.start.features, A.start.features);
  renderTerms(A.start.lr_terms, A.start.z_lr);
  $("#skip").disabled = false;
  $("#replay").disabled = true;
  $("#paperWrap").scrollTop = 0;
  status("Reading", "on");
  think("i", `${A.sentences} sentences found. Starting from a neutral claim at ${pct(A.start.risk_score)}.`, "quiet");
  const cursor = $("#cursor");
  cursor.hidden = false;
  if (autoFast) fast = true;
  for (let i = 0; i < A.steps.length; i++) {
    if (my !== token) return;
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
  if (!s.facts.length) return null;
  return s.delta > MOVE ? "up" : s.delta < -MOVE ? "down" : "noted";
}

async function applyStep(i, my) {
  const s = A.steps[i];
  const prev = i ? A.steps[i - 1] : A.start;
  const k = kind(s);
  const tag = "S" + (i + 1);
  if (!k) {
    await typeLine(tag, s.thinking[s.thinking.length - 1], "quiet", my);
    return;
  }
  s.el.classList.add("fact", k);
  const badge = el("button", "badge " + k, k === "noted" ? "0 pts" : pts(s.delta));
  badge.setAttribute("aria-label", `Sentence ${i + 1}: ${badge.textContent}`);
  badge.addEventListener("click", () => { const c = chunks.indexOf(s); if (c >= 0) focusChunk(c); });
  s.el.appendChild(badge);
  setRisk(s);
  drawSigmoid(A.steps.slice(0, i + 1), A.start);
  drawTree(s.features, prev.features);
  renderTerms(s.lr_terms, s.z_lr);
  for (let j = 1; j < s.thinking.length; j++) {
    if (my !== token) return;
    const last = j === s.thinking.length - 1;
    await typeLine(j === 1 ? tag : "", s.thinking[j], last ? "res" : "", my);
  }
  await sleep(250);
}

function finish() {
  $("#skip").disabled = true;
  $("#replay").disabled = false;
  fast = false;
  chunks = A.steps.filter((s) => s.facts.length && Math.abs(s.delta) >= MOVE);
  const f = A.final;
  think("✓", `Done. Final risk ${pct(f.risk_score)}, ${TIER_LABEL[f.tier].toUpperCase()}. ${chunkWord(chunks.length)} moved the score. Use ‹ to trace back through them.`, "res");
  status("Decision reached", "on");
  $("#prev").disabled = chunks.length === 0;
  excluded.clear();
  focusChunk(chunks.length);
  rescoreWhatIf();
}

/* ---------- decision trail (< >) ---------- */
function focusChunk(k) {
  if (!A || !$("#skip").disabled) return; // only after the score is done
  pointer = Math.max(0, Math.min(chunks.length, k));
  A.steps.forEach((s) => s.el.classList.remove("active"));
  $("#prev").disabled = pointer === 0;
  $("#next").disabled = pointer >= chunks.length;
  if (pointer === chunks.length) {
    $("#decLabel").textContent = "FINAL RISK SCORE";
    $("#stepText").textContent = chunks.length
      ? `Final decision. ${chunkWord(chunks.length)} moved the score. Press ‹ to step back through them.`
      : "Final decision. No sentence moved the score.";
    $("#whatif").hidden = false;
    showState(A.steps.length - 1, whatIfState());
    return;
  }
  $("#whatif").hidden = true;
  const s = chunks[pointer];
  const prev = s.index ? A.steps[s.index - 1] : A.start;
  s.el.classList.add("active");
  s.el.scrollIntoView({block: "center", behavior: "smooth"});
  $("#decLabel").textContent = `AFTER CHUNK ${pointer + 1} OF ${chunks.length}`;
  const why = s.facts.map((f) => `${f.label}: ${fmtVal(f.feature, f.value)}`).join(", ");
  $("#stepText").textContent = `${pts(s.delta)} (${pct(prev.risk_score)} → ${pct(s.risk_score)}) because ${why}.`;
  showState(s.index);
  think("↩", `Trace chunk ${pointer + 1}: “${s.text}” ${pts(s.delta)}.`, "quiet");
}

function showState(i, override) {
  const s = override || A.steps[i];
  const prev = i ? A.steps[i - 1] : A.start;
  setRisk(s);
  drawSigmoid(A.steps.slice(0, i + 1), A.start, override);
  drawTree(s.features, prev.features);
  renderTerms(s.lr_terms, s.z_lr);
}

/* ---------- what-if ---------- */
let whatIf = null;
function whatIfState() { return whatIf || A.final; }

async function rescoreWhatIf() {
  const body = {...A.final.features, exclude: [...excluded]};
  const r = await fetch("score", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const res = await r.json();
  const feats = {...A.final.features};
  excluded.forEach((f) => { feats[f] = info.baseline[f]; });
  whatIf = {...res, features: feats};
  renderChips(res);
  if (pointer === chunks.length) showState(A.steps.length - 1, whatIf);
}

function renderChips(res) {
  const box = $("#chips");
  box.textContent = "";
  const impact = Object.fromEntries(res.contributions.map((c) => [c.feature, c.impact]));
  info.features.forEach((f) => {
    const off = excluded.has(f);
    const lab = el("label", off ? "off" : "");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !off;
    cb.addEventListener("change", () => { cb.checked ? excluded.delete(f) : excluded.add(f); rescoreWhatIf(); });
    lab.append(cb, el("span", "", `${info.labels[f]}: ${fmtVal(f, A.final.features[f])}`));
    if (!off) {
      const d = impact[f] || 0;
      lab.append(el("b", d > MOVE ? "up" : d < -MOVE ? "down" : "", pts(d)));
    }
    box.appendChild(lab);
  });
}

/* ---------- risk readout ---------- */
function setRisk(s) {
  $("#riskNum").textContent = pct(s.risk_score);
  $("#tierPill").textContent = TIER_LABEL[s.tier];
  $("#advice").textContent = ADVICE[s.tier];
  $("#decision").className = "decision " + s.tier;
  $("#meterFill").style.transform = `scaleX(${s.risk_score})`;
  $("#pLr").textContent = s.p_lr != null ? s.p_lr.toFixed(2) : "-";
  $("#pXgb").textContent = s.p_xgb != null ? s.p_xgb.toFixed(2) : "-";
}
function setRead(f) {
  $("#readPct").textContent = "Read " + Math.round(f * 100) + "%";
  $("#readBar").style.transform = `scaleX(${f})`;
}
function resetTrail() {
  chunks = []; pointer = 0; whatIf = null; excluded.clear();
  $("#prev").disabled = $("#next").disabled = true;
  $("#stepText").textContent = "Available once the score is done";
  $("#decLabel").textContent = "RISK SCORE";
  $("#whatif").hidden = true;
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

/* ---------- logistic regression panel ---------- */
const Z0 = -5, Z1 = 5, SX0 = 42, SX1 = 410, SY0 = 186, SY1 = 12;
const zx = (z) => SX0 + (Math.max(Z0, Math.min(Z1, z)) - Z0) / (Z1 - Z0) * (SX1 - SX0);
const py = (p) => SY0 - p * (SY0 - SY1);
const sig = (z) => 1 / (1 + Math.exp(-z));
function curve(za, zb) {
  const pts_ = [];
  const n = 40;
  for (let i = 0; i <= n; i++) { const z = za + (zb - za) * i / n; pts_.push(`${zx(z).toFixed(1)},${py(sig(z)).toFixed(1)}`); }
  return "M" + pts_.join(" L");
}
function drawSigmoid(steps, start, override) {
  const g = $("#sigmoid");
  g.textContent = "";
  const tiers = info ? info.tiers : {review: 0.3, investigate: 0.6};
  g.appendChild(svg("rect", {x: SX0, y: SY1, width: SX1 - SX0, height: py(tiers.investigate) - SY1, fill: "#FBE9E7"}));
  g.appendChild(svg("rect", {x: SX0, y: py(tiers.investigate), width: SX1 - SX0, height: py(tiers.review) - py(tiers.investigate), fill: "#FDF4E1"}));
  [0, 0.5, 1].forEach((p) => {
    g.appendChild(svg("line", {x1: SX0, x2: SX1, y1: py(p), y2: py(p), stroke: "#E6E9EE"}));
    g.appendChild(svg("text", {x: SX0 - 6, y: py(p) + 3, "font-size": 10, fill: "#5B6472", "text-anchor": "end"}, String(p)));
  });
  [-4, -2, 0, 2, 4].forEach((z) => g.appendChild(svg("text", {x: zx(z), y: SY0 + 14, "font-size": 10, fill: "#5B6472", "text-anchor": "middle"}, String(z))));
  g.appendChild(svg("text", {x: (SX0 + SX1) / 2, y: SY0 + 24, "font-size": 9.5, fill: "#5B6472", "text-anchor": "middle"}, "z, log-odds from the facts read so far"));
  g.appendChild(svg("text", {x: SX0 + 6, y: py(tiers.investigate) - 5, "font-size": 9.5, fill: "#A32A22", "font-weight": 500}, `Investigate ≥ ${tiers.investigate}`));
  g.appendChild(svg("text", {x: SX0 + 6, y: py(tiers.review) - 5, "font-size": 9.5, fill: "#8A5200", "font-weight": 500}, `Review ≥ ${tiers.review}`));
  g.appendChild(svg("path", {d: curve(Z0, Z1), fill: "none", stroke: "#9AA4B2", "stroke-width": 2.5}));
  if (!start || start.z_lr == null) return;
  const facts = steps.filter((s) => s.facts.length);
  const cur = override || facts[facts.length - 1] || start;
  g.appendChild(svg("path", {d: curve(start.z_lr, cur.z_lr), fill: "none", stroke: "#0E7C86", "stroke-width": 4, "stroke-linecap": "round"}));
  g.appendChild(svg("circle", {cx: zx(start.z_lr), cy: py(sig(start.z_lr)), r: 4.5, fill: "#fff", stroke: "#0E7C86", "stroke-width": 2}));
  facts.forEach((s, n) => {
    const k = kind(s);
    const fill = k === "up" ? "#C8382F" : k === "down" ? "#2E7D4F" : "#9AA4B2";
    g.appendChild(svg("circle", {cx: zx(s.z_lr), cy: py(sig(s.z_lr)), r: 8, fill}));
    g.appendChild(svg("text", {x: zx(s.z_lr), y: py(sig(s.z_lr)) + 3.5, "font-size": 9, "font-weight": 600, fill: "#fff", "text-anchor": "middle"}, String(n + 1)));
  });
  const cx = zx(cur.z_lr), cy = py(sig(cur.z_lr));
  g.appendChild(svg("circle", {cx, cy, r: 14, fill: "none", stroke: "#14181F", "stroke-width": 1.5}));
  const right = cx > 300;
  g.appendChild(svg("text", {x: right ? cx - 18 : cx + 18, y: cy - 10, "font-size": 11, "font-weight": 600, fill: "#14181F", "text-anchor": right ? "end" : "start"},
    `z = ${cur.z_lr.toFixed(2)}, p = ${sig(cur.z_lr).toFixed(2)}`));
}

function renderTerms(terms, z) {
  const ul = $("#terms");
  ul.textContent = "";
  if (!info) return;
  const SCALE = 3;
  info.features.forEach((f) => {
    const v = terms ? terms[f] : 0;
    const li = el("li");
    li.appendChild(el("span", "", SHORT[f]));
    const track = el("div", "track");
    track.appendChild(el("div", "mid"));
    const fill = el("div", "fill");
    const w = Math.min(50, Math.abs(v) / SCALE * 50);
    // Bar starts at the centre line; a negative scale flips it to the left.
    fill.style.transform = `scaleX(${(v >= 0 ? 1 : -1) * w / 50})`;
    fill.style.background = v >= 0 ? "#C8382F" : "#2E7D4F";
    track.appendChild(fill);
    li.appendChild(track);
    li.appendChild(el("span", "num", (v >= 0 ? "+" : "") + v.toFixed(2)));
    ul.appendChild(li);
  });
  $("#zLine").textContent = z == null ? "" : `z = ${info.lr.intercept.toFixed(2)} + Σ terms = ${z.toFixed(2)}`;
}

/* ---------- XGBoost tree panel ---------- */
function buildTabs() {
  const box = $("#treeTabs");
  box.textContent = "";
  const auto = el("button", "auto on", "Auto");
  auto.addEventListener("click", () => { pinnedTree = null; refreshTabs(); redrawTree(); });
  box.appendChild(auto);
  info.trees.forEach((_, i) => {
    const b = el("button", "", "T" + (i + 1));
    b.setAttribute("aria-label", "Show tree " + (i + 1));
    b.addEventListener("click", () => { pinnedTree = i; refreshTabs(); redrawTree(); });
    box.appendChild(b);
  });
}
function refreshTabs() {
  [...$("#treeTabs").children].forEach((b, i) => b.classList.toggle("on", i === 0 ? pinnedTree === null : pinnedTree === i - 1));
}
function leafOf(node, feats) {
  const path = [node];
  while (!("leaf" in node)) {
    const next = feats[node.split] < node.split_condition ? node.yes : node.no;
    node = node.children.find((c) => c.nodeid === next);
    path.push(node);
  }
  return path;
}
let lastTreeArgs = null;
function redrawTree() { if (lastTreeArgs) drawTree(...lastTreeArgs, true); }

function drawTree(feats, prevFeats, keep) {
  if (!info) return;
  lastTreeArgs = [feats, prevFeats];
  if (pinnedTree !== null) shownTree = pinnedTree;
  else if (!keep) {
    let best = -1, bestI = shownTree;
    info.trees.forEach((t, i) => {
      const a = leafOf(t, feats).at(-1).leaf, b = leafOf(t, prevFeats).at(-1).leaf;
      if (Math.abs(a - b) > best + 1e-9) { best = Math.abs(a - b); bestI = i; }
    });
    if (best > 1e-9) shownTree = bestI;
  }
  const tree = info.trees[shownTree];
  $("#treeNo").textContent = `${shownTree + 1} of ${info.metrics.n_trees || "all"}`;
  const path = leafOf(tree, feats);
  const onIds = new Set(path.map((n) => n.nodeid));
  const prevLeaf = leafOf(tree, prevFeats).at(-1);

  // Lay out: leaves left to right, parents centred over children.
  let leafIdx = 0, maxDepth = 0;
  const nodes = [];
  (function lay(n, d) {
    maxDepth = Math.max(maxDepth, d);
    if ("leaf" in n) n._x = leafIdx++;
    else { n.children.sort((a, b) => (a.nodeid === n.yes ? -1 : 1)); n.children.forEach((c) => lay(c, d + 1)); n._x = (n.children[0]._x + n.children[1]._x) / 2; }
    n._d = d;
    nodes.push(n);
  })(tree, 0);
  const W = 640, H = 230;
  const X = (x) => 44 + (leafIdx > 1 ? x / (leafIdx - 1) : 0.5) * (W - 88);
  const Y = (d) => 20 + d * ((H - 44) / Math.max(1, maxDepth));
  const g = $("#treeSvg");
  g.textContent = "";
  nodes.forEach((n) => {
    if ("leaf" in n) return;
    n.children.forEach((c) => {
      const on = onIds.has(n.nodeid) && onIds.has(c.nodeid);
      g.appendChild(svg("path", {class: "edge" + (on ? " on" : ""), d: `M${X(n._x)},${Y(n._d) + 12} C${X(n._x)},${Y(n._d) + 34} ${X(c._x)},${Y(c._d) - 34} ${X(c._x)},${Y(c._d) - 12}`}));
      const mx = (X(n._x) + X(c._x)) / 2, my = (Y(n._d) + Y(c._d)) / 2;
      g.appendChild(svg("text", {class: "elabel", x: mx + (c.nodeid === n.yes ? -10 : 10), y: my, "text-anchor": c.nodeid === n.yes ? "end" : "start"}, c.nodeid === n.yes ? "yes" : "no"));
    });
  });
  nodes.forEach((n) => {
    const leaf = "leaf" in n;
    const label = leaf ? (n.leaf >= 0 ? "+" : "") + n.leaf.toFixed(2) : splitLabel(n);
    const w = leaf ? 52 : Math.max(86, label.length * 6.4 + 14);
    const on = onIds.has(n.nodeid);
    const grp = svg("g", {class: "node" + (leaf ? " leaf" : "") + (on ? (leaf ? " on hit" : " on") : "")});
    grp.appendChild(svg("rect", {x: X(n._x) - w / 2, y: Y(n._d) - 12, width: w, height: 24, rx: 6}));
    grp.appendChild(svg("text", {x: X(n._x), y: Y(n._d) + 4, "text-anchor": "middle"}, label));
    g.appendChild(grp);
  });
  const leaf = path.at(-1).leaf;
  const changed = prevLeaf.nodeid !== path.at(-1).nodeid;
  $("#treeNote").textContent = `This tree adds ${leaf >= 0 ? "+" : ""}${leaf.toFixed(3)} to the claim's log-odds` +
    (changed ? ` (was ${prevLeaf.leaf >= 0 ? "+" : ""}${prevLeaf.leaf.toFixed(3)} before this fact).` : ".") +
    ` XGBoost sums all trees, then applies the sigmoid.`;
}
function splitLabel(n) {
  const c = n.split_condition;
  if (n.split === "police_report") return "no police report?";
  if (n.split === "claim_amount") return `amount < $${Math.round(c).toLocaleString()}?`;
  return `${SHORT[n.split] || n.split} < ${Math.round(c * 10) / 10}?`;
}

/* ---------- wiring ---------- */
$("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) analysePdf(f, f.name); e.target.value = ""; });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", async () => {
  document.querySelectorAll("[data-sample]").forEach((x) => x.classList.toggle("on", x === b));
  const r = await fetch("samples/" + b.dataset.sample);
  if (!r.ok) { think("!", "Sample not found on the server.", "err"); return; }
  analysePdf(await r.blob(), b.dataset.sample);
}));
$("#skip").addEventListener("click", () => { fast = true; });
$("#replay").addEventListener("click", () => { if (A) { fast = false; $("#thinking").textContent = ""; play(++token); } });
$("#speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
$("#prev").addEventListener("click", () => focusChunk(pointer - 1));
$("#next").addEventListener("click", () => focusChunk(pointer + 1));
document.addEventListener("keydown", (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === "ArrowLeft" && !$("#prev").disabled) focusChunk(pointer - 1);
  if (e.key === "ArrowRight" && !$("#next").disabled) focusChunk(pointer + 1);
});
const wrap = $("#paperWrap");
["dragenter", "dragover"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.remove("drag"); }));
wrap.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analysePdf(f, f.name); });

boot();
