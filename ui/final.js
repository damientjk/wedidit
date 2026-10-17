// Claim form mode: scores with Marcus's final XGBoost model (POST /score -> src/explain.py).
// Impacts are exact TreeSHAP values in log-odds, so they add up:
//   base value + sum(impacts) = log-odds,  score = 1 / (1 + e^-log-odds).
// The score is an uncalibrated ranking score, never shown as a % chance of fraud.

const F_GROUPS = [
  ["Accident", ["Month", "WeekOfMonth", "DayOfWeek", "AccidentArea"]],
  ["Claim reported", ["MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]],
  ["Policy and history", ["BasePolicy", "Deductible", "AgentType", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "DriverRating"]],
  ["Vehicle", ["Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"]],
];
const MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const BAND_ORDER = ["none", "new", "1", "1 to 7", "8 to 15", "15 to 30", "more than 30", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7",
  "2 to 4", "more than 4", "1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8",
  "less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"];
const F_DEDUCTIBLES = [300, 400, 500, 700];
const F_INTS = new Set(["WeekOfMonth", "WeekOfMonthClaimed", "DriverRating"]);

let fin = null;          // GET /model/final
let fRes = null;         // last POST /score response
let fPointer = -1;       // -1 = summary, else index into fRes.contributions
let fReq = 0;
const fExcluded = new Set();

const fmtImp = (v) => Math.abs(v) < 0.005 ? "0.00" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);

function orderOptions(f, opts) {
  const order = /Month/.test(f) && !/Week/.test(f) ? MONTH_ORDER : /DayOfWeek/.test(f) ? DAY_ORDER : BAND_ORDER;
  return [...opts].sort((a, b) => {
    const ia = order.indexOf(String(a)), ib = order.indexOf(String(b));
    if (ia >= 0 && ib >= 0) return ia - ib;
    if (ia >= 0 || ib >= 0) return ia >= 0 ? -1 : 1;
    return String(a).localeCompare(String(b), undefined, {numeric: true});
  });
}

async function bootFinal() {
  try {
    const r = await fetch("model/final");
    if (!r.ok) throw new Error(r.status);
    fin = await r.json();
  } catch (e) {
    $("#fSub").textContent = "Final model not available on this server (GET /model/final failed).";
    return;
  }
  const form = $("#fForm");
  F_GROUPS.forEach(([title, fields]) => {
    const fs = el("fieldset");
    fs.appendChild(el("legend", "", title));
    fields.filter((f) => fin.features.includes(f)).forEach((f) => {
      const row = el("div", "frow");
      row.dataset.f = f;
      const id = "f_" + f;
      const lab = el("label", "", fin.labels[f] || f);
      lab.htmlFor = id;
      let input;
      const opts = f === "Deductible" ? F_DEDUCTIBLES : (fin.options || {})[f];
      if (opts) {
        input = el("select");
        input.add(new Option("Not provided", ""));
        orderOptions(f, opts).forEach((o) => input.add(new Option(String(o), String(o))));
      } else {
        input = el("input");
        input.type = "number";
      }
      input.id = id;
      input.name = f;
      input.addEventListener("change", () => scoreFinal());
      const imp = el("button", "imp", "");
      imp.type = "button";
      imp.setAttribute("aria-label", `Switch ${fin.labels[f] || f} off or on for a what-if`);
      row.append(lab, input, imp);
      fs.appendChild(row);
    });
    form.appendChild(fs);
  });
  form.addEventListener("submit", (e) => e.preventDefault());
  const span = 0.6; // meter shows 0 to 0.6, where all the action is
  $("#fMarkR").style.left = (fin.tiers.review / span) * 100 + "%";
  $("#fMarkI").style.left = (fin.tiers.investigate / span) * 100 + "%";
  $("#fModelSub").textContent = `Final model ${fin.model_id} (XGBoost on the Kaggle claims), exact TreeSHAP contributions in log-odds`;
  scoreFinal();
}

// Blank fields are sent as null: the model fills them with the training median/mode
// and flags the claim for a data check (see src/explain.py).
function readClaimFinal() {
  const c = {};
  fin.features.forEach((f) => {
    const v = $("#fForm").elements[f].value;
    c[f] = v === "" ? null : f === "Deductible" ? Number(v) : F_INTS.has(f) ? parseInt(v, 10) : v;
  });
  return c;
}

function filledCount() {
  return fin.features.filter((f) => $("#fForm").elements[f].value !== "").length;
}

function clearForm() {
  fin.features.forEach((f) => { $("#fForm").elements[f].value = ""; });
  fExcluded.clear();
  scoreFinal();
}

// Nothing entered yet: show an empty state instead of a score built only from typical values.
function renderEmptyFinal() {
  fRes = null;
  $("#fRisk").textContent = "-";
  $("#fTier").textContent = "Waiting for details";
  $("#fAdvice").textContent = "Fill in the claim fields. The score appears after the first one and updates after each.";
  $("#fDecision").className = "decision";
  $("#fMeter").style.transform = "scaleX(0)";
  $("#fScoreR").textContent = $("#fMargin").textContent = "-";
  $("#fStep").textContent = "The decision trail appears once there is a score.";
  $("#fPrev").disabled = $("#fNext").disabled = true;
  $("#fWhatIf").hidden = true;
  $("#fReset").hidden = true;
  document.querySelectorAll(".frow").forEach((row) => {
    row.classList.add("blank");
    row.classList.remove("ex", "hot");
    row.querySelector(".imp").textContent = "";
  });
  $("#fWaterfall").textContent = "";
  $("#fThinking").textContent = "";
  const ln = el("div", "ln quiet");
  ln.textContent = `${fin.model_id} is ready. Nothing entered yet.`;
  $("#fThinking").appendChild(ln);
}

async function scoreFinal() {
  const my = ++fReq;
  const n = filledCount();
  $("#fFilled").textContent = `${n} of ${fin.features.length}`;
  $("#fFilledBar").style.transform = `scaleX(${n / fin.features.length})`;
  if (n === 0) { renderEmptyFinal(); return; }
  const body = {...readClaimFinal(), exclude: [...fExcluded]};
  let res;
  try {
    const r = await fetch("score", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.status);
  } catch (e) {
    $("#fAdvice").textContent = "Scoring failed: " + e.message;
    return;
  }
  if (my !== fReq) return;
  fRes = res;
  renderFinal();
}

function renderFinal() {
  const r = fRes;
  $("#fRisk").textContent = r.risk_score.toFixed(2);
  $("#fTier").textContent = TIER_LABEL[r.tier];
  $("#fAdvice").textContent = r.recommendation;
  $("#fAdvice").title = r.recommendation; // full text when clamped to 3 lines
  $("#fDecision").className = "decision " + r.tier;
  $("#fMeter").style.transform = `scaleX(${Math.min(1, r.risk_score / 0.6)})`;
  $("#fScoreR").textContent = r.risk_score.toFixed(2);
  $("#fMargin").textContent = r.raw_margin.toFixed(2);

  const byF = Object.fromEntries(r.contributions.map((c) => [c.feature, c]));
  document.querySelectorAll(".frow").forEach((row) => {
    const c = byF[row.dataset.f];
    const imp = row.querySelector(".imp");
    row.classList.toggle("ex", !!c && c.excluded);
    row.classList.toggle("blank", !!c && c.imputed);
    if (!c) { imp.textContent = ""; return; }
    if (c.excluded) {
      imp.textContent = "off";
      imp.className = "imp";
      imp.title = "Switched off for the what-if: the typical value is used. Click to switch back on.";
    } else if (c.imputed) {
      imp.textContent = "typ.";
      imp.className = "imp typ";
      imp.title = `Not provided: the model used the typical value (${c.value}), which adds ${fmtImp(c.impact)} log-odds.`;
    } else {
      imp.textContent = fmtImp(c.impact);
      imp.className = "imp " + (c.impact > 0.005 ? "up" : c.impact < -0.005 ? "down" : "");
      imp.title = `${fmtImp(c.impact)} log-odds. Click to switch this field off (what-if).`;
    }
  });

  // What-if note.
  const w = $("#fWhatIf");
  if (r.sensitivity_analysis) {
    const s = r.sensitivity_analysis;
    w.hidden = false;
    w.textContent = `What-if: with ${r.excluded.map((f) => fin.labels[f]).join(", ")} set to the typical value, the score goes ${s.original_risk_score.toFixed(2)} → ${r.risk_score.toFixed(2)} (${fmtImp(s.score_change)}). Hypothetical only, it does not change this claim's real priority.`;
  } else {
    w.hidden = true;
  }
  $("#fReset").hidden = fExcluded.size === 0;
  const missing = r.missing_features.length;
  if (missing) {
    const rec = r.recommendation.replace(/^Input quality requires verification before operational routing\. /, "");
    $("#fAdvice").textContent = `Provisional: ${missing} field${missing > 1 ? "s" : ""} not provided, typical values used. ${rec}`;
    $("#fAdvice").title = $("#fAdvice").textContent;
  }
  fPointer = -1;
  focusDriver(-1);
  drawWaterfall();
  writeReasoning();
}

/* ---------- decision trail: step through the drivers ---------- */
function focusDriver(k) {
  if (!fRes) return;
  const list = fRes.contributions;
  fPointer = Math.max(-1, Math.min(list.length - 1, k));
  $("#fPrev").disabled = fPointer < 0;
  $("#fNext").disabled = fPointer >= list.length - 1;
  document.querySelectorAll(".frow.hot").forEach((r) => r.classList.remove("hot"));
  document.querySelectorAll(".wf-row.hot").forEach((r) => r.classList.remove("hot"));
  if (fPointer < 0) {
    const top = list.find((c) => !c.excluded && !c.imputed);
    $("#fStep").textContent = top
      ? `Biggest driver: ${top.label} = ${top.value} (${fmtImp(top.impact)}). Press › to step through all ${list.length} fields.`
      : "All fields excluded.";
    return;
  }
  const c = list[fPointer];
  const dir = c.excluded ? "excluded from this what-if" : c.impact > 0.005 ? "raises the score" : c.impact < -0.005 ? "lowers the score" : "has almost no effect";
  $("#fStep").textContent = `Driver ${fPointer + 1} of ${list.length}: ${c.label} = ${c.value}, ${fmtImp(c.impact)} log-odds, ${dir}.` + (c.imputed ? " (value was missing, typical value used)" : "");
  const row = document.querySelector(`.frow[data-f="${c.feature}"]`);
  if (row) { row.classList.add("hot"); row.scrollIntoView({block: "nearest", behavior: "smooth"}); }
  const wf = document.querySelector(`.wf-row[data-f="${c.feature}"]`);
  if (wf) wf.classList.add("hot");
}

/* ---------- waterfall: base value -> each field -> log-odds ---------- */
function drawWaterfall() {
  const r = fRes;
  const g = $("#fWaterfall");
  g.textContent = "";
  const TOP = 11;
  const list = r.contributions.filter((c) => !c.excluded);
  const shown = list.slice(0, TOP);
  const rest = list.slice(TOP).reduce((a, c) => a + c.impact, 0);
  const rows = [{label: "Average claim (base value)", start: 0, end: r.base_value, base: true}];
  let run = r.base_value;
  shown.forEach((c) => { rows.push({label: c.imputed ? `${c.label} (not provided)` : `${c.label} = ${c.value}`, start: run, end: run + c.impact, c}); run += c.impact; });
  if (list.length > TOP) { rows.push({label: `${list.length - TOP} other fields`, start: run, end: run + rest}); run += rest; }
  rows.push({label: "This claim (log-odds)", start: 0, end: r.raw_margin, total: true});

  const vals = rows.flatMap((x) => [x.start, x.end]).concat([0, Math.log(fin.tiers.investigate / (1 - fin.tiers.investigate))]);
  const lo = Math.min(...vals) - 0.15, hi = Math.max(...vals) + 0.15;
  const L = 250, R = 590, rowH = Math.min(26, 330 / rows.length);
  const X = (v) => L + (v - lo) / (hi - lo) * (R - L);
  g.setAttribute("viewBox", `0 0 640 ${rows.length * rowH + 34}`);
  // zero line and investigate threshold in log-odds
  const thr = Math.log(fin.tiers.investigate / (1 - fin.tiers.investigate));
  g.appendChild(svg("line", {x1: X(0), x2: X(0), y1: 4, y2: rows.length * rowH + 8, stroke: "#C5CBD3"}));
  g.appendChild(svg("line", {x1: X(thr), x2: X(thr), y1: 4, y2: rows.length * rowH + 8, stroke: "#C8382F", "stroke-dasharray": "4 3"}));
  g.appendChild(svg("text", {x: X(thr), y: rows.length * rowH + 24, "font-size": 10, fill: "#A32A22", "text-anchor": "middle"}, `investigate ≥ ${fin.tiers.investigate.toFixed(3)}`));
  g.appendChild(svg("text", {x: X(0), y: rows.length * rowH + 24, "font-size": 10, fill: "#5B6472", "text-anchor": "middle"}, "0"));
  rows.forEach((x, i) => {
    const y = 6 + i * rowH;
    const grp = svg("g", {class: "wf-row"});
    if (x.c) grp.dataset.f = x.c.feature;
    grp.appendChild(svg("rect", {class: "hl", x: 0, y: y - 2, width: 640, height: rowH - 2, rx: 4, fill: "transparent"}));
    const up = x.end >= x.start;
    const fill = x.base || x.total ? "#14181F" : up ? "#C8382F" : "#2E7D4F";
    const x0 = Math.min(X(x.start), X(x.end)), w = Math.max(2, Math.abs(X(x.end) - X(x.start)));
    grp.appendChild(svg("rect", {x: x0, y: y + 3, width: w, height: rowH - 10, rx: 3, fill, opacity: x.base || x.total ? 0.85 : 1}));
    const label = x.label.length > 40 ? x.label.slice(0, 39) + "…" : x.label;
    grp.appendChild(svg("text", {x: 8, y: y + rowH / 2 + 1, "font-weight": x.base || x.total ? 600 : 400}, label));
    const num = x.base || x.total ? x.end.toFixed(2) : fmtImp(x.end - x.start);
    const nx = up || x.base || x.total ? x0 + w + 4 : x0 - 4;
    grp.appendChild(svg("text", {class: "num", x: Math.min(nx, 634), y: y + rowH / 2 + 1, "text-anchor": up || x.base || x.total ? "start" : "end"}, num));
    g.appendChild(grp);
  });
}

/* ---------- reasoning ---------- */
function writeReasoning() {
  const r = fRes;
  const box = $("#fThinking");
  box.textContent = "";
  const line = (tag, text, cls) => {
    const ln = el("div", "ln " + (cls || ""));
    if (tag) ln.appendChild(el("span", "tag", tag));
    ln.appendChild(el("span", "", text));
    box.appendChild(ln);
  };
  line("model", `${r.model_id}, XGBoost trained on the Kaggle vehicle claims, served on ${fin.platform}.`, "quiet");
  line("start", `An average claim starts at ${r.base_value.toFixed(2)} log-odds.`);
  r.contributions.filter((c) => !c.excluded && !c.imputed).slice(0, 5).forEach((c, i) => {
    line(i === 0 ? "drivers" : "", `${c.label} = ${c.value}: ${fmtImp(c.impact)} (${c.impact >= 0 ? "raises" : "lowers"} risk)`);
  });
  const t = fin.tiers;
  const rule = r.tier === "investigate" ? `≥ ${t.investigate.toFixed(3)}` : r.tier === "review" ? `≥ ${t.review.toFixed(3)} and < ${t.investigate.toFixed(3)}` : `< ${t.review.toFixed(3)}`;
  line("result", `All ${r.contributions.length} fields sum to ${r.raw_margin.toFixed(2)} log-odds, score ${r.risk_score.toFixed(3)} ${rule}, so ${TIER_LABEL[r.tier].toUpperCase()}.`, "res");
  line("", `Tier cut-offs (provisional): review \u2265 ${t.review.toFixed(3)}, investigate \u2265 ${t.investigate.toFixed(3)}.`, "quiet");
  if (r.missing_features.length) line("missing", `${r.missing_features.length} of ${r.contributions.length} fields not provided, so typical training values were used: ${r.missing_features.map((f) => fin.labels[f]).join(", ")}. Treat the score as provisional.`, "err");
  if (r.unknown_categories.length) line("check", `Unknown values: ${r.unknown_categories.join(", ")}. Verify before routing.`, "err");
  line("note", r.safety_note, "quiet");
}

/* ---------- modes ---------- */
let finalBooted = false;
function setMode(m) {
  mode = m;
  $("#pdfLeft").hidden = $("#pdfRight").hidden = m !== "pdf";
  $("#formLeft").hidden = $("#formRight").hidden = m !== "form";
  document.querySelectorAll(".mode").forEach((b) => { b.classList.toggle("on", b.dataset.mode === m); b.setAttribute("aria-pressed", b.dataset.mode === m); });
  $("#modelChip").textContent = m === "pdf" ? pdfChip : fin ? `Final model ${fin.model_id} · XGBoost · ${fin.platform}` : "Final model";
  history.replaceState(null, "", m === "pdf" ? "#pdf" : location.pathname + location.search.replace(/[?&]sample=[^&]*/, ""));
  if (m === "form" && !finalBooted) { finalBooted = true; bootFinal().then(() => setMode(mode)); }
}

document.querySelectorAll(".mode").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("#fReset").addEventListener("click", () => { fExcluded.clear(); scoreFinal(); });
$("#fClear").addEventListener("click", clearForm);
$("#fPrev").addEventListener("click", () => focusDriver(fPointer - 1));
$("#fNext").addEventListener("click", () => focusDriver(fPointer + 1));
// Click a field's impact to exclude / include it (what-if).
$("#fForm").addEventListener("click", (e) => {
  const imp = e.target.closest(".imp");
  if (!imp || !fRes) return;
  const f = imp.parentElement.dataset.f;
  if ($("#fForm").elements[f].value === "") return; // nothing to switch off
  fExcluded.has(f) ? fExcluded.delete(f) : fExcluded.add(f);
  scoreFinal();
});
document.addEventListener("keydown", (e) => {
  if (mode !== "form" || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === "ArrowLeft" && !$("#fPrev").disabled) focusDriver(fPointer - 1);
  if (e.key === "ArrowRight" && !$("#fNext").disabled) focusDriver(fPointer + 1);
});
setMode(mode);
