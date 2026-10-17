// Set to true to always use the built-in mock scorer.
const FORCE_MOCK = false;
const API_URL = "/score";

const FIELDS = [
  {k: "claims_last_12m", label: "Claims in last 12 months", type: "number", min: 0},
  {k: "days_policy_to_incident", label: "Days from policy start to incident", type: "number", min: 0},
  {k: "vehicle_age", label: "Vehicle age (years)", type: "number", min: 0},
  {k: "claim_amount", label: "Claim amount ($)", type: "number", min: 0},
  {k: "police_report", label: "Police report filed", type: "select", opts: [[1, "Yes"], [0, "No"]]},
];

const EXAMPLES = {
  safe:       {claims_last_12m: 0, days_policy_to_incident: 540, vehicle_age: 3,  claim_amount: 1800,  police_report: 1},
  borderline: {claims_last_12m: 2, days_policy_to_incident: 90,  vehicle_age: 8,  claim_amount: 6500,  police_report: 0},
  fraud:      {claims_last_12m: 4, days_policy_to_incident: 6,   vehicle_age: 12, claim_amount: 18500, police_report: 0},
};

const $ = (s) => document.querySelector(s);
const form = $("#claim");
const excluded = new Set();
let inflight = 0;

// Mock scorer: same response shape as POST /score, supports exclude.
function mockScore(claim, exclude) {
  const raw = {
    claims_last_12m: 0.07 * claim.claims_last_12m,
    days_policy_to_incident: claim.days_policy_to_incident < 30 ? 0.14 : claim.days_policy_to_incident < 120 ? 0.05 : -0.05,
    vehicle_age: 0.008 * (claim.vehicle_age - 5),
    claim_amount: Math.min(0.12, claim.claim_amount / 150000),
    police_report: claim.police_report ? -0.06 : 0.06,
  };
  const contributions = Object.keys(raw)
    .filter((f) => !exclude.includes(f))
    .map((f) => ({feature: f, value: claim[f], impact: +raw[f].toFixed(3)}));
  const sum = contributions.reduce((a, c) => a + c.impact, 0);
  const risk = Math.max(0.01, Math.min(0.99, 0.25 + sum));
  const tier = risk < 0.35 ? "fast_track" : risk < 0.6 ? "review" : "investigate";
  contributions.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  return {risk_score: +risk.toFixed(2), tier, contributions, excluded: exclude};
}

async function score(claim, exclude) {
  if (!FORCE_MOCK) {
    try {
      const r = await fetch(API_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...claim, exclude}),
      });
      if (r.ok) { setMode(true); return await r.json(); }
    } catch (e) { /* fall through to mock */ }
  }
  setMode(false);
  return mockScore(claim, exclude);
}

function setMode(live) {
  const b = $("#mode");
  b.textContent = live ? "live API" : "mock";
  b.classList.toggle("live", live);
}

function readClaim() {
  const c = {};
  FIELDS.forEach((f) => { c[f.k] = Number(form.elements[f.k].value) || 0; });
  return c;
}

function fill(claim) {
  FIELDS.forEach((f) => { form.elements[f.k].value = claim[f.k]; });
}

const ARC_LEN = 251.3;
const COLORS = {fast_track: "#2fbf71", review: "#f2b134", investigate: "#ef4b4b"};
const LABELS = {fast_track: "Fast track", review: "Review", investigate: "Investigate"};
const ADVICE = {
  fast_track: "Low risk. Safe to pay out quickly.",
  review: "Some warning signs. A claims officer should take a look.",
  investigate: "High risk. Send to a human investigator before paying.",
};
let shown = 0;

function animateScore(to) {
  const from = shown, t0 = performance.now();
  (function tick(t) {
    const p = Math.min(1, (t - t0) / 600);
    shown = from + (to - from) * p;
    $("#score").textContent = Math.round(shown * 100) + "%";
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}

function render(res) {
  const arc = $("#arc");
  arc.style.strokeDashoffset = ARC_LEN * (1 - res.risk_score);
  arc.style.stroke = COLORS[res.tier];
  animateScore(res.risk_score);
  const t = $("#tier");
  t.textContent = LABELS[res.tier];
  t.className = "tier " + res.tier;
  $("#advice").textContent = ADVICE[res.tier];
  renderBars(res);
}

let lastContribs = [];
function renderBars(res) {
  // Keep excluded factors visible (greyed) so they can be re-ticked.
  const active = res.contributions;
  const names = new Set(active.map((c) => c.feature));
  const gone = [...excluded].filter((f) => !names.has(f)).map((f) => ({
    feature: f, value: lastContribs.find((c) => c.feature === f)?.value ?? "", impact: 0,
  }));
  const rows = [...active, ...gone];
  if (!excluded.size) lastContribs = active;
  else active.forEach((c) => { if (!lastContribs.find((l) => l.feature === c.feature)) lastContribs.push(c); });
  const max = Math.max(0.05, ...rows.map((r) => Math.abs(r.impact)));
  const ul = $("#bars");
  ul.innerHTML = "";
  rows.forEach((r) => {
    const off = excluded.has(r.feature);
    const li = document.createElement("li");
    li.className = (off ? "off " : "") + (r.impact < 0 ? "neg" : "");
    li.innerHTML = `<div class="row"><input type="checkbox" ${off ? "" : "checked"} aria-label="Include ${r.feature}">
      <span>${(FIELDS.find((f) => f.k === r.feature) || {label: r.feature}).label}</span>
      <span class="v">${r.value} &middot; ${r.impact >= 0 ? "+" : ""}${(r.impact * 100).toFixed(0)} pts</span></div>
      <div class="bar" style="width:${(Math.abs(r.impact) / max) * 100}%"></div>`;
    li.querySelector("input").addEventListener("change", (e) => {
      e.target.checked ? excluded.delete(r.feature) : excluded.add(r.feature);
      refresh();
    });
    ul.appendChild(li);
  });
}

async function refresh() {
  const id = ++inflight;
  const res = await score(readClaim(), [...excluded]);
  if (id === inflight) render(res);
}

FIELDS.forEach((f) => {
  const l = document.createElement("label");
  l.textContent = f.label;
  const el = f.type === "select" ? document.createElement("select") : document.createElement("input");
  el.name = f.k;
  if (f.type === "select") f.opts.forEach(([v, t]) => el.add(new Option(t, v)));
  else { el.type = "number"; el.min = f.min; }
  el.addEventListener("input", refresh);
  l.appendChild(el);
  form.appendChild(l);
});

document.querySelectorAll("[data-ex]").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("[data-ex]").forEach((x) => x.classList.toggle("on", x === b));
    excluded.clear(); lastContribs = [];
    fill(EXAMPLES[b.dataset.ex]);
    refresh();
  }));

fill(EXAMPLES.borderline);
refresh();
