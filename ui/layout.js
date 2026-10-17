// Draggable dividers: left/right split, and the height of the reasoning panel.
// Drag with mouse or touch, or focus a divider and use the arrow keys. Double-click resets.
// Sizes are remembered per browser (localStorage, optional).

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

function dragger(handle, {onMove, onKey, onReset, rows}) {
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("drag");
    document.body.classList.add("resizing");
    document.body.classList.toggle("rows", !!rows);
    const move = (ev) => onMove(ev);
    const up = () => {
      handle.classList.remove("drag");
      document.body.classList.remove("resizing", "rows");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
  handle.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 10 : 2;
    const d = {ArrowLeft: -step, ArrowUp: -step, ArrowRight: step, ArrowDown: step}[e.key];
    if (d === undefined) return;
    e.preventDefault();
    e.stopPropagation(); // don't also move the decision trail
    onKey(d);
  });
  handle.addEventListener("dblclick", onReset);
}

/* ---------- left / right split (percent of the grid width) ---------- */
(function () {
  const grid = document.querySelector(".grid");
  const handle = $("#splitV");
  const clamp = (v) => Math.max(30, Math.min(72, v));
  const apply = (pct) => {
    if (pct == null) { grid.style.removeProperty("--split"); handle.removeAttribute("aria-valuenow"); return; }
    grid.style.setProperty("--split", pct.toFixed(1) + "%");
    handle.setAttribute("aria-valuenow", Math.round(pct));
  };
  let cur = Number(store.get("wdi.split")) || null;
  apply(cur);
  handle.setAttribute("aria-valuemin", 30);
  handle.setAttribute("aria-valuemax", 72);
  const set = (pct) => { cur = clamp(pct); apply(cur); store.set("wdi.split", cur); };
  dragger(handle, {
    onMove: (e) => {
      const r = grid.getBoundingClientRect();
      const pad = parseFloat(getComputedStyle(grid).paddingLeft);
      set(((e.clientX - r.left - pad) / (r.width - 2 * pad)) * 100);
    },
    onKey: (d) => set((cur || 51) + d),
    onReset: () => { cur = null; apply(null); store.set("wdi.split", null); },
  });
})();

/* ---------- reasoning panel height (px), one per mode ---------- */
document.querySelectorAll(".split-h").forEach((handle) => {
  const view = handle.closest(".view");
  const key = "wdi.think." + view.id;
  const think = handle.nextElementSibling;
  const clamp = (h) => Math.max(100, Math.min(view.clientHeight - 300, h));
  const apply = (h) => h == null ? view.style.removeProperty("--think") : view.style.setProperty("--think", Math.round(h) + "px");
  let cur = Number(store.get(key)) || null;
  apply(cur);
  const set = (h) => { cur = clamp(h); apply(cur); store.set(key, Math.round(cur)); };
  dragger(handle, {
    rows: true,
    onMove: (e) => set(view.getBoundingClientRect().bottom - e.clientY - handle.offsetHeight / 2),
    onKey: (d) => set((cur || think.offsetHeight) - d * 6),
    onReset: () => { cur = null; apply(null); store.set(key, null); },
  });
});
