/* ModelMate frontend — vanilla JS, talks to the FastAPI backend on the same origin. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`);
const fmtNum = (v) => (v === null || v === undefined ? "—" : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 })) : esc(v));
const METRIC_LABEL = { roc_auc: "ROC AUC", accuracy: "Accuracy", f1: "F1", precision: "Precision", recall: "Recall",
  f1_macro: "F1 (macro)", balanced_accuracy: "Balanced acc.", r2: "R²", rmse: "RMSE", mae: "MAE" };
const TASK_LABEL = { binary: "Yes/No classification", multiclass: "Multi-class classification", regression: "Numeric prediction" };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res;
}
const getJSON = async (p, o) => (await api(p, o)).json();

let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (isError ? " error" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), isError ? 6000 : 3000);
}

/* ---------------------------------------------------------------- navigation */
function showScreen(name) {
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.screen === name));
  $$(".screen").forEach((s) => s.classList.toggle("active", s.id === `screen-${name}`));
  if (name === "models") loadModels();
}
$$(".nav-btn").forEach((b) => b.addEventListener("click", () => showScreen(b.dataset.screen)));

/* ================================================================ SCREEN 1 */
const state = { ds: null, details: null, page: 0, pageSize: 50, total: 0, edited: new Set(), stale: { stats: true, charts: true } };

const dz = $("#dropzone"), fileInput = $("#file-input");
$("#browse-btn").addEventListener("click", () => fileInput.click());
$("#replace-btn").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && uploadDataset(fileInput.files[0]));
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && uploadDataset(ev.dataTransfer.files[0]));

async function uploadDataset(file) {
  const fd = new FormData(); fd.append("file", file);
  dz.classList.add("busy");
  try {
    const r = await getJSON("/api/datasets", { method: "POST", body: fd });
    Object.assign(state, { ds: r.dataset_id, details: r.details, page: 0, edited: new Set(), stale: { stats: true, charts: true } });
    $("#dropzone").hidden = true; $("#dataset-area").hidden = false;
    $("#ds-name").textContent = r.details.file_name;
    $("#ds-shape").textContent = `${r.details.rows.toLocaleString()} rows × ${r.details.columns} columns`;
    $("#job").hidden = true;
    fillTargetSelect(r.details.column_types);
    if (!$("#m-name").value) $("#m-name").value = file.name.replace(/\.[^.]+$/, "") + " model";
    selectTab("data");
    renderDetails(r.details);
    await loadPage(0);
    toast("File uploaded");
  } catch (e) { toast(e.message, true); }
  finally { dz.classList.remove("busy"); fileInput.value = ""; }
}

/* tabs */
$$(".tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));
async function selectTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-body").forEach((b) => (b.hidden = b.id !== `tab-${name}`));
  try {
    if (name === "details") renderDetails(await getJSON(`/api/datasets/${state.ds}/details`));
    if (name === "stats" && state.stale.stats) { renderStats((await getJSON(`/api/datasets/${state.ds}/stats`)).stats); state.stale.stats = false; }
    if (name === "charts" && state.stale.charts) { renderCharts((await getJSON(`/api/datasets/${state.ds}/charts`)).charts); state.stale.charts = false; }
  } catch (e) { toast(e.message, true); }
}

/* --- Tab 1: editable data grid */
const kindOf = (col) => state.details?.column_types.find((c) => c.name === col)?.kind;
async function loadPage(page) {
  const r = await getJSON(`/api/datasets/${state.ds}/rows?offset=${page * state.pageSize}&limit=${state.pageSize}`);
  state.page = page; state.total = r.total;
  const head = `<tr><th>#</th>${r.columns.map((c) => `<th class="${kindOf(c) === "numeric" ? "num" : ""}">${esc(c)}</th>`).join("")}</tr>`;
  const body = r.rows.map((row, i) => {
    const idx = r.row_index[i];
    return `<tr><td class="rownum">${idx + 1}</td>${r.columns.map((c) => {
      const v = row[c], isNull = v === null;
      const cls = [kindOf(c) === "numeric" ? "num" : "", isNull ? "null" : "", state.edited.has(`${idx}|${c}`) ? "edited" : ""].join(" ");
      return `<td contenteditable="true" spellcheck="false" class="${cls}" data-row="${idx}" data-col="${esc(c)}">${isNull ? "" : esc(v)}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("#data-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  const pages = Math.max(1, Math.ceil(r.total / state.pageSize));
  $("#page-info").textContent = `Page ${page + 1} of ${pages.toLocaleString()} · rows ${(page * state.pageSize + 1).toLocaleString()}–${Math.min((page + 1) * state.pageSize, r.total).toLocaleString()}`;
  $("#prev-page").disabled = page === 0; $("#next-page").disabled = page >= pages - 1;
}
$("#prev-page").addEventListener("click", () => loadPage(state.page - 1).catch((e) => toast(e.message, true)));
$("#next-page").addEventListener("click", () => loadPage(state.page + 1).catch((e) => toast(e.message, true)));

const table = $("#data-table");
table.addEventListener("focusin", (e) => { if (e.target.matches("td[contenteditable]")) e.target.dataset.orig = e.target.textContent; });
table.addEventListener("keydown", (e) => {
  if (!e.target.matches("td[contenteditable]")) return;
  if (e.key === "Enter") { e.preventDefault(); e.target.blur(); }
  if (e.key === "Escape") { e.target.textContent = e.target.dataset.orig; e.target.blur(); }
});
table.addEventListener("focusout", async (e) => {
  const td = e.target;
  if (!td.matches("td[contenteditable]") || td.textContent === td.dataset.orig) return;
  const edit = { row: +td.dataset.row, column: td.dataset.col, value: td.textContent.trim() };
  try {
    await api(`/api/datasets/${state.ds}/cells`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify([edit]) });
    state.edited.add(`${edit.row}|${edit.column}`);
    td.classList.add("edited"); td.classList.remove("error"); td.classList.toggle("null", edit.value === "");
    state.stale = { stats: true, charts: true };
  } catch (err) { td.classList.add("error"); td.textContent = td.dataset.orig; toast(err.message, true); }
});

/* --- Tab 2: file details */
function renderDetails(d) {
  state.details = d;
  const item = (k, v, wide) => `<div class="${wide ? "wide" : ""}"><span>${k}</span><strong>${v}</strong></div>`;
  $("#tab-details").innerHTML = `
    <div class="kv">
      ${item("File name", esc(d.file_name))}${item("File type", esc(d.file_type))}${item("File size", fmtBytes(d.file_size_bytes))}
      ${item("Rows", d.rows.toLocaleString())}${item("Columns", d.columns)}${item("Memory in use", fmtBytes(d.memory_usage_bytes))}
      ${item("Missing cells", `${d.missing_cells.toLocaleString()} (${d.missing_pct}%)`)}${item("Duplicate rows", d.duplicate_rows.toLocaleString())}
      ${item("Cells edited", d.edited_cells)}${item("Uploaded at", new Date(d.uploaded_at).toLocaleString())}
      ${item("Stored path", esc(d.stored_path), true)}
    </div>
    <h3>Columns</h3>
    <div class="table-wrap"><table class="grid"><thead><tr><th>Column</th><th>Detected type</th><th>Storage type</th></tr></thead><tbody>
      ${d.column_types.map((c) => `<tr><td>${esc(c.name)}</td><td><span class="kind ${c.kind}">${c.kind}</span></td><td class="muted">${esc(c.pandas_dtype)}</td></tr>`).join("")}
    </tbody></table></div>`;
}

/* --- Tab 3: descriptive statistics */
function renderStats(stats) {
  const cols = ["count", "missing", "missing_pct", "unique", "mean", "std", "min", "p25", "median", "p75", "max", "skew", "top", "top_freq"];
  const label = { missing_pct: "missing %", p25: "25%", p75: "75%", top: "most common", top_freq: "freq" };
  $("#stats-table").innerHTML = `<thead><tr><th>Feature</th><th>Type</th>${cols.map((c) => `<th class="num">${label[c] || c}</th>`).join("")}</tr></thead>
    <tbody>${stats.map((s) => `<tr><td><strong>${esc(s.column)}</strong></td><td><span class="kind ${s.kind}">${s.kind}</span></td>
    ${cols.map((c) => `<td class="num">${fmtNum(s[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

/* --- Tab 4: charts (dependency-free SVG) */
function renderCharts(charts) {
  $("#charts").innerHTML = charts.map((c) => `
    <div class="chart-card">
      <h4><span>${esc(c.column)}</span><span class="kind ${c.kind}">${c.kind}</span></h4>
      <div class="sub">${{ histogram: "Distribution (histogram)", bar: "Most frequent values", timeline: "Records over time" }[c.chart]}${c.missing ? ` · ${c.missing.toLocaleString()} missing` : ""}</div>
      ${c.chart === "bar" ? hBars(c) : vBars(c)}
    </div>`).join("") || `<p class="muted">No chartable columns.</p>`;
}
function vBars({ labels, values }) {
  const W = 320, H = 170, pad = { l: 34, b: 34, t: 6, r: 4 };
  const max = Math.max(...values, 1), n = values.length, bw = (W - pad.l - pad.r) / n;
  const y = (v) => H - pad.b - (v / max) * (H - pad.b - pad.t);
  const bars = values.map((v, i) => `<rect class="bar" x="${pad.l + i * bw + 1}" y="${y(v)}" width="${Math.max(bw - 2, 1)}" height="${H - pad.b - y(v)}"><title>${esc(labels[i])}: ${v.toLocaleString()}</title></rect>`).join("");
  const step = Math.ceil(n / 4);
  const xl = labels.map((l, i) => (i % step === 0 ? `<text x="${pad.l + i * bw + bw / 2}" y="${H - pad.b + 13}" text-anchor="middle">${esc(String(l).split(" – ")[0])}</text>` : "")).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img"><text x="${pad.l - 4}" y="${pad.t + 8}" text-anchor="end">${max.toLocaleString()}</text><text x="${pad.l - 4}" y="${H - pad.b}" text-anchor="end">0</text>${bars}${xl}</svg>`;
}
function hBars({ labels, values }) {
  const W = 320, row = 18, lw = 110, H = labels.length * row + 4, max = Math.max(...values, 1);
  const bars = labels.map((l, i) => {
    const w = (values[i] / max) * (W - lw - 50), y = i * row + 2;
    const name = String(l).length > 17 ? String(l).slice(0, 16) + "…" : l;
    return `<text x="${lw - 6}" y="${y + 12}" text-anchor="end">${esc(name)}</text>
      <rect class="bar" x="${lw}" y="${y + 2}" width="${Math.max(w, 1)}" height="${row - 5}" rx="2"><title>${esc(l)}: ${values[i].toLocaleString()}</title></rect>
      <text x="${lw + w + 4}" y="${y + 12}">${values[i].toLocaleString()}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${bars}</svg>`;
}

/* --- training form */
function fillTargetSelect(cols) {
  const sel = $("#m-target");
  const guess = cols.find((c) => /churn|target|label|response|outcome|lapse/i.test(c.name)) || cols[cols.length - 1];
  sel.innerHTML = `<option value="" disabled>Select a column…</option>` +
    cols.map((c) => `<option value="${esc(c.name)}" ${c.name === guess.name ? "selected" : ""}>${esc(c.name)} (${c.kind})</option>`).join("");
  updateTargetHint();
}
$("#m-target").addEventListener("change", updateTargetHint);
function updateTargetHint() {
  const c = state.details?.column_types.find((x) => x.name === $("#m-target").value);
  $("#target-hint").textContent = !c ? "" : c.kind === "numeric"
    ? "Numeric column: ModelMate will predict a number (or a class, if it has only a few distinct values)."
    : "ModelMate will predict which category each row belongs to, with a probability.";
}

let pollTimer;
$("#train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { dataset_id: state.ds, name: $("#m-name").value.trim(), description: $("#m-desc").value.trim(), target: $("#m-target").value };
  if (!body.target) return toast("Choose the column to predict.", true);
  $("#train-btn").disabled = true;
  try {
    const { job_id } = await getJSON("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("#job").hidden = false; $("#job-result").innerHTML = ""; $("#job-log").innerHTML = "";
    clearInterval(pollTimer); pollTimer = setInterval(() => pollJob(job_id), 1000); pollJob(job_id);
  } catch (err) { toast(err.message, true); $("#train-btn").disabled = false; }
});

async function pollJob(id) {
  let j;
  try { j = await getJSON(`/api/jobs/${id}`); } catch (e) { clearInterval(pollTimer); toast(e.message, true); return; }
  $("#job-bar").style.width = `${j.progress}%`;
  $("#job-status").textContent = `${j.status[0].toUpperCase() + j.status.slice(1)} · ${j.progress}%`;
  $("#job-log").innerHTML = j.log.map((l) => `<li>${esc(l.message)}</li>`).join("");
  $("#job-log").scrollTop = 1e6;
  if (j.status === "completed" || j.status === "failed") {
    clearInterval(pollTimer); $("#train-btn").disabled = false;
    if (j.status === "failed") { $("#job-result").innerHTML = `<div class="result-box fail"><strong>Training failed</strong><p class="small">${esc(j.error)}</p></div>`; return; }
    const r = j.report;
    $("#job-result").innerHTML = `<div class="result-box">
      <div class="small muted">Best model</div><strong>${esc(r.algorithm)}</strong>
      <div class="score">${fmtNum(r.primary_score)}</div><div class="small muted">${METRIC_LABEL[r.primary_metric]} (cross-validated)</div>
      <button class="btn primary" style="margin-top:10px;width:100%" id="goto-model">Use this model to predict →</button></div>`;
    $("#goto-model").addEventListener("click", () => { showScreen("models"); openModel(j.model_id); });
    toast("Model trained and saved");
  }
}

/* ================================================================ SCREEN 2 */
let currentModel = null;
$("#refresh-models").addEventListener("click", () => loadModels());

async function loadModels() {
  try {
    const { models } = await getJSON("/api/models");
    $("#model-list").innerHTML = models.length ? models.map(modelCard).join("")
      : `<div class="empty">No models yet. Upload a dataset on the <a href="#" class="link" onclick="showScreen('uploader');return false">Data uploader</a> screen and hit TRAIN MODEL.</div>`;
    $$(".model-card").forEach((c) => c.addEventListener("click", () => openModel(c.dataset.id)));
    if (currentModel && !models.some((m) => m.id === currentModel)) { currentModel = null; $("#model-detail").hidden = true; }
    markSelected();
  } catch (e) { toast(e.message, true); }
}
function modelCard(m) {
  const feats = m.schema.map((f) => `<span class="chip" title="${esc(f.kind)}">${esc(f.name)}</span>`).join("");
  return `<button class="model-card" data-id="${m.id}">
    <div><div class="title">${esc(m.name)}</div><div class="desc">${esc(m.description) || "<i>No description</i>"}</div>
      <div class="meta">ID <b>${m.id}</b> · ${new Date(m.created_at).toLocaleDateString()}</div></div>
    <div class="meta">Predicts <b>${esc(m.target)}</b><br>${TASK_LABEL[m.task_type]}<br>Algorithm <b>${esc(m.algorithm)}</b></div>
    <div><div class="meta" style="margin-bottom:4px">Input schema · <b>${m.n_features}</b> columns</div><div class="chips">${feats}</div></div>
    <div class="scorebox"><div class="score">${fmtNum(m.primary_score)}</div><div class="meta">${METRIC_LABEL[m.primary_metric]}</div></div>
  </button>`;
}
const markSelected = () => $$(".model-card").forEach((c) => c.classList.toggle("selected", c.dataset.id === currentModel));

async function openModel(id) {
  currentModel = id; markSelected();
  const box = $("#model-detail");
  box.hidden = false; box.innerHTML = `<p class="muted">Loading model…</p>`;
  try {
    await getJSON(`/api/models/${id}/load`, { method: "POST" });
    const m = await getJSON(`/api/models/${id}`);
    renderModelDetail(m);
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

function renderModelDetail(m) {
  const metricKeys = Object.keys(m.metrics || {});
  const schemaRows = m.schema.map((f) => `<tr><td><strong>${esc(f.name)}</strong></td><td><span class="kind ${f.kind}">${f.kind}</span></td>
    <td class="small">${f.kind === "numeric" ? `${fmtNum(f.min)} … ${fmtNum(f.max)}` : f.allowed_values ? esc(f.allowed_values.slice(0, 6).join(", ")) + (f.n_categories > 6 ? ` +${f.n_categories - 6} more` : "") : ""}</td>
    <td class="small muted">${esc(f.example)}</td></tr>`).join("");
  const imp = (m.feature_importance || []).map((f) => `<div class="imp-row"><span class="name" title="${esc(f.feature)}">${esc(f.feature)}</span>
    <div class="track"><div class="fill" style="width:${(f.importance / m.feature_importance[0].importance) * 100}%"></div></div><span class="muted">${(f.importance * 100).toFixed(1)}%</span></div>`).join("");
  const lb = m.leaderboard.map((r, i) => `<tr class="${r.model === m.algorithm ? "best" : ""}"><td>${i + 1}</td><td>${esc(r.model)}</td>
    ${metricKeys.map((k) => `<td class="num">${fmtNum(r[k])}${r[k + "_std"] != null ? `<span class="muted small"> ±${r[k + "_std"]}</span>` : ""}</td>`).join("")}</tr>`).join("");

  $("#model-detail").innerHTML = `
    <div class="dataset-bar">
      <div><h2 style="margin:0">${esc(m.name)} <span class="badge-loaded">● loaded</span></h2>
      <div class="muted small">Predicts <b>${esc(m.target)}</b> · ${TASK_LABEL[m.task_type]} · trained on ${m.n_rows.toLocaleString()} rows of ${esc(m.dataset_name)}</div></div>
      <button class="btn ghost small" id="delete-model">Delete model</button>
    </div>
    <div class="detail-grid">
      <div>
        <h3>1. Prepare your file</h3>
        <p class="small muted">Your file needs these ${m.schema.length} columns (any order; extra columns are kept in the output). Blank cells are allowed.</p>
        <div class="table-wrap" style="max-height:300px"><table class="grid"><thead><tr><th>Column</th><th>Type</th><th>Expected values</th><th>Example</th></tr></thead><tbody>${schemaRows}</tbody></table></div>
        <p><a class="btn small" href="/api/models/${m.id}/template" download>⇩ Download sample input file</a></p>
      </div>
      <div>
        <h3>2. Upload data and predict</h3>
        <div class="predict-box" id="predict-box">
          <input type="file" id="predict-input" accept=".csv,.xlsx,.xls" hidden>
          <p><strong>Drop a CSV / Excel file</strong> or</p>
          <button class="btn primary" id="predict-btn">Upload file &amp; predict</button>
          <p class="small muted">Results download automatically as a file.</p>
        </div>
        <div id="predict-result"></div>
      </div>
    </div>
    <div class="detail-grid" style="margin-top:8px">
      <div><h3>What drives ${esc(m.target)}?</h3><p class="small muted">Share of the model's accuracy that depends on each feature.</p>${imp || '<p class="muted small">Not available.</p>'}</div>
      <div><h3>How the model was chosen</h3><ul class="decisions">${m.decisions.map((d) => `<li>${esc(d)}</li>`).join("")}</ul></div>
    </div>
    <h3>Model leaderboard (cross-validated)</h3>
    <div class="table-wrap"><table class="grid"><thead><tr><th>#</th><th>Model</th>${metricKeys.map((k) => `<th class="num">${METRIC_LABEL[k]}</th>`).join("")}</tr></thead><tbody>${lb}</tbody></table></div>`;

  const input = $("#predict-input"), pbox = $("#predict-box");
  $("#predict-btn").addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && runPrediction(m, input.files[0]));
  ["dragenter", "dragover"].forEach((e) => pbox.addEventListener(e, (ev) => { ev.preventDefault(); pbox.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => pbox.addEventListener(e, (ev) => { ev.preventDefault(); pbox.classList.remove("over"); }));
  pbox.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && runPrediction(m, ev.dataTransfer.files[0]));
  $("#delete-model").addEventListener("click", async () => {
    if (!confirm(`Delete model "${m.name}"? This cannot be undone.`)) return;
    try { await api(`/api/models/${m.id}`, { method: "DELETE" }); currentModel = null; $("#model-detail").hidden = true; loadModels(); toast("Model deleted"); }
    catch (e) { toast(e.message, true); }
  });
}

async function runPrediction(m, file) {
  const out = $("#predict-result"), btn = $("#predict-btn");
  const fd = new FormData(); fd.append("file", file);
  btn.disabled = true; out.innerHTML = `<p class="muted small">Scoring ${esc(file.name)}…</p>`;
  try {
    const res = await api(`/api/models/${m.id}/predict`, { method: "POST", body: fd });
    const blob = await res.blob();
    const name = (res.headers.get("Content-Disposition") || "").match(/filename="?([^"]+)"?/)?.[1] || "predictions.csv";
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    const s = JSON.parse(res.headers.get("X-Prediction-Summary") || "{}");
    const dist = s.distribution ? Object.entries(s.distribution).map(([k, v]) => `<li><b>${esc(k)}</b>: ${v.toLocaleString()} (${((v / s.rows) * 100).toFixed(1)}%)</li>`).join("")
      : `<li>Mean ${fmtNum(s.mean)} · min ${fmtNum(s.min)} · max ${fmtNum(s.max)}</li>`;
    out.innerHTML = `<div class="result-box"><strong>${s.rows?.toLocaleString()} rows scored</strong> → downloaded <code>${esc(name)}</code>
      <ul class="small" style="margin:6px 0 0">${dist}</ul>
      ${m.task_type === "binary" ? `<p class="small muted" style="margin:6px 0 0">Sort by <code>priority_rank</code> to see who to contact first.</p>` : ""}</div>`;
  } catch (e) { out.innerHTML = `<div class="result-box fail"><strong>Prediction failed</strong><p class="small">${esc(e.message)}</p></div>`; }
  finally { btn.disabled = false; $("#predict-input").value = ""; }
}
