/* T-60 dashboard UI: tabs, Run Lab, jobs SSE/logs, charts, Carbon light Plotly */
(function () {
  "use strict";

  const state = {
    runId: null,
    page: 1,
    pageSize: 40,
    tab: "all",
    jobs: [],
    sse: {},
    modules: {},
    suitePresets: {},
    logWatch: null,
    viewerJobId: null,
    runMode: "configure",
    lastTelemetry: null,
    theme: document.documentElement.getAttribute("data-theme") || "light",
    followLog: true,
    lastLogLen: 0,
  };

  const VIEWER_MODULES = new Set([
    "suite_smoke", "suite_custom", "backtest_quick", "player_rating_smoke",
  ]);

  function chartColors() {
    const dark = state.theme === "dark";
    return {
      paper: "rgba(0,0,0,0)",
      plot: "rgba(0,0,0,0)",
      font: dark ? "#f4f4f4" : "#161616",
      muted: dark ? "#a8a8a8" : "#6f6f6f",
      grid: dark ? "#393939" : "#e0e0e0",
      zero: dark ? "#525252" : "#c6c6c6",
      line: dark ? "#78a9ff" : "#0f62fe",
      marker: dark ? "#a6c8ff" : "#0043ce",
    };
  }

  function LIGHT_LAYOUT() {
    const c = chartColors();
    return {
      paper_bgcolor: c.paper,
      plot_bgcolor: c.plot,
      font: { color: c.font, family: "IBM Plex Sans, sans-serif" },
      xaxis: { gridcolor: c.grid, zerolinecolor: c.zero, color: c.muted },
      yaxis: { gridcolor: c.grid, zerolinecolor: c.zero, color: c.muted },
      margin: { t: 40, r: 20, b: 40, l: 50 },
    };
  }

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  async function api(path, opts) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts && opts.headers) },
      ...opts,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json();
  }

  function toast(msg, kind) {
    const box = $("#toasts");
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  function fmt(n, digits) {
    if (n == null || Number.isNaN(n)) return "—";
    if (typeof n === "number") return n.toFixed(digits != null ? digits : 3);
    return String(n);
  }

  function setBanner(msg) {
    const el = $("#banner");
    if (!msg) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.remove("hidden");
  }

  function cardHtml(label, value) {
    return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
  }

  function esc(v) {
    if (v == null) return "";
    return String(v).replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
    );
  }

  function mergeLayout(layout) {
    const base = LIGHT_LAYOUT();
    const L = { ...base, ...(layout || {}) };
    L.xaxis = { ...base.xaxis, ...(layout && layout.xaxis) };
    L.yaxis = { ...base.yaxis, ...(layout && layout.yaxis) };
    return L;
  }

  function applyTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    state.theme = next;
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("t60-theme", next);
    } catch (e) { /* ignore */ }
    // Re-render open Plotly charts with the new palette when possible.
    if (state.tab === "all") renderCharts("all", "#charts-all").catch(() => {});
    if (state.lastTelemetry) renderTelemetry(state.lastTelemetry);
  }

  function toggleTheme() {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  }

  // ---- Tabs ----
  function switchTab(tab) {
    state.tab = tab;
    $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === tab));
    if (tab === "all") loadAll();
    if (tab === "run") loadRunLab();
    if (tab === "player") loadPlayers();
    if (tab === "ats") loadAts();
    if (tab === "ml") loadMl();
    if (tab === "totals") loadTotals();
    if (tab === "jobs") renderJobsFull();
    if (tab === "docs") {
      loadBenchReport();
      loadHealthReport();
      initDocsToc();
    }
  }

  function healthTag(kind, label) {
    return `<span class="health-tag health-tag--${kind}">${esc(label)}</span>`;
  }

  function healthCard(label, value, hint) {
    return (
      `<div class="health-card">` +
      `<div class="label">${esc(label)}</div>` +
      `<div class="value">${value}</div>` +
      (hint ? `<div class="hint">${esc(hint)}</div>` : "") +
      `</div>`
    );
  }

  function fileStatusTag(counts) {
    const failed = counts.failed || 0;
    const xfailed = counts.xfailed || 0;
    const error = counts.error || 0;
    if (failed > 0 || error > 0) return healthTag("fail", "fail");
    if (xfailed > 0) return healthTag("xfail", "xfail");
    if ((counts.skipped || 0) > 0 && (counts.passed || 0) === 0) {
      return healthTag("skip", "skip");
    }
    return healthTag("pass", "pass");
  }

  async function loadBenchReport() {
    const body = $("#bench-report-body");
    const lede = $("#bench-report-lede");
    if (!body) return;
    try {
      const data = await api("/api/bench");
      if (!data.available) {
        body.innerHTML = `<p class="health-empty">${esc(data.message || "no bench report yet")}</p>`;
        if (lede) {
          lede.textContent =
            "Run pytest -m bench to generate output/bench/latest.json.";
        }
        return;
      }
      const totals = data.totals || {};
      const cards = [
        healthCard("Passed", String(data.passed ?? totals.passed ?? "—"), "Assertions that held"),
        healthCard("Failed", String(data.failed ?? totals.failed ?? "—"), "Unexpected failures"),
        healthCard("Xfailed", String(data.xfailed ?? totals.xfailed ?? "—"), "Known unfinished wiring"),
        healthCard("Skipped", String(data.skipped ?? totals.skipped ?? "—"), "Not collected this run"),
        healthCard("Exit status", String(data.exitstatus ?? "—"), "0 means pytest exited clean"),
      ].join("");

      const files = data.files || {};
      const rows = Object.keys(files)
        .sort()
        .map((fpath) => {
          const c = files[fpath] || {};
          return (
            `<tr>` +
            `<td class="file-col" title="${esc(fpath)}">${esc(fpath)}</td>` +
            `<td>${c.passed || 0}</td>` +
            `<td>${c.failed || 0}</td>` +
            `<td>${c.xfailed || 0}</td>` +
            `<td>${c.skipped || 0}</td>` +
            `<td>${fileStatusTag(c)}</td>` +
            `</tr>`
          );
        })
        .join("");

      const table = rows
        ? `<div class="docs-data-table-wrap"><table class="docs-data-table">` +
          `<thead><tr><th>File</th><th>Pass</th><th>Fail</th><th>Xfail</th><th>Skip</th><th>Status</th></tr></thead>` +
          `<tbody>${rows}</tbody></table></div>`
        : `<p class="health-empty">No per-file counts in this report.</p>`;

      body.innerHTML =
        `<div class="health-cards">${cards}</div>` +
        `<p class="help">Source: <code>${esc(data.source_path || "output/bench/latest.json")}</code>` +
        ` · written by <code>pytest -m bench</code></p>` +
        table;

      if (lede) {
        lede.textContent =
          `Last bench session exitstatus=${data.exitstatus}` +
          (data.source_path ? ` · ${data.source_path}` : "") +
          ".";
      }
    } catch (err) {
      body.innerHTML = `<p class="health-empty">no bench report yet</p>`;
      if (lede) lede.textContent = "Could not load formula-bench status.";
    }
  }

  function fmtHealth(n, digits) {
    if (n == null || Number.isNaN(n)) return "—";
    if (typeof n === "number") return n.toFixed(digits != null ? digits : 3);
    return String(n);
  }

  async function loadHealthReport() {
    const body = $("#health-report-body");
    const lede = $("#health-report-lede");
    const notice = $("#health-honesty");
    if (!body) return;
    try {
      const rid = state.runId || "latest";
      const q = new URLSearchParams();
      if (state.runId) q.set("run_id", state.runId);
      const data = await api(`/api/health?${q}`);

      if (notice) {
        if (data.research_only && data.banner_message) {
          notice.textContent = data.banner_message;
          notice.classList.remove("hidden");
        } else {
          notice.textContent = "";
          notice.classList.add("hidden");
        }
      }

      if (!data.n_rows) {
        body.innerHTML =
          `<p class="health-empty">no run results yet</p>` +
          `<p class="help">Select a run with a results CSV on All Data, or launch a suite smoke.</p>`;
        if (lede) {
          lede.textContent = data.results_path
            ? `Results path exists but has 0 rows: ${data.results_path}`
            : "No backtest_results.csv for the selected run.";
        }
        return;
      }

      const weekly = data.weekly || {};
      const noWeekly = weekly.status === "no_data";
      const cards = noWeekly
        ? `<p class="health-empty">Weekly report: no graded rows in the last ${data.window || 30} games.</p>`
        : [
            healthCard("Games (window)", String(weekly.n_games ?? "—"), `Last ${data.window || 30} rows`),
            healthCard("Bets", String(weekly.n_bets ?? "—"), "Active non-push ATS bets"),
            healthCard("ATS %", fmtHealth(weekly.ats_pct, 3), "Cover rate on graded bets"),
            healthCard("Spread MAE", fmtHealth(weekly.spread_mae, 2), "Lower is better"),
            healthCard("Mean CLV", fmtHealth(weekly.mean_clv, 3), "Null if CLV column missing"),
            healthCard(
              "Underperform alert",
              weekly.alert_underperformance ? "Yes" : "No",
              "ATS < 48% with ≥20 bets"
            ),
          ].join("");

      const perm = data.permutation || {};
      let permHtml;
      if (perm.status === "ok") {
        const tag = perm.passed
          ? healthTag("pass", "null destroyed edge")
          : healthTag("fail", "null retained edge");
        permHtml =
          `<div class="health-block">` +
          `<h4>Permutation null ${tag}</h4>` +
          `<p>Shuffling actual scores should destroy the model’s MAE advantage. ` +
          `If the shuffled null is as good as the real labels, treat the signal as suspect.</p>` +
          `<div class="health-cards">` +
          healthCard("Base MAE", fmtHealth(perm.base_metric, 3), "Real labels") +
          healthCard("Null mean MAE", fmtHealth(perm.null_mean, 3), `${perm.n_perm || 20} shuffles`) +
          healthCard("Pairs", String(perm.n_pairs ?? "—"), "Finite residual pairs") +
          `</div>` +
          (perm.detail ? `<p class="help">${esc(perm.detail)}</p>` : "") +
          `</div>`;
      } else {
        permHtml =
          `<div class="health-block">` +
          `<h4>Permutation null ${healthTag("missing", "skipped")}</h4>` +
          `<p>${esc(perm.reason || "Could not compute permutation null.")}</p>` +
          `</div>`;
      }

      const drift = data.drift || {};
      let driftHtml;
      if (drift.status === "ok") {
        const drows = (drift.rows || [])
          .map(
            (r) =>
              `<tr>` +
              `<td>${esc(r.feature)}</td>` +
              `<td>${fmtHealth(r.train_mean, 3)}</td>` +
              `<td>${fmtHealth(r.live_mean, 3)}</td>` +
              `<td>${fmtHealth(r.z_score, 2)}</td>` +
              `</tr>`
          )
          .join("");
        driftHtml =
          `<div class="health-block">` +
          `<h4>Feature drift ${healthTag(drows ? "warn" : "pass", drows ? "flagged" : "none")}</h4>` +
          (drows
            ? `<div class="docs-data-table-wrap"><table class="docs-data-table">` +
              `<thead><tr><th>Feature</th><th>Train mean</th><th>Live mean</th><th>z</th></tr></thead>` +
              `<tbody>${drows}</tbody></table></div>`
            : `<p class="health-empty">No features exceeded the drift z-score threshold.</p>`) +
          `</div>`;
      } else {
        driftHtml =
          `<div class="health-block">` +
          `<h4>Feature drift ${healthTag("missing", "unavailable")}</h4>` +
          `<p>${esc(drift.reason || "Feature tables were not saved for this run — drift cannot be computed.")}</p>` +
          `</div>`;
      }

      const missing = (data.columns_missing || []).length
        ? `<p class="help">Missing columns: ${esc((data.columns_missing || []).join(", "))}. ` +
          `Those metrics show as — rather than zero.</p>`
        : "";

      body.innerHTML =
        (typeof cards === "string" && cards.indexOf("health-card") >= 0
          ? `<div class="health-cards">${cards}</div>`
          : cards) +
        missing +
        permHtml +
        driftHtml +
        `<p class="help">Source: <code>${esc(data.results_path || "—")}</code>` +
        (data.quote_source ? ` · quote_source=${esc(data.quote_source)}` : "") +
        `</p>`;

      if (lede) {
        lede.textContent =
          `Run ${rid} · ${data.n_rows} rows` +
          (data.quote_source ? ` · quote_source=${data.quote_source}` : "") +
          ".";
      }
    } catch (err) {
      body.innerHTML = `<p class="health-empty">Could not load model health.</p>`;
      if (lede) lede.textContent = "Could not load model-health status.";
      if (notice) notice.classList.add("hidden");
    }
  }

  // ---- Schema forms ----
  function renderSchemaForm(container, fields, values) {
    const vals = values || {};
    container.innerHTML = "";
    (fields || []).forEach((f) => {
      if (f.name === "preset") return; // handled by dedicated preset select
      const label = document.createElement("label");
      label.appendChild(document.createTextNode(f.label || f.name));
      let input;
      const cur = vals[f.name] != null ? vals[f.name] : f.default;
      if (f.type === "bool") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!cur;
      } else if (f.type === "enum" || f.type === "run_id") {
        input = document.createElement("select");
        const opts = f.type === "run_id" ? ["", ...(state.runOptions || [])] : f.options || [];
        opts.forEach((o) => {
          const opt = document.createElement("option");
          opt.value = o;
          opt.textContent = o === "" ? "(none)" : o;
          if (String(cur) === String(o)) opt.selected = true;
          input.appendChild(opt);
        });
      } else if (f.type === "int" || f.type === "float") {
        input = document.createElement("input");
        input.type = "number";
        if (f.type === "float") input.step = "any";
        if (cur != null && cur !== "") input.value = cur;
      } else {
        input = document.createElement("input");
        input.type = "text";
        if (cur != null) input.value = cur;
      }
      input.name = f.name;
      input.dataset.field = f.name;
      label.appendChild(input);
      if (f.help) {
        const h = document.createElement("span");
        h.className = "help";
        h.textContent = f.help;
        label.appendChild(h);
      }
      container.appendChild(label);
    });
  }

  function readSchemaForm(container) {
    const out = {};
    $$("[data-field]", container).forEach((el) => {
      const name = el.dataset.field;
      if (el.type === "checkbox") out[name] = el.checked;
      else if (el.type === "number") out[name] = el.value === "" ? null : Number(el.value);
      else out[name] = el.value;
    });
    return out;
  }

  async function loadModules() {
    const mods = await api("/api/modules");
    state.modules = {};
    mods.forEach((m) => {
      state.modules[m.id] = m;
    });
    try {
      state.suitePresets = await api("/api/presets/suite");
    } catch (_) {
      state.suitePresets = {};
    }
  }

  async function loadRunLab() {
    await loadRuns(false);
    state.runOptions = (state.runs || []).map((r) => r.id);
    const suite = state.modules.suite_custom;
    const bt = state.modules.backtest_quick;
    if (suite && suite.param_schema) {
      renderSchemaForm($("#suite-form"), suite.param_schema.fields, {});
    }
    if (bt && bt.param_schema) {
      renderSchemaForm($("#backtest-form"), bt.param_schema.fields, {
        quick: true,
        fast_tuning: true,
        elo_trials: 12,
        hier_trials: 12,
        meta_trials: 18,
        window: "4",
      });
    }
    applySuitePreset($("#suite-preset").value);
  }

  function applySuitePreset(name) {
    const preset = state.suitePresets[name];
    const fields = (state.modules.suite_custom && state.modules.suite_custom.param_schema.fields) || [];
    if (!preset) {
      renderSchemaForm($("#suite-form"), fields, readSchemaForm($("#suite-form")));
      return;
    }
    renderSchemaForm($("#suite-form"), fields, { ...preset, preset: name });
    $("#run-cmd-preview").textContent =
      `Preset “${name}”: years=${preset.years || "(mini)"} elo=${preset.elo_trials} hier=${preset.hier_trials} meta=${preset.meta_trials} fast=${preset.fast_tuning}`;
  }

  // ---- Runs / All Data ----
  async function loadRuns(setDefault) {
    const runs = await api("/api/runs");
    state.runs = runs;
    const sel = $("#run-picker");
    if (!sel) return;
    const prev = state.runId;
    sel.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = runs.length ? "(latest / fallback CSV)" : "(no runs — use Run Lab)";
    sel.appendChild(opt0);
    runs.forEach((r) => {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = `${r.id}${r.has_results ? "" : " (no csv)"}`;
      sel.appendChild(o);
    });
    if (setDefault !== false) {
      if (!state.runId && runs.length) state.runId = runs.find((r) => r.has_results)?.id || runs[0].id;
      if (state.runId) sel.value = state.runId;
    } else if (prev) {
      sel.value = prev;
    }
    $("#empty-runs").classList.toggle("hidden", runs.length > 0 || !!state.runId);
  }

  async function loadSummary() {
    const rid = state.runId || "latest";
    const s = await api(`/api/runs/${encodeURIComponent(rid)}/summary`);
    const box = $("#summary-cards");
    box.innerHTML = [
      cardHtml("Games", s.n_games ?? 0),
      cardHtml("Spread MAE", fmt(s.spread_mae, 2)),
      cardHtml("Total MAE", fmt(s.total_mae, 2)),
      cardHtml("ATS hit", fmt(s.ats_hit_rate, 3)),
      cardHtml("ROI", fmt(s.roi, 3)),
      cardHtml("Actionable", s.actionable_bets ?? "—"),
      cardHtml("Quote src", s.quote_source || "—"),
    ].join("");
    if (s.research_only_banner && s.banner_message) setBanner(s.banner_message);
    else setBanner(null);
    if (window.JsonView && s.ats_status) {
      JsonView.render($("#ats-status-json"), s.ats_status, "ats_status.json");
    } else {
      $("#ats-status-json").innerHTML = "";
    }
  }

  async function loadResults() {
    const rid = state.runId || "latest";
    const q = new URLSearchParams({
      page: String(state.page),
      page_size: String(state.pageSize),
    });
    const search = $("#results-search").value.trim();
    if (search) q.set("search", search);
    const data = await api(`/api/runs/${encodeURIComponent(rid)}/results?${q}`);
    const table = $("#results-table");
    const cols = data.columns || [];
    if (!cols.length) {
      table.innerHTML = "<tr><td>No rows</td></tr>";
      $("#page-info").textContent = "";
      return;
    }
    const head = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
    const body = (data.rows || [])
      .map((row) => `<tr>${cols.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`)
      .join("");
    table.innerHTML = head + body;
    const pages = Math.max(1, Math.ceil((data.total || 0) / state.pageSize));
    $("#page-info").textContent = `Page ${state.page} / ${pages} · ${data.total} rows`;
  }

  async function loadArtifacts() {
    const rid = state.runId || "latest";
    let arts = [];
    try {
      arts = await api(`/api/runs/${encodeURIComponent(rid)}/artifacts`);
    } catch (_) {
      arts = [];
    }
    $("#artifact-list").innerHTML = arts.length
      ? arts.map((a) => `<li>${esc(a.path)} <span class="muted">(${a.size})</span></li>`).join("")
      : "<li class='muted'>No artifacts</li>";
  }

  async function renderCharts(tab, containerId) {
    const charts = await api(`/api/charts?tab=${encodeURIComponent(tab)}`);
    const box = $(containerId);
    box.innerHTML = "";
    for (const c of charts) {
      if (tab !== "all" && c.tab !== tab && c.tab !== "all") continue;
      if (tab === "all" && c.tab !== "all") continue;
      const div = document.createElement("div");
      div.className = "chart-box";
      div.id = `chart-${c.id}`;
      box.appendChild(div);
      try {
        const q = state.runId ? `?run_id=${encodeURIComponent(state.runId)}` : "";
        const fig = await api(`/api/charts/${encodeURIComponent(c.id)}${q}`);
        if (window.Plotly) {
          const data = fig.data || [];
          const layout = mergeLayout(fig.layout || { title: c.title });
          if (!data.length) {
            div.innerHTML = `<div class="muted" style="padding:1rem">${esc(c.title)}: ${(fig.layout && fig.layout.annotations && fig.layout.annotations[0] && fig.layout.annotations[0].text) || "no data"}</div>`;
          } else {
            Plotly.newPlot(div, data, layout, { responsive: true, displayModeBar: false });
          }
        } else {
          div.textContent = c.title + " (Plotly missing — check /static/js/vendor/plotly.min.js)";
        }
      } catch (e) {
        div.textContent = c.title + ": " + e.message;
      }
    }
  }

  async function loadAll() {
    await loadRuns();
    await loadSummary();
    await loadResults();
    await loadArtifacts();
    await renderCharts("all", "#charts-all");
  }

  // ---- Players ----
  async function loadPlayers() {
    const q = new URLSearchParams();
    if (state.runId) q.set("run_id", state.runId);
    const search = $("#player-search").value.trim();
    if (search) q.set("search", search);
    const data = await api(`/api/players?${q}`);
    $("#player-source").textContent = data.source
      ? `Source: ${data.source} · ${data.n} players`
      : data.message || "No snapshot";
    const rows = data.players || [];
    const cols = ["player_id", "name", "team", "offense", "defense", "mu", "rd", "games"];
    const present = cols.filter((c) => rows.some((r) => r[c] != null));
    const use = present.length ? present : Object.keys(rows[0] || { player_id: 1 });
    const table = $("#players-table");
    table.innerHTML =
      `<tr>${use.map((c) => `<th>${c}</th>`).join("")}</tr>` +
      rows
        .slice(0, 500)
        .map(
          (r, i) =>
            `<tr data-idx="${i}">${use.map((c) => `<td>${esc(r[c])}</td>`).join("")}</tr>`
        )
        .join("");
    table.onclick = (ev) => {
      const tr = ev.target.closest("tr[data-idx]");
      if (!tr) return;
      $$("#players-table tr").forEach((x) => x.classList.remove("selected"));
      tr.classList.add("selected");
      const row = rows[Number(tr.dataset.idx)];
      if (window.JsonView) JsonView.render($("#player-detail"), row, "Player detail");
      else $("#player-detail").textContent = JSON.stringify(row, null, 2);
    };
  }

  function formParams(formId) {
    const fd = new FormData($(formId));
    const o = {};
    fd.forEach((v, k) => {
      o[k] = v === "on" ? true : v;
    });
    return o;
  }

  async function loadAts() {
    const q = state.runId ? `?run_id=${encodeURIComponent(state.runId)}` : "";
    const data = await api(`/api/ats/scorecard${q}`);
    const sc = data.scorecard || {};
    const m = sc.metrics || {};
    const warns = (sc.warnings || []).map((w) =>
      `<div class="banner warn" style="margin:4px 0">${w}</div>`
    ).join("");
    $("#ats-scorecard").innerHTML = [
      cardHtml("Paired MAE", fmt(m.paired_score_mae, 2)),
      cardHtml("Model−mkt MAE", fmt(m.model_minus_market_spread_mae, 2)),
      cardHtml("Mkt-err corr", fmt(m.market_error_corr, 3)),
      cardHtml("Interval cov", fmt(m.interval_coverage, 3)),
      cardHtml("Dispersion", fmt(m.margin_dispersion_ratio, 2)),
      cardHtml("N actionable", m.n_actionable ?? "—"),
      cardHtml("ATS hit (actionable)", fmt(m.ats_hit_actionable ?? m.ats_hit)),
      cardHtml("N lean graded", m.n_lean_graded ?? "—"),
      cardHtml("ATS hit (lean)", fmt(m.ats_hit_lean)),
      cardHtml("Forecast", m.forecast_source ?? "—"),
      cardHtml("Mean |edge|", fmt(m.mean_abs_edge, 2)),
      cardHtml("ATS EV (actionable)", fmt(m.mean_ats_ev)),
      cardHtml("Lean EV (diag)", fmt(m.mean_ats_ev_lean)),
      cardHtml("N finite CLV", m.n_finite_clv ?? "—"),
      cardHtml("CLV claim OK", m.clv_claim_allowed ? "yes" : "no"),
      cardHtml("Population", m.population ?? "—"),
    ].join("") + (warns ? `<div style="grid-column:1/-1">${warns}</div>` : "");
    await renderCharts("ats", "#charts-ats");
  }

  async function loadMl() {
    const q = state.runId ? `?run_id=${encodeURIComponent(state.runId)}` : "";
    const data = await api(`/api/ml/scorecard${q}`);
    const m = (data.scorecard && data.scorecard.metrics) || {};
    $("#ml-scorecard").innerHTML = [
      cardHtml("Brier", fmt(m.brier)),
      cardHtml("Logloss", fmt(m.logloss)),
      cardHtml("N", m.n ?? "—"),
    ].join("");
    await renderCharts("ml", "#charts-ml");
  }

  async function loadTotals() {
    const q = state.runId ? `?run_id=${encodeURIComponent(state.runId)}` : "";
    const data = await api(`/api/totals/scorecard${q}`);
    const m = (data.scorecard && data.scorecard.metrics) || {};
    $("#totals-scorecard").innerHTML = [
      cardHtml("MAE", fmt(m.mae, 2)),
      cardHtml("RMSE", fmt(m.rmse, 2)),
      cardHtml("Bias", fmt(m.bias, 2)),
      cardHtml("O/U hit", fmt(m.ou_hit)),
      cardHtml("N", m.n ?? "—"),
    ].join("");
    await renderCharts("totals", "#charts-totals");
  }

  // ---- Jobs ----
  function etaLabel(sec) {
    if (sec == null) return "";
    if (sec < 60) return `~${Math.round(sec)}s`;
    if (sec < 3600) return `~${Math.round(sec / 60)}m`;
    return `~${(sec / 3600).toFixed(1)}h`;
  }

  function fmtDuration(sec) {
    if (sec == null || Number.isNaN(sec)) return "—";
    if (sec < 60) return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
    const h = Math.floor(sec / 3600);
    const m = Math.round((sec % 3600) / 60);
    return `${h}h ${m}m`;
  }

  function jobShortStatus(j) {
    const t = j.telemetry;
    if (t && t.short_status) return t.short_status;
    if (j.message && j.message.length < 80 && !j.message.includes("stage=running")) return j.message;
    if (j.stage && j.stage !== "running") return j.stage;
    return j.status || "";
  }

  function jobRowHtml(j, compact) {
    const pct = Math.max(0, Math.min(100, j.progress_pct || 0));
    const active = j.status === "running" || j.status === "queued";
    const finished = ["succeeded", "failed", "cancelled"].includes(j.status);
    const showViewer = VIEWER_MODULES.has(j.module);
    const statusLine = jobShortStatus(j);
    if (compact) {
      return `<div class="job-row ${esc(j.status)}" data-job="${esc(j.id)}">
        <div><strong>${esc(j.module)}</strong> · ${esc(j.status)}</div>
        <div class="meta">${esc(statusLine)}${etaLabel(j.eta_seconds) ? " · " + etaLabel(j.eta_seconds) : ""}</div>
        <div class="progress"><span style="width:${pct}%"></span></div>
        <div class="actions">
          ${showViewer ? `<button type="button" data-viewer="${esc(j.id)}">Viewer</button>` : ""}
          ${active ? `<button type="button" data-cancel="${esc(j.id)}">Cancel</button>` : ""}
          ${finished ? `<button type="button" data-dismiss="${esc(j.id)}">Dismiss</button>` : ""}
        </div>
      </div>`;
    }
    return `<div class="job-row ${esc(j.status)}" data-job="${esc(j.id)}">
      <div><strong>${esc(j.module)}</strong> · ${esc(j.status)}</div>
      <div class="meta">${esc(statusLine)} ${etaLabel(j.eta_seconds)} · ${esc(j.id)}</div>
      <div class="progress"><span style="width:${pct}%"></span></div>
      <div class="actions">
        ${showViewer ? `<button type="button" data-viewer="${esc(j.id)}">Viewer</button>` : ""}
        ${active ? `<button type="button" data-cancel="${esc(j.id)}">Cancel</button>` : ""}
        <button type="button" data-log="${esc(j.id)}">View log</button>
        ${finished ? `<button type="button" data-dismiss="${esc(j.id)}">Dismiss</button>` : ""}
        ${finished ? `<button type="button" data-params="${esc(j.id)}">Params</button>` : ""}
      </div>
      ${j.error ? `<div class="meta">${esc(String(j.error).split("Traceback")[0].trim().slice(0, 160))}</div>` : ""}
    </div>`;
  }

  function renderJobsLive() {
    const active = state.jobs.filter((j) =>
      ["queued", "running", "pending"].includes(j.status)
    );
    const seenOk = new Set(
      state.jobs.filter((j) => j.status === "succeeded").map((j) => j.module)
    );
    const recent = state.jobs.filter((j) => {
      if (["queued", "running", "pending"].includes(j.status)) return false;
      if (j.status === "failed" && seenOk.has(j.module)) return false;
      return true;
    }).slice(0, 6);
    const show = active.length ? [...active, ...recent.slice(0, 3)] : recent;
    $("#jobs-live").innerHTML = show.map((j) => jobRowHtml(j, true)).join("") ||
      "<div class='muted'>No jobs yet</div>";
  }

  function renderJobsFull() {
    $("#jobs-full").innerHTML = state.jobs.map((j) => jobRowHtml(j, false)).join("") || "<div class='muted'>No jobs</div>";
  }

  function setRunMode(mode) {
    state.runMode = mode;
    const configure = mode === "configure";
    $("#run-configure").classList.toggle("hidden", !configure);
    $("#run-viewer").classList.toggle("hidden", configure);
    $$(".run-mode").forEach((b) => {
      b.classList.toggle("active", b.dataset.runMode === mode);
    });
  }

  function sparkLayout() {
    return mergeLayout({
      margin: { t: 8, r: 8, b: 24, l: 36 },
      height: 110,
      showlegend: false,
      xaxis: { visible: false },
      yaxis: { title: "", tickfont: { size: 9 } },
    });
  }

  function renderSpark(elId, scores, title) {
    const el = $(elId);
    if (!el || typeof Plotly === "undefined") return;
    const y = (scores || []).slice(-40);
    if (!y.length) {
      el.innerHTML = `<div class="muted" style="padding:1rem 0;font-size:0.75rem">No trial scores yet</div>`;
      return;
    }
    Plotly.react(
      el,
      [{
        y,
        type: "scatter",
        mode: "lines+markers",
        line: { color: chartColors().line, width: 1.5 },
        marker: { size: 4, color: chartColors().marker },
        hovertemplate: "trial %{pointNumber}: %{y:.3f}<extra></extra>",
      }],
      sparkLayout(),
      { displayModeBar: false, responsive: true }
    );
  }

  function renderTelemetry(tel) {
    if (!tel) return;
    state.lastTelemetry = tel;
    const headline = $("#rv-headline");
    if (headline) headline.textContent = tel.headline || "Working…";
    const jid = $("#rv-job-id");
    if (jid) jid.textContent = tel.job_id || state.viewerJobId || "";
    const pct = Math.max(0, Math.min(100, tel.progress_pct || 0));
    const bar = $("#rv-progress-bar");
    if (bar) bar.style.width = pct + "%";
    const pl = $("#rv-progress-label");
    if (pl) pl.textContent = `${pct.toFixed(0)}% complete` + (tel.timing && tel.timing.eta_s != null
      ? ` · about ${fmtDuration(tel.timing.eta_s)} remaining`
      : "");
    const now = $(".rv-now");
    if (now) {
      const live = (tel.status || "").toLowerCase() === "running" || pct > 0 && pct < 100;
      now.classList.toggle("is-live", !!live);
    }
    const termTitle = $("#rv-terminal-title");
    if (termTitle) {
      termTitle.textContent = (tel.job_id || state.viewerJobId || "suite") + " · live log";
    }

    const pipe = $("#rv-pipeline");
    if (pipe) {
      pipe.innerHTML = (tel.pipeline || []).map((s) =>
        `<li class="pipeline-step ${esc(s.state)}">
          <span class="step-num">${s.id}</span>
          <span class="step-name">${esc(s.name)}</span>
        </li>`
      ).join("");
    }

    const season = tel.season || {};
    const seasonEl = $("#rv-season");
    if (seasonEl) {
      if (season.label) {
        seasonEl.innerHTML = `<span class="chip">${esc(season.label)}</span>` +
          (season.total
            ? `Season <strong>${season.index || "—"}</strong> of <strong>${season.total}</strong>`
            : "") +
          (season.years && season.years.length
            ? `<div class="muted" style="margin-top:0.4rem;font-size:0.8rem">Years in this run: ${esc(season.years.join(", "))}</div>`
            : "");
      } else if (season.years && season.years.length) {
        seasonEl.innerHTML = `<span class="muted">Preparing seasons:</span> ${esc(season.years.join(", "))}`;
      } else {
        seasonEl.textContent = "Waiting for season markers…";
      }
    }

    const tuners = tel.tuners || {};
    ["elo", "hier", "meta"].forEach((key) => {
      const t = tuners[key] || {};
      const panel = $(`.tuner-panel[data-tuner="${key}"]`);
      if (panel) panel.classList.toggle("running", t.state === "running");
      const stats = $(`#rv-${key}-stats`);
      if (stats) {
        const trialBit = t.total
          ? `Trial ${t.trial || 0} / ${t.total}`
          : (t.trial ? `Trial ${t.trial}` : "Not started");
        const best = t.best_mae != null ? `Best score ${fmt(t.best_mae, 3)}` : "Best score —";
        const st = t.state === "done" ? "Finished" : t.state === "running" ? "In progress" : "Waiting";
        stats.textContent = `${st} · ${trialBit} · ${best}`;
      }
      renderSpark(`#rv-${key}-spark`, t.scores || [], key);
    });

    const timing = tel.timing || {};
    const timingEl = $("#rv-timing");
    if (timingEl) {
      const blocks = [
        ["Elapsed", fmtDuration(timing.elapsed_s)],
        ["Est. remaining", fmtDuration(timing.eta_s)],
        ["Rolling avg / trial", timing.trial_sec_rolling != null ? fmtDuration(timing.trial_sec_rolling) : "—"],
        ["Last 20 avg / trial", timing.trial_sec_last20 != null ? fmtDuration(timing.trial_sec_last20) : "—"],
        ["Avg / season", timing.season_sec_avg != null ? fmtDuration(timing.season_sec_avg) : "—"],
        ["Slowest recent trial", timing.slowest_trial_s != null ? fmtDuration(timing.slowest_trial_s) : "—"],
      ];
      timingEl.innerHTML = blocks.map(([label, value]) =>
        `<div class="stat-block"><div class="label">${label}</div><div class="value">${value}</div></div>`
      ).join("");
    }
  }

  async function refreshViewerLog(jobId) {
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/log?tail=80`);
      const el = $("#rv-log-tail");
      if (!el) return;
      const text = (data.lines || []).join("\n") || "(empty log)";
      const grew = text.length > state.lastLogLen;
      el.classList.remove("muted");
      el.textContent = text;
      state.lastLogLen = text.length;
      if (state.followLog && grew) {
        el.scrollTop = el.scrollHeight;
      }
    } catch (_) {}
  }

  function initDocsToc() {
    const nav = $("#docs-toc-nav");
    if (!nav) return;
    const links = $$("a", nav);
    const sections = links
      .map((a) => document.getElementById((a.getAttribute("href") || "").slice(1)))
      .filter(Boolean);
    const onScroll = () => {
      const y = window.scrollY + 80;
      let active = sections[0];
      sections.forEach((sec) => {
        if (sec.offsetTop <= y) active = sec;
      });
      links.forEach((a) => {
        a.classList.toggle("is-active", a.getAttribute("href") === "#" + (active && active.id));
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  async function showRunViewer(jobId) {
    state.viewerJobId = jobId;
    switchTab("run");
    setRunMode("viewer");
    watchJob(jobId);
    try {
      const tel = await api(`/api/jobs/${encodeURIComponent(jobId)}/telemetry`);
      renderTelemetry(tel);
    } catch (e) {
      $("#rv-headline").textContent = "Could not load telemetry yet — waiting for log…";
    }
    await refreshViewerLog(jobId);
  }

  async function refreshJobs() {
    state.jobs = await api("/api/jobs");
    // Attach telemetry for active suite jobs (compact dock status).
    await Promise.all(
      state.jobs
        .filter((j) => VIEWER_MODULES.has(j.module) && ["queued", "running"].includes(j.status))
        .slice(0, 4)
        .map(async (j) => {
          try {
            j.telemetry = await api(`/api/jobs/${encodeURIComponent(j.id)}/telemetry`);
          } catch (_) {}
        })
    );
    renderJobsLive();
    if (state.tab === "jobs") renderJobsFull();
    state.jobs.forEach((j) => {
      if (["queued", "running"].includes(j.status) && !state.sse[j.id]) watchJob(j.id);
    });
    if (state.viewerJobId && state.runMode === "viewer") {
      const j = state.jobs.find((x) => x.id === state.viewerJobId);
      if (j && j.telemetry) renderTelemetry({ ...j.telemetry, job_id: j.id, status: j.status });
    }
  }

  function watchJob(jobId) {
    if (state.sse[jobId]) return;
    const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    state.sse[jobId] = es;
    es.onmessage = (ev) => {
      try {
        const meta = JSON.parse(ev.data);
        const idx = state.jobs.findIndex((j) => j.id === meta.id);
        if (idx >= 0) state.jobs[idx] = { ...state.jobs[idx], ...meta };
        else state.jobs.unshift(meta);
        renderJobsLive();
        if (state.tab === "jobs") renderJobsFull();
        if (state.logWatch === jobId) refreshLogModal(jobId, false);
        if (state.viewerJobId === jobId && meta.telemetry) {
          renderTelemetry(meta.telemetry);
          if (state.runMode === "viewer") refreshViewerLog(jobId);
        }
        if (["succeeded", "failed", "cancelled"].includes(meta.status)) {
          es.close();
          delete state.sse[jobId];
          if (meta.status === "succeeded") toast(`${meta.module} succeeded`, "ok");
          if (meta.status === "failed") toast(`${meta.module} failed — open Viewer / log`, "error");
        }
      } catch (_) {}
    };
    es.onerror = () => {
      es.close();
      delete state.sse[jobId];
    };
  }

  async function startJob(module, tab, params) {
    const picker = $("#run-picker");
    const runId = state.runId || (picker && picker.value) || null;
    if (runId) state.runId = runId;
    const body = {
      module,
      tab: tab || state.tab,
      params: { ...(params || {}), run_id: (params && params.run_id) || runId || undefined },
      run_id: runId || null,
    };
    try {
      const ui = await api("/api/ui-state");
      ui[tab || state.tab] = params || {};
      ui.run_id = runId;
      await api("/api/ui-state", { method: "POST", body: JSON.stringify(ui) });
    } catch (_) {}
    const meta = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    state.jobs.unshift(meta);
    renderJobsLive();
    watchJob(meta.id);
    toast(`Started ${module} (${meta.id})`, "ok");
    if (VIEWER_MODULES.has(module)) {
      await showRunViewer(meta.id);
    }
    return meta;
  }

  async function refreshLogModal(jobId, open) {
    const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/log?tail=400`);
    $("#log-modal-title").textContent = `Log · ${jobId} · ${data.status || ""}`;
    $("#log-modal-body").textContent = (data.lines || []).join("\n") || "(empty log)";
    if (open) {
      state.logWatch = jobId;
      $("#log-modal").classList.remove("hidden");
    }
  }

  // ---- Wire events ----
  function bind() {
    $$(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
    $("#run-picker").addEventListener("change", async (e) => {
      state.runId = e.target.value || null;
      state.page = 1;
      await loadSummary();
      await loadResults();
      await loadArtifacts();
      await renderCharts("all", "#charts-all");
      if (state.tab === "docs") await loadHealthReport();
    });
    $("#btn-refresh-runs").addEventListener("click", loadAll);
    $("#results-search").addEventListener("change", () => {
      state.page = 1;
      loadResults();
    });
    $("#page-prev").addEventListener("click", () => {
      if (state.page > 1) {
        state.page -= 1;
        loadResults();
      }
    });
    $("#page-next").addEventListener("click", () => {
      state.page += 1;
      loadResults();
    });
    $("#btn-export-csv").addEventListener("click", async () => {
      const r = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({ run_id: state.runId, tab: "all" }),
      });
      toast("Exported: " + r.path, "ok");
    });
    const smoke = () => startJob("suite_smoke", "run", { skip_integrity: true, preset: "smoke" });
    $("#btn-suite-smoke").addEventListener("click", smoke);
    $("#btn-empty-smoke").addEventListener("click", smoke);
    const emptyDocs = $("#btn-empty-docs");
    if (emptyDocs) emptyDocs.addEventListener("click", () => switchTab("docs"));
    $("#btn-promo-gates").addEventListener("click", () =>
      startJob("promotion_gates", "all", { run_id: state.runId })
    );

    $("#suite-preset").addEventListener("change", (e) => applySuitePreset(e.target.value));
    $("#btn-launch-suite").addEventListener("click", async () => {
      const params = readSchemaForm($("#suite-form"));
      params.preset = $("#suite-preset").value;
      $("#run-cmd-preview").textContent = "Launching suite_custom… " + JSON.stringify(params);
      await startJob("suite_custom", "run", params);
    });
    $("#btn-launch-smoke-lab").addEventListener("click", () =>
      startJob("suite_smoke", "run", { skip_integrity: true, preset: "smoke" })
    );
    $("#btn-launch-backtest").addEventListener("click", async () => {
      const params = readSchemaForm($("#backtest-form"));
      await startJob("backtest_quick", "run", params);
    });

    $$(".run-mode").forEach((b) => {
      b.addEventListener("click", () => setRunMode(b.dataset.runMode));
    });
    $("#btn-run-viewer").addEventListener("click", async () => {
      setRunMode("viewer");
      if (!state.viewerJobId) {
        const cand = state.jobs.find((j) => VIEWER_MODULES.has(j.module));
        if (cand) await showRunViewer(cand.id);
      } else {
        await showRunViewer(state.viewerJobId);
      }
    });

    $("#player-search").addEventListener("change", loadPlayers);
    $("#btn-players-export").addEventListener("click", async () => {
      const r = await api("/api/players/export" + (state.runId ? `?run_id=${encodeURIComponent(state.runId)}` : ""), {
        method: "POST",
      });
      toast("Exported: " + r.path, "ok");
    });
    $("#btn-player-smoke").addEventListener("click", () =>
      startJob("player_rating_smoke", "player", { fast_tuning: true, quick: true })
    );

    $("#btn-ats-report").addEventListener("click", () =>
      startJob("ats_reliability", "ats", { ...formParams("#ats-form"), run_id: state.runId })
    );
    $("#btn-edge-policy").addEventListener("click", () =>
      startJob("edge_policy_calib", "ats", { run_id: state.runId })
    );
    $("#btn-ats-suite").addEventListener("click", () =>
      startJob("suite_smoke", "ats", { ...formParams("#ats-form"), skip_integrity: true })
    );
    $("#btn-ats-export").addEventListener("click", async () => {
      const r = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({
          run_id: state.runId,
          filename: `ats_scorecard_${Date.now()}.csv`,
          columns: ["DATE", "EDGE", "ATS_WIN", "CALIBRATED_COVER_PROB", "POINT_CLV", "PNL"],
        }),
      });
      toast("Exported: " + r.path, "ok");
    });

    $("#btn-ml-report").addEventListener("click", () =>
      startJob("ml_calibration_report", "ml", { ...formParams("#ml-form"), run_id: state.runId })
    );
    $("#btn-ml-export").addEventListener("click", async () => {
      const r = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({
          run_id: state.runId,
          filename: `ml_scorecard_${Date.now()}.csv`,
          columns: ["DATE", "WIN_PROB", "P_HOME", "MARKET_ML", "MARKET_ML_AWAY", "ACTUAL_HOME", "ACTUAL_AWAY"],
        }),
      });
      toast("Exported: " + r.path, "ok");
    });

    $("#btn-totals-report").addEventListener("click", () =>
      startJob("totals_calibration_report", "totals", {
        ...formParams("#totals-form"),
        run_id: state.runId,
      })
    );
    $("#btn-totals-export").addEventListener("click", async () => {
      const r = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({
          run_id: state.runId,
          filename: `totals_scorecard_${Date.now()}.csv`,
          columns: ["DATE", "PRED_TOTAL", "MARKET_TOTAL", "TOTAL_ERR", "ACTUAL_HOME", "ACTUAL_AWAY", "OU_HIT"],
        }),
      });
      toast("Exported: " + r.path, "ok");
    });

    async function onJobListClick(ev) {
      const cancelId = ev.target.getAttribute("data-cancel");
      if (cancelId) {
        await api(`/api/jobs/${encodeURIComponent(cancelId)}/cancel`, { method: "POST" });
        printCancel(cancelId);
        await refreshJobs();
        return;
      }
      const dismissId = ev.target.getAttribute("data-dismiss");
      if (dismissId) {
        await api(`/api/jobs/${encodeURIComponent(dismissId)}`, { method: "DELETE" });
        await refreshJobs();
        return;
      }
      const viewerId = ev.target.getAttribute("data-viewer");
      if (viewerId) {
        await showRunViewer(viewerId);
        return;
      }
      const logId = ev.target.getAttribute("data-log");
      if (logId) {
        await refreshLogModal(logId, true);
        return;
      }
      const paramsId = ev.target.getAttribute("data-params");
      if (paramsId) {
        const j = state.jobs.find((x) => x.id === paramsId);
        if (j && window.JsonView) {
          const host = document.createElement("div");
          JsonView.render(host, { params: j.params, result_paths: j.result_paths, error: j.error }, `Job ${paramsId}`);
          $("#log-modal-title").textContent = `Params · ${paramsId}`;
          $("#log-modal-body").innerHTML = host.innerHTML;
          $("#log-modal").classList.remove("hidden");
          state.logWatch = null;
        }
      }
    }
    function printCancel(id) {
      toast(`Cancel requested: ${id}`, "ok");
      console.info("[dashboard] cancel", id);
    }
    $("#jobs-live").addEventListener("click", onJobListClick);
    $("#jobs-full").addEventListener("click", onJobListClick);

    async function clearFinished() {
      const r = await api("/api/jobs/clear-finished", { method: "POST" });
      toast(`Cleared ${r.cleared} finished jobs`, "ok");
      await refreshJobs();
    }
    $("#btn-clear-finished").addEventListener("click", clearFinished);
    $("#btn-clear-finished-drawer").addEventListener("click", clearFinished);

    $("#jobs-drawer-toggle").addEventListener("click", () => {
      const d = $("#jobs-drawer");
      d.classList.toggle("collapsed");
      $("#jobs-drawer-toggle").textContent = d.classList.contains("collapsed") ? "Show" : "Hide";
    });
    const btnExpand = $("#btn-jobs-expand");
    if (btnExpand) {
      btnExpand.addEventListener("click", () => {
        const d = $("#jobs-drawer");
        d.classList.remove("collapsed");
        d.classList.toggle("is-expanded");
        btnExpand.textContent = d.classList.contains("is-expanded") ? "Shrink" : "Expand";
        $("#jobs-drawer-toggle").textContent = "Hide";
      });
    }
    const btnTheme = $("#btn-theme");
    if (btnTheme) btnTheme.addEventListener("click", toggleTheme);
    const btnTermExpand = $("#btn-rv-terminal-expand");
    if (btnTermExpand) {
      btnTermExpand.addEventListener("click", () => {
        const dock = $("#rv-terminal-dock");
        if (!dock) return;
        dock.classList.toggle("is-expanded");
        btnTermExpand.textContent = dock.classList.contains("is-expanded") ? "Shrink" : "Expand";
      });
    }
    const btnFollow = $("#btn-rv-follow-log");
    if (btnFollow) {
      btnFollow.addEventListener("click", () => {
        state.followLog = !state.followLog;
        btnFollow.textContent = state.followLog ? "Follow · on" : "Follow · off";
        if (state.followLog) {
          const el = $("#rv-log-tail");
          if (el) el.scrollTop = el.scrollHeight;
        }
      });
      btnFollow.textContent = "Follow · on";
    }
    const btnDocsBench = $("#btn-docs-refresh-bench");
    if (btnDocsBench) btnDocsBench.addEventListener("click", () => loadBenchReport());
    const btnDocsHealth = $("#btn-docs-refresh-health");
    if (btnDocsHealth) btnDocsHealth.addEventListener("click", () => loadHealthReport());
    const btnDocsViewer = $("#btn-docs-open-viewer");
    if (btnDocsViewer) {
      btnDocsViewer.addEventListener("click", () => {
        switchTab("run");
        setRunMode("viewer");
        toast("Run Viewer — watch a live suite from Jobs or launch one in Configure", "ok");
      });
    }
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (ev) => {
        if (!localStorage.getItem("t60-theme")) {
          applyTheme(ev.matches ? "dark" : "light");
        }
      });
    }
    $("#log-modal-close").addEventListener("click", () => {
      $("#log-modal").classList.add("hidden");
      state.logWatch = null;
    });
    $("#log-modal").addEventListener("click", (ev) => {
      if (ev.target.id === "log-modal") {
        $("#log-modal").classList.add("hidden");
        state.logWatch = null;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    bind();
    await loadModules();
    await refreshJobs();
    await loadAll();
    setInterval(refreshJobs, 8000);
    setInterval(() => {
      if (state.logWatch) refreshLogModal(state.logWatch, false).catch(() => {});
    }, 2500);
  });
})();
