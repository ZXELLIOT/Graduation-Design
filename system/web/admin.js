/**
 * admin.js — 后台管理页面
 * 密码登录 + 性能监控 + 全参数调优 + 日志查看
 */

// ============================================================
// 登录
// ============================================================
const loginOverlay = document.getElementById("login-overlay");
const adminContent = document.getElementById("admin-content");
const adminPwd = document.getElementById("admin-pwd");
const loginBtn = document.getElementById("login-btn");
const loginErr = document.getElementById("login-err");
const adminNotice = document.getElementById("admin-notice");

const ADMIN_AUTH_KEY = "simcse_admin_authed_v1";
const COMPARATOR_SYNC_KEY = "simcse_comparator_sync_v1";

function markAdminAuthed() {
  try { sessionStorage.setItem(ADMIN_AUTH_KEY, "1"); } catch (e) {}
}

function hasAuthed() {
  let authed = "";
  try { authed = String(sessionStorage.getItem(ADMIN_AUTH_KEY) || ""); } catch (e) {}
  return authed === "1";
}

function emitComparatorSync(partial) {
  const payload = { ...partial, ts: Date.now() };
  try { localStorage.setItem(COMPARATOR_SYNC_KEY, JSON.stringify(payload)); } catch (e) {}
}

function esc(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function showAdminNotice(message, isError = true) {
  if (!adminNotice) return;
  adminNotice.style.display = "block";
  adminNotice.textContent = String(message || "");
  if (isError) {
    adminNotice.style.color = "#7b1e1e";
    adminNotice.style.background = "#ffe6e6";
    adminNotice.style.border = "1px solid #f2bbbb";
  } else {
    adminNotice.style.color = "#0b6e4f";
    adminNotice.style.background = "#dff5eb";
    adminNotice.style.border = "1px solid #b7e6d3";
  }
}

function clearAdminNotice() {
  if (!adminNotice) return;
  adminNotice.style.display = "none";
  adminNotice.textContent = ""; 
}

async function apiJson(url, options = {}, label = "接口") {
  let res;
  try {
    res = await fetch(url, options);
  } catch (e) {
    throw new Error(`${label}请求失败：网络异常`);
  }
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    throw new Error(`${label}返回异常：非JSON响应`);
  }
  if (!res.ok) {
    const detail = (data && data.detail) ? String(data.detail) : `HTTP ${res.status}`;
    throw new Error(`${label}失败：${detail}`);
  }
  return data;
}

async function doLogin() {
  const pwd = adminPwd.value.trim();
  if (!pwd) { loginErr.textContent = "请输入密码"; return; }
  try {
    const data = await apiJson("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pwd }),
    }, "登录接口");
    if (data.ok) {
      markAdminAuthed();
      loginOverlay.classList.add("hidden");
      adminContent.classList.remove("hidden");
      initAdmin();
    }
    else { loginErr.textContent = data.detail || "密码错误"; }
  } catch (e) { loginErr.textContent = e.message || "网络错误"; }
}
loginBtn.addEventListener("click", doLogin);
adminPwd.addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });

// ============================================================
// 初始化
// ============================================================
let perfTimer = null;
const perfSeries = {
  labels: [],
  cpu: [],
  mem: [],
  resp: [],
  sim: [],
};
const PERF_CAP = 40;
function _fmt(n, d) { const v = Number(n); return Number.isFinite(v) ? v.toFixed(d || 1) : "0.0"; }

function initAdmin() {
  if (perfTimer) clearInterval(perfTimer);
  loadPerf();
  loadLogs();
  loadSettings();
  perfTimer = setInterval(loadPerf, 2000);
}

// ============================================================
// 性能监控
// ============================================================
async function loadPerf() {
  try {
    const d = await apiJson("/api/perf", {}, "性能接口");
    const dev = d.device || {}, chat = d.chat || {};

    const cpu = Number(dev.cpu_percent);
    const mem = Number(dev.memory_percent);
    const resp = Number(chat.latest_elapsed_ms);
    const sim = Number(chat.latest_score);

    const invalid = [cpu, mem, resp, sim].some(v => !Number.isFinite(v));
    if (invalid) {
      showAdminNotice("性能接口返回异常值，已忽略本次刷新。", true);
      return;
    }

    document.getElementById("perf-time").textContent = d.timestamp || "--:--:--";
    document.getElementById("perf-cpu").textContent = `系统CPU ${_fmt(cpu)}%`;
    document.getElementById("perf-mem").textContent = `系统内存 ${_fmt(mem)}%`;
    document.getElementById("perf-resp").textContent = `响应时间 ${_fmt(resp)}ms`;
    document.getElementById("perf-sim").textContent = `相似度 ${_fmt(sim, 3)}`;

    perfSeries.labels.push(d.timestamp || "");
    perfSeries.cpu.push(cpu);
    perfSeries.mem.push(mem);
    perfSeries.resp.push(resp);
    perfSeries.sim.push(sim);

    for (let k of Object.keys(perfSeries)) if (perfSeries[k].length > PERF_CAP) perfSeries[k].shift();
    drawChart(document.getElementById("perf-cpu-chart"), perfSeries.labels, [{ color: "#2a9d8f", data: perfSeries.cpu }]);
    drawChart(document.getElementById("perf-mem-chart"), perfSeries.labels, [{ color: "#3b82f6", data: perfSeries.mem }]);
    drawChart(document.getElementById("perf-resp-chart"), perfSeries.labels, [{ color: "#ef4444", data: perfSeries.resp }]);
    drawChart(document.getElementById("perf-sim-chart"), perfSeries.labels, [{ color: "#f59e0b", data: perfSeries.sim }]);
    clearAdminNotice();
  } catch (e) {
    showAdminNotice(e.message || "性能接口异常", true);
  }
}

function drawChart(canvas, labels, seriesList) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d"), cssW = Math.max(200, canvas.clientWidth || 200), cssH = Math.max(100, canvas.clientHeight || 100);
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) { canvas.width = Math.floor(cssW * dpr); canvas.height = Math.floor(cssH * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#fffdf8"; ctx.fillRect(0, 0, cssW, cssH);
  const pad = { l: 30, r: 8, t: 6, b: 16 }, pw = Math.max(10, cssW - pad.l - pad.r), ph = Math.max(10, cssH - pad.t - pad.b);
  let yMin = Infinity, yMax = -Infinity;
  for (const s of seriesList) for (const v of s.data) if (Number.isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
  if (!Number.isFinite(yMin)) { yMin = 0; yMax = 1; }
  if (Math.abs(yMax - yMin) < 1e-6) yMax = yMin + 1;
  ctx.strokeStyle = "#e8dcc5"; ctx.lineWidth = 0.8;
  for (let i = 0; i <= 3; i++) { const y = pad.t + (ph * i) / 3; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke(); }
  ctx.fillStyle = "#6a7683"; ctx.font = "9px Microsoft YaHei"; ctx.fillText(_fmt(yMax, 1), 2, pad.t + 4); ctx.fillText(_fmt(yMin, 1), 2, pad.t + ph);
  const n = Math.max(labels.length, 1), xStep = n > 1 ? pw / (n - 1) : 0;
  for (const s of seriesList) { ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i < s.data.length; i++) { const x = pad.l + xStep * i, y = pad.t + ((yMax - s.data[i]) / (yMax - yMin)) * ph; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke(); }
}

// ============================================================
// 日志
// ============================================================
async function loadLogs() {
  try {
    const logs = await apiJson("/api/logs/latest?limit=60", {}, "日志接口");
    const panel = document.getElementById("logs-panel");
    if (!Array.isArray(logs)) {
      showAdminNotice("日志接口返回异常值：数据不是数组。", true);
      return;
    }
    if (logs.length === 0) { panel.textContent = "暂无日志"; return; }
    panel.innerHTML = logs.reverse().map(item => {
      const q = item.context_enabled
        ? esc(item.user_input || item.raw_user_input || "")
        : esc(item.raw_user_input || item.user_input || "");
      const ctxFlag = item.context_enabled ? "开" : "关";
      const aiFlag = item.ai_enhanced ? "开" : "关";
      const matchedQuery = esc(item.matched_query || item.expected_reply || "");
      const score = _fmt(item.score, 3);
      const elapsed = _fmt(item.elapsed_ms, 0);
      const aiElapsed = _fmt(item.ai_elapsed_ms, 0);
      const line5 = item.ai_enhanced
        ? `<div>AI输出: ${esc(item.ai_output || item.reply || "")}</div>`
        : `<div>输出答句: ${esc(item.reply || "")}</div>`;
      const line6 = item.ai_enhanced
        ? `<div>总时间: ${elapsed}ms | AI响应: ${aiElapsed}ms</div>`
        : `<div>时间: ${elapsed}ms</div>`;
      return `<div class="log-item">
        <div>[上下文:${ctxFlag}] [AI:${aiFlag}]</div>
        <div>用户输入: ${q}</div>
        <div>匹配问句: ${matchedQuery} | 相似度: ${score}</div>
        ${line5}
        ${line6}
      </div>`;
    }).join("");
  } catch (e) {
    showAdminNotice(e.message || "日志接口异常", true);
  }
}
document.getElementById("refresh-logs").addEventListener("click", loadLogs);
document.getElementById("clear-logs").addEventListener("click", async () => {
  if (!confirm("确认清除全部日志？")) return;
  try {
    await apiJson("/api/logs/clear", { method: "POST" }, "清空日志接口");
    loadLogs();
  } catch (e) {
    showAdminNotice(e.message || "清空日志失败", true);
  }
});

// ============================================================
// 调参：读取 & 应用
// ============================================================
function setVal(id, val) { const el = document.getElementById(id); if (!el) return; if (el.type === "checkbox") el.checked = Boolean(val); else el.value = String(val); }
function getVal(id) { const el = document.getElementById(id); if (!el) return null; if (el.type === "checkbox") return el.checked; return el.value; }

async function loadSettings() {
  try {
    const d = await apiJson("/api/settings/comparator", {}, "配置读取接口");

    const numericKeys = [
      "similarity_threshold", "rerank_query_weight", "rerank_reply_weight",
      "coarse_recall_count", "rerank_top_k", "context_max_turns",
      "max_text_len",
    ];
    const hasInvalidNumber = numericKeys.some((k) => !Number.isFinite(Number(d[k])));
    if (hasInvalidNumber) {
      showAdminNotice("配置读取接口返回异常值，部分字段不是有效数字。", true);
      return;
    }

    setVal("similarity-threshold", Number(d.similarity_threshold).toFixed(2));
    setVal("rerank-query-weight", Number(d.rerank_query_weight).toFixed(2));
    setVal("rerank-reply-weight", Number(d.rerank_reply_weight).toFixed(2));
    setVal("coarse-recall-count", Math.round(Number(d.coarse_recall_count)));
    setVal("rerank-top-k", Math.round(Number(d.rerank_top_k)));
    setVal("context-matching-enabled", d.context_matching_enabled);
    setVal("context-max-turns", Math.round(Number(d.context_max_turns)));
    setVal("max-text-len", Math.round(Number(d.max_text_len)));
    setVal("ai-enhanced", d.ai_enhanced);
    setVal("ai-timeout-sec", isFiniteNumber(Number(d.ai_timeout_sec)) ? Number(d.ai_timeout_sec).toFixed(0) : "20");
    // AI只读信息
    const setRo = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = String(val ?? ""); };
    setRo("ai-model-name", d.ai_model_name || "(未配置)");
    setRo("ai-responses-url", d.ai_responses_url || "(未配置)");
    const apiReadyEl = document.getElementById("ai-api-ready");
    if (apiReadyEl) {
      apiReadyEl.textContent = d.ai_api_ready ? "✓ 已配置" : "✗ 未配置";
      apiReadyEl.style.color = d.ai_api_ready ? "#1f9d55" : "#d03838";
    }
    document.getElementById("reload-settings").textContent = "已读取";
    emitComparatorSync({
      ai_enhanced: Boolean(d.ai_enhanced),
      context_matching_enabled: Boolean(d.context_matching_enabled),
    });
    setTimeout(() => { document.getElementById("reload-settings").textContent = "读取当前配置"; }, 1500);
  } catch (e) {
    document.getElementById("reload-settings").textContent = "读取失败";
    showAdminNotice(e.message || "配置读取失败", true);
  }
}

document.getElementById("reload-settings").addEventListener("click", loadSettings);

document.getElementById("apply-settings").addEventListener("click", async () => {
  const payload = {
    similarity_threshold: Number(getVal("similarity-threshold")),
    rerank_query_weight: Number(getVal("rerank-query-weight")),
    rerank_reply_weight: Number(getVal("rerank-reply-weight")),
    coarse_recall_count: Math.round(Number(getVal("coarse-recall-count"))),
    rerank_top_k: Math.round(Number(getVal("rerank-top-k"))),
    context_matching_enabled: getVal("context-matching-enabled"),
    context_max_turns: Math.round(Number(getVal("context-max-turns"))),
    max_text_len: Math.round(Number(getVal("max-text-len"))),
    ai_enhanced: getVal("ai-enhanced"),
    ai_timeout: Number(getVal("ai-timeout-sec")),
  };
  try {
    const hasInvalidPayload = Object.entries(payload)
      .filter(([k]) => k !== "context_matching_enabled" && k !== "ai_enhanced")
      .some(([, v]) => !Number.isFinite(Number(v)));
    if (hasInvalidPayload) {
      throw new Error("应用失败：提交参数中存在非法数值。");
    }

    await apiJson("/api/settings/comparator", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }, "配置更新接口");
    emitComparatorSync({
      ai_enhanced: Boolean(payload.ai_enhanced),
      context_matching_enabled: Boolean(payload.context_matching_enabled),
    });
    document.getElementById("settings-status").textContent = "参数已生效";
    document.getElementById("settings-status").style.color = "#1f9d55";
    clearAdminNotice();
    setTimeout(() => { document.getElementById("settings-status").textContent = ""; }, 2000);
  } catch (e) {
    document.getElementById("settings-status").textContent = "应用失败";
    document.getElementById("settings-status").style.color = "#d03838";
    showAdminNotice(e.message || "应用失败", true);
  }
});

if (hasAuthed()) {
  loginOverlay.classList.add("hidden");
  adminContent.classList.remove("hidden");
  initAdmin();
}
