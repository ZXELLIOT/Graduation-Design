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

async function doLogin() {
  const pwd = adminPwd.value.trim();
  if (!pwd) { loginErr.textContent = "请输入密码"; return; }
  try {
    const res = await fetch("/api/admin/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: pwd }) });
    const data = await res.json();
    if (data.ok) { loginOverlay.classList.add("hidden"); adminContent.classList.remove("hidden"); initAdmin(); }
    else { loginErr.textContent = data.detail || "密码错误"; }
  } catch (e) { loginErr.textContent = "网络错误"; }
}
loginBtn.addEventListener("click", doLogin);
adminPwd.addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });

// ============================================================
// 初始化
// ============================================================
let perfTimer = null;
const perfSeries = { labels: [], cpu: [], mem: [], chatAvg: [], chatP95: [] };
const PERF_CAP = 40;
function _fmt(n, d) { const v = Number(n); return Number.isFinite(v) ? v.toFixed(d || 1) : "0.0"; }

function initAdmin() { loadPerf(); loadLogs(); loadSettings(); perfTimer = setInterval(loadPerf, 2000); }

// ============================================================
// 性能监控
// ============================================================
async function loadPerf() {
  try {
    const d = await (await fetch("/api/perf")).json();
    const dev = d.device || {}, chat = d.chat || {};
    document.getElementById("perf-time").textContent = d.timestamp || "--:--:--";
    document.getElementById("perf-cpu").textContent = `CPU ${_fmt(dev.cpu_percent)}%`;
    document.getElementById("perf-mem").textContent = `内存 ${_fmt(dev.memory_percent)}%`;
    document.getElementById("perf-chat-avg").textContent = `均时延 ${_fmt(chat.avg_elapsed_ms)}ms`;
    document.getElementById("perf-chat-p95").textContent = `P95 ${_fmt(chat.p95_elapsed_ms)}ms`;
    perfSeries.labels.push(d.timestamp || ""); perfSeries.cpu.push(Number(dev.cpu_percent || 0)); perfSeries.mem.push(Number(dev.memory_percent || 0));
    perfSeries.chatAvg.push(Number(chat.avg_elapsed_ms || 0)); perfSeries.chatP95.push(Number(chat.p95_elapsed_ms || 0));
    for (let k of Object.keys(perfSeries)) if (perfSeries[k].length > PERF_CAP) perfSeries[k].shift();
    drawChart(document.getElementById("perf-device-chart"), perfSeries.labels, [{ color: "#2a9d8f", data: perfSeries.cpu }, { color: "#3b82f6", data: perfSeries.mem }]);
    drawChart(document.getElementById("perf-chat-chart"), perfSeries.labels, [{ color: "#ef4444", data: perfSeries.chatAvg }, { color: "#f59e0b", data: perfSeries.chatP95 }]);
  } catch (e) {}
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
    const logs = await (await fetch("/api/logs/latest?limit=30")).json();
    const panel = document.getElementById("logs-panel");
    if (!Array.isArray(logs) || logs.length === 0) { panel.textContent = "暂无日志"; return; }
    panel.innerHTML = logs.reverse().map(item => `<div class="log-item"><div class="log-meta">${(item.timestamp||"").replace(/T/," ")} | ${item.ai_enhanced?"AI":"本地"} | ${item.context_enabled?"Ctx开启":""}</div><div>Q: ${(item.raw_user_input||item.user_input||"").slice(0,50)}</div><div style="color:#1f9d55">A: ${(item.reply||"").slice(0,80)}</div><div style="color:#607080;font-size:0.75rem;">score=${typeof item.score==="number"?item.score.toFixed(3):"0"} | ${_fmt(item.elapsed_ms,0)}ms</div></div>`).join("");
  } catch (e) {}
}
document.getElementById("refresh-logs").addEventListener("click", loadLogs);
document.getElementById("clear-logs").addEventListener("click", async () => { if (!confirm("确认清除全部日志？")) return; await fetch("/api/logs/clear", { method: "POST" }); loadLogs(); });

// ============================================================
// 调参：读取 & 应用
// ============================================================
function setVal(id, val) { const el = document.getElementById(id); if (!el) return; if (el.type === "checkbox") el.checked = Boolean(val); else el.value = String(val); }
function getVal(id) { const el = document.getElementById(id); if (!el) return null; if (el.type === "checkbox") return el.checked; return el.value; }

async function loadSettings() {
  try {
    const d = await (await fetch("/api/settings/comparator")).json();
    setVal("similarity-threshold", Number(d.similarity_threshold).toFixed(2));
    setVal("rerank-query-weight", Number(d.rerank_query_weight).toFixed(2));
    setVal("rerank-reply-weight", Number(d.rerank_reply_weight).toFixed(2));
    setVal("coarse-recall-count", Math.round(Number(d.coarse_recall_count)));
    setVal("rerank-top-k", Math.round(Number(d.rerank_top_k)));
    setVal("context-matching-enabled", d.context_matching_enabled);
    setVal("context-max-turns", Math.round(Number(d.context_max_turns)));
    setVal("max-text-len", Math.round(Number(d.max_text_len)));
    setVal("ai-enhanced", d.ai_enhanced);
    setVal("ai-topk", Math.round(Number(d.ai_topk)));
    setVal("ai-timeout", Math.round(Number(d.ai_timeout_sec)));
    setVal("ai-model-name", d.ai_model_name || "");
    setVal("ai-responses-url", d.ai_responses_url || "");
    document.getElementById("ai-key-status").textContent = d.ai_api_ready ? "已配置" : "未配置";
    document.getElementById("ai-key-status").style.color = d.ai_api_ready ? "#1f9d55" : "#d03838";
    document.getElementById("reload-settings").textContent = "已读取";
    setTimeout(() => { document.getElementById("reload-settings").textContent = "读取当前配置"; }, 1500);
  } catch (e) { document.getElementById("reload-settings").textContent = "读取失败"; }
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
  };
  try {
    await fetch("/api/settings/comparator", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    document.getElementById("settings-status").textContent = "参数已生效";
    setTimeout(() => { document.getElementById("settings-status").textContent = ""; }, 2000);
  } catch (e) { document.getElementById("settings-status").textContent = "应用失败"; document.getElementById("settings-status").style.color = "#d03838"; }
});
