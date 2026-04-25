const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat");
const statusNode = document.getElementById("status");
const logsPanel = document.getElementById("logs-panel");
const refreshLogsBtn = document.getElementById("refresh-logs");
const clearLogsBtn = document.getElementById("clear-logs");
const similarityThresholdInput = document.getElementById("similarity-threshold");
const rerankQueryWeightInput = document.getElementById("rerank-query-weight");
const rerankReplyWeightInput = document.getElementById("rerank-reply-weight");
const contextMaxTurnsInput = document.getElementById("context-max-turns");
const contextMatchingEnabledInput = document.getElementById("context-matching-enabled");
const coarseRecallCountInput = document.getElementById("coarse-recall-count");
const rerankTopKInput = document.getElementById("rerank-top-k");
const aiEnhancedInput = document.getElementById("ai-enhanced");
const reloadSettingsBtn = document.getElementById("reload-settings");
const applySettingsBtn = document.getElementById("apply-settings");
const perfLastUpdatedNode = document.getElementById("perf-last-updated");
const perfCpuNode = document.getElementById("perf-cpu");
const perfMemNode = document.getElementById("perf-mem");
const perfChatAvgNode = document.getElementById("perf-chat-avg");
const perfChatP95Node = document.getElementById("perf-chat-p95");
const perfDeviceCanvas = document.getElementById("perf-device-chart");
const perfChatCanvas = document.getElementById("perf-chat-chart");

let history = [];
let conversationId = null;
let perfTimerId = null;
const perfSeries = {
  labels: [],
  cpu: [],
  mem: [],
  chatAvg: [],
  chatP95: [],
};
const PERF_CAP = 40;

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatResultType(resultType) {
  const key = String(resultType || "unknown");
  if (key === "matched") {
    return "本地命中";
  }
  if (key === "ai_enhanced_matched") {
    return "AI增强命中";
  }
  if (key === "below_threshold") {
    return "低于阈值拒答";
  }
  if (key === "no_candidate") {
    return "无候选拒答";
  }
  if (key === "invalid_input") {
    return "输入无效";
  }
  return key;
}

function formatContextReason(reason) {
  const key = String(reason || "unknown");
  const map = {
    short_query: "短文本(<=4字)，拼接上下文",
    trigger_keyword: "命中关键词，直接拼接",
    no_context_needed: "无需上下文，直接检索原问题",
    no_history: "无历史，不拼接",
    invalid_input: "输入无效",
    empty_input: "空输入",
    disabled_by_config: "配置关闭，不拼接",
  };
  return map[key] || key;
}

function formatAiFallbackReason(reason) {
  const key = String(reason || "none");
  const map = {
    none: "未回退",
    not_enabled: "未开启AI增强",
    no_candidate: "无候选",
    below_threshold: "低于阈值",
    ai_empty: "AI返回为空，已回退本地",
    ai_and_local_empty: "AI与本地都为空",
  };
  return map[key] || key;
}

function renderLogItem(item) {
  const ts = escapeHtml(item.timestamp || "");
  const resultType = formatResultType(item.result_type);
  const aiEnhanced = Boolean(item.ai_enhanced);
  const aiFallback = Boolean(item.ai_fallback);
  const aiFallbackReason = formatAiFallbackReason(item.ai_fallback_reason);
  const contextEnabled = Boolean(item.context_enabled);
  const contextReason = formatContextReason(item.context_reason);
  const userRaw = escapeHtml(item.raw_user_input || item.user_input || "");
  const userUsed = escapeHtml(item.user_input || "");
  const replyText = escapeHtml(item.reply || "");
  const matchedQuery = escapeHtml(item.matched_query || item.top_query || "无");
  const matchedReply = escapeHtml(item.top_reply || item.expected_reply || "无");

  const score = typeof item.score === "number" ? item.score.toFixed(4) : "0.0000";
  const elapsed = typeof item.elapsed_ms === "number" ? item.elapsed_ms.toFixed(2) : "0.00";
  const aiElapsed = typeof item.ai_elapsed_ms === "number" ? item.ai_elapsed_ms.toFixed(2) : "0.00";
  const coarseCount = Number.isFinite(Number(item.coarse_candidate_count)) ? Math.round(Number(item.coarse_candidate_count)) : 0;
  const candidateCount = Number.isFinite(Number(item.candidate_count)) ? Math.round(Number(item.candidate_count)) : 0;
  const returnedCount = Number.isFinite(Number(item.returned_count)) ? Math.round(Number(item.returned_count)) : 0;
  const topKReq = Number.isFinite(Number(item.top_k_requested)) ? Math.round(Number(item.top_k_requested)) : 0;

  const contextTrace = typeof item.context_trace === "object" && item.context_trace !== null ? item.context_trace : {};
  const hasTrigger = Boolean(contextTrace.has_trigger);
  const historyCount = Number.isFinite(Number(contextTrace.history_count)) ? Math.round(Number(contextTrace.history_count)) : 0;
  const contextJoinText = contextEnabled ? `${userUsed || "-"} / ${userRaw || "-"}` : `${userRaw || "-"}`;

  const tags = [
    `<span class="log-tag ${aiEnhanced ? "tag-ai-on" : "tag-ai-off"}">AI增强=${aiEnhanced ? "开" : "关"}</span>`,
    `<span class="log-tag ${contextEnabled ? "tag-ctx-on" : "tag-ctx-off"}">上下文=${contextEnabled ? "启用" : "关闭"}</span>`,
    `<span class="log-tag tag-result">${escapeHtml(resultType)}</span>`,
  ];
  if (aiFallback) {
    tags.push(`<span class="log-tag tag-fallback">AI回退=${escapeHtml(aiFallbackReason)}</span>`);
  }

  const debugLines = aiEnhanced
    ? [
        ["AI识别标志", aiEnhanced ? "开启" : "关闭"],
        ["响应时间", `${elapsed} ms`],
        ["是否上下文拼接", contextEnabled ? "是" : "否"],
        ["拼接后语句/原始输入", contextJoinText],
        ["最高相似度", score],
        ["AI回复", replyText || "-"],
        ["AI响应时间", `${aiElapsed} ms`],
      ]
    : [
        ["AI识别标志", aiEnhanced ? "开启" : "关闭"],
        ["响应时间", `${elapsed} ms`],
        ["是否上下文拼接", contextEnabled ? "是" : "否"],
        ["拼接后语句/原始输入", contextJoinText],
        ["最高相似度", score],
        ["匹配问句", matchedQuery],
        ["匹配答句", matchedReply],
      ];

  const debugHtml = debugLines
    .map(([label, value]) => `<div class="debug-line"><span class="debug-key">${escapeHtml(label)}</span><span class="debug-value">${value || "-"}</span></div>`)
    .join("");

  return `
    <div class="log-item">
      <div class="log-headline">
        <span class="log-ts">${ts}</span>
        <div class="log-tags">${tags.join('<span class="log-sep">|</span>')}</div>
      </div>
      <div class="debug-block">
        ${debugHtml}
      </div>
      <div class="log-stats log-stats-compact">
        <span>topK请求 ${topKReq}</span>
        <span>粗召回 ${coarseCount}</span>
        <span>可重排 ${candidateCount}</span>
        <span>返回 ${returnedCount}</span>
      </div>
      <div class="log-context">
        <strong>上下文判定:</strong> ${escapeHtml(contextReason)}
        <span>触发词:${hasTrigger ? "是" : "否"}</span>
        <span>历史轮数:${historyCount}(用于判定)</span>
      </div>
    </div>
  `;
}

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "msg-user" : "msg-bot"}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setStatus(text, isError = false) {
  statusNode.textContent = text;
  statusNode.style.color = isError ? "#d03838" : "#1f9d55";
}

async function sendMessage(message) {
  const payload = {
    message,
    history,
    conversation_id: conversationId,
  };

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`请求失败: ${res.status}`);
  }

  return await res.json();
}

async function loadLogs() {
  const res = await fetch("/api/logs/latest?limit=30");
  if (!res.ok) {
    logsPanel.textContent = "日志读取失败。";
    return;
  }
  const logs = await res.json();
  if (!Array.isArray(logs) || logs.length === 0) {
    logsPanel.textContent = "暂无日志。";
    return;
  }

  logsPanel.innerHTML = "";
  logs.reverse().forEach((item) => {
    const row = document.createElement("div");
    row.innerHTML = renderLogItem(item);
    logsPanel.appendChild(row);
  });
}

async function clearLogs() {
  const res = await fetch("/api/logs/clear", {
    method: "POST",
  });
  return await res.json();
}

async function loadComparatorSettings() {
  const res = await fetch("/api/settings/comparator");
  if (!res.ok) {
    throw new Error(`读取调参配置失败: ${res.status}`);
  }
  return await res.json();
}

async function updateComparatorSettings(payload) {
  const res = await fetch("/api/settings/comparator", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const errData = await res.json();
      detail = errData && errData.detail ? String(errData.detail) : "";
    } catch (_) {
      detail = "";
    }
    throw new Error(detail || `更新调参配置失败: ${res.status}`);
  }
  return await res.json();
}

async function loadPerfMetrics() {
  const res = await fetch("/api/perf");
  if (!res.ok) {
    throw new Error(`读取性能指标失败: ${res.status}`);
  }
  return await res.json();
}

function _fmt(n, digits = 2) {
  const v = Number(n);
  if (!Number.isFinite(v)) {
    return "0.00";
  }
  return v.toFixed(digits);
}

function drawLineChart(canvas, labels, seriesList) {
  if (!canvas || !canvas.getContext) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const cssW = Math.max(300, canvas.clientWidth || 300);
  const cssH = Math.max(150, canvas.clientHeight || 150);
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, cssW, cssH);

  const pad = { l: 34, r: 10, t: 10, b: 20 };
  const plotW = Math.max(10, cssW - pad.l - pad.r);
  const plotH = Math.max(10, cssH - pad.t - pad.b);

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of seriesList) {
    for (const v of s.data) {
      if (Number.isFinite(v)) {
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
      }
    }
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (Math.abs(yMax - yMin) < 1e-6) {
    yMax = yMin + 1;
  }

  ctx.strokeStyle = "#e8dcc5";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.t + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + plotW, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#6a7683";
  ctx.font = "11px Microsoft YaHei";
  ctx.fillText(_fmt(yMax, 1), 2, pad.t + 4);
  ctx.fillText(_fmt(yMin, 1), 2, pad.t + plotH);

  const n = Math.max(labels.length, 1);
  const xStep = n > 1 ? plotW / (n - 1) : 0;
  for (const s of seriesList) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i += 1) {
      const x = pad.l + xStep * i;
      const y = pad.t + ((yMax - s.data[i]) / (yMax - yMin)) * plotH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
}

function pushPerfPoint(perfData) {
  const device = perfData.device || {};
  const chat = perfData.chat || {};
  const ts = String(perfData.timestamp || "--:--:--");

  perfSeries.labels.push(ts);
  perfSeries.cpu.push(Number(device.cpu_percent || 0));
  perfSeries.mem.push(Number(device.memory_percent || 0));
  perfSeries.chatAvg.push(Number(chat.avg_elapsed_ms || 0));
  perfSeries.chatP95.push(Number(chat.p95_elapsed_ms || 0));

  if (perfSeries.labels.length > PERF_CAP) {
    perfSeries.labels.shift();
    perfSeries.cpu.shift();
    perfSeries.mem.shift();
    perfSeries.chatAvg.shift();
    perfSeries.chatP95.shift();
  }
}

function renderPerf(perfData) {
  const device = perfData.device || {};
  const chat = perfData.chat || {};

  perfLastUpdatedNode.textContent = String(perfData.timestamp || "--:--:--");
  perfCpuNode.textContent = `CPU ${_fmt(device.cpu_percent, 1)}%`;
  perfMemNode.textContent = `内存 ${_fmt(device.memory_percent, 1)}%`;
  perfChatAvgNode.textContent = `会话均时延 ${_fmt(chat.avg_elapsed_ms, 1)} ms`;
  perfChatP95Node.textContent = `会话P95 ${_fmt(chat.p95_elapsed_ms, 1)} ms`;

  drawLineChart(perfDeviceCanvas, perfSeries.labels, [
    { color: "#2a9d8f", data: perfSeries.cpu },
    { color: "#3b82f6", data: perfSeries.mem },
  ]);

  drawLineChart(perfChatCanvas, perfSeries.labels, [
    { color: "#ef4444", data: perfSeries.chatAvg },
    { color: "#f59e0b", data: perfSeries.chatP95 },
  ]);
}

async function refreshPerf() {
  try {
    const data = await loadPerfMetrics();
    pushPerfPoint(data);
    renderPerf(data);
  } catch (_) {
    // 性能监控失败不影响聊天主流程。
  }
}

function buildAiInfoText(data) {
  const model = data.ai_model_name || "unknown";
  const coarseRecall = Math.round(Number(data.coarse_recall_count || 0));
  const rerankTopK = Math.round(Number(data.rerank_top_k || 5));
  const apiReady = Boolean(data.ai_api_ready);
  return `模型=${model} | 统一TopK=${rerankTopK} | 粗召回=${coarseRecall} | 精排=${rerankTopK} | 密钥就绪=${apiReady ? "是" : "否"}`;
}

function setSettingsToForm(data) {
  similarityThresholdInput.value = Number(data.similarity_threshold).toFixed(2);
  rerankQueryWeightInput.value = Number(data.rerank_query_weight).toFixed(2);
  rerankReplyWeightInput.value = Number(data.rerank_reply_weight).toFixed(2);
  contextMaxTurnsInput.value = String(Math.round(Number(data.context_max_turns)));
  contextMatchingEnabledInput.checked = Boolean(data.context_matching_enabled ?? true);
  coarseRecallCountInput.value = String(Math.round(Number(data.coarse_recall_count || 0)));
  rerankTopKInput.value = String(Math.round(Number(data.rerank_top_k || 5)));
  aiEnhancedInput.checked = Boolean(data.ai_enhanced);
}

function readSettingsFromForm() {
  const payload = {
    similarity_threshold: Number(similarityThresholdInput.value),
    rerank_query_weight: Number(rerankQueryWeightInput.value),
    rerank_reply_weight: Number(rerankReplyWeightInput.value),
    context_max_turns: Number(contextMaxTurnsInput.value),
    context_matching_enabled: Boolean(contextMatchingEnabledInput.checked),
    coarse_recall_count: Number(coarseRecallCountInput.value),
    rerank_top_k: Number(rerankTopKInput.value),
    ai_enhanced: Boolean(aiEnhancedInput.checked),
  };

  if (!Number.isFinite(payload.similarity_threshold) || payload.similarity_threshold < 0 || payload.similarity_threshold > 1) {
    throw new Error("命中阈值必须在 0 到 1 之间");
  }
  if (!Number.isFinite(payload.rerank_query_weight) || payload.rerank_query_weight < 0) {
    throw new Error("问问权重不能为负数");
  }
  if (!Number.isFinite(payload.rerank_reply_weight) || payload.rerank_reply_weight < 0) {
    throw new Error("问答权重不能为负数");
  }
  if (payload.rerank_query_weight + payload.rerank_reply_weight <= 0) {
    throw new Error("问问权重和问答权重不能同时为 0");
  }
  if (!Number.isFinite(payload.context_max_turns) || payload.context_max_turns < 1 || payload.context_max_turns > 10) {
    throw new Error("上下文轮数必须在 1 到 10 之间");
  }
  if (!Number.isFinite(payload.coarse_recall_count) || payload.coarse_recall_count < 0 || payload.coarse_recall_count > 10000) {
    throw new Error("粗召回数量必须在 0 到 10000 之间");
  }
  if (!Number.isFinite(payload.rerank_top_k) || payload.rerank_top_k < 1 || payload.rerank_top_k > 200) {
    throw new Error("精排数量必须在 1 到 200 之间");
  }
  payload.context_max_turns = Math.round(payload.context_max_turns);
  payload.coarse_recall_count = Math.round(payload.coarse_recall_count);
  payload.rerank_top_k = Math.round(payload.rerank_top_k);
  return payload;
}

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  addMessage(message, "user");
  setStatus("正在检索与生成回复...");
  chatInput.value = "";
  chatInput.disabled = true;
  if (sendBtn) {
    sendBtn.disabled = true;
  }

  try {
    const data = await sendMessage(message);
    conversationId = data.conversation_id;
    addMessage(data.reply, "bot");

    history.push([message, data.reply]);
    const maxTurns = Math.max(1, Math.min(10, Math.round(Number(contextMaxTurnsInput.value) || 3)));
    if (history.length > maxTurns) {
      history = history.slice(-maxTurns);
    }

    setStatus(`已回复 | 相似度 ${Number(data.score).toFixed(4)} | 耗时 ${Number(data.elapsed_ms).toFixed(2)} ms`);
    await loadLogs();
  } catch (err) {
    console.error(err);
    setStatus("请求失败，请检查服务状态。", true);
    addMessage("系统暂时不可用，请稍后重试。", "bot");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
    if (sendBtn) {
      sendBtn.disabled = false;
    }
  }
});

clearChatBtn.addEventListener("click", () => {
  const ok = window.confirm("确认清除当前对话内容吗？");
  if (!ok) {
    return;
  }
  history = [];
  conversationId = null;
  chatWindow.innerHTML = "";
  addMessage("你好，我是 sim-检索式对话机器人。你可以直接开始聊天。", "bot");
  setStatus("对话已清除，日志不受影响");
});

refreshLogsBtn.addEventListener("click", async () => {
  await loadLogs();
});

clearLogsBtn.addEventListener("click", async () => {
  const ok = window.confirm("确认清除全部日志吗？该操作不可恢复。");
  if (!ok) {
    return;
  }

  try {
    const data = await clearLogs();
    logsPanel.textContent = "暂无日志。";
    setStatus(`日志已清除（${data.cleared_count || 0} 条）`);
  } catch (err) {
    console.error(err);
    setStatus("清除日志失败，请重试。", true);
  }
});

reloadSettingsBtn.addEventListener("click", async () => {
  try {
    const setting = await loadComparatorSettings();
    setSettingsToForm(setting);
    setStatus(`已读取当前调参配置 | AI增强=${setting.ai_enhanced ? "开" : "关"} | ${buildAiInfoText(setting)}`);
  } catch (err) {
    console.error(err);
    setStatus("读取调参配置失败，请重试。", true);
  }
});

applySettingsBtn.addEventListener("click", async () => {
  try {
    const payload = readSettingsFromForm();
    const data = await updateComparatorSettings(payload);
    setSettingsToForm(data);
    setStatus(`参数已生效 | AI增强=${data.ai_enhanced ? "开" : "关"} | ${buildAiInfoText(data)}`);
  } catch (err) {
    console.error(err);
    setStatus(`参数更新失败: ${err.message || "请重试"}`, true);
  }
});

aiEnhancedInput.addEventListener("change", async () => {
  const targetValue = Boolean(aiEnhancedInput.checked);
  const contextEnabled = Boolean(contextMatchingEnabledInput.checked);
  try {
    const data = await updateComparatorSettings({
      ai_enhanced: targetValue,
      context_matching_enabled: contextEnabled,
    });
    setSettingsToForm(data);
    setStatus(`AI增强已${data.ai_enhanced ? "开启" : "关闭"} | 上下文匹配=${data.context_matching_enabled ? "开" : "关"} | ${buildAiInfoText(data)}`);
  } catch (err) {
    aiEnhancedInput.checked = !targetValue;
    setStatus(`AI增强切换失败: ${err.message || "请重试"}`, true);
  }
});

contextMatchingEnabledInput.addEventListener("change", async () => {
  const targetValue = Boolean(contextMatchingEnabledInput.checked);
  try {
    const data = await updateComparatorSettings({ context_matching_enabled: targetValue });
    setSettingsToForm(data);
    setStatus(`上下文匹配已${data.context_matching_enabled ? "开启" : "关闭"} | ${buildAiInfoText(data)}`);
  } catch (err) {
    contextMatchingEnabledInput.checked = !targetValue;
    setStatus(`上下文匹配切换失败: ${err.message || "请重试"}`, true);
  }
});

addMessage("你好，我是 sim-检索式对话机器人。你可以直接开始聊天。", "bot");

(async () => {
  await loadLogs();
  await refreshPerf();
  if (perfTimerId === null) {
    perfTimerId = window.setInterval(refreshPerf, 2000);
  }
  try {
    const setting = await loadComparatorSettings();
    setSettingsToForm(setting);
  } catch (err) {
    console.error(err);
    setStatus("读取调参配置失败，将使用页面默认值。", true);
  }
})();