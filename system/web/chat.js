/**
 * chat.js — 对话页面
 */

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat");
const clearCtxBtn = document.getElementById("clear-ctx");
const statusNode = document.getElementById("status");
const aiToggle = document.getElementById("ai-toggle");
const ctxToggle = document.getElementById("ctx-toggle");

let history = [];
let conversationId = null;

const CHAT_STATE_KEY = "simcse_chat_state_v1";
const COMPARATOR_SYNC_KEY = "simcse_comparator_sync_v1";

function safeParse(jsonText, fallback) {
  try { return JSON.parse(jsonText); } catch (e) { return fallback; }
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
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

function persistChatState() {
  const payload = {
    history,
    conversationId,
    aiEnabled: Boolean(aiToggle.checked),
    ctxEnabled: Boolean(ctxToggle.checked),
    savedAt: Date.now(),
  };
  try { sessionStorage.setItem(CHAT_STATE_KEY, JSON.stringify(payload)); } catch (e) {}
}

function emitComparatorSync(partial) {
  const payload = { ...partial, ts: Date.now() };
  try { localStorage.setItem(COMPARATOR_SYNC_KEY, JSON.stringify(payload)); } catch (e) {}
}

function renderChatFromHistory() {
  chatWindow.innerHTML = "";
  addMessage("你好！我是 SimCSE 检索式对话机器人，可以直接开始聊天。", "bot");
  for (const pair of history) {
    if (!Array.isArray(pair) || pair.length < 2) continue;
    addMessage(String(pair[0] ?? ""), "user");
    addMessage(String(pair[1] ?? ""), "bot");
  }
}

function restoreChatState() {
  const raw = sessionStorage.getItem(CHAT_STATE_KEY);
  if (!raw) {
    renderChatFromHistory();
    return;
  }
  const state = safeParse(raw, null);
  if (!state || typeof state !== "object") {
    renderChatFromHistory();
    return;
  }
  history = Array.isArray(state.history) ? state.history : [];
  conversationId = state.conversationId || null;
  if (typeof state.aiEnabled === "boolean") aiToggle.checked = state.aiEnabled;
  if (typeof state.ctxEnabled === "boolean") ctxToggle.checked = state.ctxEnabled;
  renderChatFromHistory();
  if (history.length > 0) {
    setStatus(`已恢复会话（${history.length} 轮）`);
  }
}

async function refreshComparatorSettings(silent = false) {
  try {
    const data = await apiJson("/api/settings/comparator", {}, "配置接口");
    if (typeof data.ai_enhanced !== "boolean" || typeof data.context_matching_enabled !== "boolean") {
      throw new Error("配置接口返回异常值：开关字段缺失或类型错误");
    }
    const aiEnabled = Boolean(data.ai_enhanced);
    const ctxEnabled = Boolean(data.context_matching_enabled);
    const aiChanged = aiToggle.checked !== aiEnabled;
    const ctxChanged = ctxToggle.checked !== ctxEnabled;
    aiToggle.checked = aiEnabled;
    ctxToggle.checked = ctxEnabled;
    if (!silent && (aiChanged || ctxChanged)) {
      setStatus(`配置已同步 | AI:${aiEnabled ? "开" : "关"} 上下文:${ctxEnabled ? "开" : "关"}`);
    }
    persistChatState();
  } catch (e) {
    if (!silent) setStatus(e.message || "配置同步失败", true);
  }
}

// ============================================================
// 消息
// ============================================================

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "msg-user" : "msg-bot"}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addLoading() {
  const div = document.createElement("div");
  div.className = "msg msg-bot";
  div.id = "loading-msg";
  div.innerHTML = '<span style="display:inline-flex;gap:4px;"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span>';
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  // Animate dots
  let i = 0;
  const dots = div.querySelectorAll(".dot");
  div._interval = setInterval(() => {
    dots.forEach((d, j) => { d.style.opacity = j === i % 3 ? "1" : "0.3"; });
    i++;
  }, 300);
}

function removeLoading() {
  const el = document.getElementById("loading-msg");
  if (el) {
    if (el._interval) clearInterval(el._interval);
    el.remove();
  }
}

function setStatus(text, isError) {
  statusNode.textContent = text;
  statusNode.style.color = isError ? "#d03838" : "#1f9d55";
}

// ============================================================
// 发送
// ============================================================

async function sendMessage(message) {
  const data = await apiJson("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, conversation_id: conversationId }),
  }, "聊天接口");

  if (typeof data.reply !== "string" || !isFiniteNumber(Number(data.score)) || !isFiniteNumber(Number(data.elapsed_ms))) {
    throw new Error("聊天接口返回异常值：reply/score/elapsed_ms 字段非法");
  }
  return data;
}

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatForm.requestSubmit(); }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg) return;

  addMessage(msg, "user");
  addLoading();
  setStatus("思考中...");
  chatInput.value = "";
  chatInput.disabled = true;
  sendBtn.textContent = "发送中...";
  sendBtn.disabled = true;

  try {
    const data = await sendMessage(msg);
    conversationId = data.conversation_id;
    removeLoading();
    addMessage(data.reply, "bot");
    history.push([msg, data.reply]);
    if (history.length > 10) history = history.slice(-10);
    persistChatState();
    const aiTag = data.ai_enhanced ? " | AI增强" : "";
    setStatus(`相似度 ${Number(data.score).toFixed(3)} | ${Number(data.elapsed_ms).toFixed(0)}ms${aiTag}`);
  } catch (err) {
    removeLoading();
    setStatus(err.message || "请求失败，请检查服务状态", true);
    addMessage("系统暂时不可用，请稍后重试。", "bot");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
    sendBtn.textContent = "发送";
    sendBtn.disabled = false;
  }
});

// ============================================================
// 开关
// ============================================================

async function syncToggle(key, value) {
  const payload = {};
  if (key === "ai") payload.ai_enhanced = value;
  if (key === "ctx") payload.context_matching_enabled = value;
  try {
    await apiJson("/api/settings/comparator", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }, "配置更新接口");
    emitComparatorSync(payload);
    persistChatState();
  } catch (e) {
    setStatus(e.message || "开关更新失败", true);
  }
}

function showAiPopup(success, detail) {
  const popup = document.createElement("div");
  popup.style.cssText = "position:fixed;top:20px;right:20px;z-index:999;padding:14px 18px;border-radius:10px;font-size:0.9rem;max-width:360px;box-shadow:0 6px 20px rgba(0,0,0,0.15);animation:rise 0.3s ease;";
  popup.style.background = success ? "#dff5eb" : "#ffe1e1";
  popup.style.color = success ? "#0b6e4f" : "#7b1e1e";
  popup.style.border = success ? "1px solid #b7e6d3" : "1px solid #f2bbbb";
  popup.textContent = detail;
  document.body.appendChild(popup);
  setTimeout(() => { popup.style.opacity = "0"; popup.style.transition = "opacity 0.5s"; }, 3000);
  setTimeout(() => popup.remove(), 3500);
}

aiToggle.addEventListener("change", async () => {
  const on = aiToggle.checked;
  const t0 = performance.now();
  try {
    const data = await apiJson("/api/settings/comparator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ai_enhanced: on }),
    }, "AI开关接口");
    const ms = (performance.now() - t0).toFixed(0);
    if (data && typeof data === "object") {
      emitComparatorSync({ ai_enhanced: on });
      persistChatState();
      showAiPopup(true, `AI 增强已${on ? "开启" : "关闭"} | 响应 ${ms}ms`);
    } else {
      aiToggle.checked = !on;
      showAiPopup(false, `AI 增强${on ? "开启" : "关闭"}失败: 返回数据异常 | ${ms}ms`);
    }
  } catch (e) {
    aiToggle.checked = !on;
    showAiPopup(false, e.message || "AI 增强请求失败");
    setStatus(e.message || "AI 增强更新失败", true);
  }
});

ctxToggle.addEventListener("change", () => syncToggle("ctx", ctxToggle.checked));

// ============================================================
// 清除
// ============================================================

clearChatBtn.addEventListener("click", () => {
  history = [];
  conversationId = null;
  renderChatFromHistory();
  persistChatState();
  setStatus("对话已清空");
});

clearCtxBtn.addEventListener("click", async () => {
  try {
    const data = await apiJson("/api/context/clear", { method: "POST" }, "上下文清理接口");
    const cleared = Number(data.cleared);
    if (!Number.isFinite(cleared)) {
      throw new Error("上下文清理接口返回异常值：cleared 非数字");
    }
    history = [];
    persistChatState();
    setStatus(`上下文记忆已清除（${data.cleared} 条）`);
  } catch (e) { setStatus(e.message || "清除失败", true); }
});

window.addEventListener("storage", (evt) => {
  if (evt.key !== COMPARATOR_SYNC_KEY || !evt.newValue) return;
  const incoming = safeParse(evt.newValue, null);
  if (!incoming || typeof incoming !== "object") return;
  if (typeof incoming.ai_enhanced === "boolean") aiToggle.checked = incoming.ai_enhanced;
  if (typeof incoming.context_matching_enabled === "boolean") ctxToggle.checked = incoming.context_matching_enabled;
  persistChatState();
});

// ============================================================
// 初始化
// ============================================================

restoreChatState();
setStatus("就绪");
refreshComparatorSettings(true);
setInterval(() => refreshComparatorSettings(true), 3000);
