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
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, conversation_id: conversationId }),
  });
  if (!res.ok) throw new Error(`请求失败: ${res.status}`);
  return await res.json();
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
    const aiTag = data.ai_enhanced ? " | AI增强" : "";
    setStatus(`相似度 ${Number(data.score).toFixed(3)} | ${Number(data.elapsed_ms).toFixed(0)}ms${aiTag}`);
  } catch (err) {
    removeLoading();
    setStatus("请求失败，请检查服务状态", true);
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
  try { await fetch("/api/settings/comparator", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); } catch (e) {}
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
    const res = await fetch("/api/settings/comparator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ai_enhanced: on }),
    });
    const data = await res.json();
    const ms = (performance.now() - t0).toFixed(0);
    if (res.ok) {
      showAiPopup(true, `AI 增强已${on ? "开启" : "关闭"} | 响应 ${ms}ms`);
    } else {
      aiToggle.checked = !on;
      showAiPopup(false, `AI 增强${on ? "开启" : "关闭"}失败: ${data.detail || "未知错误"} | ${ms}ms`);
    }
  } catch (e) {
    aiToggle.checked = !on;
    showAiPopup(false, `AI 增强请求失败: 网络错误`);
  }
});

ctxToggle.addEventListener("change", () => syncToggle("ctx", ctxToggle.checked));

// ============================================================
// 清除
// ============================================================

clearChatBtn.addEventListener("click", () => {
  history = [];
  conversationId = null;
  chatWindow.innerHTML = "";
  addMessage("你好！我是 SimCSE 检索式对话机器人，可以直接开始聊天。", "bot");
  setStatus("对话已清空");
});

clearCtxBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/context/clear", { method: "POST" });
    const data = await res.json();
    setStatus(`上下文记忆已清除（${data.cleared} 条）`);
  } catch (e) { setStatus("清除失败", true); }
});

// ============================================================
// 初始化
// ============================================================

addMessage("你好！我是 SimCSE 检索式对话机器人，可以直接开始聊天。", "bot");
setStatus("就绪");
