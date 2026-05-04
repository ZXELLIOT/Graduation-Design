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

aiToggle.addEventListener("change", () => syncToggle("ai", aiToggle.checked));
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
