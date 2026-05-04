"""
system/app.py

文件作用:
    Web 主入口。
    负责接口路由与模块调用。

部署方式:
    - 本地测试: python system/app.py → 浏览器打开 http://127.0.0.1:7860
    - Linux 部署: 配置 Nginx 反向代理 + 公网域名
"""

import os
import sys
import time
import psutil
import uvicorn
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.bootstrap import get_dialog_comparator
from system.chat_logger import LOG_FILE, append_chat_log, clear_logs, latest_logs
from system.chat_service import infer
from system.comparator_settings import apply_comparator_settings, comparator_settings_payload
from system.config import ADMIN_PASSWORD

WEB_DIR = Path(CURRENT_DIR) / "web"
SERVICE_START_TS = time.time()
PROCESS = psutil.Process(os.getpid())

# 预热一次，避免首次读取出现全 0。
psutil.cpu_percent(interval=None)
PROCESS.cpu_percent(interval=None)


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str
    history: Optional[List[List[str]]] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应体。"""

    conversation_id: str
    reply: str
    score: float
    result_type: str
    elapsed_ms: float
    matched_query: Optional[str] = None
    ai_enhanced: bool = False


class ComparatorSettingsRequest(BaseModel):
    """比较器调参请求体。"""

    similarity_threshold: Optional[float] = None
    rerank_query_weight: Optional[float] = None
    rerank_reply_weight: Optional[float] = None
    context_max_turns: Optional[int] = None
    max_text_len: Optional[int] = None
    context_short_query_len: Optional[int] = None
    context_matching_enabled: Optional[bool] = None
    context_overlap_threshold: Optional[float] = None
    context_semantic_threshold: Optional[float] = None
    coarse_recall_count: Optional[int] = None
    rerank_top_k: Optional[int] = None
    ai_enhanced: Optional[bool] = None


api_app = FastAPI(title="SimCSE 检索式对话系统", version="1.0.0")

if WEB_DIR.exists():
    api_app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@api_app.get("/")
def home() -> FileResponse:
    """返回前端首页文件。"""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail=f"前端首页不存在: {index_file}")
    return FileResponse(str(index_file))


@api_app.get("/chat")
def chat_page() -> FileResponse:
    """对话页面。"""
    f = WEB_DIR / "chat.html"
    if not f.exists(): raise HTTPException(404)
    return FileResponse(str(f))


@api_app.get("/admin")
def admin_page() -> FileResponse:
    """后台管理页面。"""
    f = WEB_DIR / "admin.html"
    if not f.exists(): raise HTTPException(404)
    return FileResponse(str(f))


@api_app.post("/api/admin/login")
def admin_login(body: Dict[str, str]) -> Dict[str, Any]:
    """后台管理密码验证。"""
    pwd = body.get("password", "")
    if pwd == ADMIN_PASSWORD:
        return {"ok": True, "token": "admin"}
    return {"ok": False, "detail": "密码错误"}


@api_app.post("/api/context/clear")
def clear_context() -> Dict[str, Any]:
    """清除上下文记忆。"""
    comparator = get_dialog_comparator()
    n = comparator.clear_context_memory()
    return {"ok": True, "cleared": n}


@api_app.get("/api/meta")
def meta() -> Dict[str, str]:
    """返回服务元信息。"""
    return {
        "app": "SimCSE 检索式对话系统",
        "database_source": "LCCC-large + FAISS 双索引",
        "log_file": str(LOG_FILE),
    }


@api_app.get("/api/settings/comparator")
def get_comparator_settings() -> Dict[str, Any]:
    """读取当前比较器全部可调参数。"""
    comparator = get_dialog_comparator()
    return comparator_settings_payload(comparator)


@api_app.post("/api/settings/comparator")
def set_comparator_settings(req: ComparatorSettingsRequest) -> Dict[str, Any]:
    """热更新比较器参数。"""
    comparator = get_dialog_comparator()
    apply_comparator_settings(comparator, req)
    return {"ok": True, **comparator_settings_payload(comparator)}


@api_app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """聊天接口。"""
    conversation_id = req.conversation_id or str(uuid4())

    # 推理主链路：输入 → 检索/增强 → 结构化结果。
    result = infer(user_input=req.message, history=req.history)

    # 落盘日志，便于离线分析与问题复盘。
    append_chat_log(
        conversation_id=conversation_id,
        user_input=req.message,
        result=result,
        history=req.history,
    )
    return ChatResponse(
        conversation_id=conversation_id,
        reply=str(result["reply"]),
        score=float(result["score"]),
        result_type=str(result["result_type"]),
        elapsed_ms=float(result["elapsed_ms"]),
        matched_query=result.get("matched_q"),
        ai_enhanced=bool(result.get("ai_enhanced", False)),
    )


@api_app.get("/api/logs/latest")
def api_latest_logs(limit: int = 50) -> JSONResponse:
    """读取最近日志记录。"""
    return JSONResponse(latest_logs(limit=limit))


@api_app.post("/api/logs/clear")
def api_clear_logs() -> Dict[str, Any]:
    """清空日志文件并返回统计信息。"""
    return {"ok": True, "cleared_count": clear_logs()}


@api_app.get("/api/perf")
def api_perf() -> Dict[str, Any]:
    """返回实时性能指标（设备与对话系统）。"""
    vm = psutil.virtual_memory()
    cpu_percent = float(psutil.cpu_percent(interval=None))
    proc_cpu_percent = float(PROCESS.cpu_percent(interval=None))
    proc_mem_mb = float(PROCESS.memory_info().rss / (1024 * 1024))
    uptime_sec = float(time.time() - SERVICE_START_TS)

    logs = latest_logs(limit=120)
    elapsed_vals = [float(x.get("elapsed_ms", 0.0)) for x in logs if isinstance(x, dict)]
    elapsed_vals = [x for x in elapsed_vals if x >= 0.0]

    if elapsed_vals:
        avg_elapsed = float(sum(elapsed_vals) / len(elapsed_vals))
        p95_elapsed = float(sorted(elapsed_vals)[max(0, int(len(elapsed_vals) * 0.95) - 1)])
        latest_elapsed = float(elapsed_vals[-1])
    else:
        avg_elapsed = 0.0
        p95_elapsed = 0.0
        latest_elapsed = 0.0

    return {
        "timestamp": time.strftime("%H:%M:%S", time.localtime()),
        "device": {
            "cpu_percent": cpu_percent,
            "memory_percent": float(vm.percent),
            "memory_used_gb": float(vm.used / (1024 ** 3)),
            "memory_total_gb": float(vm.total / (1024 ** 3)),
        },
        "service": {
            "pid": int(PROCESS.pid),
            "process_cpu_percent": proc_cpu_percent,
            "process_memory_mb": proc_mem_mb,
            "uptime_sec": uptime_sec,
        },
        "chat": {
            "sample_count": int(len(elapsed_vals)),
            "latest_elapsed_ms": latest_elapsed,
            "avg_elapsed_ms": avg_elapsed,
            "p95_elapsed_ms": p95_elapsed,
        },
    }


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("  SimCSE 检索式对话系统")
        print("  本地服务: http://127.0.0.1:7860")
        print("=" * 50)
        # 启动前预热：确保首个请求不承担完整初始化延迟。
        get_dialog_comparator()
        print("\n系统就绪，正在启动 Web 服务...\n")
        uvicorn.run(api_app, host="127.0.0.1", port=7860)
    except Exception as e:
        print(f"启动失败: {e}")