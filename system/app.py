"""
system/app.py

文件作用:
    Web 主入口。
    负责接口路由与模块调用。
"""

import os
import sys
import time
import re
import atexit
import threading
import subprocess
import uvicorn
import psutil
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

WEB_DIR = Path(CURRENT_DIR) / "web"
SERVICE_START_TS = time.time()
PROCESS = psutil.Process(os.getpid())
AUTO_TUNNEL_ENABLED = os.getenv("AUTO_TUNNEL_ENABLED", "1") == "1"
TUNNEL_LOCAL_URL = os.getenv("TUNNEL_LOCAL_URL", "http://127.0.0.1:7860")
_tunnel_process: Optional[subprocess.Popen[str]] = None
_tunnel_url: str = ""

# 预热一次，避免首次读取出现全 0。
psutil.cpu_percent(interval=None)
PROCESS.cpu_percent(interval=None)


def _read_tunnel_output(proc: subprocess.Popen[str]) -> None:
    """异步读取 cloudflared 输出并提取 trycloudflare 访问地址。"""
    global _tunnel_url
    if proc.stdout is None:
        return

    for raw in proc.stdout:
        line = str(raw).strip()
        if line:
            print(f"[tunnel] {line}")
        if not _tunnel_url:
            match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
            if match:
                _tunnel_url = match.group(0)
                print(f"[tunnel] 公网访问地址: {_tunnel_url}")


def start_quick_tunnel() -> None:
    """启动 cloudflared quick tunnel（若可用）。"""
    global _tunnel_process

    if not AUTO_TUNNEL_ENABLED:
        print("[tunnel] AUTO_TUNNEL_ENABLED=0，已跳过自动内网穿透。")
        return

    if _tunnel_process is not None and _tunnel_process.poll() is None:
        return

    cmd = ["cloudflared", "tunnel", "--url", TUNNEL_LOCAL_URL]
    try:
        _tunnel_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_read_tunnel_output, args=(_tunnel_process,), daemon=True).start()
        print("[tunnel] 已启动 cloudflared quick tunnel，正在生成公网地址...")
    except FileNotFoundError:
        print("[tunnel] 未找到 cloudflared，跳过自动内网穿透。")
    except Exception as e:
        print(f"[tunnel] 启动失败: {e}")


def stop_quick_tunnel() -> None:
    """停止 cloudflared tunnel 进程。"""
    global _tunnel_process
    if _tunnel_process is None:
        return
    try:
        if _tunnel_process.poll() is None:
            _tunnel_process.terminate()
    except Exception:
        pass
    finally:
        _tunnel_process = None


atexit.register(stop_quick_tunnel)


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


api_app = FastAPI(title="日常闲聊机器人本地服务", version="1.0.0")

if WEB_DIR.exists():
    api_app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@api_app.get("/")
def home() -> FileResponse:
    """返回前端首页文件。"""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail=f"前端首页不存在: {index_file}")
    return FileResponse(str(index_file))


@api_app.get("/api/meta")
def meta() -> Dict[str, str]:
    """返回服务元信息。"""
    return {
        "app": "日常闲聊机器人",
        "database_source": "LCCC-large+ FAISS 索引",
        "log_file": str(LOG_FILE),
        "tunnel_url": _tunnel_url,
    }


@api_app.on_event("startup")
def _startup_hook() -> None:
    """服务启动后自动拉起内网穿透。"""
    start_quick_tunnel()


@api_app.on_event("shutdown")
def _shutdown_hook() -> None:
    """服务退出时关闭内网穿透进程。"""
    stop_quick_tunnel()


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

    # 推理主链路：输入 -> 检索/增强 -> 结构化结果。
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
        print("系统正在启动，请稍候...")
        # 启动前预热：确保首个请求不承担完整初始化延迟。
        get_dialog_comparator()
        print("系统就绪，正在启动本地 Web 服务: http://127.0.0.1:7860")
        uvicorn.run(api_app, host="127.0.0.1", port=7860)
    except Exception as e:
        print(f"启动界面失败: {e}")