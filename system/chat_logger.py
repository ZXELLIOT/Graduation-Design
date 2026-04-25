"""
system/chat_logger.py

文件作用:
    对话日志读写模块。
    统一维护日志路径、追加写入、读取与清空逻辑。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from system.config import PROJECT_ROOT

LOG_DIR = Path(PROJECT_ROOT) / "runtime" / "chat_logs"
LOG_FILE = LOG_DIR / "chat_log.jsonl"


def ensure_log_dir() -> None:
    """确保日志目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_chat_log(
    conversation_id: str,
    user_input: str,
    result: Dict[str, Any],
    history: Optional[List[List[str]]] = None,
) -> None:
    """追加写入对话日志（jsonl）。"""
    ensure_log_dir()

    matched_item = result.get("matched_item")
    used_query = str(result.get("used_query") or "")
    trace_raw = result.get("trace")
    trace: Dict[str, Any] = trace_raw if isinstance(trace_raw, dict) else {}
    context_raw = trace.get("context")
    retrieval_raw = trace.get("retrieval")
    context_trace: Dict[str, Any] = context_raw if isinstance(context_raw, dict) else {}
    retrieval_trace: Dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
    context_enabled = bool(context_trace.get("enabled", False))
    context_reason = str(context_trace.get("reason", "unknown"))
    context_summary = (
        f"启用({context_reason})"
        if context_enabled
        else f"关闭({context_reason})"
    )

    # 字段约定：
    # 1) user_input 为“实际检索输入”（可能是上下文拼接后的文本）
    # 2) raw_user_input 为前端原始输入
    # 3) expected_reply 用于离线评估回放，默认与 reply 一致
    log_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "conversation_id": conversation_id,
        "ai_enhanced": bool(result.get("ai_enhanced", False)),
        "ai_elapsed_ms": float(result.get("ai_elapsed_ms", 0.0)),
        "ai_fallback": bool(result.get("ai_fallback", False)),
        "ai_fallback_reason": str(result.get("ai_fallback_reason", "none")),
        "user_input": used_query or user_input,
        "raw_user_input": user_input,
        "reply": result.get("reply", ""),
        "score": float(result.get("score", 0.0)),
        "result_type": result.get("result_type", "unknown"),
        "elapsed_ms": float(result.get("elapsed_ms", 0.0)),
        "context_enabled": context_enabled,
        "context_reason": context_reason,
        "context_summary": context_summary,
        "context_trace": context_trace,
        "retrieval_trace": retrieval_trace,
        "candidate_count": int(retrieval_trace.get("candidate_count", 0)),
        "coarse_candidate_count": int(retrieval_trace.get("coarse_candidate_count", 0)),
        "returned_count": int(retrieval_trace.get("returned_count", 0)),
        "top_k_requested": int(retrieval_trace.get("top_k_requested", 0)),
        "expected_reply": result.get("expected_reply", result.get("reply", "")),
        "matched_query": result.get("matched_q"),
        "top_query": matched_item.get("query") if isinstance(matched_item, dict) else None,
        "top_reply": matched_item.get("reply") if isinstance(matched_item, dict) else None,
        "history": history or [],
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def latest_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """读取最近日志记录。"""
    ensure_log_dir()
    if not LOG_FILE.exists():
        return []

    safe_limit = max(1, min(int(limit), 200))
    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()[-safe_limit:]

    payload: List[Dict[str, Any]] = []
    for line in lines:
        try:
            payload.append(json.loads(line.strip()))
        except Exception:
            continue
    return payload


def clear_logs() -> int:
    """清空日志并返回清理条数。"""
    ensure_log_dir()
    if not LOG_FILE.exists():
        return 0

    cleared_count = 0
    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            cleared_count = sum(1 for _ in f)
    except Exception:
        cleared_count = 0

    with LOG_FILE.open("w", encoding="utf-8"):
        pass
    return int(cleared_count)
