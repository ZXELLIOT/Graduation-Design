"""
system/chat_service.py

文件作用:
    聊天推理主流程服务。
    封装“检索 -> AI增强 -> 兜底”的业务逻辑。
"""

import time
from typing import Any, Dict, List, Optional

from system.ai_enhancer import generate_ai_enhanced_reply
from system.bootstrap import get_dialog_comparator
from system.config import (
    AI_ENHANCE_API_KEY,
    AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL,
    AI_ENHANCE_TIMEOUT_SEC,
)
from system.runtime_settings import runtime_ai_settings


def infer(user_input: str, history: Optional[List[List[str]]] = None) -> Dict[str, Any]:
    """对外提供统一推理入口。"""
    comparator = get_dialog_comparator()
    t0 = time.perf_counter()

    safe_history = history or []
    # 单一 top-k：检索重排与 AI 融合统一使用比较器 rerank_top_k。
    rerank_top_k = int(getattr(comparator, "rerank_top_k", 5))

    def _safe_trace(trace_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = trace_meta if isinstance(trace_meta, dict) else {}
        context_raw = base.get("context")
        retrieval_raw = base.get("retrieval")
        context_meta: Dict[str, Any] = context_raw if isinstance(context_raw, dict) else {}
        retrieval_meta: Dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        return {
            "context": {
                "enabled": bool(context_meta.get("enabled", False)),
                "reason": str(context_meta.get("reason", "unknown")),
                "has_trigger": bool(context_meta.get("has_trigger", False)),
                "is_short_query": bool(context_meta.get("is_short_query", False)),
                "negative_hit": bool(context_meta.get("negative_hit", False)),
                "history_count": int(context_meta.get("history_count", 0)),
            },
            "retrieval": {
                "top_k_requested": int(retrieval_meta.get("top_k_requested", rerank_top_k)),
                "coarse_candidate_count": int(retrieval_meta.get("coarse_candidate_count", 0)),
                "candidate_count": int(retrieval_meta.get("candidate_count", 0)),
                "returned_count": int(retrieval_meta.get("returned_count", 0)),
                "best_score": float(retrieval_meta.get("best_score", 0.0)),
            },
        }

    # 分支 A：关闭 AI 增强时，走本地比较器直接返回。
    if not runtime_ai_settings.get("enabled", False):
        reply, score, matched_q, matched_item, result_type, used_query, trace_meta = comparator.compare_with_meta(
            user_input,
            history=safe_history,
            top_k=rerank_top_k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": str(reply),
            "score": float(score),
            "matched_q": matched_q,
            "matched_item": matched_item,
            "result_type": str(result_type),
            "elapsed_ms": float(elapsed_ms),
            "used_query": str(used_query),
            "ai_enhanced": False,
            "ai_elapsed_ms": 0.0,
            "ai_fallback": False,
            "ai_fallback_reason": "not_enabled",
            "expected_reply": str(reply),
            "trace": _safe_trace(trace_meta),
        }

    # 分支 B：开启 AI 增强时，先取本地 top-k 候选，再决定是否调用外部模型融合。
    used_query, candidates, trace_meta = comparator.get_topk_candidates_with_meta(
        user_input=user_input,
        history=safe_history,
        top_k=rerank_top_k,
    )
    trace_payload = _safe_trace(trace_meta)

    if not candidates:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": "知识库没有这个问题的回复",
            "score": 0.0,
            "matched_q": None,
            "matched_item": None,
            "result_type": "no_candidate",
            "elapsed_ms": float(elapsed_ms),
            "used_query": str(used_query),
            "ai_enhanced": True,
            "ai_elapsed_ms": 0.0,
            "ai_fallback": True,
            "ai_fallback_reason": "no_candidate",
            "expected_reply": "知识库没有这个问题的回复",
            "trace": trace_payload,
        }

    best_score, best_item = candidates[0]

    # 候选质量不足时拒答，避免低置信度回复进入 AI 融合链路。
    if float(best_score) < float(comparator.similarity_threshold):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": "知识库没有这个问题的回复",
            "score": float(best_score),
            "matched_q": None,
            "matched_item": best_item,
            "result_type": "below_threshold",
            "elapsed_ms": float(elapsed_ms),
            "used_query": str(used_query),
            "ai_enhanced": True,
            "ai_elapsed_ms": 0.0,
            "ai_fallback": True,
            "ai_fallback_reason": "below_threshold",
            "expected_reply": "知识库没有这个问题的回复",
            "trace": trace_payload,
        }

    # 候选通过阈值后再调用 AI 融合，降低无效外部请求。
    ai_t0 = time.perf_counter()
    ai_reply = generate_ai_enhanced_reply(
        contextual_user_input=str(used_query or user_input),
        candidates=candidates[:rerank_top_k],
        top_k=rerank_top_k,
        model_name=AI_ENHANCE_MODEL_NAME,
        responses_url=AI_ENHANCE_RESPONSES_URL,
        api_key=AI_ENHANCE_API_KEY,
        timeout_sec=AI_ENHANCE_TIMEOUT_SEC,
    )
    ai_elapsed_ms = (time.perf_counter() - ai_t0) * 1000.0

    ai_reply_text = str(ai_reply).strip() if ai_reply else ""
    ai_fallback = False
    ai_fallback_reason = ""
    final_reply = ai_reply_text if ai_reply_text else str(best_item.get("reply", ""))
    if not ai_reply_text:
        ai_fallback = True
        ai_fallback_reason = "ai_empty"
    if not final_reply:
        final_reply = "知识库没有这个问题的回复"
        ai_fallback = True
        ai_fallback_reason = "ai_and_local_empty"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "reply": final_reply,
        "score": float(best_score),
        "matched_q": best_item.get("query"),
        "matched_item": best_item,
        "result_type": "ai_enhanced_matched",
        "elapsed_ms": float(elapsed_ms),
        "used_query": str(used_query or user_input),
        "ai_enhanced": True,
        "ai_elapsed_ms": float(ai_elapsed_ms),
        "ai_fallback": bool(ai_fallback),
        "ai_fallback_reason": str(ai_fallback_reason or "none"),
        "expected_reply": final_reply,
        "trace": trace_payload,
    }
