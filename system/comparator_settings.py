"""
system/comparator_settings.py

文件作用:
    比较器热更新参数管理。
    负责参数校验、应用与快照输出。
"""

from typing import Any, Dict

from fastapi import HTTPException

from system.comparator import DialogComparator
from system.runtime_settings import apply_ai_runtime_settings, build_ai_meta_payload


def comparator_settings_payload(comparator: DialogComparator) -> Dict[str, Any]:
    """组装当前比较器与 AI 运行时配置快照。"""
    payload = {
        "similarity_threshold": float(comparator.similarity_threshold),
        "rerank_query_weight": float(comparator.rerank_weights[0]),
        "rerank_reply_weight": float(comparator.rerank_weights[1]),
        "context_matching_enabled": bool(getattr(comparator, "context_matching_enabled", True)),
        "context_max_turns": int(comparator.context_max_turns),
        "max_text_len": int(comparator.max_text_len),
        "context_short_query_len": int(comparator.context_short_query_len),
        "context_overlap_threshold": float(comparator.context_overlap_threshold),
        "context_semantic_threshold": float(comparator.context_semantic_threshold),
        "coarse_recall_count": int(getattr(comparator, "coarse_recall_count", 0)),
        "rerank_top_k": int(getattr(comparator, "rerank_top_k", 5)),
    }
    payload.update(build_ai_meta_payload())
    return payload


def apply_comparator_settings(comparator: DialogComparator, req: Any) -> None:
    """将请求中的参数应用到比较器与 AI 运行时。"""
    # 1) 基础阈值与权重参数
    if req.similarity_threshold is not None:
        val = float(req.similarity_threshold)
        if val < 0.0 or val > 1.0:
            raise HTTPException(status_code=400, detail="similarity_threshold 必须在 0 到 1 之间。")
        comparator.similarity_threshold = val


    # 2) 语义重排权重（问问/问答）
    query_weight = req.rerank_query_weight
    reply_weight = req.rerank_reply_weight
    if query_weight is not None or reply_weight is not None:
        q = float(query_weight) if query_weight is not None else float(comparator.rerank_weights[0])
        r = float(reply_weight) if reply_weight is not None else float(comparator.rerank_weights[1])
        if q < 0 or r < 0:
            raise HTTPException(status_code=400, detail="rerank 权重不能为负数。")
        if q + r <= 1e-12:
            raise HTTPException(status_code=400, detail="rerank 两项权重不能同时为 0。")
        comparator.rerank_weights = (q, r)

    # 3) 上下文判定相关参数
    if req.context_max_turns is not None:
        val = int(req.context_max_turns)
        if val < 1 or val > 10:
            raise HTTPException(status_code=400, detail="context_max_turns 必须在 1 到 10 之间。")
        comparator.context_max_turns = val

    if req.context_matching_enabled is not None:
        comparator.context_matching_enabled = bool(req.context_matching_enabled)

    if req.max_text_len is not None:
        val = int(req.max_text_len)
        if val < 8 or val > 256:
            raise HTTPException(status_code=400, detail="max_text_len 必须在 8 到 256 之间。")
        comparator.max_text_len = val

    if req.context_short_query_len is not None:
        val = int(req.context_short_query_len)
        if val < 2 or val > 64:
            raise HTTPException(status_code=400, detail="context_short_query_len 必须在 2 到 64 之间。")
        comparator.context_short_query_len = val

    if req.context_overlap_threshold is not None:
        val = float(req.context_overlap_threshold)
        if val < 0.0 or val > 1.0:
            raise HTTPException(status_code=400, detail="context_overlap_threshold 必须在 0 到 1 之间。")
        comparator.context_overlap_threshold = val

    if req.context_semantic_threshold is not None:
        val = float(req.context_semantic_threshold)
        if val < 0.0 or val > 1.0:
            raise HTTPException(status_code=400, detail="context_semantic_threshold 必须在 0 到 1 之间。")
        comparator.context_semantic_threshold = val

    # 4) 召回规模参数（粗召回与精排）
    if req.coarse_recall_count is not None:
        val = int(req.coarse_recall_count)
        if val < 0 or val > 10000:
            raise HTTPException(status_code=400, detail="coarse_recall_count 必须在 0 到 10000 之间。")
        comparator.coarse_recall_count = val

    if req.rerank_top_k is not None:
        val = int(req.rerank_top_k)
        if val < 1 or val > 200:
            raise HTTPException(status_code=400, detail="rerank_top_k 必须在 1 到 200 之间。")
        comparator.rerank_top_k = val

    # 5) AI 增强运行时参数
    apply_ai_runtime_settings(ai_enhanced=req.ai_enhanced)