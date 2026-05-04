"""
tests/test_eval04_ai_mode_ablation.py

评测目标:
    对比三种回复生成策略的输出效果和时间性能。
    不对回复质量做模型自动判断，只输出原始对比数据供人工分析。

三种策略:
    策略A: AI直接回复 — 直接将用户问句发给大模型，由大模型自由生成答句
    策略B: 本地检索+AI增强 — 先用本系统检索 top-k 候选，再让大模型融合候选生成
    策略C: 纯本地检索 — 仅用本系统双塔检索，不做任何 AI 增强

测试数据:
    tests/data/eval_pairs.csv (30 对 QA)

输出:
    - eval04_cases_table.csv:  每个测试用例的输入/预期/三种策略的输出 + 耗时
    - eval04_time_table.csv:   三种策略的平均耗时统计
    - speed_compare.png:       三种策略耗时对比柱状图
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.ai_enhancer import generate_ai_enhanced_reply
from system.bootstrap import load_database_columns
from system.comparator import DialogComparator
from system.config import (
    AI_ENHANCE_API_KEY,
    AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL,
    AI_ENHANCE_TIMEOUT_SEC,
)
from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================
EVAL_PAIRS_CSV = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")
RERANK_TOP_K = 5


def _extract_ai_text(resp_json: Dict[str, Any]) -> str:
    """从 Ark API responses 格式的返回中提取文本。"""
    output_text = resp_json.get("output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = resp_json.get("output", [])
    if isinstance(output, list):
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            for part in (content if isinstance(content, list) else []):
                if isinstance(part, dict):
                    txt = part.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    return ""


def _ai_direct(question: str) -> Tuple[str, float]:
    """
    策略A: AI 直接回复。

    将用户问句直接发送给大模型，不依赖任何本地知识库。
    """
    if not AI_ENHANCE_API_KEY:
        return "[未配置API Key]", 0.0

    t0 = time.perf_counter()
    prompt = (
        "请作为中文对话助手，基于用户输入给出自然、简洁、有帮助的一句话回答。"
        "\n用户输入:\n"
        f"{str(question)}"
    )
    payload = {
        "model": AI_ENHANCE_MODEL_NAME,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    }
    headers = {"Authorization": f"Bearer {AI_ENHANCE_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(AI_ENHANCE_RESPONSES_URL, headers=headers, json=payload, timeout=AI_ENHANCE_TIMEOUT_SEC)
        resp.raise_for_status()
        text = _extract_ai_text(resp.json())
        return (text if text else "[AI返回为空]"), (time.perf_counter() - t0) * 1000.0
    except Exception:
        return "[AI请求失败]", (time.perf_counter() - t0) * 1000.0


def _local_ai_enhanced(comparator: DialogComparator, question: str) -> Tuple[str, float]:
    """
    策略B: 本地检索 + AI 增强。

    先用本地比较器检索 top-k 候选答句，
    再将候选和用户问句一起发送给大模型，让大模型融合候选生成最终回复。
    """
    t0 = time.perf_counter()
    used_q, cands = comparator.get_topk_candidates(user_input=question, history=[], top_k=RERANK_TOP_K)
    if not cands:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return "知识库没有这个问题的回复", elapsed

    enhanced = generate_ai_enhanced_reply(
        contextual_user_input=str(used_q or question),
        candidates=cands[:RERANK_TOP_K],
        top_k=RERANK_TOP_K,
        model_name=AI_ENHANCE_MODEL_NAME,
        responses_url=AI_ENHANCE_RESPONSES_URL,
        api_key=AI_ENHANCE_API_KEY,
        timeout_sec=AI_ENHANCE_TIMEOUT_SEC,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    if not str(enhanced).strip():
        return str(cands[0][1].get("reply", "")), elapsed
    return str(enhanced), elapsed


def _local_only(comparator: DialogComparator, question: str) -> Tuple[str, float]:
    """
    策略C: 纯本地检索。

    只用系统双塔做语义匹配，从知识库中返回最相似的答句。
    """
    t0 = time.perf_counter()
    reply = str(comparator.compare(question, history=[], top_k=RERANK_TOP_K)[0])
    elapsed = (time.perf_counter() - t0) * 1000.0
    return reply, elapsed


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval04_ai_mode_ablation")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载测试用例
    if not os.path.exists(EVAL_PAIRS_CSV):
        raise FileNotFoundError(f"测试用例文件不存在: {EVAL_PAIRS_CSV}")
    eval_df = pd.read_csv(EVAL_PAIRS_CSV)
    eval_queries = eval_df["query"].astype(str).tolist()
    eval_expected = eval_df["response"].astype(str).tolist()

    print(f"测试用例: {len(eval_queries)} 对")
    if not AI_ENHANCE_API_KEY:
        print("[警告] 未配置 AI_ENHANCE_API_KEY，AI 相关策略将无法正常执行")

    # 初始化系统
    engine = SimCSEModelEngine()
    query_index, response_index, text_store = load_database_columns()
    query_texts = [text_store.get_query(i) for i in range(len(text_store))]
    reply_texts = [text_store.get_response(i) for i in range(len(text_store))]
    comparator = DialogComparator(
        model_engine=engine,
        query_index=query_index,
        response_index=response_index,
        doc_texts=[],
        query_texts=query_texts,
        reply_texts=reply_texts,
        similarity_threshold=0.0,
        context_matching_enabled=False,
        rerank_top_k=RERANK_TOP_K,
    )

    # ============================================================
    # 逐用例评测三种策略
    # ============================================================
    cases: List[Dict[str, Any]] = []
    times_by_mode: Dict[str, List[float]] = {
        "AI直接回复": [],
        "本地检索+AI增强": [],
        "纯本地检索": [],
    }

    for i, (q, expected) in enumerate(tqdm(
        zip(eval_queries, eval_expected),
        total=len(eval_queries),
        desc="AI消融评测",
        unit="条",
    )):
        a_reply, a_ms = _ai_direct(q)
        b_reply, b_ms = _local_ai_enhanced(comparator, q)
        c_reply, c_ms = _local_only(comparator, q)

        cases.append(
            {
                "序号": i + 1,
                "输入问句": q,
                "预期答句": expected,
                "AI直接回复_输出": a_reply,
                "AI直接回复_耗时ms": round(a_ms, 2),
                "本地+AI增强_输出": b_reply,
                "本地+AI增强_耗时ms": round(b_ms, 2),
                "纯本地检索_输出": c_reply,
                "纯本地检索_耗时ms": round(c_ms, 2),
            }
        )
        times_by_mode["AI直接回复"].append(a_ms)
        times_by_mode["本地检索+AI增强"].append(b_ms)
        times_by_mode["纯本地检索"].append(c_ms)

    # ============================================================
    # 输出用例对比表
    # ============================================================
    cases_df = pd.DataFrame(cases)
    cases_df.to_csv(os.path.join(out_dir, "eval04_cases_table.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 输出时间统计
    # ============================================================
    time_rows: List[Dict[str, Any]] = []
    for mode, times in times_by_mode.items():
        arr = np.asarray(times, dtype=np.float64)
        time_rows.append(
            {
                "策略": mode,
                "平均耗时ms": round(float(np.mean(arr)), 2),
                "最小耗时ms": round(float(np.min(arr)), 2),
                "最大耗时ms": round(float(np.max(arr)), 2),
                "标准差ms": round(float(np.std(arr)), 2),
            }
        )
    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(os.path.join(out_dir, "eval04_time_table.csv"), index=False, encoding="utf-8-sig")

    print("\n===== 三种策略耗时统计 =====")
    print(time_df.to_string(index=False))

    # 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    names = time_df["策略"].tolist()
    vals = time_df["平均耗时ms"].tolist()
    colors = ["#ef4444", "#f59e0b", "#2a9d8f"]
    bars = ax.bar(names, vals, color=colors, edgecolor="#1f2a37")
    ax.set_ylabel("平均耗时（ms）")
    ax.set_title("三种回复策略耗时对比")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v:.2f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_compare.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()
