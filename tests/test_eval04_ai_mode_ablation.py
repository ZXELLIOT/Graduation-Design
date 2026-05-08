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
    - eval04_accuracy_table.csv: 三种策略平均答句相似度统计（BGE）
    - speed_compare.png:       三种策略耗时对比柱状图
    - accuracy_compare.png:    三种策略回复准确度对比图（BGE）
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.ai_enhancer import generate_ai_enhanced_reply
from system.bootstrap import load_faiss_index, load_text_store
from system.comparator import DialogComparator
from system.config import (
    AI_ENHANCE_API_KEY,
    AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL,
    AI_ENHANCE_TIMEOUT_SEC,
)
from system.model_engine import SimCSEModelEngine
from tests.test_utils import encode_hf_texts
from tests.tests_config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================
EVAL_PAIRS_CSV = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")
EVAL_PAIR_COUNT = 50
CORPUS_SIZE = 10000
RERANK_TOP_K = 5
SIM_SCORER_MODEL = "bge-small-zh-v1.5"
SIM_BATCH_SIZE = 64


def _load_eval_pairs_line_by_line(max_count: int) -> Tuple[List[str], List[str]]:
    """按行读取测试用例，与前面评测脚本保持一致。"""
    if not os.path.exists(EVAL_PAIRS_CSV):
        raise FileNotFoundError(f"测试用例文件不存在: {EVAL_PAIRS_CSV}")

    eval_queries: List[str] = []
    eval_expected: List[str] = []
    for i, chunk in enumerate(pd.read_csv(EVAL_PAIRS_CSV, chunksize=1)):
        if i >= max_count:
            break
        row = chunk.iloc[0]
        eval_queries.append(str(row["query"]))
        eval_expected.append(str(row["response"]))

    if not eval_queries:
        raise RuntimeError("未读取到任何测试用例")
    return eval_queries, eval_expected


def _build_small_query_index(full_index: Any, n_rows: int) -> Any:
    """从全量索引重建前 n_rows 条向量，构建小索引（与前面脚本一致）。"""
    total = min(max(int(n_rows), 0), int(getattr(full_index, "ntotal", 0)))
    if total <= 0:
        raise RuntimeError("无法构建小索引：可用向量数量为0。")

    if hasattr(full_index, "reconstruct_n"):
        vecs = np.asarray(full_index.reconstruct_n(0, total), dtype=np.float32)
    else:
        vecs = np.asarray([full_index.reconstruct(i) for i in range(total)], dtype=np.float32)

    dim = int(vecs.shape[1])
    if hasattr(full_index, "hnsw"):
        hnsw_m = int(full_index.hnsw.nb_neighbors(1))
        small_index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        if hasattr(full_index.hnsw, "efSearch"):
            small_index.hnsw.efSearch = int(full_index.hnsw.efSearch)
    else:
        small_index = faiss.IndexFlatIP(dim)
    small_index.add(vecs)  # type: ignore[call-arg]
    return small_index


def _append_bge_similarity(cases_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算三种策略的预期答句 vs 实际输出 BGE 相似度。"""
    scorer_path = os.path.join(TEST_MODELS_DIR, SIM_SCORER_MODEL)
    if not os.path.exists(scorer_path):
        raise FileNotFoundError(f"相似度模型不存在: {scorer_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(scorer_path, local_files_only=True)
    model = AutoModel.from_pretrained(scorer_path, local_files_only=True).to(device)
    model.eval()

    expected = cases_df["预期答句"].astype(str).tolist()
    mode_cols = {
        "AI直接回复": "AI直接回复_输出",
        "本地检索+AI增强": "本地+AI增强_输出",
        "纯本地检索": "纯本地检索_输出",
    }

    text_list: List[str] = expected[:]
    spans: Dict[str, Tuple[int, int]] = {}
    start = len(text_list)
    for mode, col in mode_cols.items():
        outs = cases_df[col].astype(str).tolist()
        spans[mode] = (start, start + len(outs))
        text_list.extend(outs)
        start += len(outs)

    embs = encode_hf_texts(
        model=model,
        tokenizer=tokenizer,
        texts=text_list,
        device=device,
        batch_size=SIM_BATCH_SIZE,
        max_len=512,
    )
    n = len(cases_df)
    exp_emb = embs[:n]

    out_df = cases_df.copy()
    rows: List[Dict[str, Any]] = []
    for mode in ["AI直接回复", "本地检索+AI增强", "纯本地检索"]:
        s, e = spans[mode]
        out_emb = embs[s:e]
        sims = np.sum(exp_emb * out_emb, axis=1)
        sims = np.clip(sims, -1.0, 1.0)
        out_df[f"{mode}_答句相似度"] = np.round(sims, 6)
        rows.append(
            {
                "策略": mode,
                "平均答句相似度": round(float(np.mean(sims)), 6),
                "标准差": round(float(np.std(sims)), 6),
            }
        )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_df, pd.DataFrame(rows)


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
    used_q, cands, _ = comparator.get_topk_candidates_with_meta(user_input=question, history=[], top_k=RERANK_TOP_K)
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
    reply = str(comparator.compare_with_meta(question, history=[], top_k=RERANK_TOP_K)[0])
    elapsed = (time.perf_counter() - t0) * 1000.0
    return reply, elapsed


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval04_ai_mode_ablation")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载测试用例（按行读取）
    eval_queries, eval_expected = _load_eval_pairs_line_by_line(EVAL_PAIR_COUNT)

    print(f"测试用例: {len(eval_queries)} 对")
    if not AI_ENHANCE_API_KEY:
        print("[警告] 未配置 AI_ENHANCE_API_KEY，AI 相关策略将无法正常执行")

    # 初始化系统
    engine = SimCSEModelEngine()
    full_query_index = load_faiss_index()
    query_index = _build_small_query_index(full_query_index, CORPUS_SIZE)
    text_store = load_text_store(max_rows=int(getattr(query_index, "ntotal", 0)))
    comparator = DialogComparator(
        model_engine=engine,
        query_index=query_index,
        text_store=text_store,
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
    # 输出用例对比表（含 BGE 相似度）
    # ============================================================
    cases_df = pd.DataFrame(cases)
    cases_df, acc_df = _append_bge_similarity(cases_df)
    cases_df.to_csv(os.path.join(out_dir, "eval04_cases_table.csv"), index=False, encoding="utf-8-sig")
    acc_df.to_csv(os.path.join(out_dir, "eval04_accuracy_table.csv"), index=False, encoding="utf-8-sig")

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
    print("\n===== 三种策略回复准确度（BGE） =====")
    print(acc_df.to_string(index=False))

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

    # 准确度对比图（BGE）
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    acc_names = acc_df["策略"].tolist()
    acc_vals = acc_df["平均答句相似度"].tolist()
    acc_colors = ["#ef4444", "#f59e0b", "#2a9d8f"]
    bars2 = ax2.bar(acc_names, acc_vals, color=acc_colors, edgecolor="#1f2a37")
    ax2.set_ylabel("平均答句相似度")
    ax2.set_title("三种策略回复准确度对比（BGE）")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, acc_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "accuracy_compare.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()