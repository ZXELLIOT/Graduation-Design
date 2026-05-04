"""
tests/test_eval03_baselines_with_system.py

评测目标:
    使用相同的测试数据和语料库，对比本项目双塔模型与 4 个基准模型的检索效果。
    不对输出准确性做模型自动判断，只输出原始对比数据供人工分析。

评测方案:
    对每个模型，将语料库所有答句编码为向量，用户问句编码后做全量余弦搜索，
    取出相似度最高的答句。对比不同模型的输入/预期/实际输出及平均耗时。

    本项目模型使用双塔（query_encoder 编码问句 + response_encoder 编码答句），
    检索时做问答加权余弦搜索。基准模型为单塔，仅做问句-答句余弦匹配。

测试数据:
    - 测试用例: tests/data/eval_pairs.csv (30 对 QA)
    - 语料库: lccc_large.csv 中随机采样 5000 条

输出:
    - eval03_cases_table.csv:  每个测试用例的输入/预期/各模型输出
    - eval03_time_table.csv:   各模型的平均耗时
    - speed_compare.png:       各模型速度对比柱状图
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.config import DB_CSV_PATH, RERANK_WEIGHTS
from system.model_engine import SimCSEModelEngine
from tests.config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================
EVAL_PAIRS_CSV = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")
CORPUS_SIZE = 5000
BATCH_SIZE = 64
CHUNK_SIZE = 1024
MAX_LEN = 64
EPS = 1e-8


def _load_eval_and_corpus() -> Tuple[List[str], List[str], List[str], List[str]]:
    """加载测试用例和语料库。"""
    # 测试用例
    eval_df = pd.read_csv(EVAL_PAIRS_CSV)
    eval_queries = eval_df["query"].astype(str).tolist()
    eval_expected = eval_df["response"].astype(str).tolist()

    # 语料库（从 lccc_large 采样）
    corpus_df = pd.read_csv(DB_CSV_PATH, usecols=["query", "response"])
    corpus_df = corpus_df.dropna()
    rng = np.random.default_rng(2026)
    idx = sorted(rng.choice(len(corpus_df), size=min(CORPUS_SIZE, len(corpus_df)), replace=False).tolist())
    sample = corpus_df.iloc[idx].reset_index(drop=True)
    corpus_queries = sample["query"].astype(str).tolist()
    corpus_replies = sample["response"].astype(str).tolist()

    return eval_queries, eval_expected, corpus_queries, corpus_replies


def _full_cosine_best(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_replies: List[str],
) -> List[str]:
    """对预编码向量做全量余弦搜索，返回每句的最佳匹配答句。"""
    q_mat = np.asarray(query_vectors, dtype=np.float32)
    c_mat = np.asarray(corpus_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1) + EPS
    c_norms = np.linalg.norm(c_mat, axis=1) + EPS

    best_scores = np.full((q_mat.shape[0],), -1e9, dtype=np.float32)
    best_ids = np.full((q_mat.shape[0],), -1, dtype=np.int64)

    for start in range(0, c_mat.shape[0], CHUNK_SIZE):
        chunk = c_mat[start : start + CHUNK_SIZE]
        chunk_norms = c_norms[start : start + CHUNK_SIZE]
        scores = (q_mat @ chunk.T) / (q_norms[:, None] * chunk_norms[None, :])
        local_ids = np.argmax(scores, axis=1)
        local_scores = scores[np.arange(scores.shape[0]), local_ids]
        update = local_scores > best_scores
        best_scores[update] = local_scores[update]
        best_ids[update] = start + local_ids[update]

    return [corpus_replies[int(i)] if 0 <= int(i) < len(corpus_replies) else "" for i in best_ids]


def _full_weighted_best(
    query_vectors: np.ndarray,
    corpus_query_vectors: np.ndarray,
    corpus_reply_vectors: np.ndarray,
    corpus_replies: List[str],
) -> List[str]:
    """问答加权余弦搜索（本项目模型专用双塔模式）。"""
    q_mat = np.asarray(query_vectors, dtype=np.float32)
    cq_mat = np.asarray(corpus_query_vectors, dtype=np.float32)
    cr_mat = np.asarray(corpus_reply_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1) + EPS
    cq_norms = np.linalg.norm(cq_mat, axis=1) + EPS
    cr_norms = np.linalg.norm(cr_mat, axis=1) + EPS

    qw, rw = float(RERANK_WEIGHTS[0]), float(RERANK_WEIGHTS[1])
    best_scores = np.full((q_mat.shape[0],), -1e9, dtype=np.float32)
    best_ids = np.full((q_mat.shape[0],), -1, dtype=np.int64)

    for start in range(0, cq_mat.shape[0], CHUNK_SIZE):
        q_chunk = cq_mat[start : start + CHUNK_SIZE]
        r_chunk = cr_mat[start : start + CHUNK_SIZE]
        qc_norms = cq_norms[start : start + CHUNK_SIZE]
        rc_norms = cr_norms[start : start + CHUNK_SIZE]
        q_scores = (q_mat @ q_chunk.T) / (q_norms[:, None] * qc_norms[None, :])
        r_scores = (q_mat @ r_chunk.T) / (q_norms[:, None] * rc_norms[None, :])
        scores = qw * q_scores + rw * r_scores
        local_ids = np.argmax(scores, axis=1)
        local_scores = scores[np.arange(scores.shape[0]), local_ids]
        update = local_scores > best_scores
        best_scores[update] = local_scores[update]
        best_ids[update] = start + local_ids[update]

    return [corpus_replies[int(i)] if 0 <= int(i) < len(corpus_replies) else "" for i in best_ids]


def _encode_hf_batch(model, tokenizer, texts: List[str], device) -> np.ndarray:
    """HuggingFace 标准模型批量编码。"""
    arr: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [str(x) for x in texts[i : i + BATCH_SIZE]]
            tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LEN).to(device)
            out = model(**tok)
            emb = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out.last_hidden_state[:, 0, :]
            arr.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(arr) if arr else np.empty((0, 0), dtype=np.float32)


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval03_baselines_with_system")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    eval_queries, eval_expected, corpus_queries, corpus_replies = _load_eval_and_corpus()
    print(f"测试用例: {len(eval_queries)} 对, 语料库: {len(corpus_queries)} 条")

    # ============================================================
    # 评测各模型
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 全部模型的输出和耗时
    model_replies: Dict[str, List[str]] = {}
    model_times: Dict[str, float] = {}

    # --- 本项目双塔模型 ---
    print("\n----- 本项目双塔模型 -----")
    engine = SimCSEModelEngine()
    t0 = time.perf_counter()
    eval_q_vecs = np.asarray(engine.encode(eval_queries, encoder="query", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True), dtype=np.float32)
    corpus_q_vecs = np.asarray(engine.encode(corpus_queries, encoder="query", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True), dtype=np.float32)
    corpus_r_vecs = np.asarray(engine.encode(corpus_replies, encoder="response", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True), dtype=np.float32)
    replies = _full_weighted_best(eval_q_vecs, corpus_q_vecs, corpus_r_vecs, corpus_replies)
    model_replies["本项目双塔"] = replies
    model_times["本项目双塔"] = (time.perf_counter() - t0) * 1000.0 / max(len(eval_queries), 1)

    # --- 基准模型 ---
    baselines = {
        "基准_text2vec": "text2vec-base-chinese",
        "基准_bert": "bert-base-chinese",
        "基准_roberta": "chinese-roberta-wwm-ext",
        "基准_minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    }

    for short_name, folder in baselines.items():
        path = os.path.join(TEST_MODELS_DIR, folder)
        if not os.path.exists(path):
            print(f"  [跳过] {short_name}: 模型目录不存在")
            continue

        print(f"----- {short_name} -----")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True).to(device)
        model.eval()

        t1 = time.perf_counter()
        eval_vecs = _encode_hf_batch(model, tokenizer, eval_queries, device)
        corpus_vecs = _encode_hf_batch(model, tokenizer, corpus_replies, device)
        replies = _full_cosine_best(eval_vecs, corpus_vecs, corpus_replies)
        model_replies[short_name] = replies
        model_times[short_name] = (time.perf_counter() - t1) * 1000.0 / max(len(eval_queries), 1)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 输出用例对比表
    # ============================================================
    cases_data = {
        "序号": range(1, len(eval_queries) + 1),
        "输入问句": eval_queries,
        "预期答句": eval_expected,
    }
    for model_name in model_replies:
        cases_data[f"{model_name}_输出"] = model_replies[model_name]

    cases_df = pd.DataFrame(cases_data)
    cases_df.to_csv(os.path.join(out_dir, "eval03_cases_table.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 输出时间对比
    # ============================================================
    time_df = pd.DataFrame(
        {"模型": list(model_times.keys()), "平均每句耗时ms": list(model_times.values())}
    ).sort_values("平均每句耗时ms")
    time_df.to_csv(os.path.join(out_dir, "eval03_time_table.csv"), index=False, encoding="utf-8-sig")

    print("\n===== 各模型平均每句耗时 (ms) =====")
    print(time_df.to_string(index=False))

    # 柱状图
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = time_df["模型"].tolist()
    vals = time_df["平均每句耗时ms"].tolist()
    colors = ["#2a9d8f" if "双塔" in n else "#3b82f6" for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="#1f2a37")
    ax.set_ylabel("平均每句耗时（ms）")
    ax.set_title("各模型检索速度对比（语料库 5000 条）")
    ax.set_xticklabels(names, rotation=15, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_compare.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()
