"""
tests/test_eval02_retrieval_algo_db.py

评测目标:
    在相同测试数据下，对比三种检索算法的输出效果和时间性能。
    不对输出准确性做模型自动判断，只输出原始对比数据供人工分析。

三种检索算法:
    算法A: 全量余弦搜索 — 用户问句向量与语料库所有问句向量逐一计算余弦相似度
    算法B: 全量问答加权搜索 — 用户问句与语料库所有问句+答句加权计算相似度
    算法C: FAISS精排 — 高性能向量索引粗召回 + 问问/问答加权重排

测试数据:
    从 tests/data/eval_pairs.csv 读取测试用例
    从 lccc_large.csv 构建不同规模的语料库 (1k/5k/10k/50k)

输出:
    - eval02_cases_table.csv:      每个测试用例的输入/预期/三种算法输出
    - eval02_time_gradient.csv:    不同数据规模下各算法的时间对比
    - time_gradient.png:           时间增长梯度折线图
    - speed_compare.png:           三种算法的平均耗时柱状图
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.bootstrap import load_database_columns
from system.comparator import DialogComparator

from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================

# 测试用例文件（由对话历史中生成，30对 QA）
EVAL_PAIRS_CSV = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")

# 数据规模梯度（用于观察时间增长趋势）
CORPUS_SIZES = [1000, 5000, 10000, 50000]

# 全量扫描分块大小（避免单次矩阵乘法撑爆内存）
CHUNK_SIZE = 2048

# 编码批大小
BATCH_SIZE = 64

EPS = 1e-8

# 本地比较器精排数量
RERANK_TOP_K = 10


def _sample_corpus(
    query_texts: List[str],
    reply_texts: List[str],
    corpus_size: int,
    seed: int = 2026,
) -> Tuple[List[str], List[str]]:
    """从全量语料中随机采样指定数量的子集。"""
    total = min(len(query_texts), len(reply_texts))
    k = min(corpus_size, total)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(total, size=k, replace=False).tolist())
    return [query_texts[i] for i in idx], [reply_texts[i] for i in idx]


def _full_cosine_search(
    query_vectors: np.ndarray,
    corpus_query_vectors: np.ndarray,
    corpus_reply_vectors: np.ndarray,
    corpus_replies: List[str],
    query_weight: float = 1.0,
    reply_weight: float = 0.0,
) -> Tuple[List[str], float]:
    """
    全量余弦搜索。

    当 reply_weight=0 时是纯问句余弦搜索（算法A），
    当 reply_weight>0 时是问答加权搜索（算法B）。

    返回: (各用例的匹配答句列表, 总耗时ms)
    """
    t0 = time.perf_counter()
    q_mat = np.asarray(query_vectors, dtype=np.float32)
    cq_mat = np.asarray(corpus_query_vectors, dtype=np.float32)
    cr_mat = np.asarray(corpus_reply_vectors, dtype=np.float32)

    q_norms = np.linalg.norm(q_mat, axis=1) + EPS
    cq_norms = np.linalg.norm(cq_mat, axis=1) + EPS
    cr_norms = np.linalg.norm(cr_mat, axis=1) + EPS

    best_scores = np.full((q_mat.shape[0],), -1e9, dtype=np.float32)
    best_ids = np.full((q_mat.shape[0],), -1, dtype=np.int64)

    # 分块扫描避免内存溢出
    for start in range(0, cq_mat.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, cq_mat.shape[0])
        q_chunk = cq_mat[start:end]
        r_chunk = cr_mat[start:end]
        qc_norms = cq_norms[start:end]
        rc_norms = cr_norms[start:end]

        q_scores = (q_mat @ q_chunk.T) / (q_norms[:, None] * qc_norms[None, :])
        scores = query_weight * q_scores

        if reply_weight > 0.0:
            r_scores = (q_mat @ r_chunk.T) / (q_norms[:, None] * rc_norms[None, :])
            scores = scores + reply_weight * r_scores

        chunk_best = np.argmax(scores, axis=1)
        chunk_scores = scores[np.arange(scores.shape[0]), chunk_best]
        update = chunk_scores > best_scores
        best_scores[update] = chunk_scores[update]
        best_ids[update] = start + chunk_best[update]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    replies = [corpus_replies[int(i)] if 0 <= int(i) < len(corpus_replies) else "" for i in best_ids]
    return replies, elapsed_ms


def _faiss_rerank_search(
    comparator: DialogComparator,
    queries: List[str],
    reply_texts: List[str],
) -> Tuple[List[str], float]:
    """
    FAISS 精排检索（算法C）。

    使用系统的 DialogComparator，流程为:
    FAISS HNSW 索引粗召回 → 问问/问答加权重排 → 返回最优答句
    """
    t0 = time.perf_counter()
    replies: List[str] = []
    for q in queries:
        result = comparator.compare(q, history=[], top_k=RERANK_TOP_K)
        replies.append(str(result[0]))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return replies, elapsed_ms


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval02_retrieval_algo_db")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载测试用例
    if not os.path.exists(EVAL_PAIRS_CSV):
        raise FileNotFoundError(f"测试用例文件不存在: {EVAL_PAIRS_CSV}")
    eval_df = pd.read_csv(EVAL_PAIRS_CSV)
    eval_queries = eval_df["query"].astype(str).tolist()
    eval_expected = eval_df["response"].astype(str).tolist()

    print(f"加载测试用例: {len(eval_queries)} 对")

    # 加载全量语料库
    engine = SimCSEModelEngine()
    query_index, response_index, text_store = load_database_columns()
    query_texts = [text_store.get_query(i) for i in range(len(text_store))]
    reply_texts = [text_store.get_response(i) for i in range(len(text_store))]
    print(f"全量语料库规模: {len(text_store)} 条")

    # ============================================================
    # 准备：一次性编码所有测试问句（节省重复编码）
    # ============================================================
    eval_query_vectors = np.asarray(
        engine.encode(eval_queries, encoder="query", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True),
        dtype=np.float32,
    )

    # ============================================================
    # 逐规模评测
    # ============================================================
    time_records: List[Dict[str, Any]] = []
    cases_records: Dict[int, pd.DataFrame] = {}

    for size in tqdm(CORPUS_SIZES, desc="数据规模梯度评测", unit="规模"):
        # 构建当前规模的语料子集
        sub_queries, sub_replies = _sample_corpus(query_texts, reply_texts, size)
        sub_query_vectors = np.asarray(
            engine.encode(sub_queries, encoder="query", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True),
            dtype=np.float32,
        )
        sub_reply_vectors = np.asarray(
            engine.encode(sub_replies, encoder="response", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True),
            dtype=np.float32,
        )

        # 算法A: 全量问句余弦
        a_replies, a_ms = _full_cosine_search(
            eval_query_vectors, sub_query_vectors, sub_reply_vectors,
            sub_replies, query_weight=1.0, reply_weight=0.0,
        )

        # 算法B: 全量问答加权
        b_replies, b_ms = _full_cosine_search(
            eval_query_vectors, sub_query_vectors, sub_reply_vectors,
            sub_replies, query_weight=0.75, reply_weight=0.25,
        )

        # 算法C: FAISS精排（仅全量规模有索引）
        if size == CORPUS_SIZES[-1]:
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
            c_replies, c_ms = _faiss_rerank_search(comparator, eval_queries, reply_texts)
        else:
            c_replies = [""] * len(eval_queries)
            c_ms = 0.0

        # 记录时间
        time_records.append(
            {
                "数据规模": size,
                "全量问句余弦(ms)": round(a_ms / max(len(eval_queries), 1), 2),
                "全量问答加权(ms)": round(b_ms / max(len(eval_queries), 1), 2),
                "FAISS精排(ms)": round(c_ms / max(len(eval_queries), 1), 2) if c_ms > 0 else 0,
            }
        )

        # 保存用例级对比（仅最大规模输出详表）
        if size == CORPUS_SIZES[-1]:
            cases_df = pd.DataFrame(
                {
                    "序号": range(1, len(eval_queries) + 1),
                    "输入问句": eval_queries,
                    "预期答句": eval_expected,
                    "算法A_全量余弦": a_replies,
                    "算法B_问答加权": b_replies,
                    "算法C_FAISS精排": c_replies,
                }
            )
            cases_df.to_csv(
                os.path.join(out_dir, "eval02_cases_table.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    # ============================================================
    # 输出时间表格
    # ============================================================
    time_df = pd.DataFrame(time_records)
    time_df.to_csv(
        os.path.join(out_dir, "eval02_time_gradient.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 不同数据规模下各算法的平均每句耗时 (ms) =====")
    print(time_df.to_string(index=False))

    # ============================================================
    # 图表: 时间增长梯度折线图
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = time_df["数据规模"].values
    for col, color, label in [
        ("全量问句余弦(ms)", "#ef4444", "全量问句余弦"),
        ("全量问答加权(ms)", "#f59e0b", "全量问答加权"),
        ("FAISS精排(ms)", "#2a9d8f", "FAISS精排"),
    ]:
        vals = time_df[col].values
        ax.plot(sizes.astype(str), vals, marker="o", color=color, label=label, linewidth=2)
    ax.set_xlabel("语料库规模（条）")
    ax.set_ylabel("平均每句耗时（ms）")
    ax.set_title("检索算法时间增长梯度对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_gradient.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 柱状图: 全量规模下三算法耗时对比
    last_row = time_df[time_df["数据规模"] == CORPUS_SIZES[-1]]
    if not last_row.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        alg_names = ["全量问句余弦", "全量问答加权", "FAISS精排"]
        alg_cols = ["全量问句余弦(ms)", "全量问答加权(ms)", "FAISS精排(ms)"]
        alg_vals = [float(last_row[c].iloc[0]) for c in alg_cols]
        colors = ["#ef4444", "#f59e0b", "#2a9d8f"]
        bars = ax2.bar(alg_names, alg_vals, color=colors, edgecolor="#1f2a37")
        ax2.set_ylabel("平均每句耗时（ms）")
        ax2.set_title(f"全量规模 ({CORPUS_SIZES[-1]} 条) 下三种算法耗时对比")
        for b, v in zip(bars, alg_vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.2f}", ha="center", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "speed_compare.png"), dpi=300, bbox_inches="tight")
        plt.close(fig2)

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()
