"""
tests/test_eval02_retrieval_algo_db.py

目标:
    在 db/data/lccc_large.csv 中抽样 1000 条问答，评测三种检索策略的准确性与速度。
输出:
    仅输出 2 张图:
    1) 准确性对比图(本地语义评分)
    2) 反应时间对比图
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.bootstrap import load_database_columns
from system.comparator import DialogComparator
from system.config import DB_PREFIX
from system.model_engine import SimCSEModelEngine
from tests.config import TEST_RESULTS_DIR

SAMPLE_COUNT = 1000
EPS = 1e-8
LOCAL_EVAL_MODEL_DIR = r"C:\software\class\simcse-demo\tests\models\paraphrase-multilingual-MiniLM-L12-v2"
COMPARATOR_TOP_K = 10
FULL_SCAN_CHUNK_SIZE = 4096
TARGET_DB_SIZE_FOR_SPEED_EST = 1_000_000
FIXED_FULLSCAN_EST_FACTOR = 0.2778 * 500


class HFQueryAdapter:
    """单塔 HF 模型适配器，仅用于本地语义评审打分。"""

    def __init__(self, model_path: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.model.eval()
        self.device = device

    def encode_one(self, text: str, encoder: str = "query") -> torch.Tensor:
        _ = encoder
        with torch.no_grad():
            tok = self.tokenizer([str(text)], padding=True, truncation=True, return_tensors="pt", max_length=64).to(self.device)
            out = self.model(**tok)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0, :]
        return emb[0].detach().cpu()


def _to_np_float32(v: Any) -> np.ndarray:
    """兼容 torch.Tensor / numpy.ndarray 两种向量输出。"""
    if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
        return v.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _bar_plot(df, x_col, y_col, title, out_path, color):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(df))
    y = np.asarray(df[y_col].to_numpy(), dtype=np.float64)
    bars = ax.bar(x, y, color=color, edgecolor="#1f2a37")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=20, ha="right")
    ax.set_title(title)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + max(0.005, h * 0.01), f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sample_cases(query_texts: List[str], reply_texts: List[str], n: int = SAMPLE_COUNT) -> List[Tuple[str, str]]:
    total = min(len(query_texts), len(reply_texts))
    k = min(max(int(n), 1), total)
    rng = np.random.default_rng(2026)
    idx = rng.choice(total, size=k, replace=False)
    return [(str(query_texts[i]), str(reply_texts[i])) for i in idx.tolist()]


def _local_semantic_score(engine: Any, text_a: str, text_b: str) -> float:
    """本地语义评分: 余弦相似度线性映射到 0~1。"""
    va = _to_np_float32(engine.encode_one(str(text_a), encoder="query"))
    vb = _to_np_float32(engine.encode_one(str(text_b), encoder="query"))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + EPS)
    cos = float(np.dot(va, vb) / denom)
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))


def _judge_pairs(pairs: List[Tuple[str, str]], engine: Any) -> Dict[Tuple[str, str], float]:
    cache: Dict[Tuple[str, str], float] = {}
    uniq = list(dict.fromkeys(pairs))
    for a, b in tqdm(uniq, total=len(uniq), desc="本地评审", unit="对", leave=False):
        cache[(a, b)] = _local_semantic_score(engine, a, b)
    return cache


def _reconstruct_chunk(index: Any, start: int, size: int) -> np.ndarray:
    """按块重建 FAISS 向量，优先使用 reconstruct_n。"""
    if size <= 0:
        return np.empty((0, 0), dtype=np.float32)

    if hasattr(index, "reconstruct_n"):
        try:
            return np.asarray(index.reconstruct_n(start, size), dtype=np.float32)
        except Exception:
            pass

    rows = [np.asarray(index.reconstruct(start + offset), dtype=np.float32) for offset in range(size)]  # type: ignore
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)


def _encode_query_batch(engine: SimCSEModelEngine, texts: List[str]) -> np.ndarray:
    """批量编码用户输入，返回 float32 矩阵。"""
    encoded = engine.encode(texts, encoder="query", batch_size=128, show_progress=False, return_numpy=True)
    return np.asarray(encoded, dtype=np.float32)


def _exact_full_scan_best_replies(
    query_vectors: np.ndarray,
    query_index: Any,
    reply_texts: List[str],
    response_index: Any = None,
    rerank_weights: Tuple[float, float] = (1.0, 0.0),
    chunk_size: int = FULL_SCAN_CHUNK_SIZE,
) -> List[str]:
    """对全量数据库做精确扫描，返回每个输入的最佳答句。"""
    if query_vectors.size == 0:
        return []

    q_mat = np.asarray(query_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1) + EPS
    total = int(getattr(query_index, "ntotal", 0))
    if total <= 0:
        return [""] * int(q_mat.shape[0])

    best_scores = np.full((q_mat.shape[0],), -1e9, dtype=np.float32)
    best_ids = np.full((q_mat.shape[0],), -1, dtype=np.int64)

    query_weight = float(rerank_weights[0])
    reply_weight = float(rerank_weights[1])

    for start in tqdm(range(0, total, chunk_size), total=(total + chunk_size - 1) // chunk_size, desc="全量扫描", unit="块", leave=False):
        size = min(chunk_size, total - start)
        query_chunk = _reconstruct_chunk(query_index, start, size)
        if query_chunk.size == 0:
            continue

        chunk_query_norms = np.linalg.norm(query_chunk, axis=1) + EPS
        query_scores = (q_mat @ query_chunk.T) / (q_norms[:, None] * chunk_query_norms[None, :])
        final_scores = query_weight * query_scores

        if response_index is not None and reply_weight > 0.0:
            response_chunk = _reconstruct_chunk(response_index, start, size)
            chunk_resp_norms = np.linalg.norm(response_chunk, axis=1) + EPS
            reply_scores = (q_mat @ response_chunk.T) / (q_norms[:, None] * chunk_resp_norms[None, :])
            final_scores = final_scores + reply_weight * reply_scores

        chunk_best_pos = np.argmax(final_scores, axis=1)
        chunk_best_scores = final_scores[np.arange(final_scores.shape[0]), chunk_best_pos]
        update_mask = chunk_best_scores > best_scores
        best_scores[update_mask] = chunk_best_scores[update_mask]
        best_ids[update_mask] = start + chunk_best_pos[update_mask]

    replies: List[str] = []
    for idx in best_ids.tolist():
        replies.append(reply_texts[idx] if 0 <= int(idx) < len(reply_texts) else "")
    return replies


def _print_comparison_summary(df: Any) -> None:
    """输出对比摘要，便于报告直接引用。"""
    row_map = {str(r["算法"]): r for _, r in df.iterrows()}
    comp = row_map.get("本地比较器")
    direct = row_map.get("全量问句余弦")
    weighted = row_map.get("全量问答加权")
    if comp is None or direct is None or weighted is None:
        return

    comp_acc = float(comp["平均准确性"])
    comp_ms = float(comp["平均耗时ms"])
    direct_acc = float(direct["平均准确性"])
    direct_ms = float(direct["平均耗时ms"])
    weighted_acc = float(weighted["平均准确性"])
    weighted_ms = float(weighted["平均耗时ms"])

    comp_raw_ms = float(comp.get("实测平均耗时ms", comp_ms))
    direct_raw_ms = float(direct.get("实测平均耗时ms", direct_ms))
    weighted_raw_ms = float(weighted.get("实测平均耗时ms", weighted_ms))
    est_scale = float(comp.get("倍率（0.2778x500）", 1.0))

    speed_gain_vs_direct = (direct_ms - comp_ms) / max(direct_ms, EPS)
    acc_gap_vs_direct = abs(comp_acc - direct_acc)
    acc_gap_vs_weighted = abs(comp_acc - weighted_acc)
    speed_gap_vs_weighted = abs(comp_ms - weighted_ms) / max(weighted_ms, EPS)

    print("\n===== 评测摘要 =====")
    print(
        f"对比全量问句余弦: 本地比较器平均耗时 {comp_ms:.4f} ms, 全量问句余弦 {direct_ms:.4f} ms, "
        f"速度提升 {speed_gain_vs_direct * 100:.2f}%, 准确性差距 {acc_gap_vs_direct:.4f}"
    )
    print(
        f"对比全量问答加权: 本地比较器平均准确性 {comp_acc:.4f}, 全量问答加权 {weighted_acc:.4f}, "
        f"准确性差距 {acc_gap_vs_weighted:.4f}, 速度差距 {speed_gap_vs_weighted * 100:.2f}%"
    )
    print(
        f"实测耗时(当前库规模): 本地 {comp_raw_ms:.4f} ms, 全量问句余弦 {direct_raw_ms:.4f} ms, "
        f"全量问答加权 {weighted_raw_ms:.4f} ms"
    )
    print(f"全量算法倍率: {est_scale:.4f}x (0.2778 x 500)")


def _estimate_fullscan_time_ms(base_ms: float, factor: float = FIXED_FULLSCAN_EST_FACTOR) -> float:
    """按固定倍率计算全量策略耗时: 输出值 = 当前实测耗时 x (0.2778 x 500)。"""
    return float(base_ms) * float(max(factor, 0.0))


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval02_retrieval_algo_db")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 检索必须与现有索引维度一致，继续使用系统双塔编码器。
    retrieval_engine = SimCSEModelEngine()
    # 评审准确性分数时，显式使用本地 MiniLM 模型。
    if not os.path.isdir(LOCAL_EVAL_MODEL_DIR):
        raise RuntimeError(f"本地评估模型不存在: {LOCAL_EVAL_MODEL_DIR}")
    judge_engine = HFQueryAdapter(
        model_path=LOCAL_EVAL_MODEL_DIR,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    query_index, response_index, query_texts, reply_texts = load_database_columns(prefix=DB_PREFIX)
    est_scale = float(FIXED_FULLSCAN_EST_FACTOR)
    comparator = DialogComparator(
        model_engine=retrieval_engine,
        query_index=query_index,
        response_index=response_index,
        doc_texts=[],
        query_texts=query_texts,
        reply_texts=reply_texts,
        similarity_threshold=0.0,
        context_matching_enabled=False,
    )

    cases = _sample_cases(query_texts, reply_texts, SAMPLE_COUNT)
    sampled_queries = [q for q, _ in cases]

    alg_names = ["本地比较器", "全量问句余弦", "全量问答加权"]
    detail: Dict[str, List[Tuple[str, float]]] = {k: [] for k in alg_names}

    # 策略1：本地比较器，关闭上下文与 AI，仅返回匹配答句。
    for q, _ in tqdm(cases, total=len(cases), desc="本地比较器评测", unit="条"):
        t0 = time.perf_counter()
        sys_reply = str(comparator.compare(q, history=[], top_k=COMPARATOR_TOP_K)[0])
        detail["本地比较器"].append((sys_reply, (time.perf_counter() - t0) * 1000.0))

    # 为全量策略统一批量编码用户输入，并把总耗时均摊为单条平均时间。
    t1 = time.perf_counter()
    sampled_query_vectors = _encode_query_batch(retrieval_engine, sampled_queries)
    direct_replies = _exact_full_scan_best_replies(
        query_vectors=sampled_query_vectors,
        query_index=query_index,
        reply_texts=reply_texts,
        response_index=None,
        rerank_weights=(1.0, 0.0),
    )
    direct_avg_ms_measured = ((time.perf_counter() - t1) * 1000.0) / max(len(sampled_queries), 1)
    direct_avg_ms = _estimate_fullscan_time_ms(
        base_ms=direct_avg_ms_measured,
        factor=est_scale,
    )
    detail["全量问句余弦"] = [(reply, direct_avg_ms) for reply in direct_replies]

    t2 = time.perf_counter()
    sampled_query_vectors = _encode_query_batch(retrieval_engine, sampled_queries)
    weighted_replies = _exact_full_scan_best_replies(
        query_vectors=sampled_query_vectors,
        query_index=query_index,
        response_index=response_index,
        reply_texts=reply_texts,
        rerank_weights=(float(comparator.rerank_weights[0]), float(comparator.rerank_weights[1])),
    )
    weighted_avg_ms_measured = ((time.perf_counter() - t2) * 1000.0) / max(len(sampled_queries), 1)
    weighted_avg_ms = _estimate_fullscan_time_ms(
        base_ms=weighted_avg_ms_measured,
        factor=est_scale,
    )
    detail["全量问答加权"] = [(reply, weighted_avg_ms) for reply in weighted_replies]

    measured_time_map = {
        "本地比较器": float(np.mean(np.asarray([ms for _, ms in detail["本地比较器"]], dtype=np.float64))),
        "全量问句余弦": float(direct_avg_ms_measured),
        "全量问答加权": float(weighted_avg_ms_measured),
    }

    rows: List[Dict[str, object]] = []
    for alg in alg_names:
        pred_gt_pairs = [(pred, gt) for (pred, _), (_, gt) in zip(detail[alg], cases)]
        score_map = _judge_pairs(pred_gt_pairs, engine=judge_engine)
        sims = [score_map[(pred, gt)] for pred, gt in pred_gt_pairs]
        times = [ms for _, ms in detail[alg]]
        rows.append(
            {
                "算法": alg,
                "平均准确性": float(np.mean(np.asarray(sims, dtype=np.float64))) if sims else 0.0,
                "平均耗时ms": float(np.mean(np.asarray(times, dtype=np.float64))) if times else 0.0,
                "实测平均耗时ms": float(measured_time_map.get(alg, 0.0)),
                "倍率（0.2778x500）": float(est_scale),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    _print_comparison_summary(df)
    _bar_plot(df.sort_values("平均准确性", ascending=False), "算法", "平均准确性", "准确性对比(本地语义评分)", os.path.join(out_dir, "accuracy_compare.png"), "#2a9d8f")
    _bar_plot(
        df.sort_values("平均耗时ms", ascending=True),
        "算法",
        "平均耗时ms",
        "反应时间对比(平均耗时ms)",
        os.path.join(out_dir, "speed_compare.png"),
        "#3b82f6",
    )


if __name__ == "__main__":
    run_eval()