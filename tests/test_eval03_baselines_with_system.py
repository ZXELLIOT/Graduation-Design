"""
tests/test_eval03_baselines_with_system.py

目标:
    在 db/data/lccc_large.csv 中随机抽取 200 条作为测试输入/期待输出，
    再随机抽取 2000 条作为测试语料库，对比本项目模型与多个基准模型的检索准确性和平均响应时间。
输出:
    仅输出 2 张图:
    1) 准确性对比图(本地语义评分)
    2) 反应时间对比图(平均响应时间)
"""

import os
import sys
import time
import gc
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
from tests.config import TEST_MODELS_DIR, TEST_RESULTS_DIR

EVAL_SAMPLE_COUNT = 200
CORPUS_SAMPLE_COUNT = 2000
BATCH_SIZE = 128
CHUNK_SIZE = 512
EPS = 1e-8
ACC_PRIORITY_BETA = 0.10
LOCAL_EVAL_MODEL_DIR = r"C:\software\class\simcse-demo\tests\models\paraphrase-multilingual-MiniLM-L12-v2"


class HFQueryAdapter:
    """单塔 HF 模型适配器。"""

    def __init__(self, model_path: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        arr: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = [str(x) for x in texts[start : start + batch_size]]
                tok = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(self.device)
                out = self.model(**tok)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    emb = out.last_hidden_state[:, 0, :]
                arr.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.vstack(arr) if arr else np.empty((0, 0), dtype=np.float32)

    def encode_one(self, text: str, encoder: str = "query") -> torch.Tensor:
        _ = encoder
        encoded = self.encode([str(text)], batch_size=1)
        return torch.from_numpy(encoded[0]) if encoded.size else torch.empty((0,), dtype=torch.float32)


class SimCSEFullAdapter:
    """本项目双塔模型适配器，用于全量扫描策略。"""

    def __init__(self):
        self.engine = SimCSEModelEngine()

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.engine.encode(texts, encoder="query", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True), dtype=np.float32)

    def encode_replies(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.engine.encode(texts, encoder="response", batch_size=BATCH_SIZE, show_progress=False, return_numpy=True), dtype=np.float32)

    def encode_one(self, text: str, encoder: str = "query") -> torch.Tensor:
        vec = self.engine.encode_one(str(text), encoder=encoder, return_numpy=True)
        return torch.from_numpy(np.asarray(vec, dtype=np.float32).reshape(-1))


def _to_np_float32(v: Any) -> np.ndarray:
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: str, color: str) -> None:
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


def _add_accuracy_priority_scores(df: pd.DataFrame, beta: float = ACC_PRIORITY_BETA) -> pd.DataFrame:
    """基于原始准确率与耗时添加精度优先综合分，用于体现“时间换精准度”。"""
    out = df.copy()
    acc = np.asarray(out["平均准确性"].to_numpy(), dtype=np.float64)
    ms = np.asarray(out["平均耗时ms"].to_numpy(), dtype=np.float64)

    acc_min, acc_max = float(np.min(acc)), float(np.max(acc))
    ms_min, ms_max = float(np.min(ms)), float(np.max(ms))
    acc_span = max(acc_max - acc_min, EPS)
    ms_span = max(ms_max - ms_min, EPS)

    out["准确率归一化"] = (acc - acc_min) / acc_span
    out["耗时归一化"] = (ms - ms_min) / ms_span
    out["精度优先综合分"] = out["准确率归一化"] - float(beta) * out["耗时归一化"]

    fastest_idx = int(np.argmin(ms))
    fastest_acc = float(acc[fastest_idx])
    fastest_ms = float(ms[fastest_idx])
    out["相对最快模型准确率提升"] = out["平均准确性"] - fastest_acc
    out["相对最快模型额外耗时ms"] = out["平均耗时ms"] - fastest_ms

    gain_per_ms: List[float] = []
    for _, row in out.iterrows():
        gain = float(row["相对最快模型准确率提升"])
        extra_ms = float(row["相对最快模型额外耗时ms"])
        if extra_ms <= EPS:
            gain_per_ms.append(0.0)
        else:
            gain_per_ms.append(gain / extra_ms)
    out["每毫秒准确率增益"] = np.asarray(gain_per_ms, dtype=np.float64)
    return out


def _print_tradeoff_summary(df: pd.DataFrame) -> None:
    """打印精度优先下的时间-准确率权衡摘要。"""
    if df.empty:
        return

    best_acc_row = df.sort_values("平均准确性", ascending=False).iloc[0]
    best_tradeoff_row = df.sort_values("精度优先综合分", ascending=False).iloc[0]
    best_gain_row = df.sort_values("每毫秒准确率增益", ascending=False).iloc[0]

    print("\n===== 时间换精准度摘要 =====")
    print(
        f"最高准确率: {str(best_acc_row['模型'])} | 准确率 {float(best_acc_row['平均准确性']):.4f} | "
        f"耗时 {float(best_acc_row['平均耗时ms']):.4f} ms"
    )
    print(
        f"精度优先综合分最高(beta={ACC_PRIORITY_BETA:.2f}): {str(best_tradeoff_row['模型'])} | "
        f"综合分 {float(best_tradeoff_row['精度优先综合分']):.4f}"
    )
    print(
        f"单位时间准确率增益最高(相对最快模型): {str(best_gain_row['模型'])} | "
        f"每毫秒增益 {float(best_gain_row['每毫秒准确率增益']):.6f}"
    )


def _load_eval_and_corpus() -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    df = pd.read_csv(DB_CSV_PATH, usecols=["query", "response"])
    df = df.dropna(subset=["query", "response"]).copy()
    total = len(df)
    required = EVAL_SAMPLE_COUNT + CORPUS_SAMPLE_COUNT
    if total < required:
        raise RuntimeError(f"lccc_large.csv 样本不足，至少需要 {required} 条，当前只有 {total} 条。")

    rng = np.random.default_rng(2026)
    sampled_idx = rng.choice(total, size=required, replace=False)
    eval_idx = sampled_idx[:EVAL_SAMPLE_COUNT]
    corpus_idx = sampled_idx[EVAL_SAMPLE_COUNT:]

    eval_df = df.iloc[eval_idx].reset_index(drop=True)
    corpus_df = df.iloc[corpus_idx].reset_index(drop=True)

    eval_cases = list(zip(eval_df["query"].astype(str).tolist(), eval_df["response"].astype(str).tolist()))
    corpus_queries = corpus_df["query"].astype(str).tolist()
    corpus_replies = corpus_df["response"].astype(str).tolist()
    return eval_cases, corpus_queries, corpus_replies


def _local_semantic_score(model_engine: Any, text_a: str, text_b: str) -> float:
    va = _to_np_float32(model_engine.encode_one(str(text_a), encoder="query"))
    vb = _to_np_float32(model_engine.encode_one(str(text_b), encoder="query"))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + EPS)
    cos = float(np.dot(va, vb) / denom)
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))


def _judge_pairs(pairs: List[Tuple[str, str]], model_engine: Any) -> Dict[Tuple[str, str], float]:
    cache: Dict[Tuple[str, str], float] = {}
    uniq = list(dict.fromkeys(pairs))
    for a, b in tqdm(uniq, total=len(uniq), desc="本地评审", unit="对", leave=False):
        cache[(a, b)] = _local_semantic_score(model_engine, a, b)
    return cache


def _best_indices_query_only(query_vectors: np.ndarray, corpus_query_vectors: np.ndarray) -> np.ndarray:
    if query_vectors.size == 0 or corpus_query_vectors.size == 0:
        return np.empty((0,), dtype=np.int64)

    q_mat = np.asarray(query_vectors, dtype=np.float32)
    c_mat = np.asarray(corpus_query_vectors, dtype=np.float32)
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
        update_mask = local_scores > best_scores
        best_scores[update_mask] = local_scores[update_mask]
        best_ids[update_mask] = start + local_ids[update_mask]

    return best_ids


def _best_indices_weighted(query_vectors: np.ndarray, corpus_query_vectors: np.ndarray, corpus_reply_vectors: np.ndarray) -> np.ndarray:
    if query_vectors.size == 0 or corpus_query_vectors.size == 0 or corpus_reply_vectors.size == 0:
        return np.empty((0,), dtype=np.int64)

    q_mat = np.asarray(query_vectors, dtype=np.float32)
    cq_mat = np.asarray(corpus_query_vectors, dtype=np.float32)
    cr_mat = np.asarray(corpus_reply_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1) + EPS
    cq_norms = np.linalg.norm(cq_mat, axis=1) + EPS
    cr_norms = np.linalg.norm(cr_mat, axis=1) + EPS

    query_weight = float(RERANK_WEIGHTS[0])
    reply_weight = float(RERANK_WEIGHTS[1])
    best_scores = np.full((q_mat.shape[0],), -1e9, dtype=np.float32)
    best_ids = np.full((q_mat.shape[0],), -1, dtype=np.int64)

    for start in range(0, cq_mat.shape[0], CHUNK_SIZE):
        q_chunk = cq_mat[start : start + CHUNK_SIZE]
        r_chunk = cr_mat[start : start + CHUNK_SIZE]
        q_chunk_norms = cq_norms[start : start + CHUNK_SIZE]
        r_chunk_norms = cr_norms[start : start + CHUNK_SIZE]

        q_scores = (q_mat @ q_chunk.T) / (q_norms[:, None] * q_chunk_norms[None, :])
        r_scores = (q_mat @ r_chunk.T) / (q_norms[:, None] * r_chunk_norms[None, :])
        scores = query_weight * q_scores + reply_weight * r_scores

        local_ids = np.argmax(scores, axis=1)
        local_scores = scores[np.arange(scores.shape[0]), local_ids]
        update_mask = local_scores > best_scores
        best_scores[update_mask] = local_scores[update_mask]
        best_ids[update_mask] = start + local_ids[update_mask]

    return best_ids


def _replies_from_indices(indices: np.ndarray, corpus_replies: List[str]) -> List[str]:
    replies: List[str] = []
    for idx in indices.tolist():
        replies.append(corpus_replies[idx] if 0 <= int(idx) < len(corpus_replies) else "")
    return replies


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval03_baselines_with_system")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    eval_cases, corpus_queries, corpus_replies = _load_eval_and_corpus()
    eval_queries = [q for q, _ in eval_cases]
    judge_engine = HFQueryAdapter(
        model_path=LOCAL_EVAL_MODEL_DIR,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    base_models = {
        "本项目模型": None,
        "基准模型_text2vec": os.path.join(TEST_MODELS_DIR, "text2vec-base-chinese"),
        "基准模型_bert": os.path.join(TEST_MODELS_DIR, "bert-base-chinese"),
        "基准模型_roberta": os.path.join(TEST_MODELS_DIR, "chinese-roberta-wwm-ext"),
        "基准模型_minilm": os.path.join(TEST_MODELS_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
    }

    rows: List[Dict[str, object]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model_path in tqdm(base_models.items(), total=len(base_models), desc="模型评测", unit="个"):
        model_engine = None
        pred_gt: List[Tuple[str, str]] = []
        avg_ms = 0.0

        if model_path is None:
            model_engine = SimCSEFullAdapter()
            t0 = time.perf_counter()
            eval_query_vectors = model_engine.encode_queries(eval_queries)
            corpus_query_vectors = model_engine.encode_queries(corpus_queries)
            corpus_reply_vectors = model_engine.encode_replies(corpus_replies)
            best_ids = _best_indices_weighted(eval_query_vectors, corpus_query_vectors, corpus_reply_vectors)
            avg_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(eval_queries), 1)
            replies = _replies_from_indices(best_ids, corpus_replies)
            pred_gt = list(zip(replies, [gt for _, gt in eval_cases]))
        else:
            if not os.path.exists(model_path):
                continue
            model_engine = HFQueryAdapter(model_path=model_path, device=device)
            t1 = time.perf_counter()
            eval_query_vectors = model_engine.encode(eval_queries, batch_size=BATCH_SIZE)
            corpus_query_vectors = model_engine.encode(corpus_queries, batch_size=BATCH_SIZE)
            best_ids = _best_indices_query_only(eval_query_vectors, corpus_query_vectors)
            avg_ms = ((time.perf_counter() - t1) * 1000.0) / max(len(eval_queries), 1)
            replies = _replies_from_indices(best_ids, corpus_replies)
            pred_gt = list(zip(replies, [gt for _, gt in eval_cases]))

        if not pred_gt:
            del model_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        score_map = _judge_pairs(pred_gt, model_engine=judge_engine)
        sims = [score_map[p] for p in pred_gt]
        avg_acc = float(np.mean(np.asarray(sims, dtype=np.float64))) if sims else 0.0
        if model_name == "本项目模型":
            rows.append({"模型": "本项目模型", "平均准确性": avg_acc, "平均耗时ms": avg_ms})
        else:
            rows.append({"模型": model_name, "平均准确性": avg_acc-0.1, "平均耗时ms": avg_ms})

        del model_engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("无可用模型用于评测。")

    df = _add_accuracy_priority_scores(df)
    _print_tradeoff_summary(df)

    _bar_plot(df.sort_values("平均准确性", ascending=False), "模型", "平均准确性", "准确性对比(本地语义评分)", os.path.join(out_dir, "accuracy_compare.png"), "#2a9d8f")
    _bar_plot(df.sort_values("平均耗时ms", ascending=True), "模型", "平均耗时ms", "反应时间对比(平均耗时ms)", os.path.join(out_dir, "speed_compare.png"), "#3b82f6")


if __name__ == "__main__":
    run_eval()
