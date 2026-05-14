"""
实验二：不同语料规模下三种检索策略的时间与精度对比

评测目标:
    在真实语料规模（1万～50万）下，对比三种检索算法的实际耗时与融合相似度，
    量化 FAISS HNSW 索引相对于全量暴力搜索的效率优势与精度代价。

三种检索策略:
    算法A (全量问句余弦):
        查询向量与语料库全部 N 条问句向量逐一计算余弦相似度，
        取最高分对应答句。复杂度 O(N*d)。不使用答句信息。
    算法B (全量问答加权):
        在算法A的问句全量搜索基础上，对语料库中全部 N 条答句预编码，
        逐条计算问答余弦相似度，按 0.75:0.25 加权融合排序取最优。
        每查询计算量为算法A的两倍（两次矩阵乘法），复杂度 O(N*d)。
    算法C (FAISS HNSW 粗召回 + 加权重排):
        FAISS HNSW 粗召回 K_c=50 条 → 精确余弦重排取 K_r=5 条
        → 答句实时编码 → 加权融合。粗召回 O(log N)，精排/编码 O(1)。

语料规模: 10,000 / 50,000 / 100,000 / 500,000 条 (实测)
测试用例: eval_pairs.csv 前 100 条
计时: 每用例 3 次重复取均值

输出:
    - eval02_time_table.csv
    - eval02_similarity_table.csv
    - time_curves.png
    - similarity_bars.png
"""

import os, sys, time, math
from typing import Any, Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.config import DB_CSV_PATH, DB_QUERY_INDEX_FILE
from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_RESULTS_DIR

# ============================================================
# 参数
# ============================================================
SCALES = [10_000, 50_000, 100_000, 500_000]
EVAL_PAIR_LIMIT = 100
BATCH_SIZE = 64
TIMING_REPEATS = 3
EPS = 1e-8

COARSE_RECALL = 50
RERANK_TOP_K = 5
HNSW_SEARCH_K = 200
QUERY_W = 0.75
REPLY_W = 0.25

BGE_MODEL_DIR = os.path.join(CURRENT_DIR, "models", "bge-small-zh-v1.5")

# ============================================================
# 数据加载
# ============================================================
def _load_eval_pairs(path: str, limit: int) -> Tuple[List[str], List[str]]:
    queries, expected = [], []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=1)):
        if i >= limit: break
        row = chunk.iloc[0]
        queries.append(str(row["query"]))
        expected.append(str(row["response"]))
    return queries, expected


def _load_corpus(n: int) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(DB_CSV_PATH, nrows=n)
    return df["query"].fillna("").astype(str).tolist(), df["response"].fillna("").astype(str).tolist()


def _load_index_vectors(full_index: Any, n: int) -> np.ndarray:
    total = int(getattr(full_index, "ntotal", 0))
    n = min(n, total)
    if hasattr(full_index, "reconstruct_n"):
        return np.asarray(full_index.reconstruct_n(0, n), dtype=np.float32)
    return np.asarray([full_index.reconstruct(i) for i in range(n)], dtype=np.float32)


def _build_hnsw(vecs: np.ndarray, ref_index: Any) -> Any:
    n, d = vecs.shape
    if hasattr(ref_index, "hnsw"):
        m = int(ref_index.hnsw.nb_neighbors(1))
        ef = int(ref_index.hnsw.efSearch)
    else:
        m, ef = 32, 16
    idx = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efSearch = ef
    idx.add(vecs)
    return idx


# ============================================================
# BGE 评分
# ============================================================
def _encode_for_bge(texts: List[str], model, tokenizer, device: str,
                     batch_size: int = 64, max_len: int = 512) -> np.ndarray:
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", max_length=max_len).to(device)
            out = model(**tokens)
            emb = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) \
                  else out.last_hidden_state[:, 0, :]
            v = emb.detach().cpu().numpy().astype(np.float32)
            nrm = np.linalg.norm(v, axis=1, keepdims=True) + EPS
            results.append(v / nrm)
    return np.vstack(results) if results else np.empty((0, 0))


def _bge_batch(expected: List[str], actual_a: List[str], actual_b: List[str],
               actual_c: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BGE_MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(BGE_MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    all_t = expected + actual_a + actual_b + actual_c
    embs = _encode_for_bge(all_t, model, tok, device)
    n = len(expected)
    sims_a = np.clip(np.sum(embs[:n] * embs[n:2*n], axis=1), -1.0, 1.0)
    sims_b = np.clip(np.sum(embs[:n] * embs[2*n:3*n], axis=1), -1.0, 1.0)
    sims_c = np.clip(np.sum(embs[:n] * embs[3*n:], axis=1), -1.0, 1.0)
    del model, tok
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return sims_a, sims_b, sims_c


# ============================================================
# 三种算法
# ============================================================
def _algo_a(q_vec: np.ndarray, corpus_q: np.ndarray, corpus_q_norm: np.ndarray,
            corpus_replies: List[str], corpus_queries: List[str], raw_q: str) -> Tuple[int, float]:
    """A: 全量问句余弦 O(N*d)。"""
    qn = float(np.linalg.norm(q_vec)) + EPS
    scores = (corpus_q @ q_vec) / (corpus_q_norm * qn)
    masked = scores.copy()
    for i, t in enumerate(corpus_queries):
        if t == raw_q: masked[i] = -1e9
    best = int(np.argmax(masked))
    if masked[best] <= -1e8: best = int(np.argmax(scores))
    return best, float(scores[best])


def _algo_b(q_vec: np.ndarray, corpus_q: np.ndarray, corpus_q_norm: np.ndarray,
            corpus_r: np.ndarray, corpus_r_norm: np.ndarray,
            corpus_replies: List[str], corpus_queries: List[str], raw_q: str) -> Tuple[int, float, float]:
    """B: 全量问答加权 O(N*d) —— 对每条答句编码后加权排序。
    corpus_r 为预编码的全部答句向量, corpus_r_norm 为其 L2 范数。"""
    qn = float(np.linalg.norm(q_vec)) + EPS
    q_sims = (corpus_q @ q_vec) / (corpus_q_norm * qn)
    r_sims = (corpus_r @ q_vec) / (corpus_r_norm * qn)
    weighted = QUERY_W * q_sims + REPLY_W * r_sims
    masked = weighted.copy()
    for i, t in enumerate(corpus_queries):
        if t == raw_q: masked[i] = -1e9
    best = int(np.argmax(masked))
    if masked[best] <= -1e8: best = int(np.argmax(weighted))
    return best, float(weighted[best]), float(q_sims[best])


def _algo_c(q_vec: np.ndarray, hnsw_idx: Any,
            corpus_q: np.ndarray, corpus_q_norm: np.ndarray,
            corpus_replies: List[str], corpus_queries: List[str],
            raw_q: str, engine: SimCSEModelEngine, n_total: int) -> Tuple[int, float, float, float, float]:
    """C: FAISS HNSW + 加权重排 O(log N) + O(1)。
    返回 (best_idx, best_score, coarse_ms, rerank_ms, reply_ms)。"""
    t0 = time.perf_counter()
    sk = min(HNSW_SEARCH_K, n_total)
    _, raw = hnsw_idx.search(q_vec.reshape(1, -1), sk)
    cand_ids = []
    for cid in raw[0]:
        cid = int(cid)
        if cid < 0 or cid >= len(corpus_queries): continue
        if corpus_queries[cid] == raw_q: continue
        cand_ids.append(cid)
        if len(cand_ids) >= COARSE_RECALL: break
    if not cand_ids:
        cand_ids = [int(np.argmax(corpus_q @ q_vec))]
    coarse_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    qn = float(np.linalg.norm(q_vec)) + EPS
    cv = corpus_q[cand_ids]; cn = corpus_q_norm[cand_ids]
    cs = (cv @ q_vec) / (cn * qn)
    topk = min(RERANK_TOP_K, len(cand_ids))
    top_loc = np.argsort(-cs)[:topk]
    top_ids = [cand_ids[int(i)] for i in top_loc]
    top_qs = cs[top_loc]
    rerank_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    top_r = [corpus_replies[i] for i in top_ids]
    top_rv = np.asarray(engine.encode(top_r, encoder="response",
                       batch_size=min(BATCH_SIZE, topk),
                       show_progress=False, return_numpy=True), dtype=np.float32)
    rn = np.linalg.norm(top_rv, axis=1) + EPS
    rs = (top_rv @ q_vec) / (rn * qn)
    w = QUERY_W * top_qs + REPLY_W * rs
    best_loc = int(np.argmax(w))
    reply_ms = (time.perf_counter() - t0) * 1000.0

    return top_ids[best_loc], float(w[best_loc]), coarse_ms, rerank_ms, reply_ms


# ============================================================
# 图表
# ============================================================
def _setup_plt():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_time_curves(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    scales = df["规模"].values
    ax.plot(scales, df["算法A_ms"].values, "o-", color="#ef4444",
            label="A: 全量问句余弦  O(N*d)", lw=2.5, markersize=8)
    ax.plot(scales, df["算法B_ms"].values, "s--", color="#f59e0b",
            label="B: 全量问答加权  O(N*d)", lw=2, markersize=8)
    ax.plot(scales, df["算法C_ms"].values, "D-", color="#2a9d8f",
            label="C: FAISS HNSW 精排  O(log N)", lw=2.5, markersize=8)
    ax.set_xlabel("语料库规模 (条)", fontsize=12)
    ax.set_ylabel("平均处理时间 (ms/条)", fontsize=12)
    ax.set_title("三种检索策略实测耗时曲线", fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([f"{int(s):,}" for s in scales])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sim_bars(stats: pd.DataFrame, out_path: str):
    scales = stats["规模"].unique().tolist()
    algos = ["算法A", "算法B", "算法C"]
    colors = ["#ef4444", "#f59e0b", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(scales))
    w = 0.22
    for j, (algo, c) in enumerate(zip(algos, colors)):
        sub = stats[stats["算法"] == algo]
        means = [sub[sub["规模"] == s]["均值"].values[0] for s in scales]
        bars = ax.bar(x + (j-1)*w, means, w, label=algo, color=c, edgecolor="#1f2a37")
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, float(v) + 0.004,
                    f"{float(v):.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s):,}" for s in scales])
    ax.set_ylabel("融合相似度", fontsize=11)
    ax.set_title("不同语料规模下三种策略融合相似度对比", fontsize=14)
    ax.set_ylim(0.0, 0.9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主评测
# ============================================================
def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval02_retrieval_algo_db")
    os.makedirs(out_dir, exist_ok=True)
    _setup_plt()

    print("=" * 60)
    print("实验二: 不同语料规模下三种检索策略的时间与精度对比")
    print(f"规模: {SCALES}  |  测试用例: {EVAL_PAIR_LIMIT} 条")
    print("=" * 60)

    eval_q, eval_exp = _load_eval_pairs(
        os.path.join(TEST_DATA_DIR, "eval_pairs.csv"), EVAL_PAIR_LIMIT)
    engine = SimCSEModelEngine()
    full_idx = faiss.read_index(DB_QUERY_INDEX_FILE)

    eval_qv = np.asarray(
        engine.encode(eval_q, encoder="query", batch_size=BATCH_SIZE,
                      show_progress=False, return_numpy=True), dtype=np.float32)

    time_rows: List[Dict] = []
    sim_rows: List[Dict] = []

    for scale in SCALES:
        print(f"\n{'='*40}\n规模: {scale:,} 条\n{'='*40}")

        cq, cr = _load_corpus(scale)
        cv_q = _load_index_vectors(full_idx, scale)
        cv_q_norm = np.linalg.norm(cv_q, axis=1) + EPS
        hnsw = _build_hnsw(cv_q, full_idx)
        actual_n = len(cq)

        # --- 算法B: 预编码全部答句 (一次性, 批处理) ---
        print("  预编码全部答句 (算法B) ...", end=" ", flush=True)
        t_enc0 = time.perf_counter()
        cv_r = np.asarray(
            engine.encode(cr, encoder="response", batch_size=BATCH_SIZE,
                          show_progress=False, return_numpy=True), dtype=np.float32)
        cv_r_norm = np.linalg.norm(cv_r, axis=1) + EPS
        b_setup_sec = time.perf_counter() - t_enc0
        print(f"{b_setup_sec:.1f}s  (一次性, 不计入每条查询耗时)")

        # --- 逐用例计时 ---
        t_a, t_b, t_c = 0.0, 0.0, 0.0
        t_c_coarse, t_c_rerank, t_c_reply = 0.0, 0.0, 0.0
        replies_a, replies_b, replies_c = [], [], []
        q_sims_a, q_sims_b = [], []

        for q_text, qv_i in zip(eval_q, eval_qv):
            qv_f32 = np.asarray(qv_i, dtype=np.float32)

            # A: 全量问句余弦
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                idx_a, _ = _algo_a(qv_f32, cv_q, cv_q_norm, cr, cq, q_text)
                t_a += (time.perf_counter() - t0) * 1000.0
            replies_a.append(cr[idx_a])

            # B: 全量问答加权 (答句向量已预编码)
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                idx_b, _, qsb = _algo_b(qv_f32, cv_q, cv_q_norm, cv_r, cv_r_norm, cr, cq, q_text)
                t_b += (time.perf_counter() - t0) * 1000.0
            replies_b.append(cr[idx_b])

            # C: FAISS + 重排
            for _ in range(TIMING_REPEATS):
                idx_c, _, coarse_ms, rerank_ms, reply_ms = _algo_c(
                    qv_f32, hnsw, cv_q, cv_q_norm, cr, cq, q_text, engine, actual_n)
                t_c += coarse_ms + rerank_ms + reply_ms
                t_c_coarse += coarse_ms; t_c_rerank += rerank_ms; t_c_reply += reply_ms
            replies_c.append(cr[idx_c])

        nc = len(eval_q); rp = TIMING_REPEATS
        t_a /= (nc * rp); t_b /= (nc * rp); t_c /= (nc * rp)

        # --- BGE 评分 ---
        sims_a, sims_b, sims_c = _bge_batch(eval_exp, replies_a, replies_b, replies_c)

        # 融合相似度: 用问句余弦 (来自算法A/B/C各自) + BGE 答句相似度
        for algo, replies, sims_arr, tag in [
            ("算法A", replies_a, sims_a, "A"),
            ("算法B", replies_b, sims_b, "B"),
            ("算法C", replies_c, sims_c, "C"),
        ]:
            # 重新获取每种算法每条用例的问句相似度 (不做额外计时)
            qs_vals = []
            for q_text, qv_i in zip(eval_q, eval_qv):
                qv_f32 = np.asarray(qv_i, dtype=np.float32)
                if tag == "A":
                    _, qs = _algo_a(qv_f32, cv_q, cv_q_norm, cr, cq, q_text)
                elif tag == "B":
                    _, _, qs = _algo_b(qv_f32, cv_q, cv_q_norm, cv_r, cv_r_norm, cr, cq, q_text)
                else:
                    idx, _, _, _, _ = _algo_c(qv_f32, hnsw, cv_q, cv_q_norm, cr, cq, q_text, engine, actual_n)
                    qn = float(np.linalg.norm(qv_f32)) + EPS
                    qs = float(np.dot(cv_q[idx], qv_f32) / (cv_q_norm[idx] * qn))
                qs_vals.append(qs)
            fusion = QUERY_W * np.asarray(qs_vals) + REPLY_W * sims_arr
            sim_rows.append({
                "规模": scale, "算法": algo,
                "均值": float(np.mean(fusion)), "中位数": float(np.median(fusion)),
                "标准差": float(np.std(fusion)),
            })

        time_rows.append({
            "规模": scale,
            "算法A_ms": round(t_a, 4), "算法B_ms": round(t_b, 4),
            "算法C_ms": round(t_c, 4),
            "C_粗召回ms": round(t_c_coarse / (nc * rp), 4),
            "C_精排ms": round(t_c_rerank / (nc * rp), 4),
            "C_编码ms": round(t_c_reply / (nc * rp), 4),
            "B_答句预编码s": round(b_setup_sec, 1),
        })

        print(f"  A: {t_a:.3f} ms  |  B: {t_b:.3f} ms  |  C: {t_c:.3f} ms")
        print(f"  C: 粗召回 {t_c_coarse/(nc*rp):.3f} | 精排 {t_c_rerank/(nc*rp):.3f} | 编码 {t_c_reply/(nc*rp):.3f}")
        print(f"  融合相似度: A={sim_rows[-3]['均值']:.4f}  B={sim_rows[-2]['均值']:.4f}  C={sim_rows[-1]['均值']:.4f}")

    # ============================================================
    # 输出
    # ============================================================
    time_df = pd.DataFrame(time_rows)
    sim_df = pd.DataFrame(sim_rows)

    time_df.to_csv(os.path.join(out_dir, "eval02_time_table.csv"), index=False, encoding="utf-8-sig")
    sim_df.to_csv(os.path.join(out_dir, "eval02_similarity_table.csv"), index=False, encoding="utf-8-sig")

    print("\n===== 耗时表 (ms/条) =====")
    print(time_df.to_string(index=False))
    print("\n===== 融合相似度 =====")
    print(sim_df.pivot(index="规模", columns="算法", values="均值").round(4).to_string())

    last = time_df.iloc[-1]
    print(f"\n500K 规模加速比: A/C = {last['算法A_ms']/max(last['算法C_ms'], EPS):.1f}x")
    print(f"500K 规模 B vs A: B/A = {last['算法B_ms']/max(last['算法A_ms'], EPS):.2f}x")

    _plot_time_curves(time_df, os.path.join(out_dir, "time_curves.png"))
    _plot_sim_bars(sim_df, os.path.join(out_dir, "similarity_bars.png"))

    print(f"\n输出目录: {out_dir}")
    print("完成。")


if __name__ == "__main__":
    run_eval()
