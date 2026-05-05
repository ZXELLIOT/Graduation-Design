"""
tests/test_eval03_baselines_with_system.py

评测目标:
    在相同测试数据与语料库上，对比 4 个基准模型与本地模型（仅问句编码器）的检索效果。

评测流程:
    1) 先加载 4 个基准模型编码器。
    2) 对前 10000 条答句语料进行预编码（直接抛弃问句列）。
    3) eval_pairs 按行读取；每条测试问句分别用各模型编码，做全量相似度检索，取最佳输出。
    4) 记录单句平均耗时与用例明细。
    5) 使用 bge-small-zh-v1.5 计算“预期答句 vs 实际输出”语义相似度。

输出:
    - eval03_cases_table.csv:         模型名称/预期输入/预期输出/实际输出/答句相似度
    - eval03_time_table.csv:          各模型平均每句耗时(ms)
    - eval03_similarity_table.csv:    各模型平均答句相似度
    - speed_compare.png:              时间对比图
    - similarity_compare.png:         相似度对比图
"""

import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.config import DB_CSV_PATH
from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================
EVAL_PAIRS_CSV = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")
CORPUS_SIZE = 10000
EVAL_PAIR_COUNT = 200
BATCH_SIZE = 64
MAX_LEN = 128
EPS = 1e-8

BASELINES = {
    "基准_bge-small": "bge-small-zh-v1.5",
    "基准_text2vec": "text2vec-base-chinese",
    "基准_bert": "bert-base-chinese",
    "基准_minilm": "paraphrase-multilingual-MiniLM-L12-v2",
}

LOCAL_MODEL_NAME = "本地模型_问句编码器"
SIM_SCORER_MODEL = "bge-small-zh-v1.5"


def _safe_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if bool(pd.isna(cast(Any, x))):
            return ""
    except Exception:
        pass
    return str(x)


def _load_corpus_replies(n_rows: int) -> List[str]:
    """加载前 n_rows 条答句语料。"""
    df = pd.read_csv(DB_CSV_PATH, usecols=["response"], nrows=n_rows)
    return df["response"].astype(str).tolist()


def _load_eval_pairs_line_by_line(max_count: int) -> List[Tuple[str, str]]:
    """按行读取 eval_pairs。"""
    if not os.path.exists(EVAL_PAIRS_CSV):
        raise FileNotFoundError(f"测试文件不存在: {EVAL_PAIRS_CSV}")

    pairs: List[Tuple[str, str]] = []
    for i, chunk in enumerate(pd.read_csv(EVAL_PAIRS_CSV, chunksize=1)):
        if i >= max_count:
            break
        row = chunk.iloc[0]
        pairs.append((_safe_text(row["query"]), _safe_text(row["response"])))

    if not pairs:
        raise RuntimeError("未读取到测试问答对。")
    return pairs


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """标准 mean pooling。"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_hf_batch(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: str,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """HuggingFace 模型批量编码，返回已归一化向量。"""
    arr: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(x) for x in texts[i : i + batch_size]]
            tok = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_len,
            ).to(device)
            out = model(**tok)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                emb = _mean_pool(out.last_hidden_state, tok["attention_mask"])
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                raise RuntimeError("模型输出不包含 last_hidden_state/pooler_output，无法编码。")
            emb = F.normalize(emb, p=2, dim=1)
            arr.append(emb.detach().cpu().numpy().astype(np.float32))

    if not arr:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(arr)


def _encode_simcse_query(engine: SimCSEModelEngine, texts: List[str], batch_size: int) -> np.ndarray:
    """本地模型仅使用 query 编码器，并做 L2 归一化。"""
    vec = np.asarray(
        engine.encode(texts, encoder="query", batch_size=batch_size, show_progress=False, return_numpy=True),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + EPS
    return vec / norms


def _run_single_tower_retrieval(
    model_name: str,
    eval_pairs: List[Tuple[str, str]],
    corpus_replies: List[str],
    corpus_reply_vecs: np.ndarray,
    encode_query_fn: Callable[[str], np.ndarray],
) -> Tuple[List[Dict[str, str]], float]:
    """单塔逐行检索：逐句编码 + 全量相似度。"""
    rows: List[Dict[str, str]] = []
    total_ms = 0.0

    for query, expected in eval_pairs:
        t0 = time.perf_counter()
        q_vec = encode_query_fn(query)
        scores = corpus_reply_vecs @ q_vec
        best_idx = int(np.argmax(scores))
        actual = corpus_replies[best_idx] if 0 <= best_idx < len(corpus_replies) else ""
        total_ms += (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "模型名称": model_name,
                "预期输入": query,
                "预期输出": expected,
                "实际输出": actual,
            }
        )

    avg_ms = total_ms / max(len(eval_pairs), 1)
    return rows, float(avg_ms)


def _append_reply_similarity_with_bge(cases_df: pd.DataFrame, device: str) -> pd.DataFrame:
    """用 bge-small-zh-v1.5 计算预期答句与实际输出相似度。"""
    scorer_path = os.path.join(TEST_MODELS_DIR, SIM_SCORER_MODEL)
    if not os.path.exists(scorer_path):
        raise FileNotFoundError(f"相似度模型不存在: {scorer_path}")

    tokenizer = AutoTokenizer.from_pretrained(scorer_path, local_files_only=True)
    model = AutoModel.from_pretrained(scorer_path, local_files_only=True).to(device)
    model.eval()

    expected = cases_df["预期输出"].astype(str).tolist()
    actual = cases_df["实际输出"].astype(str).tolist()
    texts = expected + actual

    embs = _encode_hf_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_len=512,
        batch_size=BATCH_SIZE,
    )

    n = len(cases_df)
    exp_emb = embs[:n]
    act_emb = embs[n:]
    sims = np.sum(exp_emb * act_emb, axis=1)
    sims = np.clip(sims, -1.0, 1.0)

    out = cases_df.copy()
    out["答句相似度"] = np.round(sims, 6)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _plot_bar(
    names: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    out_path: str,
    colors: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, values, color=colors, edgecolor="#1f2a37")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha="right")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(0.005, 0.01 * ax.get_ylim()[1]),
            f"{v:.4f}",
            ha="center",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval03_baselines_with_system")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_pairs = _load_eval_pairs_line_by_line(EVAL_PAIR_COUNT)
    corpus_replies = _load_corpus_replies(CORPUS_SIZE)
    print(f"测试问答对: {len(eval_pairs)} 条")
    print(f"语料答句数: {len(corpus_replies)} 条（前 {CORPUS_SIZE} 条）")

    all_rows: List[Dict[str, str]] = []
    model_times: Dict[str, float] = {}

    # 1) 先加载 4 个基准模型并执行逐行检索
    for model_name, folder in BASELINES.items():
        model_path = os.path.join(TEST_MODELS_DIR, folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"基准模型目录不存在: {model_path}")

        print(f"\n----- {model_name} -----")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()

        corpus_reply_vecs = _encode_hf_batch(
            model=model,
            tokenizer=tokenizer,
            texts=corpus_replies,
            device=device,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
        )

        def _hf_encode_one(text: str) -> np.ndarray:
            return _encode_hf_batch(
                model=model,
                tokenizer=tokenizer,
                texts=[text],
                device=device,
                max_len=MAX_LEN,
                batch_size=1,
            )[0]

        rows, avg_ms = _run_single_tower_retrieval(
            model_name=model_name,
            eval_pairs=eval_pairs,
            corpus_replies=corpus_replies,
            corpus_reply_vecs=corpus_reply_vecs,
            encode_query_fn=_hf_encode_one,
        )
        all_rows.extend(rows)
        model_times[model_name] = avg_ms

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2) 本地模型仅加载问句编码器，其余流程与上面一致
    print(f"\n----- {LOCAL_MODEL_NAME} -----")
    engine = SimCSEModelEngine()
    local_corpus_vecs = _encode_simcse_query(engine, corpus_replies, batch_size=BATCH_SIZE)

    def _local_encode_one(text: str) -> np.ndarray:
        return _encode_simcse_query(engine, [text], batch_size=1)[0]

    local_rows, local_avg_ms = _run_single_tower_retrieval(
        model_name=LOCAL_MODEL_NAME,
        eval_pairs=eval_pairs,
        corpus_replies=corpus_replies,
        corpus_reply_vecs=local_corpus_vecs,
        encode_query_fn=_local_encode_one,
    )
    all_rows.extend(local_rows)
    model_times[LOCAL_MODEL_NAME] = local_avg_ms

    # 3) 用 BGE 计算输出相似度
    cases_df = pd.DataFrame(all_rows)
    cases_df = _append_reply_similarity_with_bge(cases_df, device=device)
    cases_df.to_csv(
        os.path.join(out_dir, "eval03_cases_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4) 时间统计与相似度统计
    time_df = pd.DataFrame(
        {
            "模型名称": list(model_times.keys()),
            "平均每句耗时ms": list(model_times.values()),
        }
    ).sort_values("平均每句耗时ms")
    time_df.to_csv(
        os.path.join(out_dir, "eval03_time_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    sim_df = (
        cases_df.groupby("模型名称", as_index=False)
        .agg(平均答句相似度=("答句相似度", "mean"))
        .sort_values("平均答句相似度", ascending=False)
    )
    sim_df.to_csv(
        os.path.join(out_dir, "eval03_similarity_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 各模型平均每句耗时(ms) =====")
    print(time_df.to_string(index=False))
    print("\n===== 各模型平均答句相似度 =====")
    print(sim_df.to_string(index=False))

    # 5) 输出两张对比图
    speed_names = time_df["模型名称"].tolist()
    speed_vals = time_df["平均每句耗时ms"].tolist()
    speed_colors = ["#2a9d8f" if "本地模型" in n else "#3b82f6" for n in speed_names]
    _plot_bar(
        names=speed_names,
        values=speed_vals,
        ylabel="平均每句耗时（ms）",
        title="各模型检索时间对比（语料库 10000 条）",
        out_path=os.path.join(out_dir, "speed_compare.png"),
        colors=speed_colors,
    )

    sim_names = sim_df["模型名称"].tolist()
    sim_vals = sim_df["平均答句相似度"].tolist()
    sim_colors = ["#2a9d8f" if "本地模型" in n else "#f59e0b" for n in sim_names]
    _plot_bar(
        names=sim_names,
        values=sim_vals,
        ylabel="平均答句相似度",
        title="各模型预期答句-实际输出相似度对比（BGE评分）",
        out_path=os.path.join(out_dir, "similarity_compare.png"),
        colors=sim_colors,
    )

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()
