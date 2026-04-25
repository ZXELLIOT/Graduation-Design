"""
tests/test_eval01_model_corr_speed.py

目标:
    分别加载各模型的问句编码器，在 tests/data 基准数据集上做统一评测。
    每个数据集按随机种子抽样 2000 条，分别评估 0/1 分类集与 0-5 打分集。
输出:
    仅输出 3 张图:
    1) 响应速度对比图(含95%置信区间)
    2) 0/1分类数据集热力图(AUC，跨种子均值)
    3) 0-5数据集对比图(Spearman，含95%置信区间)
"""

import os
import sys
import time
import math
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

from system.model_engine import SimCSEModelEngine
from tests.config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

SAMPLE_COUNT = 2000
EPS = 1e-8
EVAL_SEEDS = [2026, 2027, 2028, 2029, 2030]
CI_Z = 1.96


def _cosine_scores(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    dot = np.sum(e1 * e2, axis=1)
    norm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + EPS
    return dot / norm


def _safe_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2:
        return 0.0
    if np.std(labels) <= EPS or np.std(preds) <= EPS:
        return 0.0
    try:
        corr = float(pd.Series(labels).corr(pd.Series(preds), method="spearman"))
        if math.isnan(corr):
            return 0.0
        return corr
    except Exception:
        return 0.0


def _safe_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """不依赖 sklearn 的 AUC 计算，仅用于二分类。"""
    if labels.size < 2 or preds.size < 2:
        return 0.0
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(preds, dtype=np.float64)
    y_bin = (y >= 0.5).astype(np.int32)
    pos = int(np.sum(y_bin == 1))
    neg = int(np.sum(y_bin == 0))
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(sorted_s[j + 1] - sorted_s[i]) <= EPS:
            j += 1
        if j > i:
            avg_rank = float((i + 1 + j + 1) / 2.0)
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[y_bin == 1]))
    auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return float(max(0.0, min(1.0, auc)))


def _semantic_judgement_score(task_type: str, labels: np.ndarray, preds: np.ndarray) -> float:
    """统一语义判断能力分到0-1区间，便于跨任务解释。"""
    if task_type == "classification":
        return _safe_auc(labels, preds)

    spearman = _safe_spearman(labels, preds)
    # Spearman范围为[-1,1]，线性映射到[0,1]后更直观。
    return float(max(0.0, min(1.0, (spearman + 1.0) / 2.0)))


def _mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    ci95 = float(CI_Z * std / math.sqrt(float(arr.size)))
    return mean, ci95


def _to_np_float32(v: Any) -> np.ndarray:
    """兼容 torch.Tensor / numpy.ndarray 两种编码输出。"""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.asarray(v, dtype=np.float32)


def _infer_task_type(dataset_name: str, labels: np.ndarray) -> str:
    ds = dataset_name.lower()
    if "stsb" in ds or "sts-b" in ds:
        return "regression"

    uniq = np.unique(np.round(labels, 6))
    if uniq.size <= 2 and set(np.asarray(uniq, dtype=np.float64).tolist()).issubset({0.0, 1.0}):
        return "classification"
    return "regression"


def _load_pair_datasets(sample_count: int = SAMPLE_COUNT, random_seed: int = 2026) -> Dict[str, Tuple[List[str], List[str], np.ndarray, str]]:
    dataset_map: Dict[str, Tuple[List[str], List[str], np.ndarray, str]] = {}

    for file_name in sorted(os.listdir(TEST_DATA_DIR)):
        if not file_name.endswith(".txt"):
            continue
        path = os.path.join(TEST_DATA_DIR, file_name)
        df = pd.read_csv(path, sep="\t", header=None, names=["s1", "s2", "label"])
        df = df.dropna(subset=["s1", "s2", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        if df.empty:
            continue
        n = min(sample_count, len(df))
        df = df.sample(n=n, random_state=random_seed)
        s1 = df["s1"].astype(str).tolist()
        s2 = df["s2"].astype(str).tolist()
        dataset_name = file_name.replace("_test.txt", "").replace(".txt", "")

        # 评测中显式排除 lccc 数据集。
        if "lccc" in dataset_name.lower():
            continue

        y = np.asarray(df["label"].astype(float).tolist(), dtype=np.float32)
        task_type = _infer_task_type(dataset_name, y)
        dataset_map[dataset_name] = (s1, s2, y, task_type)

    if not dataset_map:
        raise RuntimeError("tests/data 下未找到可用句对数据。")

    return dataset_map


def _encode_with_hf(sentences: List[str], tokenizer: Any, model: Any, device: Any, batch_size: int = 128) -> np.ndarray:
    arr: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            out = model(**tokens)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0, :]
            arr.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(arr)


def _bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: str,
    color: str,
    err_col: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(df))
    y = np.asarray(df[y_col].to_numpy(), dtype=np.float64)
    yerr = None
    if err_col and err_col in df.columns:
        yerr = np.asarray(df[err_col].to_numpy(), dtype=np.float64)
    bars = ax.bar(x, y, yerr=yerr, capsize=4 if yerr is not None else 0, color=color, edgecolor="#1f2a37")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=20, ha="right")
    ax.set_title(title)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + max(0.005, abs(h) * 0.01), f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap_plot(matrix_df: pd.DataFrame, out_path: str, title: str, cbar_label: str) -> None:
    vals = matrix_df.values.astype(float)
    fig_w = max(8, 1.2 * vals.shape[1] + 2)
    fig_h = max(4.5, 0.7 * vals.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals, cmap="YlGnBu", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(list(matrix_df.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(list(matrix_df.index))
    ax.set_title(title)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval01_model_corr_speed")
    os.makedirs(out_dir, exist_ok=True)

    dataset_maps = {seed: _load_pair_datasets(SAMPLE_COUNT, random_seed=seed) for seed in EVAL_SEEDS}

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    per_dataset_rows: List[Dict[str, object]] = []
    speed_seed_rows: List[Dict[str, object]] = []

    # 本模型
    engine = SimCSEModelEngine()
    model_total_steps = sum(len(dataset_maps[seed]) for seed in EVAL_SEEDS)
    model_pbar = tqdm(
        total=model_total_steps,
        desc="本模型评测",
        unit="任务",
        dynamic_ncols=True,
        mininterval=0.2,
    )
    for seed in EVAL_SEEDS:
        dataset_map = dataset_maps[seed]
        elapsed_samples = 0
        elapsed_ms_total = 0.0
        for ds_name, (s1, s2, labels, task_type) in dataset_map.items():
            model_pbar.set_postfix_str(f"seed={seed}, ds={ds_name}")
            t0 = time.perf_counter()
            e1 = engine.encode(s1, encoder="query", batch_size=128, show_progress=False)
            e2 = engine.encode(s2, encoder="query", batch_size=128, show_progress=False)
            ms = (time.perf_counter() - t0) * 1000.0
            e1_np = _to_np_float32(e1)
            e2_np = _to_np_float32(e2)
            preds = _cosine_scores(e1_np, e2_np)
            spearman = _safe_spearman(labels, preds)
            auc = _safe_auc(labels, preds) if task_type == "classification" else np.nan
            sem_score = _semantic_judgement_score(task_type, labels, preds)

            per_dataset_rows.append(
                {
                    "模型": "本项目模型",
                    "随机种子": int(seed),
                    "数据集": ds_name,
                    "任务类型": task_type,
                    "Spearman": spearman,
                    "AUC": float(auc) if not np.isnan(auc) else np.nan,
                    "语义判断分": sem_score,
                    "样本数": int(len(labels)),
                }
            )
            elapsed_ms_total += ms
            elapsed_samples += len(labels)
            model_pbar.update(1)

        speed_seed_rows.append(
            {
                "模型": "本项目模型",
                "随机种子": int(seed),
                "平均耗时ms": elapsed_ms_total / max(elapsed_samples, 1),
            }
        )
    model_pbar.close()

    # 基准模型
    baseline_map = {
        "text2vec": "text2vec-base-chinese",
        "bert": "bert-base-chinese",
        "roberta": "chinese-roberta-wwm-ext",
        "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    }
    device = engine.device

    for short_name, folder in tqdm(
        baseline_map.items(),
        total=len(baseline_map),
        desc="基准模型评测",
        unit="个",
        dynamic_ncols=True,
        mininterval=0.2,
    ):
        path = os.path.join(TEST_MODELS_DIR, folder)
        if not os.path.exists(path):
            continue
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True).to(device)
        model.eval()

        model_name = f"基准模型_{short_name}"
        baseline_total_steps = sum(len(dataset_maps[seed]) for seed in EVAL_SEEDS)
        baseline_pbar = tqdm(
            total=baseline_total_steps,
            desc=f"{short_name}评测",
            unit="任务",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        for seed in EVAL_SEEDS:
            dataset_map = dataset_maps[seed]
            elapsed_samples = 0
            elapsed_ms_total = 0.0
            for ds_name, (s1, s2, labels, task_type) in dataset_map.items():
                baseline_pbar.set_postfix_str(f"seed={seed}, ds={ds_name}")
                t0 = time.perf_counter()
                e1_np = _encode_with_hf(s1, tokenizer, model, device=device, batch_size=128)
                e2_np = _encode_with_hf(s2, tokenizer, model, device=device, batch_size=128)
                ms = (time.perf_counter() - t0) * 1000.0
                preds = _cosine_scores(e1_np, e2_np)
                spearman = _safe_spearman(labels, preds)
                auc = _safe_auc(labels, preds) if task_type == "classification" else np.nan
                sem_score = _semantic_judgement_score(task_type, labels, preds)

                per_dataset_rows.append(
                    {
                        "模型": model_name,
                        "随机种子": int(seed),
                        "数据集": ds_name,
                        "任务类型": task_type,
                        "Spearman": spearman,
                        "AUC": float(auc) if not np.isnan(auc) else np.nan,
                        "语义判断分": sem_score,
                        "样本数": int(len(labels)),
                    }
                )
                elapsed_ms_total += ms
                elapsed_samples += len(labels)
                baseline_pbar.update(1)

            speed_seed_rows.append(
                {
                    "模型": model_name,
                    "随机种子": int(seed),
                    "平均耗时ms": elapsed_ms_total / max(elapsed_samples, 1),
                }
            )
        baseline_pbar.close()

    speed_seed_df = pd.DataFrame(speed_seed_rows)
    if speed_seed_df.empty:
        raise RuntimeError("无可用评测结果。")

    speed_rows: List[Dict[str, object]] = []
    for model_name in sorted(speed_seed_df["模型"].astype(str).unique().tolist()):
        vals = speed_seed_df[speed_seed_df["模型"] == model_name]["平均耗时ms"].to_numpy(dtype=np.float64)
        mean_v, ci_v = _mean_ci95(vals)
        speed_rows.append({"模型": model_name, "平均耗时ms": mean_v, "速度95CI": ci_v})
    speed_df = pd.DataFrame(speed_rows)

    # 只保留速度图，不再输出总Spearman图。
    _bar_plot(
        speed_df.sort_values(by="平均耗时ms", ascending=True),
        "模型",
        "平均耗时ms",
        "响应速度对比(平均耗时ms, 95%CI)",
        os.path.join(out_dir, "speed_compare.png"),
        "#3b82f6",
        err_col="速度95CI",
    )

    per_dataset_df = pd.DataFrame(per_dataset_rows)

    # 图2: 0/1分类数据集热力图(使用AUC，更适合分类任务)
    cls_df = per_dataset_df[per_dataset_df["任务类型"] == "classification"].copy()
    if cls_df.empty:
        raise RuntimeError("未找到0/1分类数据集，无法生成分类热力图。")
    cls_matrix = cls_df.pivot_table(index="模型", columns="数据集", values="AUC", aggfunc="mean").fillna(0.0)
    _heatmap_plot(
        cls_matrix,
        os.path.join(out_dir, "classification_heatmap.png"),
        "0/1分类数据集热力图(AUC)",
        "AUC",
    )

    # 图3: 0-5 数据集模型对比图（Spearman）
    reg_df = per_dataset_df[per_dataset_df["任务类型"] == "regression"].copy()
    if reg_df.empty:
        raise RuntimeError("未找到0-5打分数据集，无法生成0-5对比图。")

    reg_seed_df = reg_df.groupby(["模型", "随机种子"], as_index=False).agg(Spearman=("Spearman", "mean"))
    reg_rows: List[Dict[str, object]] = []
    for model_name in sorted(reg_seed_df["模型"].astype(str).unique().tolist()):
        vals = reg_seed_df[reg_seed_df["模型"] == model_name]["Spearman"].to_numpy(dtype=np.float64)
        mean_v, ci_v = _mean_ci95(vals)
        reg_rows.append({"模型": model_name, "Spearman": mean_v, "Spearman95CI": ci_v})
    reg_model_df = pd.DataFrame(reg_rows).sort_values(by="Spearman", ascending=False)

    _bar_plot(
        reg_model_df,
        "模型",
        "Spearman",
        "0-5数据集相似度对比(Spearman, 95%CI)",
        os.path.join(out_dir, "score_compare.png"),
        "#2a9d8f",
        err_col="Spearman95CI",
    )


if __name__ == "__main__":
    run_eval()