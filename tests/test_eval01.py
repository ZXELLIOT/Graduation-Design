"""
实验一：模型语义相似度与编码效率基准对比

评测目标:
    参照 SimCSE 论文 (Gao et al., 2021) 的标准评测方案，将本项目的
    SimCSE 双塔模型与 4 个主流中文预训练模型进行基准对比，
    评估语义相似度任务的准确性和编码效率。

评测数据集 (6个):
    0/1 二分类 (5个): ATEC, BQ, LCQMC, PAWS-X, AFQMC
    0-5 回归 (1个):  STS-B

评测模型 (5个):
    - 本项目模型: text2vec-base-chinese + SimCSE 两阶段微调
    - BGE-small-zh-v1.5
    - text2vec-base-chinese
    - bert-base-chinese
    - paraphrase-multilingual-MiniLM-L12-v2

输出:
    - metrics_table.csv
    - classification_auc_heatmap.png
    - stsb_spearman_bars.png
    - speed_bars.png
    - spearman_heatmap.png
"""

import os, sys, time, math
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
from tests.tests_config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测参数
# ============================================================
CLASSIFICATION_SAMPLE = 1500   # 0/1 数据集采样数
REGRESSION_SAMPLE = 1000       # STS-B (0-5) 采样数
EVAL_SEEDS = [2026, 2027, 2028]
MAX_LEN = 64
BATCH_SIZE = 64
CI_Z = 1.96
EPS = 1e-8
COS_THRESHOLD = 0.5

# 模型名称映射 (缩写 → 论文中全称)
MODEL_DISPLAY = {
    "mysimcse":      "本项目模型",
    "bge":           "BGE-small-zh-v1.5",
    "text2vec":      "text2vec-base-chinese",
    "bert":          "bert-base-chinese",
    "minilm":        "MiniLM-L12-v2",
}

# ============================================================
# 评估函数
# ============================================================
def _cosine_scores(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    dot = np.sum(e1 * e2, axis=1)
    norm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + EPS
    return dot / norm


def _safe_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2 or np.std(labels) <= EPS or np.std(preds) <= EPS:
        return 0.0
    try:
        c = float(pd.Series(labels).corr(pd.Series(preds), method="spearman"))
        return 0.0 if math.isnan(c) else c
    except Exception:
        return 0.0


def _safe_pearson(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2 or np.std(labels) <= EPS or np.std(preds) <= EPS:
        return 0.0
    try:
        c = float(pd.Series(labels).corr(pd.Series(preds), method="pearson"))
        return 0.0 if math.isnan(c) else c
    except Exception:
        return 0.0


def _binary_metrics(labels: np.ndarray, preds: np.ndarray, threshold: float = COS_THRESHOLD) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int32)
    y_pred_bin = (np.asarray(preds, dtype=np.float64) >= threshold).astype(np.int32)
    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    # AUC via Mann-Whitney U
    pos_scores = np.asarray(preds, dtype=np.float64)[y_true == 1]
    neg_scores = np.asarray(preds, dtype=np.float64)[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        auc = 0.5
    else:
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        pos_greater = np.sum((pos_scores[:, None] > neg_scores[None, :]).astype(np.float64))
        pos_equal = np.sum((np.abs(pos_scores[:, None] - neg_scores[None, :]) < EPS).astype(np.float64))
        auc = (pos_greater + 0.5 * pos_equal) / (n_pos * n_neg)
    return {"Accuracy": float(accuracy), "Precision": float(precision),
            "Recall": float(recall), "F1": float(f1), "AUC": float(auc)}


def _mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return (float(np.mean(arr)), 0.0) if arr.size == 1 else (0.0, 0.0)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    return mean, float(CI_Z * std / math.sqrt(float(arr.size)))


def _to_np(v: Any) -> np.ndarray:
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.asarray(v, dtype=np.float32)

# ============================================================
# 数据加载 (按任务类型区分采样数)
# ============================================================
def _load_datasets(seed: int) -> Dict[str, Tuple[List[str], List[str], np.ndarray, str]]:
    """加载所有评测数据集。0/1 采 1500 条, STS-B 采 1000 条。"""
    result = {}
    for fname in sorted(os.listdir(TEST_DATA_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(TEST_DATA_DIR, fname)
        df = pd.read_csv(path, sep="\t", header=None, names=["s1", "s2", "label"])
        df = df.dropna(subset=["s1", "s2", "label"])
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        if df.empty:
            continue

        ds_name = fname.replace("_test.txt", "").replace(".txt", "")
        if "lccc" in ds_name.lower():
            continue

        # 按任务类型选择采样数
        is_stsb = "stsb" in ds_name.lower()
        n_sample = REGRESSION_SAMPLE if is_stsb else CLASSIFICATION_SAMPLE
        n = min(n_sample, len(df))
        df = df.sample(n=n, random_state=seed)

        y = np.asarray(df["label"].astype(float).tolist(), dtype=np.float32)
        task = "regression" if is_stsb else "classification"
        result[ds_name] = (df["s1"].astype(str).tolist(),
                           df["s2"].astype(str).tolist(), y, task)
    if not result:
        raise RuntimeError("未找到评测数据集")
    return result


def _encode_hf(sentences: List[str], tokenizer, model, device, batch_size=BATCH_SIZE) -> np.ndarray:
    arr = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", max_length=MAX_LEN).to(device)
            out = model(**tokens)
            emb = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) \
                  else out.last_hidden_state[:, 0, :]
            arr.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(arr)

# ============================================================
# 图表
# ============================================================
def _bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str,
              out_path: str, color: str, err_col: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    y = np.asarray(df[y_col].to_numpy(), dtype=np.float64)
    yerr = np.asarray(df[err_col].to_numpy(), dtype=np.float64) if err_col and err_col in df.columns else None
    bars = ax.bar(x, y, yerr=yerr, capsize=5 if yerr is not None else 0,
                  color=color, edgecolor="#1f2a37", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=15, ha="right", fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(y_col)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2., h + max(0.005, abs(h) * 0.012),
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap(mat_df: pd.DataFrame, out_path: str, title: str, cbar_label: str,
             vmin=0.0, vmax=1.0) -> None:
    vals = mat_df.values.astype(float)
    fw = max(9, 1.2 * vals.shape[1] + 3)
    fh = max(5, 0.7 * vals.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.imshow(vals, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(list(mat_df.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(list(mat_df.index))
    ax.set_title(title, fontsize=14)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 主评测
# ============================================================
def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval01_model_corr_speed")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("实验一: 模型语义相似度与编码效率基准对比")
    print(f"0/1 数据集采样: {CLASSIFICATION_SAMPLE} 条  |  STS-B 采样: {REGRESSION_SAMPLE} 条")
    print(f"随机种子: {EVAL_SEEDS}")
    print("=" * 60)

    # 按种子加载
    dataset_maps = {seed: _load_datasets(seed) for seed in EVAL_SEEDS}
    # 打印每个数据集的实际条数
    ds0 = dataset_maps[EVAL_SEEDS[0]]
    for name, (s1, s2, y, task) in ds0.items():
        print(f"  {name}: {len(s1)} 条 ({task})")

    all_rows: List[Dict[str, Any]] = []

    # ---------- 本模型 ----------
    print("\n[1/5] 评测本项目模型 ...")
    engine = SimCSEModelEngine()
    for seed in EVAL_SEEDS:
        for ds_name, (s1, s2, labels, task) in tqdm(dataset_maps[seed].items(), desc=f"种子={seed}"):
            t0 = time.perf_counter()
            e1 = engine.encode(s1, encoder="query", batch_size=BATCH_SIZE, show_progress=False)
            e2 = engine.encode(s2, encoder="query", batch_size=BATCH_SIZE, show_progress=False)
            ms = (time.perf_counter() - t0) * 1000.0 / max(len(s1), 1)
            preds = _cosine_scores(_to_np(e1), _to_np(e2))
            if task == "classification":
                m = _binary_metrics(labels, preds)
                m["Spearman"] = _safe_spearman(labels, preds)
                m["Pearson"] = _safe_pearson(labels, preds)
            else:
                m = {"Spearman": _safe_spearman(labels, preds), "Pearson": _safe_pearson(labels, preds),
                     "Accuracy": float("nan"), "Precision": float("nan"),
                     "Recall": float("nan"), "F1": float("nan"), "AUC": float("nan")}
            all_rows.append({"模型": "本项目模型", "随机种子": seed, "数据集": ds_name,
                             "任务类型": task, "样本数": len(labels), "平均耗时ms": ms, **m})

    # ---------- 基准模型 ----------
    baseline_map = {
        "BGE-small-zh": "bge-small-zh-v1.5",
        "text2vec": "text2vec-base-chinese",
        "bert": "bert-base-chinese",
        "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    }
    device = engine.device
    for idx, (short, folder) in enumerate(baseline_map.items(), 2):
        model_path = os.path.join(TEST_MODELS_DIR, folder)
        if not os.path.exists(model_path):
            print(f"\n[{idx}/5] 跳过 {short}: 模型目录不存在")
            continue
        print(f"\n[{idx}/5] 评测 {short} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()
        model_name = f"基准_{short}"
        for seed in EVAL_SEEDS:
            for ds_name, (s1, s2, labels, task) in tqdm(dataset_maps[seed].items(),
                                                          desc=f"{short} seed={seed}", leave=False):
                t0 = time.perf_counter()
                e1 = _encode_hf(s1, tokenizer, model, device)
                e2 = _encode_hf(s2, tokenizer, model, device)
                ms = (time.perf_counter() - t0) * 1000.0 / max(len(s1), 1)
                preds = _cosine_scores(e1, e2)
                if task == "classification":
                    m = _binary_metrics(labels, preds)
                    m["Spearman"] = _safe_spearman(labels, preds)
                    m["Pearson"] = _safe_pearson(labels, preds)
                else:
                    m = {"Spearman": _safe_spearman(labels, preds), "Pearson": _safe_pearson(labels, preds),
                         "Accuracy": float("nan"), "Precision": float("nan"),
                         "Recall": float("nan"), "F1": float("nan"), "AUC": float("nan")}
                all_rows.append({"模型": model_name, "随机种子": seed, "数据集": ds_name,
                                 "任务类型": task, "样本数": len(labels), "平均耗时ms": ms, **m})
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 表格 & 图表
    # ============================================================
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False, encoding="utf-8-sig")

    # --- 表1: 二分类 AUC (种子均值) ---
    cls = df[df["任务类型"] == "classification"]
    cls_auc = cls.pivot_table(index="模型", columns="数据集", values="AUC", aggfunc="mean")
    print("\n===== 表: 二分类数据集 AUC 均值 =====")
    print(cls_auc.round(4).to_string())

    _heatmap(cls_auc, os.path.join(out_dir, "classification_auc_heatmap.png"),
             "二分类数据集 AUC 对比", "AUC")

    # --- 表2: 二分类全部指标 (5数据集 x 5模型平均) ---
    cls_full = cls.groupby(["模型", "数据集"], as_index=False).agg(
        Accuracy=("Accuracy", "mean"), Precision=("Precision", "mean"),
        Recall=("Recall", "mean"), F1=("F1", "mean"), AUC=("AUC", "mean"),
    )
    cls_full.to_csv(os.path.join(out_dir, "classification_full_metrics.csv"), index=False, encoding="utf-8-sig")

    # --- 表3: STS-B Spearman ---
    stsb = df[df["数据集"].str.lower().str.contains("stsb")]
    if not stsb.empty:
        stsb_agg = stsb.groupby(["模型", "随机种子"], as_index=False).agg(
            Spearman=("Spearman", "mean"), Pearson=("Pearson", "mean"),
        )
        stsb_summary = []
        for mn in sorted(stsb_agg["模型"].unique()):
            sv = stsb_agg[stsb_agg["模型"] == mn]["Spearman"].values
            pv = stsb_agg[stsb_agg["模型"] == mn]["Pearson"].values
            sm, sc = _mean_ci95(sv)
            pm, pc = _mean_ci95(pv)
            stsb_summary.append({"模型": mn, "Spearman": sm, "Spearman 95%CI": sc, "Pearson": pm})
        stsb_df = pd.DataFrame(stsb_summary).sort_values("Spearman", ascending=False)
        print("\n===== 表: STS-B Spearman =====")
        print(stsb_df.round(4).to_string(index=False))
        stsb_df.to_csv(os.path.join(out_dir, "stsb_spearman.csv"), index=False, encoding="utf-8-sig")

        _bar_plot(stsb_df, "模型", "Spearman",
                  "STS-B Spearman 秩相关系数 (95%CI)", os.path.join(out_dir, "stsb_spearman_bars.png"),
                  "#2a9d8f", err_col="Spearman 95%CI")

    # --- 表4: 编码速度 ---
    speed = df.groupby(["模型", "随机种子"], as_index=False).agg(ms=("平均耗时ms", "mean"))
    speed_s = []
    for mn in sorted(speed["模型"].unique()):
        vals = speed[speed["模型"] == mn]["ms"].values
        m, ci = _mean_ci95(vals)
        speed_s.append({"模型": mn, "平均耗时ms": m, "95%CI": ci})
    speed_df = pd.DataFrame(speed_s).sort_values("平均耗时ms")
    print("\n===== 表: 编码速度 =====")
    print(speed_df.round(2).to_string(index=False))
    speed_df.to_csv(os.path.join(out_dir, "speed_table.csv"), index=False, encoding="utf-8-sig")

    _bar_plot(speed_df, "模型", "平均耗时ms",
              "编码速度对比 (ms/句对, 95%CI)", os.path.join(out_dir, "speed_bars.png"),
              "#3b82f6", err_col="95%CI")

    # --- 全数据集 Spearman 热力图 ---
    all_spearman = df.pivot_table(index="模型", columns="数据集", values="Spearman", aggfunc="mean").fillna(0.0)
    _heatmap(all_spearman, os.path.join(out_dir, "spearman_heatmap.png"),
             "全数据集 Spearman 排序质量对比", "Spearman")

    print(f"\n输出目录: {out_dir}")
    print("完成")


if __name__ == "__main__":
    run_eval()
