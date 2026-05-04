"""
tests/test_eval01_model_corr_speed.py

评测目标:
    参照 SimCSE 论文 (Gao et al., 2021, arXiv:2104.08821) 的评测方案，
    将本项目训练的 SimCSE 双塔模型与主流中文预训练模型进行基准对比。

评测方案（适配中文）:
    本模型训练数据为中文 LCCC 对话语料，预训练基座为 text2vec-base-chinese
    (BERT-base-chinese 架构)，仅支持中文。因此全部使用中文评测数据集。

评测数据集:
    0/1 二分类数据集（5个，判断两句是否语义等价）:
        - ATEC:   支付宝金融问句相似度 (20k 句对)
        - BQ:     银行金融领域问句匹配 (10k)
        - LCQMC:  大规模中文问句匹配语料, COLING 2018 (12.5k)
        - PAWS-X: 跨语言释义检测, Google Research (2k)
        - AFQMC:  蚂蚁金融问句匹配, CLUE Benchmark (2k)

    0-5 回归数据集（1个，原生中文语义相似度评分）:
        - STS-B:  SemEval 2017 语义文本相似度基准中文翻译版 (1.4k)
          注: 中文 0-5 数据集极为稀缺，STS-B 是目前唯一广泛使用的。

评测指标（参照 SimCSE 论文）:
    0/1 数据集: Accuracy, Precision, Recall, F1, AUC
    0-5 数据集: Spearman 秩相关系数（ρ × 100）
    排序质量:    Spearman 热力图（含全6个数据集，学术标准做法）
    速度指标:   每句对平均编码耗时 (ms)

评测模型:
    - 本项目模型 (mysimcse):        SimCSE 两阶段微调双塔
    - BGE-small-zh-v1.5:            智源研究院, C-MTEB 榜首系列
    - text2vec-base-chinese:        CoSENT 中文句向量
    - bert-base-chinese:            Google 中文 BERT
    - paraphrase-multilingual-MiniLM-L12-v2: 微软多语言轻量模型

输出文件:
    - speed_compare.png:           各模型平均处理速度对比（含95%CI）
    - classification_heatmap.png:  0/1 数据集 AUC 热力图
    - spearman_heatmap.png:        全数据集 Spearman 排序质量热力图
    - regression_compare.png:      STS-B Spearman 对比（唯一原生0-5）
    - metrics_table.csv:           完整指标表格
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

# 路径初始化
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.model_engine import SimCSEModelEngine
from tests.config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# ============================================================
# 评测超参数
# ============================================================

# 每个数据集随机采样的句对数（控制评测时间）
SAMPLE_COUNT = 300

# 随机种子列表（用于多次重复评测计算置信区间）
EVAL_SEEDS = [2026, 2027, 2028]

# 每句最长截断长度
MAX_LEN = 64

# 批量编码大小
BATCH_SIZE = 64

# 95% 置信区间 Z 值
CI_Z = 1.96

# 浮点容差
EPS = 1e-8

# 余弦相似度阈值（用于将余弦值转为 0/1 预测）
COS_THRESHOLD = 0.5


def _cosine_scores(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    计算两组向量之间的余弦相似度。

    数学定义: cos(A,B) = (A·B) / (|A| × |B|)

    参数:
        e1: 形状为 [N, D] 的向量矩阵。
        e2: 形状为 [N, D] 的向量矩阵。

    返回:
        长度为 N 的一维余弦相似度数组。
    """
    dot = np.sum(e1 * e2, axis=1)
    norm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + EPS
    return dot / norm


def _safe_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    安全计算 Spearman 秩相关系数。

    Spearman 衡量两个变量的单调关系强度，范围 [-1, 1]。
    对 0-5 评分数据集，它是衡量语义相似度排序质量的权威指标。
    """
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


def _safe_pearson(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    安全计算 Pearson 线性相关系数。

    Pearson 衡量两个变量的线性相关强度，范围 [-1, 1]。
    常用于补充 Spearman（Spearman 看排序，Pearson 看线性）。
    """
    if labels.size < 2:
        return 0.0
    if np.std(labels) <= EPS or np.std(preds) <= EPS:
        return 0.0
    try:
        corr = float(pd.Series(labels).corr(pd.Series(preds), method="pearson"))
        if math.isnan(corr):
            return 0.0
        return corr
    except Exception:
        return 0.0


def _binary_metrics(labels: np.ndarray, preds: np.ndarray, threshold: float = COS_THRESHOLD) -> Dict[str, float]:
    """
    计算二分类评估指标。

    将余弦相似度按阈值二值化后，与真实 0/1 标签对比计算:
        - Accuracy:  整体预测正确率
        - Precision: 预测为正的样本中真正的正样本比例
        - Recall:    真正正样本中被预测出来的比例
        - F1:        Precision 和 Recall 的调和平均
        - AUC:       不依赖阈值的排序质量指标（用 Wilcoxon-Mann-Whitney 统计量近似）

    参数:
        labels: 真实标签 (0 或 1)。
        preds:  预测余弦相似度。
        threshold: 二值化阈值。

    返回:
        包含各指标的字典。
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_pred_bin = (np.asarray(preds, dtype=np.float64) >= threshold).astype(np.int32)

    # --- 计算混淆矩阵四要素 ---
    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)

    # --- AUC（手动计算，避免 sklearn 依赖）---
    # 基于 Wilcoxon-Mann-Whitney 统计量: AUC = P(正样本分 > 负样本分)
    pos_scores = np.asarray(preds, dtype=np.float64)[y_true == 1]
    neg_scores = np.asarray(preds, dtype=np.float64)[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        auc = 0.5
    else:
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        # 对每对 (pos, neg) 计数，pos > neg 则得分
        pos_greater = np.sum(
            (pos_scores[:, None] > neg_scores[None, :]).astype(np.float64)
        )
        pos_equal = np.sum(
            (np.abs(pos_scores[:, None] - neg_scores[None, :]) < EPS).astype(np.float64)
        )
        auc = (pos_greater + 0.5 * pos_equal) / (n_pos * n_neg)

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(auc),
    }


def _mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    """计算均值和 95% 置信区间半宽。"""
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
    """
    自动推断数据集任务类型。

    规则:
        - 名称含 stsb → regression（0-5 语义相似度评分，唯一原生中文 STS 数据集）
        - 标签仅含 0 和 1 → classification（0/1 二分类）
        - 其他 → regression

    说明:
        STS-B 是目前唯一广泛使用的中文 0-5 语义相似度数据集。
        对于二分类数据集，除用 Accuracy/AUC 评估外，也可用 Spearman
        评估余弦相似度的排序质量——这是 C-MTEB 等基准的标准做法。
    """
    ds = dataset_name.lower()
    if "stsb" in ds:
        return "regression"

    uniq = np.unique(np.round(labels, 6))
    if uniq.size <= 2 and set(np.asarray(uniq, dtype=np.float64).tolist()).issubset(
        {0.0, 1.0}
    ):
        return "classification"
    return "regression"


def _load_pair_datasets(
    sample_count: int = SAMPLE_COUNT, random_seed: int = 2026
) -> Dict[str, Tuple[List[str], List[str], np.ndarray, str]]:
    """
    加载 tests/data/ 下的标准评测句对数据集。

    数据格式: TSV，三列 (id, sentence1, sentence2, label)
    自动排除 lccc 数据集（那是训练数据，不是标准评测数据）。
    """
    dataset_map: Dict[str, Tuple[List[str], List[str], np.ndarray, str]] = {}

    for file_name in sorted(os.listdir(TEST_DATA_DIR)):
        if not file_name.endswith(".txt"):
            continue
        path = os.path.join(TEST_DATA_DIR, file_name)

        # 读取 TSV 格式数据
        df = pd.read_csv(path, sep="\t", header=None, names=["s1", "s2", "label"])
        df = df.dropna(subset=["s1", "s2", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])

        if df.empty:
            continue

        # 随机采样控制评测时间
        n = min(sample_count, len(df))
        df = df.sample(n=n, random_state=random_seed)

        s1 = df["s1"].astype(str).tolist()
        s2 = df["s2"].astype(str).tolist()
        dataset_name = file_name.replace("_test.txt", "").replace(".txt", "")

        # 排除训练/语料数据集
        if "lccc" in dataset_name.lower():
            continue

        y = np.asarray(df["label"].astype(float).tolist(), dtype=np.float32)
        task_type = _infer_task_type(dataset_name, y)
        dataset_map[dataset_name] = (s1, s2, y, task_type)

    if not dataset_map:
        raise RuntimeError("tests/data 下未找到可用句对数据。")

    return dataset_map


def _encode_with_hf(
    sentences: List[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    使用 HuggingFace 标准模型编码句子为向量。

    对于有 pooler_output 的模型取 pooler_output，
    否则取 last_hidden_state 的 [CLS] 位置（索引 0）。
    """
    arr: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=MAX_LEN,
            ).to(device)
            out = model(**tokens)
            # pooler_output: BERT 系列的句向量输出头（tanh 激活）
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                # 无 pooler 则用 [CLS] token 的最后一层隐藏状态
                emb = out.last_hidden_state[:, 0, :]
            arr.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(arr)


# ============================================================
# 图表绘制函数
# ============================================================


def _bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: str,
    color: str,
    err_col: str = "",
) -> None:
    """绘制带误差线的柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    y = np.asarray(df[y_col].to_numpy(), dtype=np.float64)
    yerr = None
    if err_col and err_col in df.columns:
        yerr = np.asarray(df[err_col].to_numpy(), dtype=np.float64)
    bars = ax.bar(
        x, y, yerr=yerr, capsize=4 if yerr is not None else 0,
        color=color, edgecolor="#1f2a37",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=20, ha="right")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(y_col)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + max(0.005, abs(h) * 0.01),
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap_plot(
    matrix_df: pd.DataFrame, out_path: str, title: str, cbar_label: str
) -> None:
    """绘制学术论文风格的热力图。"""
    vals = matrix_df.values.astype(float)
    fig_w = max(8, 1.2 * vals.shape[1] + 2)
    fig_h = max(4.5, 0.7 * vals.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(list(matrix_df.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(list(matrix_df.index))
    ax.set_title(title, fontsize=14)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(
                j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主评测流程
# ============================================================


def run_eval() -> None:
    """执行完整的模型基准对比评测。"""
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval01_model_corr_speed")
    os.makedirs(out_dir, exist_ok=True)

    # 配置中文字体
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 按多种子加载数据（用于置信区间计算）
    dataset_maps = {
        seed: _load_pair_datasets(SAMPLE_COUNT, random_seed=seed)
        for seed in EVAL_SEEDS
    }

    # ============================================================
    # 评测 1: 本项目双塔模型
    # ============================================================
    engine = SimCSEModelEngine()
    all_rows: List[Dict[str, Any]] = []
    speed_rows: List[Dict[str, Any]] = []

    print("\n===== 正在评测本项目模型 =====")
    for seed in EVAL_SEEDS:
        dataset_map = dataset_maps[seed]
        for ds_name, (s1, s2, labels, task_type) in tqdm(
            dataset_map.items(),
            total=len(dataset_map),
            desc=f"种子={seed}",
            unit="数据集",
        ):
            # 编码计时
            t0 = time.perf_counter()
            e1 = engine.encode(s1, encoder="query", batch_size=BATCH_SIZE, show_progress=False)
            e2 = engine.encode(s2, encoder="query", batch_size=BATCH_SIZE, show_progress=False)
            ms_per_pair = (time.perf_counter() - t0) * 1000.0 / max(len(s1), 1)

            # 计算余弦相似度作为预测分数
            e1_np = _to_np_float32(e1)
            e2_np = _to_np_float32(e2)
            preds = _cosine_scores(e1_np, e2_np)

            # 按任务类型计算相应指标
            if task_type == "classification":
                metrics = _binary_metrics(labels, preds)
                metrics["Spearman"] = _safe_spearman(labels, preds)
                metrics["Pearson"] = _safe_pearson(labels, preds)
            else:
                metrics = {
                    "Spearman": _safe_spearman(labels, preds),
                    "Pearson": _safe_pearson(labels, preds),
                    "Accuracy": float("nan"),
                    "Precision": float("nan"),
                    "Recall": float("nan"),
                    "F1": float("nan"),
                    "AUC": float("nan"),
                }

            all_rows.append(
                {
                    "模型": "本项目模型",
                    "随机种子": seed,
                    "数据集": ds_name,
                    "任务类型": task_type,
                    "样本数": len(labels),
                    "平均耗时ms": ms_per_pair,
                    **metrics,
                }
            )

    # ============================================================
    # 评测 2: 基准模型
    # ============================================================
    baseline_map = {
        "BGE-small-zh": "bge-small-zh-v1.5",
        "text2vec": "text2vec-base-chinese",
        "bert": "bert-base-chinese",
        "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    }
    device = engine.device

    for short_name, folder in tqdm(
        baseline_map.items(),
        total=len(baseline_map),
        desc="基准模型评测",
        unit="个",
    ):
        path = os.path.join(TEST_MODELS_DIR, folder)
        if not os.path.exists(path):
            print(f"  [跳过] 模型目录不存在: {path}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True).to(device)
        model.eval()

        model_name = f"基准_{short_name}"
        print(f"\n----- 正在评测 {model_name} -----")

        for seed in EVAL_SEEDS:
            dataset_map = dataset_maps[seed]
            for ds_name, (s1, s2, labels, task_type) in tqdm(
                dataset_map.items(),
                total=len(dataset_map),
                desc=f"{short_name} seed={seed}",
                unit="数据集",
                leave=False,
            ):
                # 编码计时
                t0 = time.perf_counter()
                e1_np = _encode_with_hf(s1, tokenizer, model, device=device)
                e2_np = _encode_with_hf(s2, tokenizer, model, device=device)
                ms_per_pair = (time.perf_counter() - t0) * 1000.0 / max(len(s1), 1)

                preds = _cosine_scores(e1_np, e2_np)

                if task_type == "classification":
                    metrics = _binary_metrics(labels, preds)
                    metrics["Spearman"] = _safe_spearman(labels, preds)
                    metrics["Pearson"] = _safe_pearson(labels, preds)
                else:
                    metrics = {
                        "Spearman": _safe_spearman(labels, preds),
                        "Pearson": _safe_pearson(labels, preds),
                        "Accuracy": float("nan"),
                        "Precision": float("nan"),
                        "Recall": float("nan"),
                        "F1": float("nan"),
                        "AUC": float("nan"),
                    }

                all_rows.append(
                    {
                        "模型": model_name,
                        "随机种子": seed,
                        "数据集": ds_name,
                        "任务类型": task_type,
                        "样本数": len(labels),
                        "平均耗时ms": ms_per_pair,
                        **metrics,
                    }
                )

        # 释放 GPU 内存
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 数据整理与图表生成
    # ============================================================
    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        raise RuntimeError("无可用评测结果。")

    # 保存完整指标表
    df_all.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False, encoding="utf-8-sig")

    # --- 图1: 速度对比 ---
    speed_group = df_all.groupby(["模型", "随机种子"], as_index=False).agg(
        平均耗时ms=("平均耗时ms", "mean")
    )
    speed_summary: List[Dict[str, Any]] = []
    for model_name in sorted(speed_group["模型"].unique()):
        vals = speed_group[speed_group["模型"] == model_name]["平均耗时ms"].values
        mean_v, ci_v = _mean_ci95(vals)
        speed_summary.append(
            {"模型": model_name, "平均耗时ms": mean_v, "速度95CI": ci_v}
        )
    speed_df = pd.DataFrame(speed_summary).sort_values("平均耗时ms")

    _bar_plot(
        speed_df,
        "模型",
        "平均耗时ms",
        "响应速度对比（每句对平均耗时 ms，95%CI）",
        os.path.join(out_dir, "speed_compare.png"),
        "#3b82f6",
        err_col="速度95CI",
    )

    # --- 图2: 0/1 分类数据集 AUC 热力图 ---
    cls_df = df_all[df_all["任务类型"] == "classification"].copy()
    if not cls_df.empty:
        cls_matrix = cls_df.pivot_table(
            index="模型", columns="数据集", values="AUC", aggfunc="mean"
        ).fillna(0.0)
        _heatmap_plot(
            cls_matrix,
            os.path.join(out_dir, "classification_heatmap.png"),
            "0/1分类数据集 AUC 对比",
            "AUC",
        )

        # 打印分类指标摘要
        print("\n===== 0/1 分类数据集指标摘要 =====")
        cls_summary = cls_df.groupby(["模型", "数据集"], as_index=False).agg(
            Accuracy=("Accuracy", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            AUC=("AUC", "mean"),
        )
        print(cls_summary.to_string(index=False))

    # --- 图3: 全数据集 Spearman 热力图（含二分类+STS-B，学术标准做法）---
    # 说明：Spearman 衡量余弦相似度排序与真实标签排序的一致性。
    # 二分类数据集（0/1标签）用 Spearman 评估排序质量是 C-MTEB 等基准的标准做法。
    all_spearman = df_all.pivot_table(
        index="模型", columns="数据集", values="Spearman", aggfunc="mean"
    ).fillna(0.0)
    _heatmap_plot(
        all_spearman,
        os.path.join(out_dir, "spearman_heatmap.png"),
        "全数据集 Spearman 排序质量对比（含0/1+0-5）",
        "Spearman",
    )

    # 单独突出 STS-B（唯一原生 0-5 数据集）的 Spearman 柱状图
    stsb_df = df_all[df_all["数据集"].str.lower().str.contains("stsb")].copy()
    if not stsb_df.empty:
        stsb_group = stsb_df.groupby(["模型", "随机种子"], as_index=False).agg(
            Spearman=("Spearman", "mean"),
            Pearson=("Pearson", "mean"),
        )
        stsb_summary = []
        for mn in sorted(stsb_group["模型"].unique()):
            sv = stsb_group[stsb_group["模型"] == mn]["Spearman"].values
            pv = stsb_group[stsb_group["模型"] == mn]["Pearson"].values
            sm, sc = _mean_ci95(sv)
            pm, _ = _mean_ci95(pv)
            stsb_summary.append({"模型": mn, "Spearman": sm, "Spearman95CI": sc, "Pearson": pm})
        stsb_model_df = pd.DataFrame(stsb_summary).sort_values("Spearman", ascending=False)
        _bar_plot(
            stsb_model_df, "模型", "Spearman",
            "STS-B 语义相似度 Spearman 对比（唯一原生0-5数据集，95%CI）",
            os.path.join(out_dir, "regression_compare.png"), "#2a9d8f",
            err_col="Spearman95CI",
        )
        print("\n===== STS-B (0-5) Spearman =====")
        print(stsb_model_df.to_string(index=False))

    print(f"\n评测完成，结果保存至: {out_dir}")


if __name__ == "__main__":
    run_eval()
