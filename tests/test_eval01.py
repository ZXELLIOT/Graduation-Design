"""
实验一：模型语义相似度与编码效率基准对比。
输出三个 CSV: 二分类指标、STS-B 回归指标、编码速度。
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

CLASSIFICATION_SAMPLE = 1500
REGRESSION_SAMPLE = 1000
SEED = 2026
MAX_LEN = 64
BATCH_SIZE = 64
EPS = 1e-8
COS_THRESHOLD = 0.5

def _cosine_scores(e1, e2):
    return np.sum(e1 * e2, axis=1)

def _safe_spearman(labels, preds):
    if labels.size < 2 or np.std(labels) <= EPS or np.std(preds) <= EPS:
        return 0.0
    try:
        c = pd.Series(labels).corr(pd.Series(preds), method="spearman")
        return 0.0 if math.isnan(c) else float(c)
    except Exception:
        return 0.0

def _binary_metrics(labels, preds):
    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = (np.asarray(preds, dtype=np.float64) >= COS_THRESHOLD).astype(np.int32)
    tp = sum((y_pred == 1) & (y_true == 1))
    fp = sum((y_pred == 1) & (y_true == 0))
    fn = sum((y_pred == 0) & (y_true == 1))
    tn = sum((y_pred == 0) & (y_true == 0))
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, EPS)
    pos = np.asarray(preds)[y_true == 1]
    neg = np.asarray(preds)[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.5
    else:
        pos_g = np.sum((pos[:, None] > neg[None, :]).astype(np.float64))
        pos_e = np.sum((np.abs(pos[:, None] - neg[None, :]) < EPS).astype(np.float64))
        auc = (pos_g + 0.5 * pos_e) / (len(pos) * len(neg))
    return {"Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec),
            "F1": float(f1), "AUC": float(auc)}

def _load_datasets():
    classification, regression = {}, {}
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
        n = min(REGRESSION_SAMPLE if "stsb" in ds_name.lower() else CLASSIFICATION_SAMPLE, len(df))
        df = df.sample(n=n, random_state=SEED)
        y = np.asarray(df["label"].astype(float), dtype=np.float32)
        entry = (df["s1"].astype(str).tolist(), df["s2"].astype(str).tolist(), y)
        (regression if "stsb" in ds_name.lower() else classification)[ds_name] = entry
    if not classification and not regression:
        raise RuntimeError("未找到评测数据集")
    return classification, regression

def _encode_hf(sentences, tokenizer, model, device):
    arr = []
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i: i + BATCH_SIZE]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            out = model(**tokens)
            emb = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
            arr.append(F.normalize(emb.float(), p=2, dim=1).cpu().numpy().astype(np.float32))
    return np.vstack(arr)

def run_eval():
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval01")
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 50)
    print("实验一: 模型语义相似度基准对比")
    print("=" * 50)
    cls_ds, reg_ds = _load_datasets()
    print(f"二分类: {list(cls_ds.keys())}  |  STS-B: {list(reg_ds.keys())}")
    cls_rows, reg_rows, speed_rows = [], [], []
    engine = SimCSEModelEngine()
    device = engine.device
    models = {
        "mysimcse": (engine.tokenizer, engine.query_encoder),
        "text2vec": None,
        "bert": None,
        "minilm": None,
    }
    baselines = {
        "text2vec": "text2vec-base-chinese",
        "bert": "bert-base-chinese",
        "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    }
    for short, folder in baselines.items():
        path = os.path.join(TEST_MODELS_DIR, folder)
        if os.path.exists(path):
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
            models[short] = (tok, model)
    dummy = ["测速句"] * 100
    for idx, (name, (tok, model)) in enumerate(models.items(), 1):
        print(f"\n[{idx}/5] {name}")
        t0 = time.perf_counter()
        _encode_hf(dummy, tok, model, device)
        speed_rows.append({"模型": name, "100句耗时ms": (time.perf_counter() - t0) * 1000})
        for ds_name, (s1, s2, labels) in cls_ds.items():
            e1, e2 = _encode_hf(s1, tok, model, device), _encode_hf(s2, tok, model, device)
            cls_rows.append({"模型": name, "数据集": ds_name, **_binary_metrics(labels, _cosine_scores(e1, e2))})
        for ds_name, (s1, s2, labels) in reg_ds.items():
            e1, e2 = _encode_hf(s1, tok, model, device), _encode_hf(s2, tok, model, device)
            reg_rows.append({"模型": name, "数据集": ds_name,
                             "Spearman": _safe_spearman(labels, _cosine_scores(e1, e2))})
        if name not in ("mysimcse",): 
            del model, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # 输出 3 个 CSV
    cls_auc = pd.DataFrame(cls_rows).pivot_table(index="模型", columns="数据集", values="AUC", aggfunc="mean")
    cls_auc.round(4).to_csv(os.path.join(out_dir, "1_classification_auc.csv"), encoding="utf-8-sig")
    print("\n===== 1. 二分类 AUC =====")
    print(cls_auc.round(4).to_string())
    reg_pivot = pd.DataFrame(reg_rows).pivot_table(index="模型", columns="数据集", values="Spearman", aggfunc="mean")
    reg_pivot.round(4).to_csv(os.path.join(out_dir, "2_stsb_spearman.csv"), encoding="utf-8-sig")
    print("\n===== 2. STS-B Spearman =====")
    print(reg_pivot.round(4).to_string())
    speed_df = pd.DataFrame(speed_rows).sort_values("100句耗时ms")
    speed_df.round(1).to_csv(os.path.join(out_dir, "3_speed.csv"), index=False, encoding="utf-8-sig")
    print("\n===== 3. 编码速度 =====")
    print(speed_df.round(1).to_string(index=False))
    print(f"\n输出: {out_dir}")
    print("完成")

if __name__ == "__main__":
    run_eval()