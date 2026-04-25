"""
tests/test_eval04_ai_mode_ablation.py

目标:
    在 db/data/lccc_large.csv 中随机抽取 200 条作为测试输入/期待输出，
    对比以下 3 种模式的准确性和响应时间:
    1) AI 直接回复
    2) 本模型 + AI 增强(禁用上下文)
    3) 本模型无 AI 增强(禁用上下文)
输出:
    仅输出 2 张图:
    1) 准确性对比图(MiniLM 本地语义评分)
    2) 反应时间对比图
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.ai_enhancer import generate_ai_enhanced_reply
from system.bootstrap import load_database_columns
from system.comparator import DialogComparator
from system.config import (
    AI_ENHANCE_API_KEY,
    AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL,
    AI_ENHANCE_TIMEOUT_SEC,
    DB_CSV_PATH,
    DB_PREFIX,
)
from system.model_engine import SimCSEModelEngine
from tests.config import TEST_MODELS_DIR, TEST_RESULTS_DIR

SAMPLE_COUNT = 200
EPS = 1e-8
BATCH_SIZE = 128
LOCAL_EVAL_MODEL_DIR = os.path.join(TEST_MODELS_DIR, "paraphrase-multilingual-MiniLM-L12-v2")


class HFQueryAdapter:
    """单塔 HF 模型适配器，用于 MiniLM 本地语义评分。"""

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


def _extract_ai_output_text(resp_json: Dict[str, Any]) -> str:
    output_text = resp_json.get("output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = resp_json.get("output", [])
    if isinstance(output, list):
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    return ""


def _generate_answer_direct_with_ai(question: str) -> str:
    if not AI_ENHANCE_API_KEY:
        return ""

    prompt = (
        "请作为中文对话助手，基于用户输入给出自然、简洁、有帮助的一句话回答。"
        "\n用户输入:\n"
        f"{str(question)}"
    )
    payload = {
        "model": AI_ENHANCE_MODEL_NAME,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {AI_ENHANCE_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(AI_ENHANCE_RESPONSES_URL, headers=headers, json=payload, timeout=AI_ENHANCE_TIMEOUT_SEC)
        resp.raise_for_status()
        return _extract_ai_output_text(resp.json())
    except Exception:
        return ""


def _to_np_float32(v: Any) -> np.ndarray:
    if hasattr(v, "detach"):
        try:
            return np.asarray(v.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        except Exception:
            pass
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _local_semantic_score(model_engine: Any, text_a: str, text_b: str) -> float:
    a = _to_np_float32(model_engine.encode_one(str(text_a), encoder="query"))
    b = _to_np_float32(model_engine.encode_one(str(text_b), encoder="query"))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    cos = float(np.dot(a, b) / denom)
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))


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


def _sample_cases_from_csv(n: int = SAMPLE_COUNT) -> List[Tuple[str, str]]:
    df = pd.read_csv(DB_CSV_PATH, usecols=["query", "response"])
    df = df.dropna(subset=["query", "response"]).copy()
    total = len(df)
    if total < n:
        raise RuntimeError(f"lccc_large.csv 样本不足，至少需要 {n} 条，当前只有 {total} 条。")

    rng = np.random.default_rng(2026)
    idx = rng.choice(total, size=n, replace=False)
    sampled = df.iloc[idx].reset_index(drop=True)
    return list(zip(sampled["query"].astype(str).tolist(), sampled["response"].astype(str).tolist()))


def _judge_pairs(pairs: List[Tuple[str, str]], model_engine: Any) -> Dict[Tuple[str, str], float]:
    cache: Dict[Tuple[str, str], float] = {}
    uniq = list(dict.fromkeys(pairs))
    for a, b in tqdm(uniq, total=len(uniq), desc="MiniLM评审", unit="对", leave=False):
        cache[(a, b)] = _local_semantic_score(model_engine, a, b)
    return cache


def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval04_ai_mode_ablation")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    system_engine = SimCSEModelEngine()
    query_index, response_index, query_texts, reply_texts = load_database_columns(prefix=DB_PREFIX)
    comparator = DialogComparator(
        model_engine=system_engine,
        query_index=query_index,
        response_index=response_index,
        doc_texts=[],
        query_texts=query_texts,
        reply_texts=reply_texts,
        similarity_threshold=0.0,
        context_matching_enabled=False,
    )

    judge_engine = HFQueryAdapter(
        model_path=LOCAL_EVAL_MODEL_DIR,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    cases = _sample_cases_from_csv(SAMPLE_COUNT)

    mode_names = ["AI直接回复", "本地+AI增强", "本地检索"]
    detail: Dict[str, List[Tuple[str, float]]] = {k: [] for k in mode_names}

    for q, _ in tqdm(cases, total=len(cases), desc="模式评测", unit="条"):
        t0 = time.perf_counter()
        ai_reply = _generate_answer_direct_with_ai(q)
        detail["AI直接回复"].append((str(ai_reply), (time.perf_counter() - t0) * 1000.0))

        t1 = time.perf_counter()
        used_q, cands = comparator.get_topk_candidates(user_input=q, history=[], top_k=5)
        if cands:
            enhanced_reply = generate_ai_enhanced_reply(
                contextual_user_input=str(used_q or q),
                candidates=cands[:5],
                top_k=5,
                model_name=AI_ENHANCE_MODEL_NAME,
                responses_url=AI_ENHANCE_RESPONSES_URL,
                api_key=AI_ENHANCE_API_KEY,
                timeout_sec=AI_ENHANCE_TIMEOUT_SEC,
            )
            if not str(enhanced_reply).strip():
                enhanced_reply = str(cands[0][1].get("reply", ""))
        else:
            enhanced_reply = "知识库没有这个问题的回复"
        detail["本地+AI增强"].append((str(enhanced_reply), (time.perf_counter() - t1) * 1000.0))

        t2 = time.perf_counter()
        local_reply = str(comparator.compare(q, history=[], top_k=5)[0])
        detail["本地检索"].append((local_reply, (time.perf_counter() - t2) * 1000.0))

    rows: List[Dict[str, object]] = []
    for mode in mode_names:
        pred_gt_pairs = [(pred, gt) for (pred, _), (_, gt) in zip(detail[mode], cases)]
        score_map = _judge_pairs(pred_gt_pairs, model_engine=judge_engine)
        sims = [score_map[(pred, gt)] for pred, gt in pred_gt_pairs]
        times = [ms for _, ms in detail[mode]]
        rows.append(
            {
                "模式": mode,
                "平均准确性": float(np.mean(np.asarray(sims, dtype=np.float64))) if sims else 0.0,
                "平均耗时ms": float(np.mean(np.asarray(times, dtype=np.float64))) if times else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    _bar_plot(df.sort_values("平均准确性", ascending=False), "模式", "平均准确性", "准确性对比(MiniLM本地语义评分)", os.path.join(out_dir, "accuracy_compare.png"), "#2a9d8f")
    _bar_plot(df.sort_values("平均耗时ms", ascending=True), "模式", "平均耗时ms", "反应时间对比(平均耗时ms)", os.path.join(out_dir, "speed_compare.png"), "#3b82f6")


if __name__ == "__main__":
    run_eval()
