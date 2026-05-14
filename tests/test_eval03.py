"""
实验三：AI 增强模式消融实验

评测目标:
    对比三种回复生成策略的输出效果和响应耗时，通过典型案例的定性分析
    揭示本地检索与 AI 增强各自的优劣。

三种策略:
    策略A (AI直接回复):  纯大模型生成，不依赖知识库
    策略B (本地检索+AI增强): 双塔检索 top-5 + 大模型融合润色
    策略C (纯本地检索):  仅用双塔语义匹配，从知识库返回最相似答句

快速化: API 调用通过 ThreadPoolExecutor 以 8 并发执行

测试用例: eval_pairs.csv 前 30 条
输出: eval03_cases_table.csv, eval03_time_table.csv, speed_compare.png
"""

import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.ai_enhancer import generate_ai_enhanced_reply
from system.bootstrap import load_faiss_index, load_text_store
from system.comparator import DialogComparator
from system.config import (
    AI_ENHANCE_API_KEY, AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL, AI_ENHANCE_TIMEOUT_SEC,
)
from system.model_engine import SimCSEModelEngine
from tests.tests_config import TEST_DATA_DIR, TEST_RESULTS_DIR

# ============================================================
# 参数
# ============================================================
EVAL_PAIR_COUNT = 30
CORPUS_SIZE = 10_000
RERANK_TOP_K = 5
BATCH_SIZE = 64
API_CONCURRENCY = 8
API_TIMEOUT = 25

# ============================================================
# API
# ============================================================
def _extract_text(resp_json: dict) -> str:
    t = resp_json.get("output_text", "")
    if isinstance(t, str) and t.strip(): return t.strip()
    for item in resp_json.get("output", []):
        for part in (item.get("content", []) if isinstance(item, dict) else []):
            txt = part.get("text", "")
            if isinstance(txt, str) and txt.strip(): return txt.strip()
    return ""


def _call_api(prompt: str, max_retries: int = 2) -> Tuple[str, float]:
    t0 = time.perf_counter()
    payload = {"model": AI_ENHANCE_MODEL_NAME,
               "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    headers = {"Authorization": f"Bearer {AI_ENHANCE_API_KEY}",
               "Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(AI_ENHANCE_RESPONSES_URL, headers=headers,
                                 json=payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            text = _extract_text(resp.json())
            return (text if text else "[空]"), (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            if attempt < max_retries: time.sleep(1.0)
    return f"[失败]", (time.perf_counter() - t0) * 1000.0

# ============================================================
# 图表
# ============================================================
def _setup_plt():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 主评测
# ============================================================
def run_eval() -> None:
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(TEST_RESULTS_DIR, "eval03_ai_ablation")
    os.makedirs(out_dir, exist_ok=True)
    _setup_plt()

    print("=" * 60)
    print("实验三: AI 增强模式消融实验")
    print(f"测试用例: {EVAL_PAIR_COUNT} 条  |  API 并发: {API_CONCURRENCY}")
    print("=" * 60)

    # 加载数据
    eval_path = os.path.join(TEST_DATA_DIR, "eval_pairs.csv")
    eval_q, eval_exp = [], []
    for i, chunk in enumerate(pd.read_csv(eval_path, chunksize=1)):
        if i >= EVAL_PAIR_COUNT: break
        row = chunk.iloc[0]
        eval_q.append(str(row["query"]))
        eval_exp.append(str(row["response"]))

    if not AI_ENHANCE_API_KEY:
        print("[WARN] AI_ENHANCE_API_KEY 未配置")

    # 初始化
    engine = SimCSEModelEngine()
    full_idx = load_faiss_index()
    total_n = int(getattr(full_idx, "ntotal", CORPUS_SIZE))
    n_rows = min(CORPUS_SIZE, total_n)
    if hasattr(full_idx, "reconstruct_n"):
        vecs = np.asarray(full_idx.reconstruct_n(0, n_rows), dtype=np.float32)
    else:
        vecs = np.asarray([full_idx.reconstruct(i) for i in range(n_rows)], dtype=np.float32)
    dim = int(vecs.shape[1])
    m = int(full_idx.hnsw.nb_neighbors(1)) if hasattr(full_idx, "hnsw") else 32
    ef = int(full_idx.hnsw.efSearch) if hasattr(full_idx, "hnsw") else 16
    small_idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    small_idx.hnsw.efSearch = ef
    small_idx.add(vecs)
    text_store = load_text_store(max_rows=n_rows)
    comparator = DialogComparator(
        model_engine=engine, query_index=small_idx, text_store=text_store,
        similarity_threshold=0.0, context_matching_enabled=False,
        rerank_top_k=RERANK_TOP_K,
    )

    # ---- Step 1: 策略C 纯本地 ----
    print("\n[1/3] 策略C: 纯本地检索 ...")
    replies_c, times_c = [], []
    for q in eval_q:
        t0 = time.perf_counter()
        r = str(comparator.compare_with_meta(q, history=[], top_k=1)[0])
        replies_c.append(r)
        times_c.append((time.perf_counter() - t0) * 1000.0)

    # ---- Step 2: 策略A/B 并发 API ----
    print("[2/3] 策略A/B API 并发调用 ...")
    # B 先取候选
    b_cands = []
    for q in eval_q:
        _, cands, _ = comparator.get_topk_candidates_with_meta(
            user_input=q, history=[], top_k=RERANK_TOP_K)
        b_cands.append(cands[:RERANK_TOP_K] if cands else [])

    api_tasks = []
    for i, q in enumerate(eval_q):
        prompt_a = ("你是中文对话助手。请基于用户输入给出自然、简洁的一句话回复。不要解释。"
                    f"\n用户输入: {q}")
        api_tasks.append(("A", i, prompt_a))
    for i, q in enumerate(eval_q):
        if b_cands[i]:
            top_r = [item[1].get("reply", "") for item in b_cands[i][:3]]
            prompt_b = (
                "你是中文助手。请在候选中选最合适答复并润色成一句自然中文。"
                "如果你觉得这些候选回答都不好，请按你的想法直接生成更好的回答。"
                "只输出最终答复，不要解释。"
                f"\n用户输入:{q}"
                f"\n候选:\n" + "\n".join(f"{idx}. {r}" for idx, r in enumerate(top_r, 1))
            )
        else:
            prompt_b = prompt_a
        api_tasks.append(("B", i, prompt_b))

    results_a: Dict[int, Tuple[str, float]] = {}
    results_b: Dict[int, Tuple[str, float]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as ex:
        futures = {ex.submit(_call_api, t[2]): t for t in api_tasks}
        for future in as_completed(futures):
            tag, idx, _ = futures[future]
            try:
                text, ms = future.result()
            except Exception as e:
                text, ms = f"[异常:{e}]", 0.0
            if tag == "A": results_a[idx] = (text, ms)
            else: results_b[idx] = (text, ms)
            done += 1
            if done % 10 == 0: print(f"  API: {done}/{len(api_tasks)}")

    # ---- 整理结果 ----
    replies_a, times_a = [], []
    replies_b, times_b = [], []
    for i, q in enumerate(eval_q):
        ra, ta = results_a.get(i, ("[缺失]", 0.0))
        replies_a.append(ra); times_a.append(ta)
        rb, tb_api = results_b.get(i, ("[缺失]", 0.0))
        t0 = time.perf_counter()
        comparator.get_topk_candidates_with_meta(user_input=q, history=[], top_k=RERANK_TOP_K)
        tb_local = (time.perf_counter() - t0) * 1000.0
        replies_b.append(rb); times_b.append(tb_local + tb_api)

    # ---- Step 3: 输出 ----
    print("[3/3] 输出结果 ...")
    cases_df = pd.DataFrame({
        "序号": range(1, EVAL_PAIR_COUNT + 1),
        "输入问句": eval_q, "预期答句": eval_exp,
        "AI直接回复_输出": replies_a, "AI直接回复_耗时ms": [round(t, 1) for t in times_a],
        "本地+AI增强_输出": replies_b, "本地+AI增强_耗时ms": [round(t, 1) for t in times_b],
        "纯本地检索_输出": replies_c, "纯本地检索_耗时ms": [round(t, 1) for t in times_c],
    })
    cases_df.to_csv(os.path.join(out_dir, "eval03_cases_table.csv"),
                    index=False, encoding="utf-8-sig")

    # 时间统计
    time_data = []
    for name, times in [("AI直接回复", times_a), ("本地检索+AI增强", times_b), ("纯本地检索", times_c)]:
        arr = np.asarray(times)
        time_data.append({"策略": name, "平均耗时ms": round(float(np.mean(arr)), 1),
                          "最小ms": round(float(np.min(arr)), 1),
                          "最大ms": round(float(np.max(arr)), 1),
                          "标准差ms": round(float(np.std(arr)), 1)})
    time_df = pd.DataFrame(time_data)
    time_df.to_csv(os.path.join(out_dir, "eval03_time_table.csv"), index=False, encoding="utf-8-sig")

    print("\n===== 耗时 =====")
    print(time_df.to_string(index=False))

    # 打印典型案例 (选 6 条差异明显的)
    print("\n===== 典型用例 =====")
    interesting = []
    for i in range(EVAL_PAIR_COUNT):
        ta = times_a[i]; tb = times_b[i]; tc = times_c[i]
        ra = str(replies_a[i])[:80]; rb = str(replies_b[i])[:80]; rc = str(replies_c[i])[:80]
        # 选三种回复都不相同的, 或 AI 回复失败的
        scores = len(set([ra, rb, rc]))
        if scores >= 2 or "[失败" in rb or "[失败" in ra:
            interesting.append((i, scores))
    interesting.sort(key=lambda x: -x[1])
    for idx, _ in interesting[:8]:
        print(f"\nCase {idx+1}: {eval_q[idx][:60]}")
        print(f"  Expected: {eval_exp[idx][:80]}")
        print(f"  A (AI直接, {times_a[idx]:.0f}ms): {str(replies_a[idx])[:100]}")
        print(f"  B (AI增强, {times_b[idx]:.0f}ms): {str(replies_b[idx])[:100]}")
        print(f"  C (纯检索, {times_c[idx]:.0f}ms): {str(replies_c[idx])[:100]}")

    # 耗时柱状图
    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = ["AI直接回复", "本地检索+AI增强", "纯本地检索"]
    vals = [float(time_df[time_df["策略"] == n]["平均耗时ms"].iloc[0]) for n in names]
    colors = ["#ef4444", "#f59e0b", "#2a9d8f"]
    bars = ax.bar(names, vals, color=colors, edgecolor="#1f2a37", width=0.45)
    ax.set_ylabel("平均耗时 (ms)", fontsize=11)
    ax.set_title("三种回复策略耗时对比", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, float(v) + max(10, float(v)*0.02),
                f"{float(v):.0f}", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_compare.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    run_eval()
