import argparse
import os
import sys
from typing import List

import torch
from tqdm.auto import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    # 保证从不同目录执行时都能找到 src。
    sys.path.append(CURRENT_DIR)

from src.data_loader import DataLoader
from src.config import (
    SIMILARITY_THRESHOLD,
    TRAIN_CORPUS_PATH,
    TEST_CORPUS_PATH,
    TRAIN_EMBEDDINGS_CACHE_PATH,
)
from src.model import SimCSEEncoder
from src.matcher import DialogMatcher


def maybe_slice_dataset(
    queries: List[str], replies: List[str], max_samples: int
):
    """按上限截断数据集，用于快速测试。"""
    if max_samples <= 0:
        return queries, replies
    keep = min(len(queries), max_samples)
    return queries[:keep], replies[:keep]


def show_two_module_result(matcher: DialogMatcher, query: str, gold_reply: str = ""):
    """展示单条输入在两个比较模块下的结果。"""
    result = matcher.evaluate_two_modules(query)
    qq = result["query_query"]
    qr = result["wordvec_qa"]

    print("\n---------------- 单条对比结果 ----------------")
    print(f"输入问句: {query}")
    if gold_reply:
        print(f"标准答句: {gold_reply}")

    print("\n[模块1] 问句-问句比较")
    print(f"得分: {qq['score']:.4f}")
    print(f"命中问句: {qq['matched_query']}")
    print(f"输出答句: {qq['reply']}")

    print("\n[模块2] 词向量问答比较")
    print(f"得分: {qr['score']:.4f}")
    print(f"命中问句: {qr['matched_query']}")
    print(f"输出答句: {qr['reply']}")

    print("\n[最终选择]")
    print(f"选中模块: {result['selected_method']}")
    print(f"最终得分: {result['selected_score']:.4f}")
    print(f"最终答句: {result['selected_reply']}")


def evaluate_performance(
    matcher: DialogMatcher,
    test_queries: List[str],
    test_replies: List[str],
    show_examples: int,
    top_k: int,
    semantic_threshold: float,
):
    """在测试集上统计性能指标。"""
    total = len(test_queries)
    if total == 0:
        raise ValueError("测试集为空")

    passed_threshold_count = 0
    selected_score_sum = 0.0
    query_query_win = 0
    wordvec_qa_win = 0
    none_count = 0
    query_query_pass_count = 0
    wordvec_qa_pass_count = 0
    topk_hit_count = 0
    semantic_hit_count = 0

    example_rows = []
    text_embedding_cache = {}

    def get_text_embedding(text: str):
        cached = text_embedding_cache.get(text)
        if cached is not None:
            return cached
        emb = matcher.encoder.encode([text])
        text_embedding_cache[text] = emb
        return emb

    for idx, (query, gold_reply) in enumerate(
        tqdm(
            zip(test_queries, test_replies),
            total=total,
            desc="性能评估进度",
            unit="sample",
            dynamic_ncols=True,
        ),
        start=1,
    ):
        result = matcher.evaluate_two_modules(query)

        selected_reply = result["selected_reply"]
        selected_method = result["selected_method"]
        selected_score = float(result["selected_score"])
        passed_threshold = bool(result["passed_threshold"])

        user_emb = matcher.encoder.encode([query])
        qq_scores = matcher._query_query_similarity(query, user_emb)
        qr_scores = matcher._query_reply_similarity(query)

        qq_best_score = float(torch.max(qq_scores).item())
        qr_best_score = float(torch.max(qr_scores).item())

        if passed_threshold:
            passed_threshold_count += 1

        selected_score_sum += selected_score

        if selected_method == "query-query":
            query_query_win += 1
        elif selected_method == "wordvec-qa":
            wordvec_qa_win += 1
        else:
            none_count += 1

        if qq_best_score >= SIMILARITY_THRESHOLD:
            query_query_pass_count += 1
        if qr_best_score >= SIMILARITY_THRESHOLD:
            wordvec_qa_pass_count += 1

        combined_scores = torch.maximum(qq_scores, qr_scores)
        k = max(1, min(int(top_k), int(combined_scores.numel())))
        top_indices = torch.topk(combined_scores, k=k).indices.tolist()
        topk_replies = [matcher.replies[int(i)] for i in top_indices]
        is_topk_hit = gold_reply in topk_replies
        if is_topk_hit:
            topk_hit_count += 1

        selected_emb = get_text_embedding(selected_reply)
        gold_emb = get_text_embedding(gold_reply)
        semantic_score = float(torch.matmul(selected_emb, gold_emb.transpose(0, 1)).item())
        is_semantic_hit = semantic_score >= semantic_threshold
        if is_semantic_hit:
            semantic_hit_count += 1

        if len(example_rows) < max(0, show_examples):
            example_rows.append(
                {
                    "query": query,
                    "gold_reply": gold_reply,
                    "selected_reply": selected_reply,
                    "selected_method": selected_method,
                    "selected_score": selected_score,
                    "passed_threshold": passed_threshold,
                    "is_topk_hit": is_topk_hit,
                    "semantic_score": semantic_score,
                    "is_semantic_hit": is_semantic_hit,
                }
            )

    threshold_pass_rate = passed_threshold_count / total
    avg_selected_score = selected_score_sum / total
    topk_hit_rate = topk_hit_count / total
    semantic_hit_rate = semantic_hit_count / total
    query_query_pass_rate = query_query_pass_count / total
    wordvec_qa_pass_rate = wordvec_qa_pass_count / total

    return {
        "total": total,
        "passed_threshold_count": passed_threshold_count,
        "threshold_pass_rate": threshold_pass_rate,
        "avg_selected_score": avg_selected_score,
        "topk_hit_count": topk_hit_count,
        "topk_hit_rate": topk_hit_rate,
        "semantic_hit_count": semantic_hit_count,
        "semantic_hit_rate": semantic_hit_rate,
        "query_query_win": query_query_win,
        "wordvec_qa_win": wordvec_qa_win,
        "none_count": none_count,
        "query_query_pass_count": query_query_pass_count,
        "query_query_pass_rate": query_query_pass_rate,
        "wordvec_qa_pass_count": wordvec_qa_pass_count,
        "wordvec_qa_pass_rate": wordvec_qa_pass_rate,
        "examples": example_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="统一测试脚本：模型性能 + 双模块比较结果")
    parser.add_argument("--max-test-samples", type=int, default=500, help="最多评估多少条测试样本，0 表示不限制")
    parser.add_argument("--max-index-samples", type=int, default=50000, help="检索库最多保留多少条，0 表示不限制")
    parser.add_argument("--show-examples", type=int, default=3, help="输出前几条样例")
    parser.add_argument("--compare-samples", type=int, default=3, help="额外展示多少条双模块比较结果")
    parser.add_argument("--query", type=str, default="", help="指定一条问句进行双模块比较")
    parser.add_argument("--use-cache", action="store_true", help="启用向量缓存（默认关闭，避免测试时反复更新缓存）")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K 命中率中的 K")
    parser.add_argument("--semantic-threshold", type=float, default=0.7, help="语义命中阈值（余弦分数）")
    parser.add_argument("--skip-performance", action="store_true", help="跳过整体验证，只看双模块比较")
    parser.add_argument("--skip-compare", action="store_true", help="跳过双模块比较，只看性能指标")
    return parser.parse_args()


def main():
    args = parse_args()

    train_loader = DataLoader(file_path=TRAIN_CORPUS_PATH)
    train_queries, train_replies = train_loader.load_corpus()
    test_loader = DataLoader(file_path=TEST_CORPUS_PATH)
    test_queries, test_replies = test_loader.load_corpus()

    if not train_queries:
        print("训练检索库为空，无法评估。")
        return
    if not test_queries:
        print("测试集为空，无法评估。")
        return

    sampled_train_queries, sampled_train_replies = maybe_slice_dataset(
        queries=train_queries,
        replies=train_replies,
        max_samples=args.max_index_samples,
    )
    sampled_test_queries, sampled_test_replies = maybe_slice_dataset(
        queries=test_queries,
        replies=test_replies,
        max_samples=args.max_test_samples,
    )

    print("\n================= 数据划分 =================")
    print(f"训练库样本数: {len(train_queries)}")
    print(f"实际检索库样本数: {len(sampled_train_queries)}")
    print(f"测试集样本数: {len(sampled_test_queries)}")

    print("\n================= 加载模型并构建检索器 =================")
    encoder = SimCSEEncoder()
    matcher = DialogMatcher(
        encoder,
        sampled_train_queries,
        sampled_train_replies,
        use_cache=args.use_cache,
        cache_path=TRAIN_EMBEDDINGS_CACHE_PATH,
    )
    print(f"缓存开关: {'开启' if args.use_cache else '关闭'}")

    if not args.skip_performance:
        print("\n================= 开始性能评估 =================")
        metrics = evaluate_performance(
            matcher=matcher,
            test_queries=sampled_test_queries,
            test_replies=sampled_test_replies,
            show_examples=args.show_examples,
            top_k=args.top_k,
            semantic_threshold=args.semantic_threshold,
        )

        print("\n================= 性能报告 =================")
        print(f"测试样本总数: {metrics['total']}")
        print(f"通过阈值数: {metrics['passed_threshold_count']}")
        print(f"通过阈值比例: {metrics['threshold_pass_rate']:.4f}")
        print(f"Top-{args.top_k} 命中数: {metrics['topk_hit_count']}")
        print(f"Top-{args.top_k} 命中率: {metrics['topk_hit_rate']:.4f}")
        print(f"语义命中数(阈值={args.semantic_threshold}): {metrics['semantic_hit_count']}")
        print(f"语义命中率: {metrics['semantic_hit_rate']:.4f}")
        print(f"平均最终分数: {metrics['avg_selected_score']:.4f}")
        print(f"模块通过次数(query-query): {metrics['query_query_pass_count']}")
        print(f"模块通过比例(query-query): {metrics['query_query_pass_rate']:.4f}")
        print(f"模块通过次数(wordvec-qa): {metrics['wordvec_qa_pass_count']}")
        print(f"模块通过比例(wordvec-qa): {metrics['wordvec_qa_pass_rate']:.4f}")
        print(f"模块胜出次数(query-query): {metrics['query_query_win']}")
        print(f"模块胜出次数(wordvec-qa): {metrics['wordvec_qa_win']}")
        print(f"无有效模块次数: {metrics['none_count']}")

        if metrics["examples"]:
            print("\n================= 样例展示 =================")
            for i, row in enumerate(metrics["examples"], start=1):
                print(f"\n样例 {i}")
                print(f"问句: {row['query']}")
                print(f"标准答句: {row['gold_reply']}")
                print(f"系统答句: {row['selected_reply']}")
                print(f"模块: {row['selected_method']}")
                print(f"分数: {row['selected_score']:.4f}")
                print(f"通过阈值: {row['passed_threshold']}")
                print(f"是否 Top-{args.top_k} 命中: {row['is_topk_hit']}")
                print(f"语义分数: {row['semantic_score']:.4f}")
                print(f"是否语义命中: {row['is_semantic_hit']}")

    if not args.skip_compare:
        print("\n================= 双模块比较展示 =================")
        if args.query.strip():
            show_two_module_result(matcher, args.query.strip())
        else:
            show_count = max(1, min(args.compare_samples, len(sampled_test_queries)))
            for i in range(show_count):
                show_two_module_result(
                    matcher=matcher,
                    query=sampled_test_queries[i],
                    gold_reply=sampled_test_replies[i],
                )


if __name__ == "__main__":
    main()
