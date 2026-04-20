import csv
import json
import os
import sys
import re
import time
import argparse
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_server import (
    TRAIN_JSON_PATH,
    VALID_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_NEG_CSV_PATH,
    VALID_NEG_CSV_PATH,
    TEST_NEG_CSV_PATH,
    LARGE_JSON_PATH,
    LARGE_CSV_PATH
)

def clean_text(text: str):
    """
    对输入的原始对话文本进行标准化清洗。
    
    1. 去除首尾空格。
    2. 合并中文汉字之间多余的空格，保持文本连贯。
    3. 剔除过短或过长的语句。
    4. 剔除包含网址、图片标记或纯符号的无效内容。
    
    参数:
        text: 待处理的字符串。
    返回:
        清洗后的文本或者在内容无效时返回空。
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    # 自动识别并合并汉字间的空格
    text = re.sub(
        r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef](?:\s+[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])*',
        lambda m: m.group(0).replace(' ', ''),
        text,
    )

    # 规范句长：过短通常由于回复不充分，过长不便于显示与计算
    if len(text) < 2 or len(text) > 128:
        return None
    # 剔除常见的非对话干扰内容
    if re.search(r'http|www|\[img\]|\[URL\]|\[.*?\]', text):
        return None
    if re.match(r'^[\W_]+$', text):
        return None

    # 合并多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    return text


def load_first_pairs_from_json(json_path: str, stage_name: str) -> List[Tuple[str, str]]:
    """
    从原始数据文件中读取会话记录，并提取出每一组会话的前两轮对话。
    该操作主要用于从多轮对话中提取典型的“问-答”关系对。
    
    参数:
        json_path: 原始数据文件的位置。
        stage_name: 当前处理的阶段名称（如 训练集/测试集）。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到指定的原始数据文件: {json_path}")

    load_start = time.perf_counter()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[{stage_name}] 原始数据加载成功，总对话条数: {len(data):,}，加载用时: {time.perf_counter() - load_start:.2f}秒")

    pairs: List[Tuple[str, str]] = []
    # 遍历每个会话，提取前两个元素作为初筛后的问答对
    for dialog_session in tqdm(data, desc=f"{stage_name} 正在进行清洗与提取"):
        if not isinstance(dialog_session, list) or len(dialog_session) < 2:
            continue

        clean_q = clean_text(dialog_session[0])
        clean_a = clean_text(dialog_session[1])
        if clean_q and clean_a:
            pairs.append((clean_q, clean_a))

    print(f"[{stage_name}] 提取结束。共获得有效对话对: {len(pairs):,}")
    return pairs


def write_csv(
    pairs: List[Tuple[str, str]],
    output_csv_path: str,
    with_negative: bool,
    seed: int = 42,
):
    """统一写出 CSV。

    - with_negative=True: 输出三列 query/response/negative_response（训练用）。
    - with_negative=False: 输出两列 query/response（数据库语料用）。
    """
    if len(pairs) == 0:
        raise ValueError("数据源为空，无法生成结果文件。")

    queries = [x[0] for x in pairs]
    responses = [x[1] for x in pairs]
    n = len(responses)

    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if with_negative:
            # 训练表格结构：提问、正确回答、无关回答
            rng = np.random.default_rng(seed)
            neg_idx = rng.integers(0, n, size=n, dtype=np.int64)
            idx_arr = np.arange(n, dtype=np.int64)
            clash = neg_idx == idx_arr
            while np.any(clash):
                neg_idx[clash] = rng.integers(0, n, size=int(np.sum(clash)), dtype=np.int64)
                clash = neg_idx == idx_arr

            writer.writerow(["query", "response", "negative_response"])
            for i in tqdm(
                range(n),
                desc=f"正在生成训练结果表格 ({os.path.basename(output_csv_path)})"
            ):
                writer.writerow([queries[i], responses[i], responses[int(neg_idx[i])]])
        else:
            # 数据库语料结构：仅保留两列 query/response
            writer.writerow(["query", "response"])
            for query, response in tqdm(
                pairs,
                desc=f"正在生成两列语料库 ({os.path.basename(output_csv_path)})"
            ):
                writer.writerow([query, response])


def build_standard_datasets(
    train_json_path: str,
    valid_json_path: str,
    test_json_path: str,
    large_json_path: str,
    output_train_neg_csv: str,
    output_valid_neg_csv: str,
    output_test_neg_csv: str,
    output_large_csv: str,
    seed: int = 42,
) -> Tuple[int, int, int, int]:
    """构建CSV。"""

    total_start = time.perf_counter()
    
    train_pairs = load_first_pairs_from_json(train_json_path, "训练集")
    valid_pairs = load_first_pairs_from_json(valid_json_path, "验证集")
    test_pairs = load_first_pairs_from_json(test_json_path, "测试集")
    large_pairs = load_first_pairs_from_json(large_json_path, "语料库(LCCC-large)")

    write_csv(train_pairs, output_train_neg_csv, with_negative=True, seed=seed)
    write_csv(valid_pairs, output_valid_neg_csv, with_negative=True, seed=seed + 1)
    write_csv(test_pairs, output_test_neg_csv, with_negative=True, seed=seed + 2)
    write_csv(large_pairs, output_large_csv, with_negative=False)

    print(f"数据处理完成，总耗时: {time.perf_counter() - total_start:.2f}s")
    return len(train_pairs), len(valid_pairs), len(test_pairs), len(large_pairs)


def main():
    parser = argparse.ArgumentParser(description="数据清洗与标准输出")
    parser.add_argument("--train_json", type=str, default=TRAIN_JSON_PATH)
    parser.add_argument("--valid_json", type=str, default=VALID_JSON_PATH)
    parser.add_argument("--test_json", type=str, default=TEST_JSON_PATH)
    parser.add_argument("--large_json", type=str, default=LARGE_JSON_PATH)
    parser.add_argument("--train_neg_csv", type=str, default=TRAIN_NEG_CSV_PATH)
    parser.add_argument("--valid_neg_csv", type=str, default=VALID_NEG_CSV_PATH)
    parser.add_argument("--test_neg_csv", type=str, default=TEST_NEG_CSV_PATH)
    parser.add_argument("--large_csv", type=str, default=LARGE_CSV_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_count, valid_count, test_count, large_count = build_standard_datasets(
        train_json_path=args.train_json,
        valid_json_path=args.valid_json,
        test_json_path=args.test_json,
        large_json_path=args.large_json,
        output_train_neg_csv=args.train_neg_csv,
        output_valid_neg_csv=args.valid_neg_csv,
        output_test_neg_csv=args.test_neg_csv,
        output_large_csv=args.large_csv,
        seed=args.seed,
    )

    print("\n" + "=" * 50)
    print("数据处理完成")
    print(f"训练集: {args.train_neg_csv} ({train_count:,} 对)")
    print(f"验证集: {args.valid_neg_csv} ({valid_count:,} 对)")
    print(f"测试集: {args.test_neg_csv} ({test_count:,} 对)")
    print(f"语料库: {args.large_csv} ({large_count:,} 对)")
    print("=" * 50)

if __name__ == "__main__":
    main()