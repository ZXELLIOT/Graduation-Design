"""
train/dataset_preprocessor.py

文件作用:
    对原始对话数据进行清洗、抽取与导出。
    生成训练用正负样本 CSV 和语料库 CSV。
"""

import csv
import json
import os
import sys
import time
import numpy as np
from typing import List, Optional, Tuple
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_prep_config import (
    CORPUS_DATASET_HEADERS,
    DEFAULT_PREPROCESS_SEED,
    TRAIN_JSON_PATH,
    VALID_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_NEG_CSV_PATH,
    VALID_NEG_CSV_PATH,
    TEST_NEG_CSV_PATH,
    LARGE_JSON_PATH,
    LARGE_CSV_PATH,
    PREPROCESS_LOG_PREFIX,
    PREPROCESS_STAGE_NAMES,
    TEXT_MAX_LENGTH,
    TEXT_MIN_LENGTH,
    TRAIN_DATASET_HEADERS,
)

def log_info(message: str) -> None:
    print(f"{PREPROCESS_LOG_PREFIX} {message}")

def clean_text(text: str) -> Optional[str]:
    """
    对输入的原始对话文本进行标准化清洗。
    规则:
        1. 去除多余空格。
        2. 剔除过短或过长的语句。
    参数:
        text: 待处理的字符串。
    返回:
        清洗后的文本。
    """
    if not isinstance(text, str):
        text = str(text)
    text = "".join(text.split())
    if len(text) < TEXT_MIN_LENGTH or len(text) > TEXT_MAX_LENGTH:
        return None
    return text

def load_first_pairs_from_json(json_path: str, stage_name: str) -> List[Tuple[str, str]]:
    """
    从原始数据文件中读取数据，并提取出每一组会话的前两句作为问句和答句。
    参数:
        json_path: 原始数据文件的位置。
        stage_name: 当前处理的阶段名称（如 [训练集]/[验证集]/[测试集]/[语料库]）。
    返回:
        清洗后问答对的列表，每个元素是一个 (query, response) 元组。
    """
    load_start = time.perf_counter()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log_info(f"{stage_name} 原始数据加载成功，总对话条数: {len(data):,}，加载用时: {time.perf_counter() - load_start:.2f}秒")
    pairs: List[Tuple[str, str]] = []
    for dialog_session in tqdm(data, desc=f"{PREPROCESS_LOG_PREFIX} {stage_name} 清洗与提取", unit="条", dynamic_ncols=True):
        if not isinstance(dialog_session, list) or len(dialog_session) < 2:
            continue
        clean_q = clean_text(dialog_session[0])
        clean_a = clean_text(dialog_session[1])
        if clean_q and clean_a:
            pairs.append((clean_q, clean_a))
    log_info(f"{stage_name} 提取结束。共获得有效对话对: {len(pairs):,}")
    return pairs

def write_csv(
    pairs: List[Tuple[str, str]],
    output_csv_path: str,
    with_negative: bool,
    seed: int = DEFAULT_PREPROCESS_SEED,
):
    
    """将清洗后的样本写出到 CSV 文件。
    规则:
        1. with_negative=True: 输出三列 query/response/negative_response（训练用）。
        2. with_negative=False: 输出两列 query/response（语料库入库用）。
    参数:
        pairs: 已清洗的问答对列表。
        output_csv_path: 目标 CSV 输出路径。
        with_negative: 是否生成负样本列。
        seed: 随机种子，保证负样本可复现。
    返回:
        无返回值。
    """
    queries = [x[0] for x in pairs]
    responses = [x[1] for x in pairs]
    n = len(responses)
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if with_negative:
            rng = np.random.default_rng(seed)
            neg_idx = rng.integers(0, n, size=n, dtype=np.int64)
            idx_arr = np.arange(n, dtype=np.int64)
            clash = neg_idx == idx_arr
            while np.any(clash):
                neg_idx[clash] = rng.integers(0, n, size=int(np.sum(clash)), dtype=np.int64)
                clash = neg_idx == idx_arr
            writer.writerow(TRAIN_DATASET_HEADERS)
            for i in tqdm(
                range(n),
                desc=f"{PREPROCESS_LOG_PREFIX} 生成训练CSV ({os.path.basename(output_csv_path)})",
                unit="条",
                dynamic_ncols=True,
            ):
                writer.writerow([queries[i], responses[i], responses[int(neg_idx[i])]])
        else:
            writer.writerow(CORPUS_DATASET_HEADERS)
            for query, response in tqdm(
                pairs,
                desc=f"{PREPROCESS_LOG_PREFIX} 生成语料CSV ({os.path.basename(output_csv_path)})",
                unit="条",
                dynamic_ncols=True,
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
    seed: int = DEFAULT_PREPROCESS_SEED,
) -> Tuple[int, int, int, int]:
    
    """构建训练/验证/测试与语料库CSV。
    参数:
        train_json_path: 训练集 JSON 路径。
        valid_json_path: 验证集 JSON 路径。
        test_json_path: 测试集 JSON 路径。
        large_json_path: 大规模语料 JSON 路径。
        output_train_neg_csv: 训练集输出 CSV 路径（含负样本）。
        output_valid_neg_csv: 验证集输出 CSV 路径（含负样本）。
        output_test_neg_csv: 测试集输出 CSV 路径（含负样本）。
        output_large_csv: 语料库输出 CSV 路径（不含负样本）。
        seed: 随机种子。
    返回:
        (训练集条数, 验证集条数, 测试集条数, 语料库条数)。
    """

    total_start = time.perf_counter()
    train_pairs = load_first_pairs_from_json(train_json_path, PREPROCESS_STAGE_NAMES["train"])
    valid_pairs = load_first_pairs_from_json(valid_json_path, PREPROCESS_STAGE_NAMES["valid"])
    test_pairs = load_first_pairs_from_json(test_json_path, PREPROCESS_STAGE_NAMES["test"])
    large_pairs = load_first_pairs_from_json(large_json_path, PREPROCESS_STAGE_NAMES["large"])
    write_csv(train_pairs, output_train_neg_csv, with_negative=True, seed=seed)
    write_csv(valid_pairs, output_valid_neg_csv, with_negative=True, seed=seed + 1)
    write_csv(test_pairs, output_test_neg_csv, with_negative=True, seed=seed + 2)
    write_csv(large_pairs, output_large_csv, with_negative=False)
    log_info(f"数据处理完成，总耗时: {time.perf_counter() - total_start:.2f}s")
    return len(train_pairs), len(valid_pairs), len(test_pairs), len(large_pairs)

def main():

    """脚本入口。"""

    train_count, valid_count, test_count, large_count = build_standard_datasets(
        train_json_path=TRAIN_JSON_PATH,
        valid_json_path=VALID_JSON_PATH,
        test_json_path=TEST_JSON_PATH,
        large_json_path=LARGE_JSON_PATH,
        output_train_neg_csv=TRAIN_NEG_CSV_PATH,
        output_valid_neg_csv=VALID_NEG_CSV_PATH,
        output_test_neg_csv=TEST_NEG_CSV_PATH,
        output_large_csv=LARGE_CSV_PATH,
        seed=DEFAULT_PREPROCESS_SEED,
    )

    print("=" * 60)
    log_info("处理完成，输出汇总如下")
    log_info(f"训练集: {TRAIN_NEG_CSV_PATH} ({train_count:,} 对)")
    log_info(f"验证集: {VALID_NEG_CSV_PATH} ({valid_count:,} 对)")
    log_info(f"测试集: {TEST_NEG_CSV_PATH} ({test_count:,} 对)")
    log_info(f"语料库: {LARGE_CSV_PATH} ({large_count:,} 对)")
    print("=" * 60)

if __name__ == "__main__":
    main()