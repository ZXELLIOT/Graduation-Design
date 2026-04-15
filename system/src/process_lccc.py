import os
import json
import time
import csv
import re
from tqdm.auto import tqdm

LCCC_BASE_DIR = r"C:\\Users\\13713\\个人信息\\毕业设计\\LCCC-base-split"
TRAIN_JSON_PATH = os.path.join(LCCC_BASE_DIR, "LCCC-base_train.json")
VALID_JSON_PATH = os.path.join(LCCC_BASE_DIR, "LCCC-base_valid.json")   
TEST_JSON_PATH = os.path.join(LCCC_BASE_DIR, "LCCC-base_test.json")

OUTPUT_DIR = r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_TRAIN_CSV = os.path.join(OUTPUT_DIR, "lccc_train.csv")
OUTPUT_VALID_CSV = os.path.join(OUTPUT_DIR, "lccc_valid.csv")
OUTPUT_TEST_CSV = os.path.join(OUTPUT_DIR, "lccc_test.csv")

def clean_text(text):
    """
    清洗文本：
    1. 去除中文词语与标点符号间的多余空格 (包括连续单字空格)
    2. 去除无效字符、URL、过短或过长的句子
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 去除首尾空白
    text = text.strip()
    
    # 2. 匹配一个或多个由空格分隔的中文字符或标点符号序列，并移除其中的空格
    text = re.sub(
        r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef](?:\s+[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])*',
        lambda m: m.group(0).replace(' ', ''),
        text
    )
    
    # 3. 长度过滤
    if len(text) < 2 or len(text) > 128: 
        return None
        
    # 4. 去除 URL 和 HTML 残留标签
    if re.search(r'http|www|\[img\]|\[URL\]|\[.*?\]', text):
        return None
        
    # 5. 去除纯符号或无意义字符
    if re.match(r'^[\W_]+$', text):
        return None
        
    # 6. 规范化剩余空白字符（将多个空格或换行合并为一个空格）
    text = re.sub(r'\s+', ' ', text)
    
    return text

def _process_json_to_csv(json_path, output_csv_path, stage_name):
    """
    读取单个 JSON 文件，清洗数据，并保存为 CSV
    """
    if not os.path.exists(json_path):
        print(f"文件不存在 - {json_path}")
        return 0

    try:
        load_start = time.perf_counter()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        load_elapsed = time.perf_counter() - load_start
        print(f"[{stage_name}] JSON 读取完成，Session 数: {len(data):,}，耗时: {load_elapsed:.2f}s")
    except json.JSONDecodeError:
        print(f"错误：JSON 解析失败 - {json_path}")
        return 0

    extract_start = time.perf_counter()
    valid_pair_count = 0
    
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "response"])
        
        for dialog_session in tqdm(
            data, 
            desc=f"{stage_name} 处理进度", 
            unit="session", 
            dynamic_ncols=True
        ):
            if not isinstance(dialog_session, list) or len(dialog_session) < 2:
                continue

            for idx in range(len(dialog_session) - 1):
                raw_q = dialog_session[idx]
                raw_a = dialog_session[idx + 1]

                clean_q = clean_text(raw_q)
                clean_a = clean_text(raw_a)

                if clean_q and clean_a:
                    writer.writerow([clean_q, clean_a])
                    valid_pair_count += 1
                    

    extract_elapsed = time.perf_counter() - extract_start
    print(f"[{stage_name}] 完成。有效对: {valid_pair_count:,}，耗时: {extract_elapsed:.2f}s")
    return valid_pair_count

def main():
    print(f"正在提取数据: {LCCC_BASE_DIR}")
    total_start = time.perf_counter()
    
    train_count = _process_json_to_csv(TRAIN_JSON_PATH, OUTPUT_TRAIN_CSV, "训练集")
    valid_count = _process_json_to_csv(VALID_JSON_PATH, OUTPUT_VALID_CSV, "验证集")
    test_count = _process_json_to_csv(TEST_JSON_PATH, OUTPUT_TEST_CSV, "测试集")
    
    total_elapsed = time.perf_counter() - total_start
    
    print("\n" + "="*50)
    print("数据处理完成")
    print(f"训练集: {OUTPUT_TRAIN_CSV} ({train_count:,} 对)")
    print(f"验证集: {OUTPUT_VALID_CSV} ({valid_count:,} 对)")
    print(f"测试集: {OUTPUT_TEST_CSV} ({test_count:,} 对)")
    print(f"总耗时: {total_elapsed:.2f} 秒")
    print("="*50)

if __name__ == "__main__":
    main()