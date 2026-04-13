import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import pandas as pd
from tqdm.auto import tqdm
from src.config import RAW_TRAIN_PATH, RAW_TEST_PATH, TRAIN_CORPUS_PATH, TEST_CORPUS_PATH

class TextPreprocessor:
    def __init__(self, raw_path=RAW_TRAIN_PATH, output_path=TRAIN_CORPUS_PATH):
        # 输入是原始文本，输出是标准化后的 csv。
        self.raw_path = raw_path
        self.output_path = output_path
        # 识别网址，后续用于删除。
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # 仅保留常用中文、英文、数字和基础标点，其余字符清理掉。
        self.special_char_pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、~（）()<>《》\-+*&#@%：:]')

    def _count_lines(self):
        """统计文件总行数，用于进度条百分比显示。"""
        total = 0
        with open(self.raw_path, "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
        return total

    def clean_text(self, text):
        """
        文本清洗：去网址、去异常符号、压缩空白。
        """
        if not isinstance(text, str):
            return ""
        # 1. 去掉句子中的网址
        text = self.url_pattern.sub('', text)
        # 2. 去除异常的颜文字、表情符和非法符号（仅保留中英文、数字及基础标点）
        text = self.special_char_pattern.sub('', text)
        # 3. 去除剩下的标点符号（比如清洗完颜文字后可能只剩下括号等）
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', text):
            # 如果清洗后不含任何有效文字或数字，直接丢弃。
            return ""
        # 4. 把中间多个连续空格压缩为一个
        text = re.sub(r'\s+', ' ', text)
        # 5. 去除两端多余空格
        text = text.strip()
        return text

    def is_valid_sentence(self, text, min_length=2, max_length=128):
        """
        判断清理后的句子是否可用于训练。
        """
        if len(text) < min_length:
            return False
        if len(text) > max_length:
            return False
        # 全数字通常不具备对话语义，这里过滤掉。
        if text.isdigit():
            return False
        return True

    def process(self):
        """读取原始文本并生成标准语料文件。"""
        print(f"开始处理原始数据：{self.raw_path}")
        total_start = time.perf_counter()
        queries = []
        replies = []
        # 1. 读取原始数据
        try:
            f = open(self.raw_path, 'r', encoding='utf-8')
        except FileNotFoundError:
            print("未找到原始数据文件！请检查路径。")
            return False

        # 2) 先统计总行数，这样进度条能显示百分比和剩余时间。
        count_start = time.perf_counter()
        total_lines = self._count_lines()
        count_elapsed = time.perf_counter() - count_start
        print(f"原始文件总行数: {total_lines}（统计耗时 {count_elapsed:.2f} 秒）")

        # 3) 按行处理，拆分问句和答句
        process_start = time.perf_counter()
        with f:
            for line in tqdm(
                f,
                total=total_lines,
                desc="预处理进度",
                unit="line",
                dynamic_ncols=True,
            ):
                # 原始行格式约定为：问句\t答句
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    raw_q, raw_r = parts[0], parts[1]

                    # 去掉分词空格后再做统一清洗
                    clean_q = self.clean_text(raw_q.replace(" ", ""))
                    clean_r = self.clean_text(raw_r.replace(" ", ""))

                    # 统一长度和内容过滤
                    if self.is_valid_sentence(clean_q) and self.is_valid_sentence(clean_r):
                        queries.append(clean_q)
                        replies.append(clean_r)
        process_elapsed = time.perf_counter() - process_start

        # 3) 组装表格并去重
        # query/reply 一起去重，可以保留“同问不同答”的数据。
        df = pd.DataFrame({'query': queries, 'reply': replies})
        original_count = len(df)
        # 以 query+reply 共同去重，保留一问多答的信息
        df.drop_duplicates(subset=['query', 'reply'], keep='first', inplace=True)
        final_count = len(df)
        # 4. 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # 5. 保存为标准化的 CSV
        save_start = time.perf_counter()
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        save_elapsed = time.perf_counter() - save_start
        total_elapsed = time.perf_counter() - total_start
        print("\n--- 语料预处理报告 ---")
        print(f"原始有效行数: {original_count}")
        print(f"去重后行数: {final_count}")
        print(f"已过滤冗余与异常数据: {original_count - final_count} 条")
        print(f"逐行处理耗时: {process_elapsed:.2f} 秒")
        print(f"CSV 保存耗时: {save_elapsed:.2f} 秒")
        print(f"总耗时: {total_elapsed:.2f} 秒")
        print(f"处理完成！标准语料库已保存至: {self.output_path}")
        return True


def process_train_test_corpus():
    """分别处理训练集和测试集原始语料。"""
    print("\n================ 处理训练集语料 ================")
    train_ok = TextPreprocessor(raw_path=RAW_TRAIN_PATH, output_path=TRAIN_CORPUS_PATH).process()

    print("\n================ 处理测试集语料 ================")
    test_ok = TextPreprocessor(raw_path=RAW_TEST_PATH, output_path=TEST_CORPUS_PATH).process()

    return bool(train_ok and test_ok)

if __name__ == "__main__":
    process_train_test_corpus()