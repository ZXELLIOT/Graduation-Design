import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pandas as pd
from src.config import RAW_CORPUS_PATH, CORPUS_PATH

class TextPreprocessor:
    def __init__(self, raw_path=RAW_CORPUS_PATH, output_path=CORPUS_PATH):
        self.raw_path = raw_path
        self.output_path = output_path
        # 匹配URL网址
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # 匹配常见的颜文字和简单的特殊符号（追加了对 + 号等常见符号的保留）
        self.special_char_pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、~（）()<>《》\-+*&#@%：:]')

    def clean_text(self, text):
        """
        去噪
        """
        if not isinstance(text, str):
            return ""
        # 1. 去掉句子中的网址
        text = self.url_pattern.sub('', text)
        # 2. 去除异常的颜文字、表情符和非法符号（仅保留中英文、数字及基础标点）
        text = self.special_char_pattern.sub('', text)
        # 3. 去除剩下的标点符号（比如清洗完颜文字后可能只剩下括号等）
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', text):
            return ""
        # 4. 把中间多个连续空格压缩为一个
        text = re.sub(r'\s+', ' ', text)
        # 5. 去除两端多余空格
        text = text.strip()
        return text

    def is_valid_sentence(self, text, min_length=2, max_length=128):
        """
        判断清理后的句子是否符合长度要求
        """
        if len(text) < min_length:
            return False
        if len(text) > max_length:
            return False
        # 如果全是数字，也认为是无效的对话
        if text.isdigit():
            return False
        return True

    def process(self):
        print(f"开始处理原始数据：{self.raw_path}")
        queries = []
        replies = []
        # 1. 读取原始数据
        try:
            with open(self.raw_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("未找到原始数据文件！请检查路径。")
            return False
        # 2. 逐行清洗与规则过滤
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                raw_q, raw_r = parts[0], parts[1]
                
                # 移除数据集中因 Tokenizer 产生的分词空格
                clean_q = raw_q.replace(" ", "").strip()
                clean_r = raw_r.replace(" ", "").strip()
                
                # 校验非空并保证对话具有基础长度
                if len(clean_q) >= 2 and len(clean_r) > 0:
                    queries.append(clean_q)
                    replies.append(clean_r)
        # 3. 构建 DataFrame 并进行数据集去重
        df = pd.DataFrame({'query': queries, 'reply': replies})
        original_count = len(df)
        # 剔除完全重复的问答
        df.drop_duplicates(subset=['query'], keep='first', inplace=True)
        final_count = len(df)
        # 4. 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # 5. 保存为标准化的 CSV
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        print("\n--- 语料预处理报告 ---")
        print(f"原始有效行数: {original_count}")
        print(f"去重后行数: {final_count}")
        print(f"已过滤冗余与异常数据: {original_count - final_count} 条")
        print(f"处理完成！标准语料库已保存至: {self.output_path}")
        return True

if __name__ == "__main__":
    processor = TextPreprocessor()
    processor.process()