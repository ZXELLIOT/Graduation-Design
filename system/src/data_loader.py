import pandas as pd
from src.config import CORPUS_PATH

class DataLoader:
    def __init__(self, file_path=CORPUS_PATH):
        self.file_path = file_path

    def load_corpus(self):
        """读取对话语料"""
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8')
            queries = df['query'].tolist()
            replies = df['reply'].tolist()
            print(f"成功加载语料库，共 {len(queries)} 条对话数据。")
            return queries, replies
        except Exception as e:
            print(f"语料库加载失败。\n错误信息: {e}")
            return [], []