import pandas as pd
from src.config import SIMILARITY_CORPUS_PATH


class DataLoader:
    def __init__(self, file_path=SIMILARITY_CORPUS_PATH):
        # 默认读取预处理后生成的标准语料。
        self.file_path = file_path

    def load_corpus(self):
        """读取对话语料，要求至少包含 query 和 reply 两列。"""
        try:
            # 统一按 utf-8 读取，避免中文乱码。
            df = pd.read_csv(self.file_path, encoding='utf-8')
            required_columns = {"query", "reply"}
            if not required_columns.issubset(df.columns):
                # 列名不满足要求时直接报错，避免后面静默失败。
                raise ValueError(f"缺少必要列: {required_columns - set(df.columns)}")

            # 保持行顺序不变，直接拆成两个列表供匹配模块使用。
            queries = df['query'].tolist()
            replies = df['reply'].tolist()
            print(f"成功加载语料库，共 {len(queries)} 条对话数据。")
            return queries, replies
        except Exception as e:
            # 统一兜底返回空列表，让上层决定是否中断流程。
            print(f"语料库加载失败。\n错误信息: {e}")
            return [], []