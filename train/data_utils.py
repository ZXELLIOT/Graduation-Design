"""
train/data_utils.py
数据管理：CSV 读取 + Dataset 构建。
"""
import pandas as pd
from torch.utils.data import Dataset

class QueryDataset(Dataset):
    """仅问句数据集。"""
    def __init__(self, queries):
        self.queries = queries
    def __len__(self):
        return len(self.queries)
    def __getitem__(self, idx):
        return self.queries[idx]

class TripletDataset(Dataset):
    """三元组数据集：(query, response, negative_response)。"""
    def __init__(self, queries, responses, negatives):
        self.queries = queries
        self.responses = responses
        self.negatives = negatives
    def __len__(self):
        return len(self.queries)
    def __getitem__(self, idx):
        return self.queries[idx], self.responses[idx], self.negatives[idx]

class DataManager:
    """从CSV加载并构建Dataset。"""

    @staticmethod
    def load_both(csv_path, nrows=None):
        """一次读取CSV，同时返回 QueryDataset 和 TripletDataset。"""
        print(f'加载数据: {csv_path} (nrows={nrows})')
        df = pd.read_csv(csv_path, nrows=nrows)[['query', 'response', 'negative_response']].fillna('')
        queries = df['query'].astype(str).tolist()
        responses = df['response'].astype(str).tolist()
        negatives = df['negative_response'].astype(str).tolist()
        return QueryDataset(queries), TripletDataset(queries, responses, negatives)