"""
db/encoder_module.py
问句向量化 → FAISS 索引入库。向量位置 = CSV 行号。
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from system.model_engine import SimCSEModelEngine
from db.db_config import DB_CSV_PATH, DB_QUERY_INDEX_FILE

def build_index(csv_path=DB_CSV_PATH, index_path=DB_QUERY_INDEX_FILE, n_samples=None, batch_size=128):
    """读取 CSV 问句列 → query_encoder 编码 → 写入 FAISS 索引。"""
    engine = SimCSEModelEngine()
    nrows = n_samples if n_samples and n_samples > 0 else None
    df = pd.read_csv(csv_path, usecols=["query"], nrows=nrows).fillna("")
    queries = df["query"].astype(str).tolist()
    vecs = engine.encode(queries, encoder="query", batch_size=batch_size, return_numpy=True)
    vecs = np.asarray(vecs, dtype="float32")
    dim = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"编码完成: {index_path} ({len(queries)} 条)")

if __name__ == "__main__":
    build_index(n_samples=10, batch_size=64)