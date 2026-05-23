"""验证 FAISS 索引是否正常：加载索引 → 搜索召回 → 输出结果。"""

import os, sys, numpy as np, faiss, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_config import DB_CSV_PATH, DB_QUERY_INDEX_FILE
from system.model_engine import SimCSEModelEngine

index = faiss.read_index(DB_QUERY_INDEX_FILE)
df = pd.read_csv(DB_CSV_PATH, nrows=index.ntotal)
engine = SimCSEModelEngine()

print(f"索引: {type(index).__name__}  维度: {index.d}  条数: {index.ntotal}  CSV: {len(df)} 行\n")

q = str(df.iloc[0]["query"])
vec = np.asarray(engine.encode_one(q, encoder="query", return_numpy=True), dtype="float32").reshape(1, -1)
_, ids = index.search(vec, 3)

print(f"查询: {q}\n")
for rank, idx in enumerate(ids[0]):
    print(f"  Top{rank + 1}  索引号: {idx}  行号: {idx}")
    print(f"       问: {df.iloc[idx]['query']}")
    print(f"       答: {df.iloc[idx]['response']}\n")