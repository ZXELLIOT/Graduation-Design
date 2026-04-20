from pathlib import Path

# 项目根目录：system/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- 模型目录 ---
MODEL_DIR = str(PROJECT_ROOT / "model")
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")

# --- 数据库配置 ---
# 运行时与编码均默认使用 db/data 目录
DB_PREFIX = "large"
DB_DATA_DIR = str(PROJECT_ROOT / "db" / "data")
DB_CSV_PATH = str(Path(DB_DATA_DIR) / "lccc_large.csv")
DB_QUERY_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_query.index")
DB_RESPONSE_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_response.index")

# --- 对话匹配逻辑参数 ---
# SIMILARITY_THRESHOLD: 判定回复是否可采纳的语义相似度得分阈值
SIMILARITY_THRESHOLD = 0.65

# RERANK_WEIGHTS: 加权重排权重系数 (问问相似度权重, 问答相似度权重)
RERANK_WEIGHTS = (0.75, 0.25)