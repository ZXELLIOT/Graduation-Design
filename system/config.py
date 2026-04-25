"""
system/config.py

文件作用:
	系统运行配置。
	统一维护模型目录、数据库路径、阈值与重排参数。
"""

import os
from pathlib import Path

# 项目根目录: system/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型目录
MODEL_DIR = str(PROJECT_ROOT / "model")
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")

# 数据库配置
# 运行时与编码默认使用 db/data 目录。
DB_PREFIX = "large"
DB_DATA_DIR = str(PROJECT_ROOT / "db" / "data")
DB_CSV_PATH = str(Path(DB_DATA_DIR) / "lccc_large.csv")
DB_QUERY_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_query.index")
DB_RESPONSE_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_response.index")

# 对话匹配参数
# SIMILARITY_THRESHOLD: 判定回复是否可采纳的语义相似度阈值。
SIMILARITY_THRESHOLD = 0.5

# RERANK_WEIGHTS: 加权重排权重 (问问相似度权重, 问答相似度权重)。
RERANK_WEIGHTS = (0.75, 0.25)

# CONTEXT_MAX_TURNS: 上下文拼接的最大历史轮数。
CONTEXT_MAX_TURNS = 3

# MAX_TEXT_LEN: 输入文本与上下文拼接结果的最大长度。
MAX_TEXT_LEN = 64

# CONTEXT_SHORT_QUERY_LEN: 判定短问句的长度阈值。
CONTEXT_SHORT_QUERY_LEN = 14

# CONTEXT_OVERLAP_THRESHOLD: 启用上下文判定的词面重叠阈值。
CONTEXT_OVERLAP_THRESHOLD = 0.34

# CONTEXT_SEMANTIC_THRESHOLD: 启用上下文判定的语义相似阈值。
CONTEXT_SEMANTIC_THRESHOLD = 0.58

# AI 增强配置（默认关闭）
AI_ENHANCED_DEFAULT = os.getenv("AI_ENHANCED_DEFAULT", "0") == "1"
AI_ENHANCE_TOPK = int(os.getenv("AI_ENHANCE_TOPK", "5"))
AI_ENHANCE_MODEL_NAME = os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-mini-260215")
AI_ENHANCE_RESPONSES_URL = os.getenv("ARK_RESPONSES_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
AI_ENHANCE_API_KEY = os.getenv("ARK_API_KEY", "ark-8580122c-c8a0-4863-bf54-2cf9dfbb4fd5-ccb90")
AI_ENHANCE_TIMEOUT_SEC = float(os.getenv("ARK_TIMEOUT_SEC", "20"))