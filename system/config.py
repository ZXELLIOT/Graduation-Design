"""
system/config.py
文件作用:
    系统运行配置。
    统一维护模型目录、数据库路径、阈值、重排参数与 AI 增强配置。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 从项目根目录的.env文件加载环境变量
try:
    from dotenv import load_dotenv
    _env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass
# 模型目录
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")
# 数据库文件目录
DB_DATA_DIR = PROJECT_ROOT / "db" / "data"
# FAISS 问句索引文件
DB_QUERY_INDEX_FILE = str(DB_DATA_DIR / "querydata")
# 语料 CSV 路径
DB_CSV_PATH = str(DB_DATA_DIR / "system_data.csv")
# 运行时加载上限
DB_LOAD_MAX_ROWS = 2000000

# 语义相似度阈值
SIMILARITY_THRESHOLD = 0.5
# 语义重排权重
RERANK_WEIGHTS = (0.75, 0.25)
# FAISS 粗召回数量
COARSE_RECALL_COUNT = 50
# Top-K 候选数量
DEFAULT_TOP_K = 3
# 输入文本与上下文拼接结果的最大长度
MAX_TEXT_LEN = 64

# 是否启用上下文拼接
CONTEXT_MATCHING_ENABLED = 0
# 上下文拼接的最大历史轮数
CONTEXT_MAX_TURNS = 3

# 是否开启 AI 增强
AI_ENHANCED_DEFAULT = 0
# 大模型名称
AI_ENHANCE_MODEL_NAME = os.getenv("AI_ENHANCE_MODEL_NAME", "")
# API 地址
AI_ENHANCE_RESPONSES_URL = os.getenv("AI_ENHANCE_RESPONSES_URL", "")
# API Key
AI_ENHANCE_API_KEY = os.getenv("AI_ENHANCE_API_KEY", "")
# 请求超时时间
AI_ENHANCE_TIMEOUT_SEC = 20

# 管理员密码
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")