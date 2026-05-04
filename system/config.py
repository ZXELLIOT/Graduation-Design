"""
system/config.py

文件作用:
    系统运行配置。
    统一维护模型目录、数据库路径、阈值、重排参数与 AI 增强配置。
"""

import os
from pathlib import Path

# ============================================================
# 项目路径
# ============================================================

# 项目根目录: system/config.py 的上级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 从项目根目录的 .env 文件加载环境变量（需 python-dotenv）
try:
    from dotenv import load_dotenv
    _env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

# 模型目录
MODEL_DIR = str(PROJECT_ROOT / "model")
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")

# ============================================================
# 数据库路径配置
# ============================================================

# 数据库前缀（用于拼接索引文件名，如 large_faiss_db_query.index）
DB_PREFIX = "large"

# 数据库文件目录（CSV 语料 + FAISS 索引）
DB_DATA_DIR = str(PROJECT_ROOT / "db" / "data")

# 语料 CSV 路径（query, response 两列）
DB_CSV_PATH = str(Path(DB_DATA_DIR) / "lccc_large.csv")

# FAISS 双索引文件路径
DB_QUERY_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_query.index")
DB_RESPONSE_INDEX_FILE = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db_response.index")

# ============================================================
# 对话匹配参数
# ============================================================

# 语义相似度阈值：候选回复的融合分高于此值才采纳，否则拒答
SIMILARITY_THRESHOLD = 0.5

# 语义重排权重：(问问相似度权重, 问答相似度权重)
# 问问相似度 = 用户输入向量与候选问句向量的余弦相似度
# 问答相似度 = 用户输入向量与候选答句向量的余弦相似度
# 通常问问权重更高，因为问句匹配比答句匹配更直接
RERANK_WEIGHTS = (0.75, 0.25)

# 上下文拼接的最大历史轮数（从最近一轮往前取）
CONTEXT_MAX_TURNS = 3

# 输入文本与上下文拼接结果的最大长度（字符数）
MAX_TEXT_LEN = 64

# 短问句长度阈值：低于此值的输入自动触发上下文拼接
CONTEXT_SHORT_QUERY_LEN = 14

# 词面重叠阈值（Jaccard 相似度）：超过此值触发上下文拼接
CONTEXT_OVERLAP_THRESHOLD = 0.34

# 语义相似阈值：当前输入与历史输入的语义相似度超过此值时触发上下文拼接
CONTEXT_SEMANTIC_THRESHOLD = 0.58

# ============================================================
# AI 增强配置（默认关闭，需通过 .env 或前端面板开启）
# ============================================================

# 启动时默认是否开启 AI 增强
AI_ENHANCED_DEFAULT = os.getenv("AI_ENHANCED_DEFAULT", "0") == "1"

# AI 融合时使用的候选数量
AI_ENHANCE_TOPK = int(os.getenv("AI_ENHANCE_TOPK", "5"))

# 大模型名称
AI_ENHANCE_MODEL_NAME = os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-pro-260215")

# API 地址
AI_ENHANCE_RESPONSES_URL = os.getenv(
    "ARK_RESPONSES_URL",
    "https://ark.cn-beijing.volces.com/api/v3/responses",
)

# API Key（必须在 .env 中配置 ARK_API_KEY，不提供默认值避免密钥泄露）
AI_ENHANCE_API_KEY = os.getenv("ARK_API_KEY", "")

# 请求超时时间（秒）
AI_ENHANCE_TIMEOUT_SEC = float(os.getenv("ARK_TIMEOUT_SEC", "20"))

# ============================================================
# 后台管理密码（修改此值或在 .env 中设置 ADMIN_PASSWORD）
# ============================================================
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "simcse2024")
