"""
tests/config.py
文件作用:
    测试阶段本地路径配置。
    统一维护测试数据、模型与结果目录。
"""

from pathlib import Path
import os

# 测试目录与项目目录
TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
# 测试使用的模型与数据目录（独立于 system/config.py）
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")
TEST_DATA_DIR = str(TEST_ROOT / "data")
TEST_MODELS_DIR = str(TEST_ROOT / "models")
TEST_RESULTS_DIR = str(TEST_ROOT / "results")
# 测试中固定读取 db/data 目录，避免隐式回退导致结果不一致
DB_DATA_DIR = str(PROJECT_ROOT / "db" / "data")
# 检索评测中的语义评审配置（优先使用 AI 评审，失败时可回退本地编码器）
RETRIEVAL_JUDGE_MODE = os.getenv("RETRIEVAL_JUDGE_MODE", "auto")  # auto / ark / local
ARK_RESPONSES_URL = os.getenv("ARK_RESPONSES_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
ARK_MODEL_NAME = os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-mini-260215")
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_TIMEOUT_SEC = float(os.getenv("ARK_TIMEOUT_SEC", "20"))