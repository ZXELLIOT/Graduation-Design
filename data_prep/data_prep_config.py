"""
data_prep/data_prep_config.py

文件作用:
    数据预处理模块的路径与参数配置。
    统一管理原始数据路径、输出路径、文本清洗阈值等常量。
"""

from pathlib import Path

# ============================================================
# 原始数据集路径 (LCCC-small: 训练/验证/测试)
# ============================================================
LCCC_small_DIR = Path(__file__).resolve().parent / "LCCC-small"

# 原始 JSON 数据文件
TRAIN_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_train.json")
VALID_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_valid.json")
TEST_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_test.json")

# 预处理后输出的 CSV 文件（含正负样本三元组）
TRAIN_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_train_neg.csv")
VALID_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_valid_neg.csv")
TEST_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_test_neg.csv")

# ============================================================
# 语料库路径 (LCCC-large: 系统知识库)
# ============================================================
LCCC_large_DIR = Path(__file__).resolve().parent / "LCCC-large"

# 原始大规模语料 JSON
LARGE_JSON_PATH = str(LCCC_large_DIR / "LCCC-large.json")

# 语料库输出 CSV（仅 query/response 两列，不含负样本）
LARGE_CSV_PATH = str(LCCC_large_DIR / "system_data.csv")

# ============================================================
# 文本清洗参数
# ============================================================

# 文本最小长度（字符数）：过短的对话轮次视为噪声丢弃
TEXT_MIN_LENGTH = 2

# 文本最大长度（字符数）：过长的对话轮次截断以保证训练效率
TEXT_MAX_LENGTH = 128

# ============================================================
# 数据预处理随机种子（保证负采样可复现）
# ============================================================
DEFAULT_PREPROCESS_SEED = 42

# ============================================================
# CSV 输出列名
# ============================================================

# 训练/验证/测试 CSV 列名（含负样本，供第二阶段有监督训练使用）
TRAIN_DATASET_HEADERS = ("query", "response", "negative_response")

# 语料库 CSV 列名（仅正样本对，供 FAISS 索引入库使用）
CORPUS_DATASET_HEADERS = ("query", "response")

# ============================================================
# 日志与阶段名称
# ============================================================

# 控制台日志前缀
PREPROCESS_LOG_PREFIX = "[数据预处理]"

# 各处理阶段的显示名称
PREPROCESS_STAGE_NAMES = {
    "train": "训练集",
    "valid": "验证集",
    "test": "测试集",
    "large": "语料库",
}