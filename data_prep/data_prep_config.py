"""
data_prep/data_prep_config.py

文件作用:
	定义数据清洗过程中使用的文件路径和必要常量。
"""
from pathlib import Path

# 用于训练模型数据集的路径
LCCC_small_DIR = Path(__file__).resolve().parent / "LCCC-small"
TRAIN_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_train.json")	#训练集
VALID_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_valid.json") #验证集
TEST_JSON_PATH = str(LCCC_small_DIR / "LCCC-small_test.json") #测试集
# 输出路径
TRAIN_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_train_neg.csv")
VALID_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_valid_neg.csv")
TEST_NEG_CSV_PATH = str(LCCC_small_DIR / "lccc_test_neg.csv")

# 用于系统知识库的路径
LCCC_large_DIR = Path(__file__).resolve().parent / "LCCC-large"
LARGE_JSON_PATH = str(LCCC_large_DIR / "LCCC-large.json")
# 输出路径
LARGE_CSV_PATH = str(LCCC_large_DIR / "system_data.csv")

# 日志前缀
PREPROCESS_LOG_PREFIX = "[数据预处理]"

# 数据预处理随机种子
DEFAULT_PREPROCESS_SEED = 42

# 数据预处理阶段名称
PREPROCESS_STAGE_NAMES = {
	"train": "训练集",
	"valid": "验证集",
	"test": "测试集",
	"large": "语料库",
}

# 每个文本的最大和最小长度（字符数）
TEXT_MIN_LENGTH = 2
TEXT_MAX_LENGTH = 128

# csv写出配置
TRAIN_DATASET_HEADERS = ("query", "response", "negative_response")
CORPUS_DATASET_HEADERS = ("query", "response")