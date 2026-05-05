"""
train/train_config.py

文件作用:
    训练模块配置。
    统一管理训练数据、预训练模型、模型输出路径。
"""

from pathlib import Path

# 训练模块根目录
TRAIN_ROOT = Path(__file__).resolve().parent

# ============================================================
# 训练数据目录
# ============================================================
TRAIN_DATA_DIR = TRAIN_ROOT / "data"

# 训练/验证/测试 CSV（由 data_prep 模块生成后复制到此处）
TRAIN_NEG_CSV_PATH = str(TRAIN_DATA_DIR / "lccc_train_neg.csv")
VALID_NEG_CSV_PATH = str(TRAIN_DATA_DIR / "lccc_valid_neg.csv")
TEST_NEG_CSV_PATH = str(TRAIN_DATA_DIR / "lccc_test_neg.csv")

# ============================================================
# 预训练模型目录
# ============================================================
TRAIN_MODEL_DIR = TRAIN_ROOT / "model"

# 本地预训练模型（text2vec-base-chinese，优先使用避免在线下载）
LOCAL_PRETRAINED_DIR = str(TRAIN_MODEL_DIR / "text2vec-base-chinese")

# ============================================================
# 模型输出目录
# ============================================================
OUTPUT_MODEL_DIR = str(TRAIN_ROOT / "output")

