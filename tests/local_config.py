from pathlib import Path

# 测试目录与项目目录
TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent

# 测试使用的模型与数据目录（独立于 system/config.py）
SIMCSE_MODEL_DIR = str(PROJECT_ROOT / "model" / "mysimcse")
TEST_DATA_DIR = str(TEST_ROOT / "data")
TEST_MODELS_DIR = str(TEST_ROOT / "models")
TEST_RESULTS_DIR = str(TEST_ROOT / "results")

# 测试中需要读取的问答数据目录：优先使用 db/data，不存在则回退到 train/data
_db_data_dir = PROJECT_ROOT / "db" / "data"
_train_data_dir = PROJECT_ROOT / "train" / "data"
if _db_data_dir.exists():
    DB_DATA_DIR = str(_db_data_dir)
else:
    DB_DATA_DIR = str(_train_data_dir)
