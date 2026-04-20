from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LCCC_small_DIR = DATA_DIR / "LCCC-small"
TRAIN_JSON_PATH = str(LCCC_small_DIR / "LCCC-base_train.json")
VALID_JSON_PATH = str(LCCC_small_DIR / "LCCC-base_valid.json")
TEST_JSON_PATH = str(LCCC_small_DIR / "LCCC-base_test.json")
TRAIN_NEG_CSV_PATH = str(DATA_DIR / "lccc_train_neg.csv")
VALID_NEG_CSV_PATH = str(DATA_DIR / "lccc_valid_neg.csv")
TEST_NEG_CSV_PATH = str(DATA_DIR / "lccc_test_neg.csv")

LCCC_large_DIR = DATA_DIR / "LCCC-large"
LARGE_JSON_PATH = str(LCCC_large_DIR / "LCCD.json")
LARGE_CSV_PATH = str(DATA_DIR / "lccc_large.csv")

LOCAL_PRETRAINED_DIR = str((DATA_DIR / "text2vec-base-chinese").resolve())

OUTPUT_MODEL_DIR = str((PROJECT_ROOT / "model").resolve())