import os

# 获取项目根目录绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据与模型配置
RAW_CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw_dialogues.txt")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.csv")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "data", "corpus_embeddings.pt")
LOCAL_TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "model", "my-simcse")
USE_EMBEDDINGS_CACHE = True

def has_local_model() -> bool:
    """检查本地模型产物是否齐全。"""
    return os.path.exists(os.path.join(LOCAL_TRAINED_MODEL_DIR, "model.pt")) and os.path.exists(
        os.path.join(LOCAL_TRAINED_MODEL_DIR, "vocab.json")
    )

# 模型路径
MODEL_NAME = LOCAL_TRAINED_MODEL_DIR

# 最低分阈值
SIMILARITY_THRESHOLD = 0.5

# 输入文本最大长度
MAX_LENGTH = 64

# 向后兼容 matcher/model 的配置字段
SENTENCE_POOLING = "mean"
SIMCSE_TEMPERATURE = 0.05