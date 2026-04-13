import os

# 当前文件位于 system/src，下退两级就是 system 根目录。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 训练/测试原始语料（由 process_lccc.py 生成）。
RAW_TRAIN_PATH = os.path.join(DATA_DIR, "raw_dialogues_train.txt")
RAW_TEST_PATH = os.path.join(DATA_DIR, "raw_dialogues_test.txt")

# 训练/测试标准语料（由 preprocess.py 生成）。
TRAIN_CORPUS_PATH = os.path.join(DATA_DIR, "corpus_train.csv")
TEST_CORPUS_PATH = os.path.join(DATA_DIR, "corpus_test.csv")

# 当前相似度判断默认使用训练集。
SIMILARITY_CORPUS_PATH = TRAIN_CORPUS_PATH

# 训练集与测试集缓存分开存放，避免相互覆盖。
TRAIN_EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "corpus_embeddings_train.pt")
TEST_EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "corpus_embeddings_test.pt")

# 本地训练产物目录，默认放在 model/my-simcse。
LOCAL_TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "model", "my-simcse")
# 是否允许读取和写入向量缓存。
USE_EMBEDDINGS_CACHE = True

def has_local_model() -> bool:
    """检查本地模型关键文件是否都在。"""
    return os.path.exists(os.path.join(LOCAL_TRAINED_MODEL_DIR, "model.pt")) and os.path.exists(
        os.path.join(LOCAL_TRAINED_MODEL_DIR, "vocab.json")
    )

# 推理侧统一从这个目录读取模型。
MODEL_NAME = LOCAL_TRAINED_MODEL_DIR

# 最低通过分：低于该值时返回兜底回复。
SIMILARITY_THRESHOLD = 0.5

# 低于阈值时，是否基于相近语料组织一段参考回答。
ENABLE_FALLBACK_REPLY = True

# 参考回答最多融合多少条相近语料。
FALLBACK_TOP_K = 3

# 输入句子最大长度，超出部分会被截断。
MAX_LENGTH = 64

# 模型训练温度参数，控制对比学习中正负样本的区分度。
SIMCSE_TEMPERATURE = 0.05