import os

# 获取项目根目录绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#数据与模型配置
RAW_CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw_dialogues.txt")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.csv")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "data", "corpus_embeddings.pt")
# 采用预训练模型
MODEL_NAME = "shibing624/text2vec-base-chinese"
# 相似度阈值判定
SIMILARITY_THRESHOLD = 0.5
# 文本最大截断长度
MAX_LENGTH = 64