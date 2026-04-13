import os
import glob

# 获取项目根目录绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#数据与模型配置
RAW_CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw_dialogues.txt")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.csv")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "data", "corpus_embeddings.pt")

# 预训练模型本地缓存目录（用户提供）
HF_CACHE_MODEL_DIR = r"C:\Users\13713\.cache\huggingface\hub\models--shibing624--text2vec-base-chinese"
LOCAL_TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "modle", "my-simcse")
# 为 True 时只允许使用本地模型；找不到则直接报错
STRICT_LOCAL_MODEL = True


def _resolve_model_name():
	"""优先使用本地训练模型；其次本地缓存模型；最后回退到在线模型名。"""
	# 若已经训练完成，优先加载本地模型目录
	if os.path.exists(os.path.join(LOCAL_TRAINED_MODEL_DIR, "config.json")):
		return LOCAL_TRAINED_MODEL_DIR

	snapshots_dir = os.path.join(HF_CACHE_MODEL_DIR, "snapshots")
	if os.path.isdir(snapshots_dir):
		candidates = sorted(glob.glob(os.path.join(snapshots_dir, "*")))
		if candidates:
			# 取最新快照目录
			return candidates[-1]

	if os.path.isdir(HF_CACHE_MODEL_DIR):
		return HF_CACHE_MODEL_DIR

	if STRICT_LOCAL_MODEL:
		raise FileNotFoundError("未找到本地 SimCSE 模型，请先训练或下载到本地后再启动。")

	# 仅在允许非严格本地模式时回退到在线模型名
	return "shibing624/text2vec-base-chinese"


# 模型路径（本地快照优先）
MODEL_NAME = _resolve_model_name()

# 匹配分数低于该值时返回兜底回复
SIMILARITY_THRESHOLD = 0.5

# 输入文本最大长度（超过会截断）
MAX_LENGTH = 64

# 句向量池化策略：cls / mean / first_last_avg
SENTENCE_POOLING = "first_last_avg"

# SimCSE 温度系数（与对比学习打分一致）
SIMCSE_TEMPERATURE = 0.05
