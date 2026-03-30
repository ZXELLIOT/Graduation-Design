import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH

class SimCSEEncoder:
    def __init__(self):
        print(f"正在初始化 SimCSE 编码器，加载模型 ({MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        # 自动检测是否可以使用 GPU 显卡加速计算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # 设置为推理模式
        print(f"模型加载完毕！系统当前运行在: {self.device}")

    def encode(self, texts):
        """将文字转换为高维语义向量"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=MAX_LENGTH
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 取 [CLS] token 的输出作为整个句子的语义表示
            embeddings = outputs.last_hidden_state[:, 0]
            # 进行 L2 归一化以备进行余弦距离计算
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()