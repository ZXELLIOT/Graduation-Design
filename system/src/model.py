import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH

class SimCSEEncoder:
    def __init__(self):
        print(f"正在初始化 SimCSE ，加载模型 ({MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        # 自动检测是否可以使用 GPU 显卡加速计算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # 设置为推理模式
        print(f"模型加载完毕！系统当前运行在: {self.device}")

    def encode(self, texts, batch_size=32):
        """
        将文本转换为语义向量
        通过批处理 (Batching) 控制显存占用
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
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
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)