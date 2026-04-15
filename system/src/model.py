# simcse_encoder.py (修复版本)
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# 默认优先使用项目本地模型目录（若存在）
# 本地模型目录路径相对于本文件上级目录的 `model/`
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))

def _detect_local_model(path: str) -> bool:
    """检测指定路径下是否存在可用的模型文件（简单检查）。"""
    if not path:
        return False
    if not os.path.isdir(path):
        return False
    # 只要存在配置或权重文件之一，即视为本地模型目录
    candidates = ['config.json', 'pytorch_model.bin', 'tf_model.h5', 'flax_model.msgpack']
    for c in candidates:
        if os.path.exists(os.path.join(path, c)):
            return True
    return False

# --- 模型配置 ---
MODEL_NAME = "shibing624/text2vec-base-chinese"
MAX_LENGTH = 128  # 输入文本最大长度，超长会截断
SENTENCE_POOLING = "cls"  # 句向量聚合方式: "cls"
SIMCSE_TEMPERATURE = 0.05 # 相似度计算温度参数
# --- 配置结束 ---

class SimCSEEncoder:
    def __init__(self, model_name=None):
        """
        初始化 SimCSE 编码器
        
        Args:
            model_name (str, optional): HuggingFace 模型名称或本地路径。
                                      如果为 None，则使用默认的开源模型。
        """
        # 允许传入不同的模型名称
        self.model_name = model_name or MODEL_NAME
        print(f"正在加载模型: {self.model_name}")
        
        # 1. 决定是否使用本地模型目录
        local_path_used = False
        if model_name and _detect_local_model(model_name):
            self.model_name = model_name
            local_path_used = True
        elif not model_name and _detect_local_model(LOCAL_MODEL_DIR):
            self.model_name = LOCAL_MODEL_DIR
            local_path_used = True

        # 2. 加载分词器与模型
        # 如果使用本地模型，则以离线模式加载（local_files_only=True）以避免联网请求
        local_files_only = True if local_path_used else False
        print(f"分词器加载路径: {self.model_name} (local_files_only={local_files_only})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(self.model_name, local_files_only=local_files_only)
        
        # 3. 选择计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device.type.upper()}")
        self.model.to(self.device)
        
        # 4. 设置模型状态
        self.pooling = SENTENCE_POOLING
        self.model.eval()
        print(f"模型加载完毕 | Pooling Strategy: {self.pooling}")

    def _l2_normalize(self, embeddings, eps=1e-8):
        """
        对向量进行 L2 归一化，使向量的模长为 1。
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
            
        embeddings = embeddings.float()
        # 处理NaN和inf值
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=eps)

    def _masked_mean_pool(self, token_embeddings, attention_mask, eps=1e-8):
        """
        基于 attention_mask 的加权平均池化，用于忽略填充 (padding) 部分。
        """
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=eps)
        return summed / counts

    def _sentence_pooling(self, outputs, attention_mask):
        """
        根据设定的策略，从模型的输出中提取整句的向量表示。
        """
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]

        if self.pooling == "mean":
            return self._masked_mean_pool(outputs.last_hidden_state, attention_mask)

        if self.pooling == "first_last_avg":
            hidden_states = outputs.hidden_states
            first_hidden = hidden_states[1] if len(hidden_states) > 1 else hidden_states[0]
            last_hidden = hidden_states[-1]
            avg_hidden = 0.5 * (first_hidden + last_hidden)
            return self._masked_mean_pool(avg_hidden, attention_mask)

        raise ValueError(f"不支持的 pooling 策略: {self.pooling}")

    def encode(self, texts, batch_size=32):
        """
        将文本列表转换为语义向量。
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
                need_hidden_states = self.pooling == "first_last_avg"
                outputs = self.model(**inputs, output_hidden_states=need_hidden_states)
                
                embeddings = self._sentence_pooling(outputs, inputs["attention_mask"])
                
                embeddings = self._l2_normalize(embeddings)
                
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)