# simcse_encoder.py
import os
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\model\mysimcse"
MAX_LENGTH = 128
SENTENCE_POOLING = "cls"

class SimCSEEncoder:
    def __init__(self):
        """加载本地模型并准备编码器。

        要求：在常量 MODEL_DIR 指定的位置存在已导出的模型目录。
        初始化后可调用 `encode` 与 `encode_one` 将文本转换为向量。
        """
        self.model_dir = MODEL_DIR
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"本地模型目录不存在或不完整：{self.model_dir}。"
            )
        # 加载 模型和分词器
        print(f"加载本地模型: {self.model_dir} (local_files_only=True)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        q_dir = os.path.join(self.model_dir, 'query_encoder')
        r_dir = os.path.join(self.model_dir, 'response_encoder')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_encoder = AutoModel.from_pretrained(q_dir, local_files_only=True)
        self.response_encoder = AutoModel.from_pretrained(r_dir, local_files_only=True)
        self.query_encoder.to(self.device)
        self.response_encoder.to(self.device)
        self.query_encoder.eval()
        self.response_encoder.eval()
        self._single_model = False
        self.pooling = SENTENCE_POOLING
        print(f"模型加载完成，设备: {self.device}, pooling={self.pooling}")

    def _l2_normalize(self, embeddings, eps=1e-8):
        """
        对向量进行 L2 归一化，使向量的模长为 1。
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
            
        embeddings = embeddings.float()
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        if embeddings.ndim == 1:
            norm = torch.linalg.vector_norm(embeddings, ord=2)
            return embeddings / torch.clamp(norm, min=eps)

        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=eps)

    def _masked_mean_pool(self, token_embeddings, attention_mask, eps=1e-8):
        """
        基于 attention_mask 的加权平均池化
        """
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=eps)
        return summed / counts

    def _sentence_pooling(self, outputs, attention_mask):
        """
        从模型的输出中提取整句的向量表示。
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

    def encode(self, texts, batch_size=32, encoder: str = 'query', return_numpy: bool = False):
        """
        将文本列表转换为语义向量。
        参数：
        - texts: 单条字符串或字符串列表；
        - batch_size: 编码时使用的批次大小；
        - encoder: 指定使用哪一套编码器，取值为 'query' 或 'response'；
        - return_numpy: 若为 True 则返回 CPU 上的 numpy 数组，否则返回 torch 张量。
        返回：形状为 (样本数, 向量维度) 的向量集合。
        """
        if isinstance(texts, str):
            texts = [texts]

        if encoder not in ("query", "response"):
            raise ValueError("encoder 参数只接受 'query' 或 'response'")

        model = self.query_encoder if encoder == 'query' else self.response_encoder

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                need_hidden_states = self.pooling == "first_last_avg"
                if self.device.type == 'cuda':
                    try:
                        from torch.cuda.amp import autocast
                        with autocast():
                            outputs = model(**inputs, output_hidden_states=need_hidden_states)
                    except Exception:
                        outputs = model(**inputs, output_hidden_states=need_hidden_states)
                else:
                    outputs = model(**inputs, output_hidden_states=need_hidden_states)

                embeddings = self._sentence_pooling(outputs, inputs["attention_mask"])
                embeddings = self._l2_normalize(embeddings)
                all_embeddings.append(embeddings.cpu())

        merged = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0))
        return merged.numpy() if return_numpy else merged

    def encode_one(self, text, encoder: str = 'query', return_numpy: bool = False):
        """对单条文本进行编码并返回向量。

        参数：
        - text: 待编码的文本；
        - encoder: 使用的编码器，'query' 或 'response'；
        - return_numpy: 若为 True 返回 numpy 数组。
        返回：一维向量（torch 张量或 numpy 数组）。
        """
        if encoder not in ("query", "response"):
            raise ValueError("encoder must be 'query' or 'response'")

        model = self.query_encoder if encoder == 'query' else self.response_encoder

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            need_hidden_states = self.pooling == "first_last_avg"
            outputs = model(**inputs, output_hidden_states=need_hidden_states)
            emb = self._sentence_pooling(outputs, inputs["attention_mask"])
            emb = self._l2_normalize(emb)
            emb = emb.cpu()

        if return_numpy:
            return emb.numpy().reshape(-1)
        return emb.reshape(-1)