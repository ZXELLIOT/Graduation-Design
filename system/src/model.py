import torch
from transformers import AutoModel, AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, SENTENCE_POOLING, SIMCSE_TEMPERATURE

class SimCSEEncoder:
    def __init__(self):
        print(f"正在加载模型 ({MODEL_NAME})...")
        # 分词器将文字转换为模型可读的数字
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        # 模型将数字进一步转换为语义向量
        self.model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        # 优先使用 GPU，没有则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pooling = SENTENCE_POOLING
        # 推理模式：只做预测，不更新参数
        self.model.eval()
        print(f"模型加载完毕，系统当前运行在: {self.device} | pooling={self.pooling}")

    def _l2_normalize(self, embeddings, eps=1e-8):
        """手动实现 L2 归一化。"""
        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=eps)

    def _masked_mean_pool(self, token_embeddings, attention_mask, eps=1e-8):
        """按 attention mask 做平均池化，忽略 padding。"""
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=eps)
        return summed / counts

    def _sentence_pooling(self, outputs, attention_mask):
        """支持 SimCSE 池化策略。"""
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
                # 前向计算：得到每个位置的隐藏层输出
                need_hidden_states = self.pooling == "first_last_avg"
                outputs = self.model(**inputs, output_hidden_states=need_hidden_states)
                # 按配置策略得到整句表示
                embeddings = self._sentence_pooling(outputs, inputs["attention_mask"])
                # 归一化后更适合做相似度计算
                embeddings = self._l2_normalize(embeddings)
                all_embeddings.append(embeddings.cpu())
                
        # 合并每一批的结果
        return torch.cat(all_embeddings, dim=0)

    def simcse_similarity(self, query_emb, corpus_emb, temperature=SIMCSE_TEMPERATURE, eps=1e-8):
        """SimCSE 打分：温度缩放后的点积，再映射为 0-1 置信分。"""
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if corpus_emb.dim() == 1:
            corpus_emb = corpus_emb.unsqueeze(0)

        q = self._l2_normalize(query_emb, eps=eps)
        c = self._l2_normalize(corpus_emb, eps=eps)

        logits = torch.matmul(q, c.transpose(0, 1)) / max(temperature, eps)
        scores = torch.sigmoid(logits)

        if scores.size(0) == 1:
            return scores.squeeze(0)
        return scores
