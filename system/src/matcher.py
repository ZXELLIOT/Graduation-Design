# src/matcher.py
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

class DialogMatcher:
    def __init__(self, encoder, queries=None, replies=None, cache_dir=None, similarity_threshold=0.5, enable_fallback=True, fallback_top_k=3):
        """
        匹配组件：负责管理向量缓存并在运行时执行相似度检索。

        参数说明：
        - encoder: 文本编码器实例，用于将文本转换为向量。
        - queries/replies: 可选的问答文本列表；若不提供，将尝试从缓存恢复文本映射。
        - cache_dir: 缓存目录，用于存储或读取向量与文本映射。
        - similarity_threshold: 匹配阈值。
        - enable_fallback: 是否在匹配不足时提供备用回复。
        - fallback_top_k: 备用回复候选数量限制。
        """
        self.encoder = encoder
        self.queries = queries
        self.replies = replies
        self.cache_dir = cache_dir or os.getcwd()
        self.similarity_threshold = similarity_threshold
        self.enable_fallback = enable_fallback
        self.fallback_top_k = fallback_top_k
        self.query_embeddings, self.reply_embeddings = self._load_or_create_embeddings()

    def _load_or_create_embeddings(self):
        """检查并加载向量缓存，若不存在则计算并保存"""
        query_vec_path = os.path.join(self.cache_dir, 'train_query_embeddings.pt')
        reply_vec_path = os.path.join(self.cache_dir, 'train_reply_embeddings.pt')
        queries_path = os.path.join(self.cache_dir, 'train_queries.pt')
        replies_path = os.path.join(self.cache_dir, 'train_replies.pt')
        
        if os.path.exists(query_vec_path) and os.path.exists(reply_vec_path):
            print(f"检测到已存在的缓存文件，正在加载")
            try:
                query_embeddings = torch.load(query_vec_path, map_location='cpu')
                reply_embeddings = torch.load(reply_vec_path, map_location='cpu')
                self.queries = torch.load(queries_path)
                self.replies = torch.load(replies_path)

                print(f"缓存加载成功")
            except Exception as e:
                print(f"缓存加载失败: {e}")
                return self._create_and_save_embeddings()
        else:
            print(f"未找到缓存文件，正在生成")
            return self._create_and_save_embeddings()
        
        return query_embeddings, reply_embeddings
    
    def _create_and_save_embeddings(self):
        """计算并保存新的嵌入向量"""
        if self.queries is None or self.replies is None:
            raise ValueError("缺少语料文本")

        encode_start_time = time.perf_counter()

        print(f"正在编码 {len(self.queries)} 个问题")
        query_embeddings_list = []
        batch_size = getattr(self, 'encode_batch_size', None) or 128
        for i in tqdm(range(0, len(self.queries), batch_size), desc="编码问题", unit="batch"):
            batch_texts = self.queries[i:i+batch_size]
            emb = self.encoder.encode(batch_texts, batch_size=batch_size)
            query_embeddings_list.append(emb)
        if len(query_embeddings_list) > 0:
            query_embeddings = torch.cat(query_embeddings_list, dim=0)
        else:
            if hasattr(self.encoder, 'query_encoder'):
                dim = getattr(self.encoder.query_encoder.config, 'hidden_size', 0)
            elif hasattr(self.encoder, 'model'):
                dim = getattr(self.encoder.model.config, 'hidden_size', 0)
            else:
                dim = 0
            query_embeddings = torch.empty((0, dim))

        print(f"正在编码 {len(self.replies)} 个回答")
        reply_embeddings_list = []
        for i in tqdm(range(0, len(self.replies), batch_size), desc="编码回答", unit="batch"):
            batch_texts = self.replies[i:i+batch_size]
            emb = self.encoder.encode(batch_texts, batch_size=batch_size, encoder='response')
            reply_embeddings_list.append(emb)
        if len(reply_embeddings_list) > 0:
            reply_embeddings = torch.cat(reply_embeddings_list, dim=0)
        else:
            if hasattr(self.encoder, 'response_encoder'):
                dim = getattr(self.encoder.response_encoder.config, 'hidden_size', 0)
            elif hasattr(self.encoder, 'model'):
                dim = getattr(self.encoder.model.config, 'hidden_size', 0)
            else:
                dim = 0
            reply_embeddings = torch.empty((0, dim))

        encode_elapsed = time.perf_counter() - encode_start_time

        print(f"编码完成，耗时: {encode_elapsed:.2f} 秒。正在保存缓存")

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(query_embeddings, os.path.join(self.cache_dir, 'train_query_embeddings.pt'))
        torch.save(reply_embeddings, os.path.join(self.cache_dir, 'train_reply_embeddings.pt'))
        torch.save(self.queries, os.path.join(self.cache_dir, 'train_queries.pt'))
        torch.save(self.replies, os.path.join(self.cache_dir, 'train_replies.pt'))

        print(f"缓存已保存至: {self.cache_dir}")

        return query_embeddings, reply_embeddings

    def _cosine_similarity(self, a, b):
        """计算余弦相似度（输入向量应已归一化）。

        参数：
        - a: 张量，形状可以为 (n, d) 或 (d,)；
        - b: 张量，形状可以为 (m, d) 或 (d,)；
        返回：相似度张量，形状为 (n, m) 或降维后的向量。
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        
        return torch.matmul(a, b.transpose(0, 1))

    def match(self, user_input: str, top_k_for_rerank=5):
        """
        根据用户输入，计算相似度并返回匹配结果。
        Args:
            user_input (str): 用户输入的文本。
            top_k_for_rerank (int): 第一阶段筛选的候选数量。
        Returns:
            tuple: (best_reply, best_score, matched_query)
        """
        if self.query_embeddings is None or not self.queries:
            return "问句库为空，暂无法回答问题。", 0.0, None
        if self.reply_embeddings is None or not self.replies:
            return "答句库为空，暂无法回答问题。", 0.0, None
        # 编码用户输入
        user_query_emb = self.encoder.encode_one(user_input, encoder='query')

        # --- 第一阶段：问句匹配 ---
        question_similarities = self._cosine_similarity(user_query_emb, self.query_embeddings) 
        if question_similarities.dim() > 1:
            question_similarities = question_similarities.squeeze(0) 

        # 取相似度最高的 top_k 个索引作为候选
        top_k = min(top_k_for_rerank, len(self.queries) if self.queries else 0)
        top_question_values, top_question_indices = torch.topk(question_similarities, k=top_k)
    
        # --- 第二阶段：答句重排序 ---
        # 1. 获取候选回答的向量
        candidate_reply_indices = top_question_indices.cpu().numpy().tolist()
        candidate_reply_embeddings = self.reply_embeddings[candidate_reply_indices]

        # 2. 计算用户与候选回答的相似度
        response_similarities = self._cosine_similarity(user_query_emb, candidate_reply_embeddings).squeeze(0)

        # 3. 计算加权总分，问句分 * 0.5 + 回答分 * 0.5
        q_scores = top_question_values.cpu() 
        a_scores = response_similarities.cpu()
        final_scores = (q_scores * 0.5) + (a_scores * 0.5)

        # 4. 找到加权后分数最高的
        best_idx_in_list = torch.argmax(final_scores)
        best_global_idx = candidate_reply_indices[best_idx_in_list]
        
        # 最终得分使用加权分
        best_score = final_scores[best_idx_in_list].item()
        best_reply = self.replies[best_global_idx]
        matched_query = self.queries[best_global_idx]

        # 阈值判断
        if best_score < self.similarity_threshold:
            if self.enable_fallback:
                snippets = []
                scores = top_question_values.cpu().numpy().tolist()
                indices = top_question_indices.cpu().numpy().tolist()
                
                for score, idx in zip(scores, indices):
                    if score < self.similarity_threshold * 0.7: 
                        continue
                    reply = (self.replies[idx] or "").strip()
                    if reply and reply not in snippets:
                        snippets.append(f"（相似度:{score:.2f}）{reply}")
                
                if not snippets:
                    fallback_reply = "这个问题在当前语料里没有直接答案。你可以换个更具体的问法，我再试着帮你找。"
                else:
                    combined = "；".join(snippets[:self.fallback_top_k])
                    fallback_reply = f"这个问题在当前语料里没有直接答案。我根据几条最接近的信息整理了一个参考方向：{combined}。"
                
                return fallback_reply, best_score, None
            else:
                return "抱歉，我目前的知识库中没有足够相关的信息来回答您的问题。", best_score, None

        return best_reply, best_score, matched_query