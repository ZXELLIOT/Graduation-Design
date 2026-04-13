import os
import torch
from src.config import (
    SIMILARITY_THRESHOLD,
    EMBEDDINGS_CACHE_PATH,
    MODEL_NAME,
    SENTENCE_POOLING,
    SIMCSE_TEMPERATURE,
)
from src.comparison import SimilarityComparator


class DialogMatcher:
    def __init__(self, encoder, queries, replies):
        self.encoder = encoder
        self.queries = queries
        self.replies = replies
        # 预先计算轻量特征，减少每次检索时的重复开销
        self.query_char_sets = [set(q) for q in self.queries]
        self.query_lengths = [max(1, len(q)) for q in self.queries]
        if len(self.queries) > 0:
            print(f"当前加载的语料库数据量：{len(self.queries)} 条")
            
            # 引入向缓存，避免每次启动时重复提取特征向量
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                try:
                    print("检测到本地向量缓存，正在加载")
                    # 缓存放在 CPU，避免额外占用显存
                    cache_obj = torch.load(EMBEDDINGS_CACHE_PATH, map_location='cpu')

                    # 兼容旧版缓存（仅 tensor）
                    if isinstance(cache_obj, torch.Tensor):
                        cached_embeddings = cache_obj
                        cached_model_name = None
                        cached_pooling = None
                        cached_temperature = None
                        cached_score_version = None
                    else:
                        cached_embeddings = cache_obj.get("embeddings")
                        cached_model_name = cache_obj.get("model_name")
                        cached_pooling = cache_obj.get("pooling")
                        cached_temperature = cache_obj.get("simcse_temperature")
                        cached_score_version = cache_obj.get("score_version")

                    rows_ok = (
                        isinstance(cached_embeddings, torch.Tensor)
                        and cached_embeddings.size(0) == len(self.queries)
                    )
                    model_ok = (cached_model_name is None) or (cached_model_name == MODEL_NAME)
                    pooling_ok = (cached_pooling is None) or (cached_pooling == SENTENCE_POOLING)
                    temperature_ok = (cached_temperature is None) or (cached_temperature == SIMCSE_TEMPERATURE)
                    score_ok = (cached_score_version is None) or (cached_score_version == "dual-branch-v1")

                    if rows_ok and model_ok and pooling_ok and temperature_ok and score_ok:
                        self.corpus_embeddings = cached_embeddings
                        print("向量缓存加载成功！")
                        return

                    if not rows_ok:
                        print("检测到语料已更新，正在重新构建")
                    elif not pooling_ok:
                        print("检测到池化策略变更，正在重新构建")
                    elif not temperature_ok:
                        print("检测到 SimCSE 温度参数变更，正在重新构建")
                    elif not score_ok:
                        print("检测到打分版本变更，正在重新构建")
                    else:
                        print("缓存尺寸与当前数据不符（检测到语料已更新），正在重新构建...")
                except Exception as e:
                    print(f"缓存读取异常: {e}，将全量重新计算...")
            
            print("正在计算所有问句的特征向量（首次构建耗时较长，请耐心等待）...")
            self.corpus_embeddings = self.encoder.encode(self.queries)
            
            # 保存缓存，后续可直接读取
            print("特征向量计算完毕，正在保存")
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "pooling": SENTENCE_POOLING,
                    "simcse_temperature": SIMCSE_TEMPERATURE,
                    "score_version": "dual-branch-v1",
                    "embeddings": self.corpus_embeddings,
                },
                EMBEDDINGS_CACHE_PATH,
            )
            print("本地缓存建立成功")
        else:
            print("提示：知识库为空，请先添加数据。")
            self.corpus_embeddings = None

    def _char_overlap_scores(self, user_text):
        """计算字符重叠比例。"""
        user_chars = set(user_text)
        if not user_chars:
            return torch.zeros(len(self.queries), dtype=torch.float32)

        overlaps = []
        for q_chars in self.query_char_sets:
            union_len = len(user_chars | q_chars)
            if union_len == 0:
                overlaps.append(0.0)
            else:
                overlaps.append(len(user_chars & q_chars) / union_len)
        return torch.tensor(overlaps, dtype=torch.float32)

    def _length_consistency_scores(self, user_text):
        """比较长度接近程度，越接近分数越高。"""
        user_len = max(1, len(user_text))
        scores = []
        for q_len in self.query_lengths:
            ratio = min(user_len, q_len) / max(user_len, q_len)
            scores.append(ratio)
        return torch.tensor(scores, dtype=torch.float32)

    def _ngrams(self, text, n=2):
        """把一句话拆成若干“两个字一组”的片段。"""
        if len(text) < n:
            return set([text]) if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _ngram_overlap_scores(self, user_text):
        """计算两个字片段的重叠比例。"""
        user_ngrams = self._ngrams(user_text, n=2)
        if not user_ngrams:
            return torch.zeros(len(self.queries), dtype=torch.float32)

        scores = []
        for q in self.queries:
            q_ngrams = self._ngrams(q, n=2)
            union_len = len(user_ngrams | q_ngrams)
            if union_len == 0:
                scores.append(0.0)
            else:
                scores.append(len(user_ngrams & q_ngrams) / union_len)
        return torch.tensor(scores, dtype=torch.float32)

    def _custom_similarity(self, user_text, user_emb):
        # 主分：语义相似度（自定义余弦实现）
        cosine_scores = self._cosine_similarity(user_emb, self.corpus_embeddings)

        # 辅助分1：字符重叠比例
        overlap_scores = self._char_overlap_scores(user_text)

        # 辅助分2：两个字片段重叠比例
        ngram_scores = self._ngram_overlap_scores(user_text)

        # 辅助分3：长度接近度
        length_scores = self._length_consistency_scores(user_text)

        # 最终分：按权重合并 4 个分数
        return (
            0.75 * cosine_scores
            + 0.10 * overlap_scores
            + 0.10 * ngram_scores
            + 0.05 * length_scores
        )

    def _cosine_similarity(self, a, b, eps=1e-8):
        """手动实现余弦相似度，返回 a 与 b 每行之间的相似度矩阵。"""
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)

        numerator = torch.matmul(a, b.transpose(0, 1))
        a_norm = torch.linalg.vector_norm(a, ord=2, dim=1, keepdim=True)
        b_norm = torch.linalg.vector_norm(b, ord=2, dim=1, keepdim=True).transpose(0, 1)
        denominator = torch.clamp(a_norm * b_norm, min=eps)
        scores = numerator / denominator

        if scores.size(0) == 1:
            return scores.squeeze(0)
        return scores

    def _semantic_similarity(self, user_emb):
        """使用 SimCSE 专用打分函数进行匹配。"""
        return self.encoder.simcse_similarity(user_emb, self.corpus_embeddings)

    def get_best_match(self, user_text):
        """计算两路分数并找出最佳回答。"""
        if self.corpus_embeddings is None or len(self.queries) == 0:
            return "知识库为空，暂无法回答问题。", 0.0, None
        # 1. 编码用户的实时输入
        user_emb = self.encoder.encode([user_text])

        # 2) 并行计算两套分数：当前比对方式 + SimCSE 纯语义
        current_scores = self._custom_similarity(user_text, user_emb)
        semantic_scores = self._semantic_similarity(user_emb)

        # 3) 对比分数，选择最高分对应结果
        decision = SimilarityComparator.compare(current_scores, semantic_scores)
        best_idx = decision["best_idx"]
        best_score = decision["best_score"]
        self.last_selected_method = decision["method"]

        # 4) 分数太低时，返回兜底回复
        if best_score < SIMILARITY_THRESHOLD:
            return "抱歉，我目前的知识库中没有足够相关的信息来回答您的问题。", best_score, None
        matched_query = self.queries[best_idx]
        reply = self.replies[best_idx]

        return reply, best_score, matched_query