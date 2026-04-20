import re
import numpy as np
import torch
from typing import List


class DialogComparator:
    """
    语义对话匹配器
    将用户输入编码为向量后，在向量库中检索候选问答并进行加权重排。
    """

    def __init__(
        self,
        model_engine,
        query_index,
        response_index,
        doc_texts,
        similarity_threshold,
        rerank_weights=(0.75, 0.25)
    ):
        """
        初始化匹配器。

        参数:
            model_engine: 文本编码引擎（需提供 encode_one 方法）。
            query_index: 预加载的问句索引对象。
            response_index: 预加载的答句索引对象。
            doc_texts: 预加载的文本映射列表。
            similarity_threshold: 置信度阈值，高于该分值才返回命中回复。
            rerank_weights: 加权重排权重 (问问相似权重, 问答相似权重)。
        """
        self.model_engine = model_engine
        self.similarity_threshold = similarity_threshold
        self.rerank_weights = rerank_weights
        self.query_index = query_index
        self.response_index = response_index
        self.doc_texts: List[dict] = doc_texts
        self.queries: List[dict] = doc_texts
        self.fast_candidate_multiplier = 50
        self.fast_candidate_min = 200

    @staticmethod
    def _normalize_input(text: str) -> str:
        """轻量清洗输入文本。"""
        return " ".join((text or "").strip().split())

    @staticmethod
    def _is_invalid_after_clean(text: str) -> bool:
        """清洗后输入有效性校验：过滤纯数字、纯英文数字、无中文无语义的短串。"""
        if not text:
            return True
        if len(text) < 2:
            return True
        # 全数字
        if text.isdigit():
            return True
        # 全英文/数字/下划线
        if re.fullmatch(r"[A-Za-z0-9_]+", text):
            return True
        # 只有符号
        if re.fullmatch(r"[\W_]+", text):
            return True
        return False

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        """返回 L2 归一化后的向量。"""
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    @staticmethod
    def _cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两向量余弦相似度。"""
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 <= 1e-12 or norm2 <= 1e-12:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _search_candidate_ids(self, query_vector: torch.Tensor, top_k: int = 5) -> List[int]:
        """
        使用高性能索引先召回候选 id。
        """
        total = int(self.query_index.ntotal)
        # 利用索引做快速候选召回（只取 id，不直接使用其分数）。
        candidate_k = min(max(top_k * self.fast_candidate_multiplier, self.fast_candidate_min), total)
        query_np = query_vector.detach().cpu().numpy().astype("float32").reshape(1, -1)
        _, indices = self.query_index.search(query_np, candidate_k)  # type: ignore

        candidate_ids: List[int] = []
        visited = set()
        for idx in indices[0].tolist():
            if idx == -1 or idx >= len(self.doc_texts):
                continue
            if idx in visited:
                continue
            visited.add(idx)
            candidate_ids.append(int(idx))
            if len(candidate_ids) >= top_k:
                break
        return candidate_ids

    def compare(self, user_input: str):
        """
        双路加权匹配逻辑：
        1. 从数据库检索最相近的 top_k 条候选；
        2. 计算用户输入与候选回复的语义相似度；
        3. 按权重融合两路分数，选出最优回复。

        参数:
            user_input: 用户自然语言输入。
        """
        # 先做输入清洗，保证后续编码阶段输入稳定。
        cleaned_input = self._normalize_input(str(user_input) if user_input is not None else "")
        if not cleaned_input:
            return "您似乎没有输入任何内容，请试着问我点什么吧。", 0.0, None

        # 清洗后再次校验
        if self._is_invalid_after_clean(cleaned_input):
            return "您输入的内容我暂时无法理解，可以尝试换一种说法吗？", 0.0, None

        # 1. 编码用户输入（后续不再额外归一化）
        user_query_emb = self.model_engine.encode_one(cleaned_input, encoder="query")

        # 2. 使用高性能检索召回候选 id（top-5）
        candidate_ids = self._search_candidate_ids(user_query_emb, top_k=5)

        if not candidate_ids:
            return "知识库没有这个问题的回复", 0.0, None

        q_query_np = user_query_emb.detach().cpu().reshape(-1).float().numpy()

        # 3. 对候选做双分数计算并按 0.75/0.25 加权。
        scored_candidates = []
        for idx in candidate_ids:
            item = self.doc_texts[idx]
            cand_query = np.asarray(self.query_index.reconstruct(int(idx)), dtype=np.float32)  # type: ignore
            cand_resp = np.asarray(self.response_index.reconstruct(int(idx)), dtype=np.float32)  # type: ignore

            query_sim = self._cosine_sim(q_query_np, cand_query)
            reply_sim = self._cosine_sim(q_query_np, cand_resp)
            final_score = self.rerank_weights[0] * query_sim + self.rerank_weights[1] * reply_sim
            scored_candidates.append((float(final_score), item))

        if not scored_candidates:
            return "知识库没有这个问题的回复", 0.0, None

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_item = scored_candidates[0]

        # 4. 阈值校验
        if best_score < self.similarity_threshold:
            debug_reply = (
                "知识库没有这个问题的回复\n"
                f"最高分候选问句: {best_item.get('query', '')}\n"
                f"最高分候选答句: {best_item.get('reply', '')}"
            )
            return debug_reply, best_score, None

        return best_item["reply"], best_score, best_item.get("query")