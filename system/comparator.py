"""
system/comparator.py

文件作用:
    对话匹配核心模块。

核心检索流程（6步）:
    步骤1: 输入清洗 — 去空格、截断、过滤无效输入
    步骤2: 上下文判定 — 检测短文本/指代关键词，决定是否拼接历史
    步骤3: Query 编码 — 将文本转为 768 维语义向量
    步骤4: FAISS 粗召回 — 在 query_index 中快速召回 top-N 候选索引
    步骤5: 加权重排 — 对候选分别计算问问相似度和问答相似度，加权融合排序
    步骤6: 阈值过滤 — 最高分低于阈值则拒答，否则返回最优回复
"""

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CandidateItem = Dict[str, Any]
ScoredCandidate = Tuple[float, CandidateItem]
TraceMeta = Dict[str, Any]


# ============================================================
# 对话匹配器 (DialogComparator)
# 核心六大步骤：清洗 → 上下文 → 编码 → 粗召回 → 重排 → 过滤
# ============================================================


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
        query_texts: Optional[List[str]] = None,
        reply_texts: Optional[List[str]] = None,
        text_store: Any = None,
        rerank_weights=(0.75, 0.25),
        context_max_turns=3,
        max_text_len=64,
        context_short_query_len=14,
        context_overlap_threshold=0.34,
        context_semantic_threshold=0.58,
        context_matching_enabled: bool = True,
        coarse_recall_count: int = 500,
        rerank_top_k: int = 5,
    ):
        """
        初始化匹配器。

        参数:
            model_engine: 文本编码引擎（需提供 encode_one 方法）。
            query_index: 预加载的问句索引对象。
            response_index: 预加载的答句索引对象。
            doc_texts: 预加载的文本映射列表。
            similarity_threshold: 置信度阈值，高于该分值才返回命中回复。
            rerank_weights: 语义重排权重 (问问相似权重, 问答相似权重)。
            context_max_turns: 上下文拼接的最大历史轮数。
            max_text_len: 输入与拼接结果最大长度。
            context_short_query_len: 触发上下文判定时的短句阈值。
            context_overlap_threshold: 词面重叠阈值。
            context_semantic_threshold: 上下文语义相似阈值。
            context_matching_enabled: 是否启用上下文匹配（默认启用）。
            coarse_recall_count: 粗召回候选数量（>0 时固定使用；=0 时按自适应公式）。
            rerank_top_k: 精排阶段默认保留数量。
        """
        self._eps = 1e-12
        self.model_engine = model_engine
        self.similarity_threshold = min(max(float(similarity_threshold), 0.0), 1.0)
        self.rerank_weights = self._normalize_rerank_weights(rerank_weights)
        self.query_index = query_index
        self.response_index = response_index
        self.doc_texts: List[dict] = doc_texts if isinstance(doc_texts, list) else []
        self.text_store = text_store
        if text_store is not None:
            self.doc_count = len(text_store)
        elif query_texts is not None and reply_texts is not None:
            self.query_texts = [str(x) for x in query_texts]
            self.reply_texts = [str(x) for x in reply_texts]
            self.doc_count = min(len(self.query_texts), len(self.reply_texts))
        else:
            self.query_texts = [str(item.get("query", "") or "") for item in self.doc_texts]
            self.reply_texts = [str(item.get("reply", "") or "") for item in self.doc_texts]
            self.doc_count = min(len(self.query_texts), len(self.reply_texts))
        self.fast_candidate_multiplier = 50
        self.fast_candidate_min = 200
        # 显式粗召回数量：0 表示启用自适应计算（兼容旧逻辑）。
        self.coarse_recall_count = max(0, int(coarse_recall_count))
        # 精排默认数量：当上游未指定 top_k 时使用。
        self.rerank_top_k = max(1, int(rerank_top_k))
        total_docs = self.doc_count
        # 超大库下适当收紧候选规模，避免重排阶段开销线性放大。
        if total_docs >= 3000000:
            self.fast_candidate_multiplier = 20
            self.fast_candidate_min = 80
        elif total_docs >= 1000000:
            self.fast_candidate_multiplier = 30
            self.fast_candidate_min = 120
        # 上下文匹配约束：最多 3 轮、单条文本最多 64 字
        self.context_max_turns = max(1, int(context_max_turns))
        self.max_text_len = max(8, int(max_text_len))
        # 上下文启用判定阈值
        self.context_short_query_len = max(2, int(context_short_query_len))
        self.context_overlap_threshold = min(max(float(context_overlap_threshold), 0.0), 1.0)
        self.context_semantic_threshold = min(max(float(context_semantic_threshold), 0.0), 1.0)
        self.context_matching_enabled = bool(context_matching_enabled)
        # 上下文缓存固定保留最近 2 轮（用户+系统）。
        self.context_cache_turns = 2
        # 上下文记忆（独立于对话历史，可通过 /api/context/clear 归零）
        self.context_memory: List[str] = []

        # 轻量查询向量缓存：减少短时间内重复文本的重复编码开销。
        self._query_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_vec_cache_max_size = 1024

    def clear_context_memory(self) -> int:
        """清除上下文记忆，返回清除的条数。"""
        n = len(self.context_memory)
        self.context_memory.clear()
        return n

    def _normalize_rerank_weights(self, rerank_weights: Tuple[float, float]) -> Tuple[float, float]:
        """规范化重排权重，确保非负且和不为 0。"""
        rq = float(rerank_weights[0]) if isinstance(rerank_weights, (list, tuple)) and len(rerank_weights) > 0 else 0.75
        rr = float(rerank_weights[1]) if isinstance(rerank_weights, (list, tuple)) and len(rerank_weights) > 1 else 0.25
        rq = max(0.0, rq)
        rr = max(0.0, rr)
        if rq + rr <= self._eps:
            return 0.75, 0.25
        return rq, rr

    def _prepare_user_input(self, user_input: Any) -> str:
        """统一输入清洗与有效性校验，返回可用于检索的文本。"""
        text = self._normalize_input(str(user_input) if user_input is not None else "")
        text = self._truncate_text(text)
        if not text or self._is_invalid_after_clean(text):
            return ""
        return text

    def _encode_query_cached(self, text: str) -> np.ndarray:
        """对 query 文本做带缓存编码，返回 shape=[dim] 的 float32 向量。"""
        normalized = self._normalize_input(text)
        if not normalized:
            return np.empty((0,), dtype=np.float32)

        cached = self._query_vec_cache.get(normalized)
        if cached is not None:
            self._query_vec_cache.move_to_end(normalized)
            return cached

        vec = self.model_engine.encode_one(normalized, encoder="query", return_numpy=True)
        vec_np = np.asarray(vec, dtype=np.float32).reshape(-1)

        self._query_vec_cache[normalized] = vec_np
        if len(self._query_vec_cache) > self._query_vec_cache_max_size:
            self._query_vec_cache.popitem(last=False)

        return vec_np

    def _get_query_text(self, idx: int) -> str:
        """按索引获取问句文本。"""
        if idx < 0 or idx >= self.doc_count:
            return ""
        if self.text_store is not None:
            return self.text_store.get_query(idx)
        return self.query_texts[idx] if hasattr(self, "query_texts") else ""

    def _get_reply_text(self, idx: int) -> str:
        """按索引获取答句文本。"""
        if idx < 0 or idx >= self.doc_count:
            return ""
        if self.text_store is not None:
            return self.text_store.get_response(idx)
        return self.reply_texts[idx] if hasattr(self, "reply_texts") else ""

    def _build_item(self, idx: int) -> CandidateItem:
        """按索引构建候选条目。"""
        return {
            "query": self._get_query_text(idx),
            "reply": self._get_reply_text(idx),
            "query_idx": int(idx),
            "reply_idx": int(idx),
            "csv_idx": int(idx),
        }


    @staticmethod
    def _normalize_input(text: str) -> str:
        """轻量清洗输入文本。"""
        return " ".join((text or "").strip().split())

    def _truncate_text(self, text: str) -> str:
        """统一文本长度，限制为最多 64 字。"""
        if not text:
            return ""
        return text[: self.max_text_len]

    def _merge_context_with_budget(self, current: str, recent_contexts: List[str]) -> str:
        """在长度预算内拼接上下文，并保证当前问题完整保留。"""
        current_text = self._truncate_text(current)
        if not current_text:
            return ""

        max_len = self.max_text_len
        # 当前输入优先级最高：若已接近上限，直接使用当前输入。
        if len(current_text) >= max_len - 4:
            return current_text

        # 先为当前输入预留空间，只在剩余预算内追加历史。
        budget_left = max_len - len(current_text)
        if budget_left <= 1:
            return current_text

        selected: List[str] = []
        used = 0
        for text in reversed(recent_contexts):
            t = self._truncate_text(self._normalize_input(text))
            if not t:
                continue

            needed = len(t)
            if used + needed > budget_left:
                continue

            selected.append(t)
            used += needed

        if not selected:
            return current_text

        selected.reverse()
        return "".join(selected + [current_text]).strip()

    def build_contextual_query_with_meta(self, current_text: str, history: Optional[List[Any]]) -> Tuple[str, TraceMeta]:
        """构造上下文查询：开启时拼接最近 N 条用户输入，关闭时仅返回当前输入。"""
        current = self._truncate_text(current_text)
        meta: TraceMeta = {
            "enabled": False, "reason": "disabled_by_config",
            "history_count": 0, "current_text_len": int(len(current)),
        }

        if not self.context_matching_enabled:
            return current, meta

        # 收集历史用户输入
        recent: List[str] = []
        if history:
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_text = self._normalize_input(str(turn[0]))
                    if user_text and not self._is_invalid_after_clean(user_text):
                        recent.append(self._truncate_text(user_text))
        recent = recent[-self.context_cache_turns:]

        # 合并上下文记忆中的历史
        all_context = list(self.context_memory[-self.context_cache_turns:]) + recent

        if not all_context:
            return current, meta

        contextual = self._merge_context_with_budget(current, all_context)
        # 将当前输入存入上下文记忆
        self.context_memory.append(self._truncate_text(self._normalize_input(current_text)))
        if len(self.context_memory) > self.context_cache_turns * 2:
            self.context_memory = self.context_memory[-self.context_cache_turns:]

        meta.update({
            "enabled": True, "reason": "context_enabled",
            "history_count": int(len(all_context)),
            "contextual_query": contextual,
            "current_query": current,
        })
        return contextual, meta

    def _step12_prepare_query(
        self,
        user_input: str,
        history: Optional[List[Any]],
    ) -> Tuple[str, TraceMeta]:
        """
        步骤1+2：输入清洗 + 上下文拼接判定。

        步骤1 — 输入清洗:
            - 去多余空格、截断过长文本
            - 过滤纯数字、纯符号等无效输入

        步骤2 — 上下文拼接判定（多信号）:
            - 短文本（≤4字）→ 直接拼接最近历史
            - 命中关键词（你/它/这个...）→ 直接拼接
            - 否则保持原输入不拼接
        """
        cleaned_input = self._prepare_user_input(user_input)
        if not cleaned_input:
            return "", {
                "enabled": False,
                "reason": "invalid_input",
                "has_trigger": False,
                "is_short_query": False,
                "negative_hit": False,
                "history_count": 0,
                "current_text_len": 0,
                "recent_contexts": [],
            }

        contextual_query, context_meta = self.build_contextual_query_with_meta(cleaned_input, history)
        return contextual_query, context_meta

    def _step34_encode_and_recall(
        self,
        contextual_query: str,
        top_k_safe: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        步骤3+4：Query 编码 + FAISS 粗召回。

        步骤3 — 编码:
            将拼接后的文本通过 query_encoder 转为 768 维语义向量。
            使用 LRU 缓存避免短时间内相同文本重复编码。

        步骤4 — 粗召回:
            在 query_index 中做近似最近邻搜索，召回 coarse_recall_count 个候选。
            粗召回只取索引 ID，不做精细打分（速度优先）。
        """
        user_query_np = self._encode_query_cached(contextual_query)
        if user_query_np.size == 0:
            return user_query_np, []
        candidate_ids = self._search_candidate_ids(user_query_np, top_k=top_k_safe)
        return user_query_np, candidate_ids

    def _step5_rerank_with_query_reply_similarity(
        self,
        user_query_np: np.ndarray,
        candidate_ids: List[int],
        top_k_safe: int,
    ) -> List[ScoredCandidate]:
        """
        步骤5：加权重排（语义融合打分）。

        对每个候选同时计算两个相似度:
            - 问问相似度: 用户向量 vs 候选问句向量（query_index.reconstruct）
            - 问答相似度: 用户向量 vs 候选答句向量（response_index.reconstruct）

        最终得分 = rerank_query_weight × 问问相似度 + rerank_reply_weight × 问答相似度

        为什么这样设计?
            单靠问问相似度可能找错回复（问句相似但答句不匹配），
            加入问答相似度作为修正项能更准确地衡量"这个回复是否合适"。
        """
        if self.query_index is None:
            return []

        valid_ids: List[int] = [int(idx) for idx in candidate_ids if 0 <= int(idx) < self.doc_count]
        if not valid_ids:
            return []

        # 重构候选问句向量
        cand_query_vecs = [np.asarray(self.query_index.reconstruct(idx), dtype=np.float32) for idx in valid_ids]
        query_mat = np.vstack(cand_query_vecs)

        # 有 response_index 时做加权融合，否则纯用问句相似度
        has_response = self.response_index is not None
        if has_response:
            cand_resp_vecs = [np.asarray(self.response_index.reconstruct(idx), dtype=np.float32) for idx in valid_ids]
            resp_mat = np.vstack(cand_resp_vecs)

        q_vec = user_query_np.astype(np.float32, copy=False)
        q_norm = float(np.linalg.norm(q_vec)) + self._eps
        query_norms = np.linalg.norm(query_mat, axis=1) + self._eps
        query_sims = (query_mat @ q_vec) / (query_norms * q_norm)

        if has_response:
            resp_norms = np.linalg.norm(resp_mat, axis=1) + self._eps
            reply_sims = (resp_mat @ q_vec) / (resp_norms * q_norm)
            final_scores = self.rerank_weights[0] * query_sims + self.rerank_weights[1] * reply_sims
        else:
            final_scores = query_sims

        order = np.argsort(-final_scores)
        top_order = order[:top_k_safe]

        scored_candidates: List[ScoredCandidate] = []
        for pos in top_order:
            idx = valid_ids[int(pos)]
            scored_candidates.append((float(final_scores[int(pos)]), self._build_item(idx)))
        return scored_candidates

    def _get_topk_candidates_impl(
        self,
        user_input: str,
        history: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> Tuple[str, List[ScoredCandidate], TraceMeta]:
        """按可读 6 步流程执行检索与重排，返回候选与元信息。"""
        top_k_safe = max(1, int(top_k or self.rerank_top_k))

        # 步骤1+2：输入判断/清洗 + 上下文判定与拼接
        contextual_query, context_meta = self._step12_prepare_query(user_input=user_input, history=history)
        if not contextual_query:
            return contextual_query, [], {
                "context": context_meta,
                "retrieval": {
                    "top_k_requested": top_k_safe,
                    "candidate_count": 0,
                    "coarse_candidate_count": 0,
                },
            }

        # 步骤3+4：编码 + 粗召回
        user_query_np, candidate_ids = self._step34_encode_and_recall(
            contextual_query=contextual_query,
            top_k_safe=top_k_safe,
        )
        if user_query_np.size == 0:
            return contextual_query, [], {
                "context": context_meta,
                "retrieval": {
                    "top_k_requested": top_k_safe,
                    "candidate_count": 0,
                    "coarse_candidate_count": 0,
                },
            }

        if not candidate_ids:
            return contextual_query, [], {
                "context": context_meta,
                "retrieval": {
                    "top_k_requested": top_k_safe,
                    "candidate_count": 0,
                    "coarse_candidate_count": 0,
                },
            }

        # 步骤5：问句/答句相似度加权重排，返回 top-k
        scored_candidates = self._step5_rerank_with_query_reply_similarity(
            user_query_np=user_query_np,
            candidate_ids=candidate_ids,
            top_k_safe=top_k_safe,
        )
        valid_ids_count = int(len([int(idx) for idx in candidate_ids if 0 <= int(idx) < self.doc_count]))

        retrieval_meta: TraceMeta = {
            "top_k_requested": int(top_k_safe),
            "coarse_candidate_count": int(len(candidate_ids)),
            "candidate_count": valid_ids_count,
            "returned_count": int(len(scored_candidates)),
            "coarse_recall_count": int(getattr(self, "coarse_recall_count", 0)),
            "rerank_top_k": int(getattr(self, "rerank_top_k", top_k_safe)),
        }
        if scored_candidates:
            retrieval_meta["best_score"] = float(scored_candidates[0][0])

        return contextual_query, scored_candidates, {"context": context_meta, "retrieval": retrieval_meta}

    def get_topk_candidates_with_meta(
        self,
        user_input: str,
        history: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> Tuple[str, List[ScoredCandidate], TraceMeta]:
        """返回 top-k 候选及可解释元信息。"""
        return self._get_topk_candidates_impl(user_input=user_input, history=history, top_k=top_k)

    def compare_with_meta(
        self,
        user_input: str,
        history: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> Tuple[str, float, Optional[str], Optional[CandidateItem], str, str, TraceMeta]:
        """执行比较并返回可解释元信息。"""
        cleaned_input = self._prepare_user_input(user_input)
        if not cleaned_input:
            trace_meta: TraceMeta = {
                "context": {
                    "enabled": False,
                    "reason": "invalid_input",
                    "has_trigger": False,
                    "is_short_query": False,
                    "negative_hit": False,
                    "history_count": 0,
                    "current_text_len": 0,
                    "recent_contexts": [],
                },
                "retrieval": {
                    "top_k_requested": max(1, int(top_k or self.rerank_top_k)),
                    "candidate_count": 0,
                    "coarse_candidate_count": 0,
                },
            }
            return "输入有误，请检查后重试。", 0.0, None, None, "invalid_input", "", trace_meta

        top_k_safe = max(1, int(top_k or self.rerank_top_k))
        contextual_query, scored_candidates, trace_meta = self._get_topk_candidates_impl(
            user_input=cleaned_input,
            history=history,
            top_k=top_k_safe,
        )
        if not scored_candidates:
            return "知识库没有这个问题的回复", 0.0, None, None, "no_candidate", contextual_query, trace_meta

        best_score, best_item = scored_candidates[0]
        if best_score < self.similarity_threshold:
            return "知识库没有这个问题的回复", best_score, None, best_item, "below_threshold", contextual_query, trace_meta

        return best_item["reply"], best_score, best_item.get("query"), best_item, "matched", contextual_query, trace_meta

    @staticmethod
    def _is_invalid_after_clean(text: str) -> bool:
        """清洗后输入有效性校验：过滤纯数字、纯符号、无语义短串。"""
        if not text:
            return True
        if len(text) < 2:
            return True
        # 全数字
        if text.isdigit():
            return True
        # 仅下划线
        if re.fullmatch(r"_+", text):
            return True
        # 只有符号
        if re.fullmatch(r"[\W_]+", text):
            return True
        return False

    @staticmethod
    def _cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两向量余弦相似度。"""
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 <= 1e-12 or norm2 <= 1e-12:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _search_candidate_ids(self, query_vector: np.ndarray, top_k: int = 5) -> List[int]:
        """
        使用高性能索引先召回候选 id。

        说明:
            这里返回候选集合（可能大于 top_k），再由后续重排逻辑计算最终分数。
        """
        if self.query_index is None:
            return []

        total = int(self.query_index.ntotal)
        if total <= 0:
            return []

        # 利用索引做快速候选召回（只取 id，不直接使用其分数）。
        # 优先使用显式粗召回数量；未配置时回退自适应策略。
        if self.coarse_recall_count > 0:
            candidate_k = min(max(self.coarse_recall_count, top_k), total)
        else:
            candidate_k = min(max(top_k * self.fast_candidate_multiplier, self.fast_candidate_min), total)
        if candidate_k <= 0:
            return []

        query_np = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        _, indices = self.query_index.search(query_np, candidate_k)  # type: ignore

        candidate_ids: List[int] = []
        visited = set()
        for idx in indices[0].tolist():
            if idx == -1 or idx >= self.doc_count:
                continue
            if idx in visited:
                continue
            visited.add(idx)
            candidate_ids.append(int(idx))
        return candidate_ids

    def compare(
        self,
        user_input: str,
        history: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> Tuple[str, float, Optional[str], Optional[CandidateItem], str, str]:
        """
        语义匹配逻辑：
        1. 从数据库检索最相近的 top_k 条候选；
        2. 计算用户输入与候选问答向量的语义相似度；
        3. 按语义重排权重选出最优回复。

        参数:
            user_input: 用户自然语言输入。

        返回:
            (回复文本, 最终分数, 命中问句, 命中条目, 结果类型, 最终编码问句)
        """
        reply, score, matched_q, matched_item, result_type, contextual_query, _ = self.compare_with_meta(
            user_input=user_input,
            history=history,
            top_k=top_k,
        )
        return reply, score, matched_q, matched_item, result_type, contextual_query

    def get_topk_candidates(
        self,
        user_input: str,
        history: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> Tuple[str, List[ScoredCandidate]]:
        """
        返回按融合分排序的 top-k 候选。

        返回:
            (最终编码问句, [(融合分, 候选条目), ...])
        """
        contextual_query, scored_candidates, _ = self._get_topk_candidates_impl(
            user_input=user_input,
            history=history,
            top_k=top_k,
        )
        return contextual_query, scored_candidates