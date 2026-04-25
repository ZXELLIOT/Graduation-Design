"""
system/comparator.py

文件作用:
    对话匹配核心模块。
    负责输入清洗、上下文拼接、候选召回和语义重排。
"""

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CandidateItem = Dict[str, Any]
ScoredCandidate = Tuple[float, CandidateItem]
TraceMeta = Dict[str, Any]


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
        if query_texts is not None and reply_texts is not None:
            self.query_texts: List[str] = [str(x) for x in query_texts]
            self.reply_texts: List[str] = [str(x) for x in reply_texts]
        else:
            self.query_texts = [str(item.get("query", "") or "") for item in self.doc_texts]
            self.reply_texts = [str(item.get("reply", "") or "") for item in self.doc_texts]
        self.doc_count = min(len(self.query_texts), len(self.reply_texts))
        self.queries: List[str] = self.query_texts
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
        # 仅当输入包含这些“指向上文”的词语时才触发上下文匹配
        self.context_trigger_pattern = re.compile(
            r"(你|它|这个|那款|哪里|怎么|多少|那个|这件|那件|这里|那里|上文|前文|后文|上下文|"
            r"上面|前面|上个|上一|上次|刚才|之前|继续|接着|"
            r"\bthis\b|\bthat\b|\bit\b|\bwhere\b|\bhow\b|\bmuch\b|\bmore\b|\bagain\b)",
            re.IGNORECASE,
        )
        # 明确切换话题时，禁止上下文拼接
        self.context_negative_pattern = re.compile(
            r"(另外|顺便|换个|新问题|题外话|另一个问题|by\s+the\s+way|new\s+question|another\s+topic)",
            re.IGNORECASE,
        )

        # 轻量查询向量缓存：减少短时间内重复文本的重复编码开销。
        self._query_vec_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_vec_cache_max_size = 1024

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
        return self.query_texts[idx]

    def _get_reply_text(self, idx: int) -> str:
        """按索引获取答句文本。"""
        if idx < 0 or idx >= self.doc_count:
            return ""
        return self.reply_texts[idx]

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
    def _tokenize_for_overlap(text: str) -> set:
        """提取用于词面重叠计算的中英文词元。"""
        if not text:
            return set()
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
        return {t for t in tokens if t}

    def _lexical_overlap_ratio(self, current_text: str, recent_contexts: List[str]) -> float:
        """计算当前输入与最近上下文的词面重叠比例（Jaccard）。"""
        if not current_text or not recent_contexts:
            return 0.0

        current_tokens = self._tokenize_for_overlap(current_text)
        if not current_tokens:
            return 0.0

        context_tokens = set()
        for text in recent_contexts:
            context_tokens |= self._tokenize_for_overlap(text)

        if not context_tokens:
            return 0.0

        inter = len(current_tokens & context_tokens)
        union = len(current_tokens | context_tokens)
        if union == 0:
            return 0.0
        return inter / union

    def _semantic_recent_similarity(self, current_text: str, recent_contexts: List[str], early_stop: float = 1.0) -> float:
        """计算当前输入与最近历史的最高语义相似度。"""
        if not current_text or not recent_contexts:
            return 0.0

        try:
            current_np = self._encode_query_cached(current_text)
            if current_np.size == 0:
                return 0.0

            best = 0.0
            for text in reversed(recent_contexts):
                if not text:
                    continue
                hist_np = self._encode_query_cached(text)
                if hist_np.size == 0:
                    continue
                sim = self._cosine_sim(current_np, hist_np)
                if sim > best:
                    best = sim
                if best >= early_stop:
                    break
            return float(best)
        except Exception:
            # 语义计算失败时退化为词面规则，不阻塞主流程
            return 0.0

    @staticmethod
    def _normalize_input(text: str) -> str:
        """轻量清洗输入文本。"""
        return " ".join((text or "").strip().split())

    def _truncate_text(self, text: str) -> str:
        """统一文本长度，限制为最多 64 字。"""
        if not text:
            return ""
        return text[: self.max_text_len]

    def _extract_recent_context(self, history: Optional[List[Any]]) -> List[str]:
        """缓存并提取最近 2 轮上下文（用户+系统）。"""
        if not history:
            self._recent_dialog_cache = []
            return []

        rounds: List[Tuple[str, str]] = []
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_part = turn[0]
                bot_part = turn[1]
                if isinstance(user_part, str) and isinstance(bot_part, str):
                    user_clean = self._normalize_input(user_part)
                    bot_clean = self._normalize_input(bot_part)
                    if user_clean and bot_clean and not self._is_invalid_after_clean(user_clean):
                        rounds.append((self._truncate_text(user_clean), self._truncate_text(bot_clean)))
            elif isinstance(turn, dict):
                # 若是 message 列表格式(dict)，该函数不做复杂配对，避免引入歧义。
                continue

        if not rounds:
            self._recent_dialog_cache = []
            return []

        self._recent_dialog_cache: List[Tuple[str, str]] = rounds[-self.context_cache_turns :]
        context_lines: List[str] = []
        for user_text, bot_text in self._recent_dialog_cache:
            context_lines.append(f"用户:{user_text} 系统:{bot_text}")

        return context_lines

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

            needed = len(t) + 1  # 包含分隔空格
            if used + needed > budget_left:
                continue

            selected.append(t)
            used += needed

        if not selected:
            return current_text

        selected.reverse()
        return " ".join(selected + [current_text]).strip()

    def _evaluate_context_need(self, current_text: str, recent_contexts: Optional[List[str]] = None) -> TraceMeta:
        """按规则判断是否需要上下文：短文本或命中关键词才启用。"""
        current = self._truncate_text(self._normalize_input(current_text))
        contexts = recent_contexts or []

        meta: TraceMeta = {
            "enabled": False,
            "reason": "empty_input",
            "has_trigger": False,
            "is_short_query": False,
            "overlap_ratio": 0.0,
            "semantic_ratio": 0.0,
            "semantic_checked": False,
            "context_score": 0.0,
            "negative_hit": False,
            "history_count": int(len(contexts)),
            "current_text_len": int(len(current)),
            "recent_contexts": list(contexts),
        }

        if not current:
            return meta

        negative_hit = False
        has_trigger = bool(self.context_trigger_pattern.search(current))
        is_short_query = len(current) <= 4

        meta.update(
            {
                "has_trigger": bool(has_trigger),
                "is_short_query": bool(is_short_query),
                "overlap_ratio": 0.0,
                "negative_hit": bool(negative_hit),
            }
        )

        if not contexts:
            meta["reason"] = "no_history"
            return meta

        if is_short_query:
            meta["enabled"] = True
            meta["reason"] = "short_query"
            meta["context_score"] = 1.0
            return meta

        # 关键词即触发：只要命中“指向上文”关键词并且存在历史，直接启用上下文。
        if has_trigger:
            meta["enabled"] = True
            meta["reason"] = "trigger_keyword"
            meta["context_score"] = 1.0
            return meta

        meta["reason"] = "no_context_needed"
        return meta

    def build_contextual_query_with_meta(self, current_text: str, history: Optional[List[Any]]) -> Tuple[str, TraceMeta]:
        """构造上下文查询并返回判定元信息。"""
        current = self._truncate_text(current_text)
        if not self.context_matching_enabled:
            return current, {
                "enabled": False,
                "reason": "disabled_by_config",
                "has_trigger": False,
                "is_short_query": False,
                "overlap_ratio": 0.0,
                "semantic_ratio": 0.0,
                "semantic_checked": False,
                "context_score": 0.0,
                "negative_hit": False,
                "history_count": 0,
                "current_text_len": int(len(current)),
                "recent_contexts": [],
            }

        recent_contexts = self._extract_recent_context(history)
        context_meta = self._evaluate_context_need(current, recent_contexts)
        if not bool(context_meta.get("enabled", False)):
            return current, context_meta
        if not recent_contexts:
            context_meta["enabled"] = False
            context_meta["reason"] = "no_history"
            return current, context_meta

        contextual_query = self._merge_context_with_budget(current, recent_contexts)
        context_meta["contextual_query"] = contextual_query
        context_meta["current_query"] = current
        context_meta["context_used"] = [x for x in recent_contexts if x and x in contextual_query and x != current]
        return contextual_query, context_meta

    def _build_contextual_query(self, current_text: str, history: Optional[List[Any]]) -> str:
        """构造上下文查询：最近 3 轮用户文本 + 当前输入。"""
        contextual_query, _ = self.build_contextual_query_with_meta(current_text=current_text, history=history)
        return contextual_query

    def _need_context_match(self, current_text: str, recent_contexts: Optional[List[str]] = None) -> bool:
        """使用多信号判定是否启用上下文匹配。"""
        return bool(self._evaluate_context_need(current_text, recent_contexts).get("enabled", False))

    def _step12_prepare_query(
        self,
        user_input: str,
        history: Optional[List[Any]],
    ) -> Tuple[str, TraceMeta]:
        """步骤1+2：输入清洗，并按规则决定是否拼接上下文。"""
        cleaned_input = self._prepare_user_input(user_input)
        if not cleaned_input:
            return "", {
                "enabled": False,
                "reason": "invalid_input",
                "has_trigger": False,
                "is_short_query": False,
                "overlap_ratio": 0.0,
                "semantic_ratio": 0.0,
                "semantic_checked": False,
                "context_score": 0.0,
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
        """步骤3+4：对查询编码，并进行粗召回。"""
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
        """步骤5：对候选问句/答句做加权重排，返回 top-k。"""
        if self.query_index is None or self.response_index is None:
            return []

        valid_ids: List[int] = [int(idx) for idx in candidate_ids if 0 <= int(idx) < self.doc_count]
        if not valid_ids:
            return []

        cand_query_vecs: List[np.ndarray] = []
        cand_resp_vecs: List[np.ndarray] = []
        for idx in valid_ids:
            cand_query_vecs.append(np.asarray(self.query_index.reconstruct(idx), dtype=np.float32))  # type: ignore
            cand_resp_vecs.append(np.asarray(self.response_index.reconstruct(idx), dtype=np.float32))  # type: ignore

        query_mat = np.vstack(cand_query_vecs)
        resp_mat = np.vstack(cand_resp_vecs)

        q_vec = user_query_np.astype(np.float32, copy=False)
        q_norm = float(np.linalg.norm(q_vec)) + self._eps
        query_norms = np.linalg.norm(query_mat, axis=1) + self._eps
        resp_norms = np.linalg.norm(resp_mat, axis=1) + self._eps

        query_sims = (query_mat @ q_vec) / (query_norms * q_norm)
        reply_sims = (resp_mat @ q_vec) / (resp_norms * q_norm)
        final_scores = self.rerank_weights[0] * query_sims + self.rerank_weights[1] * reply_sims

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
                    "overlap_ratio": 0.0,
                    "semantic_ratio": 0.0,
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