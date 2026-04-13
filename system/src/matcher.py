import os
import time
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from src.config import (
    SIMILARITY_THRESHOLD,
    TRAIN_EMBEDDINGS_CACHE_PATH,
    MODEL_NAME,
    SIMCSE_TEMPERATURE,
    USE_EMBEDDINGS_CACHE,
    ENABLE_FALLBACK_REPLY,
    FALLBACK_TOP_K,
)


class DialogMatcher:
    def __init__(
        self,
        encoder,
        queries,
        replies,
        use_cache=USE_EMBEDDINGS_CACHE,
        cache_path=TRAIN_EMBEDDINGS_CACHE_PATH,
    ):
        # 保存基础对象，后续检索直接复用。
        self.encoder = encoder
        self.queries = queries
        self.replies = replies
        # 当前实例是否启用向量缓存（可按场景覆盖全局配置）。
        self.use_cache = bool(use_cache)
        self.cache_path = cache_path
        # 记录本次最终采用的模块，方便界面层展示。
        self.last_selected_method = "unknown"
        # 预先计算轻量特征，减少每次检索时的重复开销
        self.query_char_sets = [set(q) for q in self.queries]
        self.query_lengths = [max(1, len(q)) for q in self.queries]
        # 从编码器中取出词表与嵌入表，用于词向量匹配模块
        self.stoi = getattr(self.encoder.vocab, "stoi", {})
        self.unk_id = self.stoi.get("<UNK>", 1)
        self.word_embedding_table = self.encoder.model.embedding.weight.detach().cpu()
        if len(self.queries) > 0:
            print(f"当前加载的语料库数据量：{len(self.queries)} 条")
            
            # 读取向量缓存，避免每次启动时重复提取特征向量
            if self.use_cache and os.path.exists(self.cache_path):
                try:
                    cache_load_start = time.perf_counter()
                    print("检测到本地向量缓存，正在加载")
                    # 缓存放在 CPU，避免额外占用显存
                    cache_obj = torch.load(self.cache_path, map_location='cpu')

                    # 兼容旧版缓存（仅 tensor）
                    if isinstance(cache_obj, torch.Tensor):
                        cached_query_embeddings = cache_obj
                        cached_reply_embeddings = None
                        cached_query_word_embeddings = None
                        cached_reply_word_embeddings = None
                        cached_model_name = None
                        cached_pooling = None
                        cached_temperature = None
                        cached_score_version = None
                    else:
                        cached_query_embeddings = cache_obj.get("query_embeddings")
                        if cached_query_embeddings is None:
                            # 兼容旧字段
                            cached_query_embeddings = cache_obj.get("embeddings")
                        cached_reply_embeddings = cache_obj.get("reply_embeddings")
                        cached_query_word_embeddings = cache_obj.get("query_word_embeddings")
                        cached_reply_word_embeddings = cache_obj.get("reply_word_embeddings")
                        cached_model_name = cache_obj.get("model_name")
                        cached_temperature = cache_obj.get("simcse_temperature")
                        cached_score_version = cache_obj.get("score_version")

                    rows_ok = (
                        isinstance(cached_query_embeddings, torch.Tensor)
                        and cached_query_embeddings.size(0) == len(self.queries)
                        and isinstance(cached_reply_embeddings, torch.Tensor)
                        and cached_reply_embeddings.size(0) == len(self.replies)
                        and isinstance(cached_query_word_embeddings, torch.Tensor)
                        and cached_query_word_embeddings.size(0) == len(self.queries)
                        and isinstance(cached_reply_word_embeddings, torch.Tensor)
                        and cached_reply_word_embeddings.size(0) == len(self.replies)
                    )
                    model_ok = (cached_model_name is None) or (cached_model_name == MODEL_NAME)
                    temperature_ok = (cached_temperature is None) or (cached_temperature == SIMCSE_TEMPERATURE)
                    score_ok = (cached_score_version is None) or (cached_score_version == "dual-module-v2")

                    # 只有“数据行数、模型信息、打分版本”都能对齐时才复用缓存。
                    if rows_ok and model_ok and temperature_ok and score_ok:
                        self.query_embeddings = cached_query_embeddings
                        self.reply_embeddings = cached_reply_embeddings
                        self.query_word_embeddings = cached_query_word_embeddings
                        self.reply_word_embeddings = cached_reply_word_embeddings
                        cache_load_elapsed = time.perf_counter() - cache_load_start
                        print(f"向量缓存加载成功！耗时: {cache_load_elapsed:.2f} 秒")
                        return

                    if not rows_ok:
                        print("检测到语料已更新，正在重新构建")
                    elif not temperature_ok:
                        print("检测到 SimCSE 温度参数变更，正在重新构建")
                    elif not score_ok:
                        print("检测到打分版本变更，正在重新构建")
                    else:
                        print("缓存尺寸与当前数据不符（检测到语料已更新），正在重新构建...")
                except Exception as e:
                    print(f"缓存读取异常: {e}，将全量重新计算...")
            
            build_start = time.perf_counter()
            print("正在计算问句与答句特征向量（首次构建耗时较长，请耐心等待）...")
            # 句向量用于模块1：问句与问句匹配。
            self.query_embeddings = self.encoder.encode(self.queries)
            self.reply_embeddings = self.encoder.encode(self.replies)
            # 词向量均值用于模块2：问句与问答联合打分。
            self.query_word_embeddings = self._build_word_embeddings(self.queries)
            self.reply_word_embeddings = self._build_word_embeddings(self.replies)
            build_elapsed = time.perf_counter() - build_start
            
            if self.use_cache:
                # 保存缓存，后续可直接读取
                print("特征向量计算完毕，正在保存")
                torch.save(
                    {
                        "model_name": MODEL_NAME,
                        "simcse_temperature": SIMCSE_TEMPERATURE,
                        "score_version": "dual-module-v2",
                        "query_embeddings": self.query_embeddings,
                        "reply_embeddings": self.reply_embeddings,
                        "query_word_embeddings": self.query_word_embeddings,
                        "reply_word_embeddings": self.reply_word_embeddings,
                    },
                    self.cache_path,
                )
                print(f"本地缓存建立成功，总耗时: {build_elapsed:.2f} 秒")
            else:
                print(f"特征向量计算完毕（当前实例未启用缓存），总耗时: {build_elapsed:.2f} 秒")
        else:
            print("提示：知识库为空，请先添加数据。")
            # 空库场景下统一置空，避免后续误用。
            self.query_embeddings = None
            self.reply_embeddings = None
            self.query_word_embeddings = None
            self.reply_word_embeddings = None

    def _text_to_word_vector(self, text):
        """将文本映射为词向量平均表示（字符级）。"""
        # 逐字查词表，未知字符回退到 <UNK>。
        ids = [self.stoi.get(ch, self.unk_id) for ch in list(text)]
        if not ids:
            ids = [self.unk_id]
        id_tensor = torch.tensor(ids, dtype=torch.long)
        vecs = self.word_embedding_table[id_tensor]
        # 对句内字符向量求平均，并做归一化。
        mean_vec = vecs.mean(dim=0, keepdim=True)
        return F.normalize(mean_vec, p=2, dim=1).squeeze(0)

    def _build_word_embeddings(self, texts):
        """批量构建词向量表示并返回 CPU tensor。"""
        # 这里通常会处理很多文本，进度条可以直观看到是否卡住。
        embeddings = [
            self._text_to_word_vector(t)
            for t in tqdm(texts, desc="词向量构建进度", unit="text", dynamic_ncols=True)
        ]
        return torch.stack(embeddings, dim=0)

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

    def _query_query_similarity(self, user_text, user_emb):
        """模块1：以用户问句为中心，与语料问句做综合匹配。"""
        # 主分：语义相似度（自定义余弦实现）
        cosine_scores = self._cosine_similarity(user_emb, self.query_embeddings)

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

    def _query_reply_similarity(self, user_text):
        """模块2：词向量问答匹配（问句词向量+答句词向量联合打分）。"""
        user_text_vec = self._text_to_word_vector(user_text).unsqueeze(0)
        query_word_scores = self._cosine_similarity(user_text_vec, self.query_word_embeddings)
        reply_word_scores = self._cosine_similarity(user_text_vec, self.reply_word_embeddings)
        # 问答联合打分：更偏向问句匹配，同时考虑答句语义贴合
        return 0.6 * query_word_scores + 0.4 * reply_word_scores

    def _cosine_similarity(self, a, b, eps=1e-8):
        """手动实现余弦相似度，返回 a 与 b 每行之间的相似度矩阵。"""
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)

        # 分子是点积，分母是模长乘积。
        numerator = torch.matmul(a, b.transpose(0, 1))
        a_norm = torch.linalg.vector_norm(a, ord=2, dim=1, keepdim=True)
        b_norm = torch.linalg.vector_norm(b, ord=2, dim=1, keepdim=True).transpose(0, 1)
        denominator = torch.clamp(a_norm * b_norm, min=eps)
        scores = numerator / denominator

        if scores.size(0) == 1:
            return scores.squeeze(0)
        return scores

    def _top_indices(self, scores, top_k=3):
        """取分数最高的若干下标。"""
        if scores.numel() == 0:
            return []
        k = max(1, min(int(top_k), int(scores.numel())))
        top_vals = torch.topk(scores, k=k)
        return [int(i) for i in top_vals.indices.tolist()]

    def _build_fallback_reply(self, user_text, qq_scores, qr_scores):
        """在低分场景下，根据相近语料组织一段参考回答。"""
        # 两个模块各取若干候选，再按分数合并排序。
        qq_ids = self._top_indices(qq_scores, top_k=FALLBACK_TOP_K)
        qr_ids = self._top_indices(qr_scores, top_k=FALLBACK_TOP_K)

        merged = []
        for idx in qq_ids:
            merged.append((idx, float(qq_scores[idx].item())))
        for idx in qr_ids:
            merged.append((idx, float(qr_scores[idx].item())))

        # 同一下标可能重复出现，这里保留更高分。
        best_by_idx = {}
        for idx, score in merged:
            old = best_by_idx.get(idx)
            if old is None or score > old:
                best_by_idx[idx] = score

        ranked = sorted(best_by_idx.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[: max(1, FALLBACK_TOP_K)]

        snippets = []
        for idx, _ in ranked:
            reply = (self.replies[idx] or "").strip()
            if not reply:
                continue
            if reply not in snippets:
                snippets.append(reply)

        if not snippets:
            return "这个问题在当前语料里没有直接答案。你可以换个更具体的问法，我再试着帮你找。"

        if len(snippets) == 1:
            return (
                "这个问题在当前语料里没有一条完全对应的回答。"
                f"结合最接近的信息，我建议你先从这点考虑：{snippets[0]}"
            )

        combined = "；".join(snippets[: max(1, FALLBACK_TOP_K)])
        return (
            "这个问题在当前语料里没有直接答案。"
            "我根据几条最接近的信息整理了一个参考方向："
            f"{combined}。"
        )

    def evaluate_two_modules(self, user_text):
        """测试两种模块并返回各自得分、命中索引和最终决策。"""
        if self.query_embeddings is None or len(self.queries) == 0:
            # 没有语料时直接返回固定结构，调用方无需额外判空。
            return {
                "input": user_text,
                "query_query": {"idx": None, "score": 0.0, "matched_query": None, "reply": None},
                "wordvec_qa": {"idx": None, "score": 0.0, "matched_query": None, "reply": None},
                "selected_method": "none",
                "selected_idx": None,
                "selected_score": 0.0,
                "selected_reply": "知识库为空，暂无法回答问题。",
                "selected_query": None,
                "passed_threshold": False,
            }

        user_emb = self.encoder.encode([user_text])

        # 模块1：用户问句 vs 语料问句
        qq_scores = self._query_query_similarity(user_text, user_emb)
        qq_idx = int(torch.argmax(qq_scores).item())
        qq_score = qq_scores[qq_idx].item()

        # 模块2：用户问句 vs 问答词向量联合分
        qr_scores = self._query_reply_similarity(user_text)
        qr_idx = int(torch.argmax(qr_scores).item())
        qr_score = qr_scores[qr_idx].item()

        # 两个模块取分数更高者。
        if qq_score >= qr_score:
            selected_method = "query-query"
            selected_idx = qq_idx
            selected_score = qq_score
        else:
            selected_method = "wordvec-qa"
            selected_idx = qr_idx
            selected_score = qr_score

        # 统一阈值过滤，避免低相关结果直接返回给用户。
        passed_threshold = selected_score >= SIMILARITY_THRESHOLD
        if passed_threshold:
            selected_reply = self.replies[selected_idx]
            selected_query = self.queries[selected_idx]
        else:
            if ENABLE_FALLBACK_REPLY:
                selected_reply = self._build_fallback_reply(user_text, qq_scores, qr_scores)
            else:
                selected_reply = "抱歉，我目前的知识库中没有足够相关的信息来回答您的问题。"
            selected_query = None

        return {
            "input": user_text,
            "query_query": {
                "idx": qq_idx,
                "score": qq_score,
                "matched_query": self.queries[qq_idx],
                "reply": self.replies[qq_idx],
            },
            "wordvec_qa": {
                "idx": qr_idx,
                "score": qr_score,
                "matched_query": self.queries[qr_idx],
                "reply": self.replies[qr_idx],
            },
            "selected_method": selected_method,
            "selected_idx": selected_idx,
            "selected_score": selected_score,
            "selected_reply": selected_reply,
            "selected_query": selected_query,
            "passed_threshold": passed_threshold,
        }

    def get_best_match(self, user_text):
        """双模块检索：问句-问句 与 问句-答句并行计算，选分数更高者。"""
        result = self.evaluate_two_modules(user_text)
        self.last_selected_method = result["selected_method"]
        if not result["passed_threshold"]:
            return result["selected_reply"], result["selected_score"], None
        return result["selected_reply"], result["selected_score"], result["selected_query"]