import torch
from src.config import SIMILARITY_THRESHOLD

class DialogMatcher:
    def __init__(self, encoder, queries, replies):
        self.encoder = encoder
        self.queries = queries
        self.replies = replies
        if len(self.queries) > 0:
            print("正在为知识库建立向量索引...")
            # 预计算：把库里的所有问题都提前变成向量存起来
            self.corpus_embeddings = self.encoder.encode(self.queries)
            print("向量索引构建成功！")
        else:
            print("警告：知识库为空，请先添加数据！")
            self.corpus_embeddings = None

    def get_best_match(self, user_text):
        """计算余弦相似度并找出最佳回答"""
        if self.corpus_embeddings is None or len(self.queries) == 0:
            return "知识库为空，暂无法回答问题。", 0.0, None
        # 1. 编码用户的实时输入
        user_emb = self.encoder.encode([user_text])
        # 2. 向量点乘（即余弦相似度），找出与库中哪一句话最相似
        similarities = torch.cosine_similarity(user_emb, self.corpus_embeddings)
        # 3. 取得分最高的一项
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        # 4. 最低阈值回复
        if best_score < SIMILARITY_THRESHOLD:
            return "抱歉，我目前的知识库中没有足够相关的信息来回答您的问题。", best_score, None
        matched_query = self.queries[best_idx]
        reply = self.replies[best_idx]

        return reply, best_score, matched_query