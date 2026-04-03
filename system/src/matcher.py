import os
import torch
from src.config import SIMILARITY_THRESHOLD, EMBEDDINGS_CACHE_PATH

class DialogMatcher:
    def __init__(self, encoder, queries, replies):
        self.encoder = encoder
        self.queries = queries
        self.replies = replies
        if len(self.queries) > 0:
            print(f"当前加载的语料库数据量：{len(self.queries)} 条")
            
            # 引入向缓存，避免每次启动时重复提取特征向量
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                try:
                    print("检测到本地向量缓存，正在加载...")
                    # 默认使用 CPU 进行相似度计算运算以节省显存
                    cached_embeddings = torch.load(EMBEDDINGS_CACHE_PATH, map_location='cpu')
                    # 校验尺寸以确认缓存和数据是否对应
                    if cached_embeddings.size(0) == len(self.queries):
                        self.corpus_embeddings = cached_embeddings
                        print("向量缓存加载成功！")
                        return
                    else:
                        print("缓存尺寸与当前数据不符（检测到语料已更新），正在重新构建...")
                except Exception as e:
                    print(f"缓存读取异常: {e}，将全量重新计算...")
            
            print("正在计算所有问句的特征向量（首次构建耗时较长，请耐心等待）...")
            self.corpus_embeddings = self.encoder.encode(self.queries)
            
            # 将计算好的特征矩阵保存到本地以备下一次启动使用
            print("特征向量计算完毕，正在保存至本地硬盘...")
            torch.save(self.corpus_embeddings, EMBEDDINGS_CACHE_PATH)
            print("本地缓存建立成功！")
        else:
            print("提示：知识库为空，请先添加数据。")
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