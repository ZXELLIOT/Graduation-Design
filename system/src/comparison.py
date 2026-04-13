import torch


class SimilarityComparator:
    """比较两套相似度分数，并返回最高分对应的结果。"""

    @staticmethod
    def _best_of(scores):
        if scores is None or scores.numel() == 0:
            raise ValueError("scores 不能为空")

        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        return best_idx, best_score

    @classmethod
    def compare(cls, current_scores, semantic_scores):
        """在“当前比对方式”和“SimCSE 纯语义”间选取最高分。"""
        cur_idx, cur_score = cls._best_of(current_scores)
        sem_idx, sem_score = cls._best_of(semantic_scores)

        # 分数相同默认保留当前策略，减少行为抖动。
        if sem_score > cur_score:
            return {
                "method": "semantic",
                "best_idx": sem_idx,
                "best_score": sem_score,
            }

        return {
            "method": "current",
            "best_idx": cur_idx,
            "best_score": cur_score,
        }
