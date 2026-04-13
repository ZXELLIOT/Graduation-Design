import os
import sys
import unittest

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(CURRENT_DIR)
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from src.comparison import SimilarityComparator
from src.matcher import DialogMatcher


class DummyEncoder:
    def __init__(self, emb):
        self.emb = emb

    def encode(self, texts):
        return self.emb


class SimilarityComparatorTests(unittest.TestCase):
    def test_selects_semantic_when_higher(self):
        current_scores = torch.tensor([0.61, 0.72, 0.58], dtype=torch.float32)
        semantic_scores = torch.tensor([0.60, 0.74, 0.57], dtype=torch.float32)

        decision = SimilarityComparator.compare(current_scores, semantic_scores)

        self.assertEqual(decision["method"], "semantic")
        self.assertEqual(decision["best_idx"], 1)
        self.assertAlmostEqual(decision["best_score"], 0.74, places=6)

    def test_selects_current_when_higher(self):
        current_scores = torch.tensor([0.83, 0.42], dtype=torch.float32)
        semantic_scores = torch.tensor([0.80, 0.50], dtype=torch.float32)

        decision = SimilarityComparator.compare(current_scores, semantic_scores)

        self.assertEqual(decision["method"], "current")
        self.assertEqual(decision["best_idx"], 0)
        self.assertAlmostEqual(decision["best_score"], 0.83, places=6)

    def test_tie_prefers_current(self):
        current_scores = torch.tensor([0.60, 0.71], dtype=torch.float32)
        semantic_scores = torch.tensor([0.71, 0.40], dtype=torch.float32)

        decision = SimilarityComparator.compare(current_scores, semantic_scores)

        self.assertEqual(decision["method"], "current")
        self.assertEqual(decision["best_idx"], 1)
        self.assertAlmostEqual(decision["best_score"], 0.71, places=6)

    def test_raises_on_empty_scores(self):
        with self.assertRaises(ValueError):
            SimilarityComparator.compare(torch.tensor([]), torch.tensor([0.5]))


class DialogMatcherComparisonFlowTests(unittest.TestCase):
    def _build_matcher(self):
        matcher = DialogMatcher.__new__(DialogMatcher)
        matcher.queries = ["q0", "q1"]
        matcher.replies = ["r0", "r1"]
        matcher.corpus_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        matcher.encoder = DummyEncoder(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        return matcher

    def test_get_best_match_uses_higher_method_score(self):
        matcher = self._build_matcher()

        matcher._custom_similarity = lambda user_text, user_emb: torch.tensor([0.60, 0.55])
        matcher._semantic_similarity = lambda user_emb: torch.tensor([0.58, 0.82])

        reply, score, matched_q = matcher.get_best_match("any")

        self.assertEqual(reply, "r1")
        self.assertEqual(matched_q, "q1")
        self.assertAlmostEqual(score, 0.82, places=6)
        self.assertEqual(matcher.last_selected_method, "semantic")

    def test_get_best_match_fallback_when_below_threshold(self):
        matcher = self._build_matcher()

        matcher._custom_similarity = lambda user_text, user_emb: torch.tensor([0.20, 0.31])
        matcher._semantic_similarity = lambda user_emb: torch.tensor([0.29, 0.45])

        reply, score, matched_q = matcher.get_best_match("any")

        self.assertIsNone(matched_q)
        self.assertAlmostEqual(score, 0.45, places=6)
        self.assertIn("抱歉", reply)


if __name__ == "__main__":
    unittest.main()
