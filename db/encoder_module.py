import os
import sys
import argparse
from typing import List

import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from system.model_engine import SimCSEModelEngine
from system.config import DB_CSV_PATH, DB_DATA_DIR, DB_PREFIX


class DataLoader:
    """数据读取模块：负责从 CSV 中高效读取问答语料。"""

    @staticmethod
    def _count_csv_rows(csv_path: str) -> int:
        """快速统计 CSV 文件中的总记录行数（不含标题）。"""
        with open(csv_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f) - 1
        return max(total, 0)

    @staticmethod
    def load_corpus(csv_path: str, n_samples=None):
        """
        加载语料并返回 queries, replies。

        - n_samples > 0: 读取前 n_samples 条
        - n_samples <= 0 或 None: 读取全量
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"指定的语料文件未找到: {csv_path}")

        total_rows = DataLoader._count_csv_rows(csv_path)
        sample_limit = int(n_samples) if n_samples is not None else 0
        use_limit = sample_limit > 0
        target_rows = min(total_rows, sample_limit) if use_limit else total_rows

        queries: List[str] = []
        replies: List[str] = []

        # 分块读取避免一次性占满内存。
        chunk_size = 100000
        progress = tqdm(total=target_rows, desc="正在分块加载语料库", unit="条")

        loaded = 0
        for chunk in pd.read_csv(csv_path, usecols=["query", "response"], chunksize=chunk_size, engine="c"):
            chunk = chunk.fillna("")
            q = chunk["query"].astype(str).tolist()
            r = chunk["response"].astype(str).tolist()

            if use_limit and loaded + len(q) > sample_limit:
                keep = sample_limit - loaded
                q = q[:keep]
                r = r[:keep]

            queries.extend(q)
            replies.extend(r)
            loaded += len(q)
            progress.update(len(q))

            if use_limit and loaded >= sample_limit:
                break

        progress.close()
        return queries, replies


class VectorDB:
    """
    向量数据库读写模块：
    1. 写入 query/response 双索引
    2. 从双索引和 CSV 恢复文本映射
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.query_index = self._build_index(dimension)
        self.response_index = self._build_index(dimension)
        self.query_texts: List[str] = []
        self.reply_texts: List[str] = []
        self.doc_texts: List[dict] = []

    @staticmethod
    def _to_numpy_float32(embeddings):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings.astype("float32")

    @staticmethod
    def _build_index(dimension: int):
        return faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

    @staticmethod
    def _save_index(index_obj, file_path: str):
        target_dir = os.path.dirname(os.path.abspath(file_path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        faiss.write_index(index_obj, file_path)

    @staticmethod
    def _read_index_safe(file_path: str):
        return faiss.read_index(file_path)

    @staticmethod
    def _file_pack(path: str) -> dict:
        return {
            "query_index": path + "_query.index",
            "response_index": path + "_response.index",
        }

    def _rebuild_doc_texts(self):
        """按索引位置重建内存视图（query/reply 一一对应）。"""
        self.doc_texts = []
        pair_count = min(len(self.query_texts), len(self.reply_texts))
        for idx in range(pair_count):
            self.doc_texts.append(
                {
                    "query": self.query_texts[idx],
                    "reply": self.reply_texts[idx],
                    "query_idx": idx,
                    "reply_idx": idx,
                    "csv_idx": idx,
                }
            )

    def add_data(
        self,
        queries: List[str],
        query_embeddings: torch.Tensor,
        replies: List[str],
        reply_embeddings: torch.Tensor,
    ):
        """批量写入索引与文本映射。"""
        query_np = self._to_numpy_float32(query_embeddings)
        reply_np = self._to_numpy_float32(reply_embeddings)

        self.query_index.add(query_np)  # type: ignore
        self.response_index.add(reply_np)  # type: ignore

        self.query_texts.extend(queries)
        self.reply_texts.extend(replies)
        self._rebuild_doc_texts()

        print(f"成功导入批次数据：知识库当前记录总条数 = {len(self.doc_texts)}")

    def save(self, path: str):
        """写入双索引文件。"""
        target_dir = os.path.dirname(os.path.abspath(path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        files = self._file_pack(path)

        save_bar = tqdm(total=2, desc="正在保存数据库文件", unit="项", dynamic_ncols=True)
        self._save_index(self.query_index, files["query_index"])
        save_bar.update(1)
        self._save_index(self.response_index, files["response_index"])
        save_bar.update(1)
        save_bar.close()

        print(f"向量索引与文本语料已成功保存至: {path}")

class CorpusEncoder:
    """写入入口模块：读取语料 -> 编码 -> 写入双索引。"""

    def __init__(self, model_engine, data_dir: str):
        self.model_engine = model_engine
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def encode_corpus_to_data(self, csv_path: str, n_samples=None, batch_size: int = 0, prefix: str = ""):
        queries, replies = DataLoader.load_corpus(csv_path, n_samples=n_samples)

        print("正在编码问句向量 (Encoder: query)...")
        query_tensor = self.model_engine.encode(
            queries,
            encoder="query",
            batch_size=batch_size,
            show_progress=True,
        )

        print("正在编码答句向量 (Encoder: response)...")
        reply_tensor = self.model_engine.encode(
            replies,
            encoder="response",
            batch_size=batch_size,
            show_progress=True,
        )

        faiss_db_path = os.path.join(self.data_dir)
        print("正在构建FAISS索引数据库...")

        db = VectorDB(dimension=768)
        db.add_data(
            queries,
            query_tensor,
            replies,
            reply_tensor,
        )
        db.save(faiss_db_path)

        print(f"知识库已成功持久化至数据库: {faiss_db_path}")
        return {
            "faiss_db": faiss_db_path,
            "size": len(queries),
        }


def main():
    parser = argparse.ArgumentParser(description="语料入库工具")
    parser.add_argument("--data_csv", type=str, default=DB_CSV_PATH)
    parser.add_argument("--data_dir", type=str, default=DB_DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--prefix", type=str, default=DB_PREFIX)
    args = parser.parse_args()

    engine = SimCSEModelEngine()
    encoder = CorpusEncoder(model_engine=engine, data_dir=args.data_dir)

    result = encoder.encode_corpus_to_data(
        csv_path=args.data_csv,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        prefix=args.prefix,
    )
    print(f"处理完成: {result}")


if __name__ == "__main__":
    main()