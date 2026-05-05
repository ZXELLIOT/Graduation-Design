"""
db/encoder_module.py

文件作用:
    问句向量化与索引入库。

核心流程:
    分块读取 CSV 问句列 → query_encoder 编码 → 写入 FAISS 索引
    FAISS 索引中向量位置 = CSV 行号，答句通过行号从磁盘按需读取。

索引策略:
    - 小规模 (<10000): HNSW 图索引
    - 大规模 (>10000): IVF-PQ 压缩索引
"""

import os
import sys
import argparse
from typing import Generator, List

import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from system.model_engine import SimCSEModelEngine
from db.db_config import DB_CSV_PATH, DB_INDEX_PATH

DEFAULT_VECTOR_DIM = 768


class DataLoader:
    """CSV 问句读取，支持全量/采样/分块流式。"""

    @staticmethod
    def _count_rows(csv_path: str) -> int:
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)

    @staticmethod
    def resolve_target_rows(csv_path: str, n_samples=None) -> int:
        total = DataLoader._count_rows(csv_path)
        limit = int(n_samples) if n_samples is not None and int(n_samples) > 0 else 0
        return min(total, limit) if limit else total

    @staticmethod
    def iter_queries(
        csv_path: str,
        n_samples=None,
        start_offset: int = 0,
        chunk_size: int = 10000,
        target_rows: int | None = None,
    ) -> Generator[List[str], None, None]:
        """分块流式读取问句列，支持断点续跑。"""
        target = (
            target_rows
            if target_rows is not None
            else DataLoader.resolve_target_rows(csv_path, n_samples=n_samples)
        )
        if start_offset >= target:
            return

        remaining = target - start_offset
        scanned = 0
        emitted = 0

        for chunk in pd.read_csv(csv_path, usecols=["query"], chunksize=chunk_size, engine="c"):
            if scanned >= target:
                break

            chunk = chunk.fillna("")
            queries = chunk["query"].astype(str).tolist()
            keep = min(len(queries), target - scanned)

            if start_offset > scanned:
                skip = min(keep, start_offset - scanned)
                queries = queries[skip:]
                keep -= skip

            if keep <= 0:
                scanned += len(chunk["query"])
                continue

            queries = queries[:keep]
            scanned += len(chunk["query"])
            emitted += len(queries)
            yield queries

            if emitted >= remaining:
                break


class VectorDB:
    """FAISS 问句索引读写。向量位置 = CSV 行号。"""

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIM):
        self.dimension = dimension
        self.index = self._build_index(dimension)

    @staticmethod
    def _to_float32(embeddings):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings.astype("float32")

    @staticmethod
    def _build_index(dimension: int, use_ivf=False, n_total=0):
        if use_ivf and n_total > 10000:
            nlist = int(4 * (n_total ** 0.5))
            quantizer = faiss.IndexFlatIP(dimension)
            idx = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8, faiss.METRIC_INNER_PRODUCT)
            idx.nprobe = 10
            return idx
        return faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

    def add(self, embeddings: torch.Tensor):
        vecs = self._to_float32(embeddings)
        self.index.add(vecs)  # type: ignore[call-arg]

    def save(self, path: str):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, dimension: int = DEFAULT_VECTOR_DIM):
        if not os.path.exists(path):
            raise FileNotFoundError(f"索引文件不存在: {path}")
        db = cls(dimension=dimension)
        db.index = faiss.read_index(path)
        return db


class CorpusEncoder:
    """编码入库：读 CSV 问句 → 编码 → 写入索引。"""

    def __init__(self, model_engine):
        self.model_engine = model_engine

    def encode_corpus_to_data(
        self,
        csv_path: str,
        index_path: str,
        n_samples=0,
        batch_size: int = 128,
        chunk_size: int = 10000,
        checkpoint_every: int = 5,
    ):
        """
        编码入库主流程。

        参数:
            csv_path:      语料 CSV 路径。
            index_path:    FAISS 索引输出路径 (如 db/data/querydata)。
            n_samples:     样本上限 (<=0 表示全量)。
            batch_size:    编码批大小。
            chunk_size:    流式分块行数。
            checkpoint_every: 每 N 块存一次检查点。
        """
        total = DataLoader.resolve_target_rows(csv_path, n_samples=n_samples if n_samples > 0 else None)
        processed = 0
        db = VectorDB(dimension=DEFAULT_VECTOR_DIM)

        if processed >= total:
            print("已完成，无需重新编码。")
            return {"index_path": index_path, "size": processed, "finished": True}

        pbar = tqdm(total=total, initial=processed, desc="编码入库", unit="条", dynamic_ncols=True)
        chunk_idx = 0
        total_chunks = -(-total // chunk_size) if chunk_size > 0 else 0  # ceil division
        for queries in DataLoader.iter_queries(
            csv_path, n_samples=n_samples if n_samples > 0 else None,
            start_offset=processed, chunk_size=chunk_size,
            target_rows=total,
        ):
            chunk_idx += 1

            # 编码
            query_tensor = self.model_engine.encode(
                queries, encoder="query", batch_size=batch_size, show_progress=False,
            )
            db.add(query_tensor)
            processed += len(queries)
            pbar.update(len(queries))
            pbar.set_postfix(分块=f"{chunk_idx}/{total_chunks}")

            # 检查点
            if chunk_idx % checkpoint_every == 0:
                db.save(index_path)
        pbar.close()

        # 最终落盘
        db.save(index_path)

        print(f"编码完成: {index_path} ({processed} 条)")
        return {"index_path": index_path, "size": processed, "finished": True}


def main():
    parser = argparse.ArgumentParser(description="问句向量索引入库")
    parser.add_argument("--data_csv", type=str, default=DB_CSV_PATH)
    parser.add_argument("--index_path", type=str, default=DB_INDEX_PATH)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    args = parser.parse_args()

    engine = SimCSEModelEngine()
    encoder = CorpusEncoder(model_engine=engine)
    result = encoder.encode_corpus_to_data(
        csv_path=args.data_csv,
        index_path=args.index_path,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"结果: {result}")


if __name__ == "__main__":
    main()