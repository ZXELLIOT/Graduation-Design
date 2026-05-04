"""
db/encoder_module.py

文件作用:
    语料向量化与索引入库模块。

核心流程:
    1. 分块读取语料 CSV → 2. 双塔编码 (query/response) →
    3. 写入 FAISS 双索引 → 4. 定期检查点落盘（支持断点续跑）

索引策略:
    - 小规模数据 (<10000): HNSW 图索引（精度高、内存占用大）
    - 大规模数据 (>10000): IVF-PQ 压缩索引（牺牲少量精度换取内存效率）
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Generator, List, Tuple

import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from system.model_engine import SimCSEModelEngine
from db.config import DB_CSV_PATH, DB_BUILD_BASE_PATH

# 默认向量维度（bert-base-chinese / text2vec-base-chinese 均为 768）
# 若更换为其他预训练模型（如 1024 维），只需修改此处即可
DEFAULT_VECTOR_DIM = 768


# ============================================================
# 数据读取模块 (DataLoader)
# 负责从 CSV 中高效读取语料，支持全量/采样/分块流式三种模式
# ============================================================


class DataLoader:
    """
    数据读取模块：从 CSV 中加载问答语料。

    三种读取模式:
        load_corpus:      一次性加载全量数据到内存
        iter_corpus:      分块流式迭代（大语料推荐，避免内存溢出）
        resolve_target_rows: 仅查询行数，不加载数据
    """

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

    @staticmethod
    def resolve_target_rows(csv_path: str, n_samples=None) -> int:
        """计算本次任务计划处理的总行数。"""
        total_rows = DataLoader._count_csv_rows(csv_path)
        sample_limit = int(n_samples) if n_samples is not None else 0
        use_limit = sample_limit > 0
        return min(total_rows, sample_limit) if use_limit else total_rows

    @staticmethod
    def iter_corpus(
        csv_path: str,
        n_samples=None,
        start_offset: int = 0,
        chunk_size: int = 100000,
        show_progress: bool = False,
    ) -> Generator[Tuple[List[str], List[str]], None, None]:
        """按分块流式读取语料，支持从指定偏移断点继续。"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"指定的语料文件未找到: {csv_path}")

        target_rows = DataLoader.resolve_target_rows(csv_path, n_samples=n_samples)
        if start_offset < 0 or start_offset > target_rows:
            raise ValueError(f"start_offset 越界: {start_offset}, target_rows={target_rows}")

        remaining = target_rows - start_offset
        progress = tqdm(
            total=remaining,
            desc="正在分块加载语料库",
            unit="条",
            disable=not show_progress,
            leave=False,
        )

        scanned = 0
        emitted = 0
        for chunk in pd.read_csv(csv_path, usecols=["query", "response"], chunksize=chunk_size, engine="c"):
            if scanned >= target_rows:
                break

            chunk = chunk.fillna("")
            q = chunk["query"].astype(str).tolist()
            r = chunk["response"].astype(str).tolist()

            current_len = len(q)
            if current_len <= 0:
                continue

            keep_len = min(current_len, target_rows - scanned)
            if keep_len <= 0:
                break

            q = q[:keep_len]
            r = r[:keep_len]

            skip_len = 0
            if start_offset > scanned:
                skip_len = min(keep_len, start_offset - scanned)

            if skip_len > 0:
                q = q[skip_len:]
                r = r[skip_len:]

            scanned += keep_len

            if not q:
                continue

            emitted += len(q)
            progress.update(len(q))
            yield q, r

            if emitted >= remaining:
                break

        progress.close()


# ============================================================
# 向量数据库模块 (VectorDB)
# 负责构建和读写 FAISS 双索引（query 索引 + response 索引）
# ============================================================


class VectorDB:
    """
    向量数据库读写模块。

    两个索引的作用:
        query_index:    存储历史问句向量 → 用户输入与历史问句匹配，粗召回候选
        response_index: 存储对应答句向量 → 重排时计算用户输入与候选答句的相似度

    为什么需要双索引？
        单索引只能回答"用户输入和哪个历史问句最像"。
        双索引额外计算"用户输入和候选答句的匹配度"，通过加权融合
        (问问相似度, 问答相似度) 得到更准确的最终排序。
    """

    def __init__(self, dimension: int = 768):
        """初始化向量数据库对象。

        参数:
            dimension: 向量维度。
        返回:
            无返回值。
        """
        self.dimension = dimension
        self.query_index = self._build_index(dimension)
        self.response_index = self._build_index(dimension)
        # 入库阶段仅构建向量索引，文本映射在加载阶段由 CSV 恢复，避免占用大量内存。
        self.doc_texts: List[dict] = []

    @staticmethod
    def _to_numpy_float32(embeddings):
        """将输入向量转为 numpy.float32。

        参数:
            embeddings: torch.Tensor 或 numpy.ndarray。
        返回:
            numpy.float32 数组。
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings.astype("float32")

    @staticmethod
    def _build_index(dimension: int, use_ivf=False, n_total=0):
        """
        按数据规模选择合适的 FAISS 索引结构。

        两种索引策略:
            HNSW (默认):  基于图的近似最近邻搜索，精度高但内存占用大。
                          适合百万级以下的数据，构建快、查询快。
            IVF-PQ:       先聚类粗筛（IVF）再做乘积量化（PQ）压缩。
                          内存占用约为 HNSW 的 1/10，适合千万级以上数据。
                          代价是精度略低，需要先训练聚类中心。

        参数:
            dimension: 向量维度（默认 768，与 bert-base-chinese 对齐）。
            use_ivf:   是否强制使用 IVF-PQ 压缩索引。
            n_total:   预估数据规模（用于计算 IVF 聚类数）。
        """
        if use_ivf and n_total > 10000:
            # --- IVF-PQ 压缩索引 ---
            # nlist: 聚类中心数，经验公式 4 * sqrt(n)
            nlist = int(4 * (n_total ** 0.5))
            # m: 子向量个数，768维分成 8 个 96 维子向量分别量化
            m = 8
            # 粗量化器：用 FlatIP（精确内积搜索）作为第一阶段粗筛
            quantizer = faiss.IndexFlatIP(dimension)
            # 每个子向量用 8 bit 编码（256 个聚类中心）
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            # nprobe: 检索时探测的聚类数，10 是速度与精度的常用平衡点
            index.nprobe = 10
            return index
        # --- HNSW 图索引（默认）---
        # 32: 每个节点连接数，越大精度越高但构建越慢
        return faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

    @staticmethod
    def _save_index(index_obj, file_path: str):
        """保存单个 FAISS 索引。

        参数:
            index_obj: FAISS 索引对象。
            file_path: 目标文件路径。
        返回:
            无返回值。
        """
        target_dir = os.path.dirname(os.path.abspath(file_path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        faiss.write_index(index_obj, file_path)

    @staticmethod
    def _read_index_safe(file_path: str):
        """读取单个 FAISS 索引文件。

        参数:
            file_path: 索引文件路径。
        返回:
            FAISS 索引对象。
        """
        return faiss.read_index(file_path)

    @staticmethod
    def _file_pack(path: str) -> dict:
        """根据基础路径构建双索引文件路径。

        参数:
            path: 基础文件路径前缀。
        返回:
            含 query/response 索引路径的字典。
        """
        return {
            "query_index": path + "_query.index",
            "response_index": path + "_response.index",
        }

    def add_data(
        self,
        queries: List[str],
        query_embeddings: torch.Tensor,
        replies: List[str],
        reply_embeddings: torch.Tensor,
    ):
        """批量写入索引与文本映射。"""
        if len(queries) != len(replies):
            raise ValueError("写入失败：问句与答句数量不一致。")

        query_np = self._to_numpy_float32(query_embeddings)
        reply_np = self._to_numpy_float32(reply_embeddings)

        self.query_index.add(query_np)  # type: ignore
        self.response_index.add(reply_np)  # type: ignore

        total = int(self.query_index.ntotal)  # type: ignore
        print(f"成功导入批次数据：知识库当前记录总条数 = {total}")

    def save(self, path: str):
        """写入双索引文件。"""
        target_dir = os.path.dirname(os.path.abspath(path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        files = self._file_pack(path)
        print("正在保存数据库文件 (query/response 索引)...")
        self._save_index(self.query_index, files["query_index"])
        self._save_index(self.response_index, files["response_index"])

        print(f"向量索引与文本语料已成功保存至: {path}")

    @classmethod
    def load(cls, path: str, dimension: int = 768):
        """从磁盘恢复双索引对象。"""
        files = cls._file_pack(path)
        if not os.path.exists(files["query_index"]) or not os.path.exists(files["response_index"]):
            raise FileNotFoundError("索引文件不存在，无法恢复。")

        db = cls(dimension=dimension)
        db.query_index = cls._read_index_safe(files["query_index"])
        db.response_index = cls._read_index_safe(files["response_index"])
        return db

# ============================================================
# 语料编码入库模块 (CorpusEncoder)
# 串联 "读取语料 → 双塔编码 → 写入双索引" 全流程
# ============================================================


class CorpusEncoder:
    """
    语料编码入库主流程。

    流程:
        1. 分块流式读取 CSV 语料（避免一次性加载撑爆内存）
        2. 逐块用双塔模型编码为 768 维向量
        3. 写入 FAISS 双索引（query_index + response_index）
        4. 每 N 个分块保存一次检查点（断点续跑，防止中断后从头开始）
    """

    def __init__(self, model_engine, data_dir: str):
        """初始化语料编码器。

        参数:
            model_engine: 文本编码引擎。
            data_dir: 数据库存储目录。
        返回:
            无返回值。
        """
        self.model_engine = model_engine
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def _state_path(base_path: str) -> str:
        """构建断点状态文件路径。"""
        return base_path + "_state.json"

    @staticmethod
    def _save_state(state_path: str, payload: dict) -> None:
        """原子写入断点状态文件。"""
        tmp_path = state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, state_path)

    @staticmethod
    def _load_state(state_path: str) -> dict:
        """读取断点状态文件。"""
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_state_payload(
        csv_path: str,
        n_samples,
        processed_rows: int,
        total_rows: int,
        finished: bool,
    ) -> dict:
        """构建状态记录。"""
        return {
            "csv_path": os.path.abspath(csv_path),
            "n_samples": int(n_samples) if n_samples is not None else 0,
            "processed_rows": int(processed_rows),
            "total_rows": int(total_rows),
            "finished": bool(finished),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def encode_corpus_to_data(
        self,
        csv_path: str,
        n_samples=None,
        batch_size: int = 0,
        chunk_size: int = 100000,
        checkpoint_every: int = 5,
        resume: bool = True,
    ):
        """执行语料入库全流程。

        参数:
            csv_path: 语料 CSV 路径。
            n_samples: 读取样本上限，None/<=0 表示全量。
            batch_size: 编码批大小。
            chunk_size: 流式读取分块大小。
            checkpoint_every: 每处理多少分块执行一次磁盘检查点。
            resume: 是否从历史检查点继续。
        返回:
            包含数据库路径和写入规模的结果字典。
        """
        faiss_db_path = os.path.join(self.data_dir)
        state_path = self._state_path(faiss_db_path)
        files = VectorDB._file_pack(faiss_db_path)

        total_rows = DataLoader.resolve_target_rows(csv_path, n_samples=n_samples)
        processed_rows = 0
        db = None

        if resume and os.path.exists(state_path) and os.path.exists(files["query_index"]) and os.path.exists(files["response_index"]):
            print("检测到历史检查点，正在尝试断点续跑...")
            state = self._load_state(state_path)
            state_csv = os.path.abspath(str(state.get("csv_path", "")))
            curr_csv = os.path.abspath(csv_path)
            if state_csv != curr_csv:
                raise ValueError(f"检查点 csv_path 不一致: {state_csv} != {curr_csv}")

            state_n_samples = int(state.get("n_samples", 0))
            curr_n_samples = int(n_samples) if n_samples is not None else 0
            if state_n_samples != curr_n_samples:
                raise ValueError(f"检查点 n_samples 不一致: {state_n_samples} != {curr_n_samples}")

            processed_rows = int(state.get("processed_rows", 0))
            db = VectorDB.load(faiss_db_path, dimension=DEFAULT_VECTOR_DIM)

            q_total = int(db.query_index.ntotal)  # type: ignore
            r_total = int(db.response_index.ntotal)  # type: ignore
            if q_total != r_total:
                raise ValueError(f"索引损坏: query_ntotal={q_total}, response_ntotal={r_total}")
            if q_total != processed_rows:
                raise ValueError(f"检查点与索引数量不一致: state={processed_rows}, index={q_total}")

            print(f"断点续跑就绪，已完成 {processed_rows}/{total_rows} 条。")
        else:
            print("未发现有效检查点，开始全新编码任务。")
            db = VectorDB(dimension=DEFAULT_VECTOR_DIM)

        if db is None:
            raise RuntimeError("向量数据库初始化失败。")

        if processed_rows >= total_rows:
            print("检查点显示任务已完成，将直接返回。")
            return {
                "faiss_db": faiss_db_path,
                "size": int(processed_rows),
                "resumed": bool(resume),
                "finished": True,
            }

        safe_batch_size = max(1, int(batch_size))
        safe_chunk_size = max(1, int(chunk_size))
        safe_checkpoint_every = max(1, int(checkpoint_every))

        chunk_idx = 0
        for queries, replies in DataLoader.iter_corpus(
            csv_path,
            n_samples=n_samples,
            start_offset=processed_rows,
            chunk_size=safe_chunk_size,
            show_progress=False,
        ):
            if len(queries) != len(replies):
                raise ValueError("分块数据异常：问句与答句数量不一致。")

            chunk_idx += 1
            chunk_rows = len(queries)
            tqdm.write(f"\n开始处理第 {chunk_idx} 个分块，本块 {chunk_rows} 条...")

            phase_progress = tqdm(
                total=3,
                desc=f"分块{chunk_idx}进度",
                unit="阶段",
                dynamic_ncols=True,
                leave=False,
            )

            phase_progress.set_postfix_str("1/3 问句编码")
            query_tensor = self.model_engine.encode(
                queries,
                encoder="query",
                batch_size=safe_batch_size,
                show_progress=False,
            )
            phase_progress.update(1)

            phase_progress.set_postfix_str("2/3 答句编码")
            reply_tensor = self.model_engine.encode(
                replies,
                encoder="response",
                batch_size=safe_batch_size,
                show_progress=False,
            )
            phase_progress.update(1)

            phase_progress.set_postfix_str("3/3 写入索引")
            db.add_data(
                queries,
                query_tensor,
                replies,
                reply_tensor,
            )
            phase_progress.update(1)
            phase_progress.close()

            processed_rows += chunk_rows
            progress_ratio = (processed_rows / total_rows * 100.0) if total_rows > 0 else 100.0
            tqdm.write(
                f"第 {chunk_idx} 块完成：本块 {chunk_rows} 条，累计 {processed_rows}/{total_rows} ({progress_ratio:.2f}%)"
            )

            # 定期落盘：索引 + 状态，防止中断后前功尽弃。
            if (chunk_idx % safe_checkpoint_every) == 0:
                db.save(faiss_db_path)
                self._save_state(
                    state_path,
                    self._build_state_payload(
                        csv_path=csv_path,
                        n_samples=n_samples,
                        processed_rows=processed_rows,
                        total_rows=total_rows,
                        finished=False,
                    ),
                )
                tqdm.write(f"检查点已保存：{processed_rows}/{total_rows}")

        # 最终落盘，标记完成。
        db.save(faiss_db_path)
        self._save_state(
            state_path,
            self._build_state_payload(
                csv_path=csv_path,
                n_samples=n_samples,
                processed_rows=processed_rows,
                total_rows=total_rows,
                finished=True,
            ),
        )

        print(f"知识库已成功持久化至数据库: {faiss_db_path}")
        return {
            "faiss_db": faiss_db_path,
            "size": int(processed_rows),
            "resumed": bool(resume),
            "finished": True,
        }


def main():
    """命令行入口。

    参数:
        无（参数由 argparse 解析）。
    返回:
        无返回值。
    """
    parser = argparse.ArgumentParser(description="语料入库工具")
    parser.add_argument("--data_csv", type=str, default=DB_CSV_PATH)
    # 注意：这里是索引基础路径（不含 _query/_response 后缀），不是目录。
    parser.add_argument("--data_dir", type=str, default=DB_BUILD_BASE_PATH)
    parser.add_argument("--batch_size", type=int, default=128)
    # 默认全量编码；传入 >0 时按条数截取。
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args()

    engine = SimCSEModelEngine()
    encoder = CorpusEncoder(model_engine=engine, data_dir=args.data_dir)

    result = encoder.encode_corpus_to_data(
        csv_path=args.data_csv,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        checkpoint_every=args.checkpoint_every,
        resume=bool(args.resume),
    )
    print(f"处理完成: {result}")


if __name__ == "__main__":
    main()