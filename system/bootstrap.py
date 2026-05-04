"""
system/bootstrap.py

文件作用:
    系统启动与知识库加载。
    统一维护比较器单例，避免重复初始化。
"""

import os
from typing import Any, List, Optional, Tuple

import faiss
import pandas as pd

from system.comparator import DialogComparator
from system.model_engine import SimCSEModelEngine
from system.config import (
    DB_DATA_DIR,
    DB_CSV_PATH,
    DB_QUERY_INDEX_FILE,
    DB_RESPONSE_INDEX_FILE,
    DB_PREFIX,
    SIMILARITY_THRESHOLD,
    RERANK_WEIGHTS,
    CONTEXT_MAX_TURNS,
    MAX_TEXT_LEN,
    CONTEXT_SHORT_QUERY_LEN,
    CONTEXT_OVERLAP_THRESHOLD,
    CONTEXT_SEMANTIC_THRESHOLD,
)

# 全局单例对象，用于在多个请求间复用同一个比较器。
_dialog_comparator_instance: Optional[DialogComparator] = None


def _read_faiss_index_safely(index_path: str) -> Any:
    """
    安全加载 FAISS 索引文件。

    加载策略（优先级从高到低）:
        1. 内存映射只读模式 (IO_FLAG_MMAP | IO_FLAG_READ_ONLY)
           — 共享内存，多进程友好，不复制数据
        2. 内存映射模式（仅 IO_FLAG_MMAP）
           — 适用于低版本 faiss 绑定不支持双 flag 的情况
        3. 常规读取模式
           — 兜底方案，会复制索引数据到进程内存
    """
    io_flag_mmap = int(getattr(faiss, "IO_FLAG_MMAP", 0))
    io_flag_ro = int(getattr(faiss, "IO_FLAG_READ_ONLY", 0))
    io_flags = io_flag_mmap | io_flag_ro

    if io_flags != 0:
        try:
            return faiss.read_index(index_path, io_flags)
        except TypeError:
            # 低版本绑定可能不支持第二个参数，继续回退。
            pass
        except Exception:
            # 内存映射失败时回退常规读取。
            pass

    return faiss.read_index(index_path)


def check_kb_exists() -> Tuple[bool, List[str]]:
    """检查数据库运行依赖文件是否存在。"""
    required_files = [DB_QUERY_INDEX_FILE, DB_RESPONSE_INDEX_FILE, DB_CSV_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    return len(missing_files) == 0, missing_files


def validate_database() -> Tuple[bool, str]:
    """快速校验数据库文件可读取。

    说明:
    1. 冷启动阶段避免再次完整解析 FAISS 大索引（初始化时还会真正加载一次），
       否则会产生重复 I/O 和明显启动延迟。
    2. 这里只做“可读性 + 基础结构”检查，确保主流程尽快启动。
    """
    try:
        # CSV: 检查列结构，仅读取首行即可。
        pd.read_csv(DB_CSV_PATH, usecols=["query", "response"], nrows=1)

        # FAISS 索引: 只做快速可读性检查，不做完整反序列化。
        for index_path in (DB_QUERY_INDEX_FILE, DB_RESPONSE_INDEX_FILE):
            with open(index_path, "rb") as f:
                head = f.read(16)
                if not head:
                    raise ValueError(f"索引文件为空: {index_path}")
    except Exception as e:
        return False, str(e)
    return True, "数据库文件可读。"




def load_database_columns(prefix: str) -> Tuple[Any, Any, List[str], List[str]]:
    """加载数据库索引与文本列（轻量模式，不构建 doc_texts 字典列表）。"""
    db_base_path = os.path.join(DB_DATA_DIR, f"{prefix}_faiss_db")
    query_index_path = db_base_path + "_query.index"
    response_index_path = db_base_path + "_response.index"

    query_index = _read_faiss_index_safely(query_index_path)
    try:
        response_index = _read_faiss_index_safely(response_index_path)
    except MemoryError:
        # 大库场景下 response_index 可能触发 bad_alloc，兜底复用 query_index 保证流程可运行。
        print("[WARN] response_index 加载内存不足，已退化为复用 query_index。")
        response_index = query_index

    required_rows = min(int(query_index.ntotal), int(response_index.ntotal))
    pair_df = pd.read_csv(DB_CSV_PATH, usecols=["query", "response"], nrows=required_rows)

    csv_queries = pair_df["query"].astype(str).tolist()
    csv_replies = pair_df["response"].astype(str).tolist()
    return query_index, response_index, csv_queries, csv_replies


def initialize_system() -> DialogComparator:
    """执行系统初始化并返回比较器实例。"""
    print("==========================================================")
    print("              日常闲聊机器人本地服务 启动中             ")
    print("==========================================================\n")

    # 步骤1：运行依赖检查 — 确认 CSV 和 FAISS 索引文件存在且可读
    print("[1/4] 正在检查知识库数据索引...")
    exists_ok, missing_files = check_kb_exists()
    if not exists_ok:
        msg = "未能找到数据库文件:\n" + "\n".join(missing_files)
        raise FileNotFoundError(msg)

    validate_ok, validate_msg = validate_database()
    if not validate_ok:
        raise RuntimeError(f"数据库异常: {validate_msg}")

    # 步骤2：加载双塔编码引擎 — query encoder + response encoder
    print("[2/4] 正在加载语义模型引擎...")
    engine = SimCSEModelEngine()

    # 步骤3：加载 FAISS 双索引与文本列 — query_index + response_index + CSV
    print("[3/4] 正在主流程加载数据库...")
    query_index, response_index, query_texts, reply_texts = load_database_columns(prefix=DB_PREFIX)

    # 步骤4：组装比较器 — 统一承载召回、重排、阈值与上下文策略
    print("[4/4] 正在初始化匹配模块...")
    comparator = DialogComparator(
        model_engine=engine,
        query_index=query_index,
        response_index=response_index,
        doc_texts=[],
        query_texts=query_texts,
        reply_texts=reply_texts,
        similarity_threshold=SIMILARITY_THRESHOLD,
        rerank_weights=RERANK_WEIGHTS,
        context_max_turns=CONTEXT_MAX_TURNS,
        max_text_len=MAX_TEXT_LEN,
        context_short_query_len=CONTEXT_SHORT_QUERY_LEN,
        context_overlap_threshold=CONTEXT_OVERLAP_THRESHOLD,
        context_semantic_threshold=CONTEXT_SEMANTIC_THRESHOLD,
        context_matching_enabled=True,
    )

    total_pairs = int(comparator.doc_count)
    print(f"启动成功，当前知识库规模：{total_pairs} 条。")
    print("系统已就绪，正在准备本地页面...\n")
    return comparator


def get_dialog_comparator() -> DialogComparator:
    """获取全局唯一比较器，首次调用时自动初始化。"""
    global _dialog_comparator_instance
    if _dialog_comparator_instance is None:
        _dialog_comparator_instance = initialize_system()
    return _dialog_comparator_instance




