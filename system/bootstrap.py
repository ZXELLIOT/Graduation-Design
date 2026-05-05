"""
system/bootstrap.py

文件作用:
    系统启动与知识库加载。
    统一维护比较器单例，避免重复初始化。
"""

import os
import threading
import time
from typing import Any, List, Optional, Tuple

import faiss

import pandas as pd
from tqdm.auto import tqdm

from system.comparator import DialogComparator
from system.model_engine import SimCSEModelEngine
from system.text_store import TextStore
from system.config import (
    DB_CSV_PATH,
    DB_QUERY_INDEX_FILE,
    DB_LOAD_MAX_ROWS,
    SIMILARITY_THRESHOLD,
    RERANK_WEIGHTS,
    CONTEXT_MAX_TURNS,
    MAX_TEXT_LEN,
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
        except (TypeError, RuntimeError, OSError, ValueError):
            pass

    return faiss.read_index(index_path)


def check_kb_exists() -> Tuple[bool, List[str]]:
    """检查数据库运行依赖文件是否存在。"""
    required_files = [DB_QUERY_INDEX_FILE, DB_CSV_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    return len(missing_files) == 0, missing_files


def validate_database() -> Tuple[bool, str]:
    """快速校验数据库文件可读取。"""
    try:
        pd.read_csv(DB_CSV_PATH, usecols=["query", "response"], nrows=1)
        with open(DB_QUERY_INDEX_FILE, "rb") as f:
            head = f.read(16)
            if not head:
                raise ValueError(f"索引文件为空: {DB_QUERY_INDEX_FILE}")
    except Exception as e:
        return False, str(e)
    return True, "数据库文件可读。"


def load_faiss_index() -> Any:
    """加载 FAISS 问句索引（独立步骤，带可预估总进度）。"""
    index_path = DB_QUERY_INDEX_FILE
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    result: dict = {"index": None, "error": None}
    done = threading.Event()
    expected_seconds = 30.0

    def _load_worker() -> None:
        try:
            result["index"] = _read_faiss_index_safely(index_path)
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    threading.Thread(target=_load_worker, daemon=True).start()

    spinner = ["|", "/", "-", "\\"]
    tick = 0
    t0 = time.perf_counter()
    shown_percent = 0

    with tqdm(total=100, desc="FAISS索引读取", unit="%", dynamic_ncols=True, leave=True) as pbar:
        while not done.wait(0.12):
            elapsed = time.perf_counter() - t0
            estimated = min(99, int((elapsed / expected_seconds) * 100))
            if estimated > shown_percent:
                pbar.update(estimated - shown_percent)
                shown_percent = estimated
            pbar.set_postfix_str(
                f"{index_size_mb:.1f}MB {spinner[tick % len(spinner)]} 已{elapsed:.1f}s/预估{expected_seconds:.0f}s"
            )
            tick += 1

        elapsed = time.perf_counter() - t0
        if result["error"] is not None:
            raise result["error"]

        query_index = result["index"]
        pbar.update(100 - shown_percent)
        pbar.set_postfix_str(f"读取完成, 总耗时 {elapsed:.1f}s")

    return query_index


def load_text_store(max_rows: Optional[int] = None) -> TextStore:
    """加载 CSV 文本偏移索引（全量加载，带进度条）。

    max_rows 为 None 时自动计数全量行数；传入整数则截断到指定行数。
    """
    if max_rows is None:
        try:
            with open(DB_CSV_PATH, "r", encoding="utf-8") as f:
                max_rows = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            max_rows = 0
    return TextStore(DB_CSV_PATH, max_rows=max_rows, show_progress=True) if max_rows > 0 else TextStore(DB_CSV_PATH)


def initialize_system() -> DialogComparator:
    """执行系统初始化并返回比较器实例。"""
    print("==========================================================")
    print("              日常闲聊机器人本地服务 启动中             ")
    print("==========================================================\n")

    # 步骤1：运行依赖检查 — 确认 CSV 和 FAISS 索引文件存在且可读
    print("[1/5] 检查数据文件...")
    exists_ok, missing_files = check_kb_exists()
    if not exists_ok:
        msg = "未能找到数据库文件:\n" + "\n".join(missing_files)
        raise FileNotFoundError(msg)

    validate_ok, validate_msg = validate_database()
    if not validate_ok:
        raise RuntimeError(f"数据库异常: {validate_msg}")

    # 步骤2：加载双塔编码引擎 — query encoder + response encoder
    print("[2/5] 加载语义模型...")
    engine = SimCSEModelEngine()

    # 步骤3：加载 FAISS 问句索引（mmap 只读模式，虚拟内存映射）
    print("[3/5] 加载向量索引...")
    query_index = load_faiss_index()

    full_total = int(getattr(query_index, "ntotal", 0))
    max_rows = max(1, int(DB_LOAD_MAX_ROWS))
    effective_rows = min(full_total, max_rows)

    # 步骤4：加载 CSV 语料偏移索引（只截断 CSV，不重建向量索引）
    print("[4/5] 加载语料数据...")
    text_store = load_text_store(max_rows=effective_rows)

    # 步骤5：组装比较器 — 答句相似度通过 CSV 行号实时编码计算
    print("[5/5] 初始化匹配引擎...")
    comparator = DialogComparator(
        model_engine=engine,
        query_index=query_index,
        doc_texts=[],
        text_store=text_store,
        similarity_threshold=SIMILARITY_THRESHOLD,
        rerank_weights=RERANK_WEIGHTS,
        context_max_turns=CONTEXT_MAX_TURNS,
        max_text_len=MAX_TEXT_LEN,
        context_matching_enabled=True,
    )

    if full_total > effective_rows:
        print(f"启动成功，知识库规模：{effective_rows:,} 条（原始索引 {full_total:,} 条）\n")
    else:
        print(f"启动成功，知识库规模：{effective_rows:,} 条\n")
    return comparator


def get_dialog_comparator() -> DialogComparator:
    """获取全局唯一比较器，首次调用时自动初始化。"""
    global _dialog_comparator_instance
    if _dialog_comparator_instance is None:
        _dialog_comparator_instance = initialize_system()
    return _dialog_comparator_instance
