"""
system/text_store.py

文件作用:
    轻量级 CSV 文本存储，通过行偏移索引实现按需读取，
    避免将百万级文本全量加载到内存（832 MB → ~30 MB）。
"""

import os
import csv
from typing import List, Optional


class TextStore:
    """轻量文本存储：记录每行的文件偏移量，按需读取。"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.offsets: List[int] = []  # 每行在文件中的起始字节位置
        self._build_index()

    def _build_index(self):
        """扫描 CSV，记录每行文件偏移量。"""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "rb") as f:
            # 跳过表头
            header = f.readline()
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def get_row(self, idx: int) -> Optional[str]:
        """按索引读取一行 CSV 原始文本。"""
        if idx < 0 or idx >= len(self.offsets):
            return None
        with open(self.csv_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            return f.readline().strip()

    def get_query(self, idx: int) -> str:
        """读取 query 列。"""
        row = self.get_row(idx)
        if row is None:
            return ""
        # CSV 格式: query,response
        parts = row.split(",", 1)
        return parts[0].strip() if parts else ""

    def get_response(self, idx: int) -> str:
        """读取 response 列。"""
        row = self.get_row(idx)
        if row is None:
            return ""
        parts = row.split(",", 1)
        return parts[1].strip() if len(parts) > 1 else ""
