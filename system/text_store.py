"""
system/text_store.py

文件作用:
    轻量级 CSV 文本存储，通过行偏移索引实现按需读取，
    避免将百万级文本全量加载到内存（832 MB → ~30 MB）。
"""

import csv
import os
from typing import List, Optional

from tqdm.auto import tqdm


class TextStore:
    """轻量文本存储：记录每行的文件偏移量，按需读取。"""

    def __init__(self, csv_path: str, max_rows: Optional[int] = None, show_progress: bool = False):
        self.csv_path = csv_path
        self.offsets: List[int] = []
        self._build_index(max_rows, show_progress)

    def _build_index(self, max_rows: Optional[int] = None, show_progress: bool = False):
        """扫描 CSV，记录每行文件偏移量（最多 max_rows 行）。"""
        if not os.path.exists(self.csv_path):
            return

        pbar = None
        if show_progress and max_rows:
            pbar = tqdm(
                total=max_rows, desc="CSV语料索引", unit="行",
                dynamic_ncols=True, leave=True,
            )

        try:
            with open(self.csv_path, "rb") as f:
                # 跳过表头
                f.readline()
                batch = 0
                while True:
                    if max_rows is not None and len(self.offsets) >= max_rows:
                        break
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    self.offsets.append(offset)
                    batch += 1
                    if pbar and batch >= 1000:
                        pbar.update(batch)
                        batch = 0
                if pbar and batch > 0:
                    pbar.update(batch)
        finally:
            if pbar:
                pbar.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def get_row(self, idx: int) -> Optional[str]:
        """按索引读取一行 CSV 原始文本。"""
        if idx < 0 or idx >= len(self.offsets):
            return None
        with open(self.csv_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            return f.readline().strip()

    def _parse_csv_row(self, idx: int) -> Optional[List[str]]:
        """按索引读取并正确解析一行 CSV（支持引号包裹的含逗号字段）。"""
        raw = self.get_row(idx)
        if raw is None:
            return None
        return next(csv.reader([raw]))

    def get_query(self, idx: int) -> str:
        """读取 query 列。"""
        parts = self._parse_csv_row(idx)
        return parts[0].strip() if parts else ""

    def get_response(self, idx: int) -> str:
        """读取 response 列。"""
        parts = self._parse_csv_row(idx)
        return parts[1].strip() if parts and len(parts) > 1 else ""
