"""
db/config.py

文件作用:
    数据库构建模块配置。
    统一提供语料路径、索引目录与默认前缀。
"""

from pathlib import Path

from system.config import DB_CSV_PATH, DB_DATA_DIR, DB_PREFIX

# 默认索引基础路径（不含 _query/_response 后缀），与 system/app.py 的加载命名保持一致。
DB_BUILD_BASE_PATH = str(Path(DB_DATA_DIR) / f"{DB_PREFIX}_faiss_db")

__all__ = [
    "DB_CSV_PATH",
    "DB_DATA_DIR",
    "DB_PREFIX",
    "DB_BUILD_BASE_PATH",
]
