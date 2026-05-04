"""
db/db_config.py

文件作用:
    数据库构建模块配置。
"""

from system.config import DB_CSV_PATH, DB_DATA_DIR

# 问句索引输出路径
DB_INDEX_PATH = str(DB_DATA_DIR) + "/query.index"

__all__ = ["DB_CSV_PATH", "DB_DATA_DIR", "DB_INDEX_PATH"]
