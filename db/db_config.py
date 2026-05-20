"""
db/db_config.py
文件作用:
    数据库构建模块配置。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 数据库文件目录
DB_DATA_DIR = str(PROJECT_ROOT / "db" / "data")
# 语料 CSV 路径
DB_CSV_PATH = str(Path(DB_DATA_DIR) / "system_data.csv")
# FAISS 问句索引文件
DB_QUERY_INDEX_FILE = str(Path(DB_DATA_DIR) / "querydata")