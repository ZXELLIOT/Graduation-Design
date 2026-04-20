import os
import sys
import time
import gradio as gr
import faiss
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.model_engine import SimCSEModelEngine
from system.comparator import DialogComparator
from system.config import (
    DB_DATA_DIR,
    DB_CSV_PATH,
    DB_QUERY_INDEX_FILE,
    DB_RESPONSE_INDEX_FILE,
    DB_PREFIX,
    SIMILARITY_THRESHOLD,
    RERANK_WEIGHTS,
)

def check_kb_exists():
    """
    检查数据库文件是否存在。
    """
    required_files = [
        DB_QUERY_INDEX_FILE,
        DB_RESPONSE_INDEX_FILE,
        DB_CSV_PATH,
    ]
    return all(os.path.exists(f) for f in required_files)


def validate_database():
    """
    校验数据库文件可打开，不读取实际数据内容。
    """
    try:
        with open(DB_CSV_PATH, "r", encoding="utf-8"):
            pass
        faiss.read_index(DB_QUERY_INDEX_FILE)
        faiss.read_index(DB_RESPONSE_INDEX_FILE)
    except Exception as e:
        return False
    return True


def load_database_assets(prefix: str):
    """
    在主流程中加载数据库索引与文本映射。
    """
    db_base_path = os.path.join(DB_DATA_DIR, f"{prefix}_faiss_db")
    query_index_path = db_base_path + "_query.index"
    response_index_path = db_base_path + "_response.index"

    query_index = faiss.read_index(query_index_path)
    response_index = faiss.read_index(response_index_path)

    required_rows = min(int(query_index.ntotal), int(response_index.ntotal))
    pair_df = pd.read_csv(DB_CSV_PATH, usecols=["query", "response"], nrows=required_rows)

    csv_queries = pair_df["query"].astype(str).tolist()
    csv_replies = pair_df["response"].astype(str).tolist()
    pair_count = min(len(csv_queries), len(csv_replies))
    doc_texts = []
    for idx in range(pair_count):
        doc_texts.append(
            {
                "query": csv_queries[idx],
                "reply": csv_replies[idx],
                "query_idx": idx,
                "reply_idx": idx,
                "csv_idx": idx,
            }
        )
    return query_index, response_index, doc_texts


def initialize_system():
    """
    执行系统初始化核心流程。
    
    1. 验证本地数据索引是否准备就绪。
    2. 加载预训练的语义提取模型（模型引擎）。
    3. 初始化对话匹配模块并从缓存中载入大规模向量数据。
    
    返回:
        已准备就绪的对话匹配器实例。
    """
    print("==========================================================")
    print("                  检索式中文对话系统 启动中               ")
    print("==========================================================\n")

    # 第一步：初始化数据库加载器并校验数据库完整性
    print("[1/4] 正在检查知识库数据索引...")
    if not check_kb_exists():
        msg = (
            f"未能找到数据库文件。\n"
        )
        raise FileNotFoundError(msg)

    if not validate_database():
        raise RuntimeError("数据库异常")

    # 第二步：加载负责将文本转化为语义向量的模型引擎
    print("[2/4] 正在加载语义模型引擎...")
    engine = SimCSEModelEngine()

    # 第三步：在主流程加载数据库索引与文本映射
    print("[3/4] 正在主流程加载数据库...")
    query_index, response_index, doc_texts = load_database_assets(prefix=DB_PREFIX)

    # 第四步：创建匹配器实例（仅负责输入处理、检索与决策）
    print("[4/4] 正在初始化匹配模块...")
    comparator = DialogComparator(
        model_engine=engine,
        query_index=query_index,
        response_index=response_index,
        doc_texts=doc_texts,
        similarity_threshold=SIMILARITY_THRESHOLD,
        rerank_weights=RERANK_WEIGHTS,
    )

    # 输出加载总结信息
    total_pairs = len(comparator.queries) if comparator.queries else 0
    print(f"启动成功，当前知识库规模：{total_pairs} 条。")
    print("系统已就绪，正在准备交互界面...\n")

    return comparator

# 全局单例对象，用于在多个请求间复用同一个匹配器
dialog_comparator_instance = None

def get_dialog_matcher():
    """
    获取全局唯一的对话匹配器。
    如果尚未初始化，则执行完整初始化流程；如果已存在，则直接返回，避免重复加载。
    """
    global dialog_comparator_instance
    if dialog_comparator_instance is None:
        dialog_comparator_instance = initialize_system()
    return dialog_comparator_instance

def predict(user_input: str, history: list) -> str:
    """
    根据用户输入在知识库中检索并返回最合适的回答。

    参数：
        user_input: 用户的原始提问文本。
        history: 对话历史列表。

    返回：
        检索到的最佳回答文本。如果相似度过低，则返回系统预设的提示信息。
    """
    _ = history

    # 调用核心匹配逻辑进行检索
    comparator = get_dialog_matcher()
    t0 = time.perf_counter()

    reply, score, matched_q = comparator.compare(user_input)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if matched_q:
        diagnostic_log = f"\n\n> 匹配问题:「{matched_q}」 ｜ 相似度: {score:.4f} ｜ 耗时: {elapsed_ms:.2f} ms"
    else:
        diagnostic_log = f"\n\n> 未匹配到高置信度答案，最高相似度: {score:.4f} ｜ 耗时: {elapsed_ms:.2f} ms"

    return reply + diagnostic_log

# 构建 Web 界面
demo = gr.ChatInterface(
    fn=predict,
    title="检索式中文对话系统 (SimCSE)",
    description=(
        "系统使用 LCCC 语料进行检索。"
    ),
    examples=["最近有什么好看的电影推荐吗？", "毕业设计进度有点卡住了，好焦虑", "今天天气真不错～"],
)

if __name__ == "__main__":
    # 启动界面
    try:
        print("系统正在启动，请稍候...")
        get_dialog_matcher()
        print("系统就绪，正在启动 Web 界面...")
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        print(f"启动界面失败: {e}")