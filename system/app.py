import os
import sys
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import SimCSEEncoder
from src.data_loader import DataLoader
from src.matcher import DialogMatcher

SIMILARITY_THRESHOLD = 0.5
ENABLE_FALLBACK_REPLY = True
FALLBACK_TOP_K = 3
TRAIN_CSV_PATH = r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data\lccc_train.csv"
CACHE_DIR = r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data"

def initialize_system():
    """
    初始化并返回用于检索回复的匹配组件实例。

    说明：
    1. 加载语义编码器；
    2. 检查并尝试加载本地缓存；若无缓存则从语料生成并保存；
    3. 用编码器与语料初始化匹配组件并返回。
    返回值：匹配组件实例（可用于在线查询）。
    """
    print("==========================================================")
    print("                  检索式中文对话系统 启动中               ")
    print("==========================================================\n")

    # 1. 初始化编码器模型
    print("[1/4] 加载语义编码器...")
    encoder = SimCSEEncoder()

    # 2. 检查本地缓存：若同时存在向量与文本映射则使用缓存
    print("[2/4] 检查本地缓存...")
    query = None
    response = None
    query_vec = os.path.join(CACHE_DIR, 'train_query_embeddings.pt')
    reply_vec = os.path.join(CACHE_DIR, 'train_reply_embeddings.pt')
    queries_pkl = os.path.join(CACHE_DIR, 'train_queries.pt')
    replies_pkl = os.path.join(CACHE_DIR, 'train_replies.pt')

    use_cache = os.path.exists(query_vec) and os.path.exists(reply_vec) and os.path.exists(queries_pkl) and os.path.exists(replies_pkl)

    if use_cache:
        print("检测到有效缓存")
    else:
        print("未检测到有效缓存，正在从语料文件加载并生成缓存")
        query, response = DataLoader.load_corpus(TRAIN_CSV_PATH, n_samples=100000)

    # 3. 初始化匹配器：匹配器内部会决定是否从缓存加载或重新生成缓存
    print("[3/4] 初始化匹配组件并加载/生成缓存...")
    matcher = DialogMatcher(
        encoder=encoder,
        queries=query,
        replies=response,
        cache_dir=CACHE_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD,
        enable_fallback=ENABLE_FALLBACK_REPLY,
        fallback_top_k=FALLBACK_TOP_K
    )

    # 4. 完成信息
    total_pairs = len(matcher.queries) if matcher and matcher.queries is not None else 0
    print(f"[4/4] 语料处理完成，共 {total_pairs} 条问答对。")
    print("系统核心模块加载完成，准备启动界面...\n")

    # 返回匹配器实例以供后续调用
    return matcher

# 延迟初始化，避免导入模块时触发完整启动流程
dialog_matcher_instance = None

def get_dialog_matcher():
    """按需初始化并复用对话匹配器实例。"""
    global dialog_matcher_instance
    if dialog_matcher_instance is None:
        dialog_matcher_instance = initialize_system()
    return dialog_matcher_instance

def predict(user_input: str, history: list) -> str:
    """处理一次用户输入并返回回复。

    参数：
    - user_input: 用户输入的文本；
    - history: 对话历史（当前未使用，仅保留接口一致性）。

    返回：
    - 字符串形式的回复；若未找到可信答案，会返回提示并附带简短诊断信息。
    """
    _ = history

    normalized_input = (user_input or "").strip()
    if not normalized_input:
        return "请输入有效的内容"

    # 在此处按需初始化匹配器并进行检索
    matcher = get_dialog_matcher()

    reply, score, matched_q = matcher.match(user_input, top_k_for_rerank=5)

    if matched_q:
        diagnostic_log = f"\n\n> 匹配问题:「{matched_q}」 ｜ 相似度: {score:.4f}"
    else:
        diagnostic_log = f"\n\n> 未匹配到高置信度答案，最高相似度: {score:.4f}"

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