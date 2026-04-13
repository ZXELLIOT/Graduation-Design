import os
import sys

# 确保从任意工作目录启动时都能找到 src 包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import gradio as gr

from src.data_loader import DataLoader
from src.model import SimCSEEncoder
from src.matcher import DialogMatcher
from src.preprocess import TextPreprocessor


def initialize_system():
    """按固定顺序初始化系统核心组件。"""
    print("==========================================================")
    print("             基于 SimCSE 的检索式中文对话系统           ")
    print("==========================================================\n")

    print("[1/4] 执行语料数据预处理...")
    preprocessor = TextPreprocessor()
    ok = preprocessor.process()
    if ok is False:
        raise RuntimeError("语料预处理失败，请检查原始数据文件与路径配置。")

    print("\n[2/4] 加载对话语料...")
    data_loader = DataLoader()
    queries, replies = data_loader.load_corpus()
    if not queries or not replies:
        raise RuntimeError("语料加载为空，请确认 data/corpus.csv 是否生成且内容有效。")

    print("\n[3/4] 初始化 SimCSE 语义模型...")
    encoder = SimCSEEncoder()

    print("\n[4/4] 构建句子特征向量与匹配索引...")
    # 尝试加载缓存加速启动，否则全量计算
    matcher = DialogMatcher(encoder, queries, replies)
    
    print("\n[√] 系统核心模块加载完成，准备启动 Web 界面...\n")
    return matcher

# 延迟初始化，避免导入模块时触发完整启动流程
matcher = None


def get_matcher():
    """按需初始化并复用 matcher，减少重复加载开销。"""
    global matcher
    if matcher is None:
        matcher = initialize_system()
    return matcher


def predict(user_input: str, history: list) -> str:
    """
    接收用户输入，计算相似度并返回最佳匹配回复。
    """
    _ = history  # gradio 会传入历史消息，这里保留接口但不参与检索

    normalized_input = (user_input or "").strip()
    if not normalized_input:
        return "请输入有效的内容。"

    try:
        local_matcher = get_matcher()
        reply, score, matched_q = local_matcher.get_best_match(normalized_input)
    except Exception as e:
        return f"系统暂时不可用，请稍后重试。\n\n> 详细信息: {e}"

    selected_method = getattr(local_matcher, "last_selected_method", "unknown")
    
    if matched_q:
        diagnostic_log = (
            f"\n\n> 匹配问句:「*{matched_q}*」"
            f" | 相似度: **{score:.4f}**"
            f" | 评分分支: **{selected_method}**"
        )
    else:
        diagnostic_log = (
            "\n\n> 当前相似度得分低于设定阈值，无法给出准确回复"
            f" | 最大相似度: **{score:.4f}**"
            f" | 评分分支: **{selected_method}**"
        )
        
    return reply + diagnostic_log


# 构建 Web 界面
demo = gr.ChatInterface(
    fn=predict,
    title="检索式中文对话系统 (SimCSE)",
    description=(
        "**计算机专业本科毕业设计** | **核心架构:** 基于预训练深度对比学习模型与余弦相似度检索。\n"
        "系统使用开源对话基准 LCCC 语料进行检索，并在本地缓存句子特征向量以提高启动和响应速度。"
    ),
    examples=["最近有什么好看的电影推荐吗？", "毕业设计进度有点卡住了，好焦虑", "今天天气真不错～"],
)

if __name__ == "__main__":
    # 启动界面
    try:
        # 主进程启动前初始化一次，避免首条消息等待模型加载
        get_matcher()
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        print(f"启动界面失败: {e}")