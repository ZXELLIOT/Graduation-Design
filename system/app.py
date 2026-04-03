import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from src.data_loader import DataLoader
from src.model import SimCSEEncoder
from src.matcher import DialogMatcher
from src.preprocess import TextPreprocessor

def initialize_system():
    print("==========================================================")
    print("             基于 SimCSE 的检索式中文对话系统           ")
    print("==========================================================\n")

    print("[1/4] 执行语料数据预处理...")
    preprocessor = TextPreprocessor()
    preprocessor.process()

    print("\n[2/4] 加载对话语料...")
    data_loader = DataLoader()
    queries, replies = data_loader.load_corpus()

    print("\n[3/4] 初始化 SimCSE 语义模型...")
    encoder = SimCSEEncoder()

    print("\n[4/4] 构建句子特征向量与匹配索引...")
    # 尝试加载缓存加速启动，否则全量计算
    matcher = DialogMatcher(encoder, queries, replies)
    
    print("\n[√] 系统核心模块加载完成，准备启动 Web 界面...\n")
    return matcher

# 初始化系统
matcher = initialize_system()


def predict(user_input: str, history: list) -> str:
    """
    接收用户输入，计算相似度并返回最佳匹配回复。
    """
    if not user_input or not user_input.strip():
        return "请输入有效的内容。"
    
    reply, score, matched_q = matcher.get_best_match(user_input)
    
    if matched_q:
        diagnostic_log = f"\n\n> 匹配问句:「*{matched_q}*」| 相似度: **{score:.4f}**"
    else:
        diagnostic_log = f"\n\n> 当前相似度得分低于设定阈值，无法给出准确回复 | 最大相似度: **{score:.4f}**"
        
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
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        print(f"启动界面失败: {e}")