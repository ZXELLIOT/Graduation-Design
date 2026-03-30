import sys
import os
import gradio as gr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import DataLoader
from src.model import SimCSEEncoder
from src.matcher import DialogMatcher
from src.preprocess import TextPreprocessor

print("==================================")
print("     基于 SimCSE 的中文对话系统     ")
print("==================================\n")
# 在系统启动前，先执行一次数据清洗与预处理，确保最新语料被加载
print("[1/4] 语料数据预处理...")
preprocessor = TextPreprocessor()
preprocessor.process()
# 初始化数据加载器
print("\n[2/4] 加载对话语料...")
data_loader = DataLoader()
queries, replies = data_loader.load_corpus()
# 初始化预训练语言模型
print("\n[3/4] 初始化 SimCSE 语义模型...")
encoder = SimCSEEncoder()
# 进行相似度计算
print("\n[4/4] 进行对话匹配相似度计算模块...")
matcher = DialogMatcher(encoder, queries, replies)
print("\n✅ 系统所有模块加载完成，正在启动 Web UI...\n")

def predict(user_input):
    """前端交互函数"""
    if not user_input.strip():
        return "请输入有效内容。"
    reply, score, matched_q = matcher.get_best_match(user_input)
    if matched_q:
        log_info = f"\n\n*(系统诊断: 最佳匹配问句:「{matched_q}」| 相似度置信得分: {score:.4f})*"
    else:
        log_info = f"\n\n*(系统诊断: 无高于设定阈值的匹配结果 | 最大相似度: {score:.4f})*"
    return reply + log_info

# 构建 Web 界面
demo = gr.ChatInterface(
    fn=predict,
    title="基于 SimCSE 的中文对话系统",
    description="本平台为毕业设计展示 Demo。\n系统基于对比学习提取语句的密集向量表示（Dense Embeddings），通过余弦相似度进行上下文检索匹配。",
    examples=["你好啊！", "能聊聊南京邮电大学吗？", "今天天气真不错啊", "帮我查一下快递", "完全超出语料的闲聊测试"]
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)