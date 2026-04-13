import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.data_loader import DataLoader
from src.model import SimCSEEncoder
from src.matcher import DialogMatcher

def main():
    query = input("请输入测试问句: ").strip()
    if not query:
        print("输入为空，测试结束。")
        return

    data_loader = DataLoader()
    queries, replies = data_loader.load_corpus()
    if not queries:
        print("语料为空，无法测试。")
        return

    encoder = SimCSEEncoder()
    matcher = DialogMatcher(encoder, queries, replies)

    result = matcher.evaluate_two_modules(query)

    print("\n================= 模块测试结果 =================")
    print(f"输入: {result['input']}")

    qq = result["query_query"]
    print("\n[模块1] 问句-问句匹配")
    print(f"得分: {qq['score']:.4f}")
    print(f"命中问句: {qq['matched_query']}")
    print(f"输出答句: {qq['reply']}")

    qr = result["wordvec_qa"]
    print("\n[模块2] 词向量问答匹配")
    print(f"得分: {qr['score']:.4f}")
    print(f"命中问句: {qr['matched_query']}")
    print(f"输出答句: {qr['reply']}")

    print("\n[最终决策]")
    print(f"选中模块: {result['selected_method']}")
    print(f"最终得分: {result['selected_score']:.4f}")
    print(f"最终输出: {result['selected_reply']}")

if __name__ == "__main__":
    main()