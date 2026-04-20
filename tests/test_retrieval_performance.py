import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

# 导入本项目模型
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.model_engine import SimCSEModelEngine
from system.comparator import DialogComparator
from tests.local_config import DB_DATA_DIR, TEST_RESULTS_DIR

# 缓存基准模型以避免重复加载
_baseline_models = {}

def get_baseline_embeddings(sentences, model_path, batch_size=128):
    """加载基准模型并提取 [CLS] 向量"""
    global _baseline_models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_path not in _baseline_models:
        load_bar = tqdm(total=2, desc=f"正在加载基准模型 {os.path.basename(model_path)}", unit="步", leave=False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            load_bar.update(1)
            model = AutoModel.from_pretrained(model_path).to(device)
            load_bar.update(1)
            model.eval()
            _baseline_models[model_path] = (tokenizer, model)
        finally:
            load_bar.close()
    else:
        tokenizer, model = _baseline_models[model_path]
    
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="正在提取基准向量", unit="批", leave=False):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            outputs = model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
            
    return torch.vstack(all_embeddings)

def load_lccc_test_cases(count=500):
    """从本地 LCCC 测试集 CSV 中抽取测试用例"""
    csv_path = os.path.join(DB_DATA_DIR, "lccc_test_neg.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到测试集文件: {csv_path}")
    
    # 读取 CSV (已知列名为 query, response)
    csv_bar = tqdm(total=1, desc="正在读取LCCC测试集", unit="步", leave=False)
    df = pd.read_csv(csv_path)
    csv_bar.update(1)
    csv_bar.close()
    # 取前 count 条作为基准
    test_cases = []
    head_df = df.head(count)
    for _, row in tqdm(head_df.iterrows(), total=len(head_df), desc="正在构建测试用例", unit="条", leave=False):
        test_cases.append((str(row['query']), str(row['response'])))
    return test_cases

def run_retrieval_logic_comparison():
    # 1. 环境初始化
    engine = SimCSEModelEngine()
    comparator = DialogComparator(
        model_engine=engine,
        data_dir=DB_DATA_DIR,
        similarity_threshold=0.0,
    )
    comparator.initialize_index(prefix="train")
    
    # 2. 准备 1000 条测试数据
    print(f"\n[1/3] 正在从本地测试集抽取 1000 条基准数据...")
    test_cases = load_lccc_test_cases(1000)
    
    results = []
    method_scores = {"Cosine_Search": [], "Robust_Search": [], "Model_Engine": []}

    print("\n[2/3] 开始三种逻辑对比评测...")
    eval_details = []

    for q, expected in tqdm(test_cases, desc="评测进行中", unit="条"):
        # 编码当前测试问句
        q_vec_raw = engine.encode_one(q, encoder="query")
        if isinstance(q_vec_raw, np.ndarray):
            q_vec_tensor = torch.from_numpy(q_vec_raw).float()
        else:
            q_vec_tensor = q_vec_raw.float()
        q_vec = q_vec_tensor.unsqueeze(0).to(engine.device)
        
        # --- 逻辑 1: 问句向量 Top-1 直接检索 ---
        scores, candidates = comparator._db_search(q_vec_tensor, top_k=5)
        if not candidates:
            reply_1 = ""
            reply_2 = ""
            reply_3 = "知识库没有这个问题的回复"
        else:
            reply_1 = candidates[0]["reply"]

            # --- 逻辑 2: Top-K 候选二次重排（临时 0.5/0.5 加权） ---
            best_score = float("-inf")
            best_reply = candidates[0]["reply"]
            for i, item in enumerate(candidates):
                reply_emb = engine.encode_one(item["reply"], encoder="response")
                if isinstance(reply_emb, np.ndarray):
                    reply_tensor = torch.from_numpy(reply_emb).float()
                else:
                    reply_tensor = reply_emb.float()
                reply_sim = float(torch.matmul(q_vec_tensor, reply_tensor))
                mix_score = 0.5 * float(scores[i]) + 0.5 * reply_sim
                if mix_score > best_score:
                    best_score = mix_score
                    best_reply = item["reply"]
            reply_2 = best_reply

            # --- 逻辑 3: 生产逻辑 compare ---
            reply_3, _, _ = comparator.compare(q)
        
        # 记录每种策略的结果，供专家人工审核打分
        eval_details.append({
            "测试问句": q,
            "标准回答": expected,
            "策略A(简单匹配)": reply_1,
            "策略A评分": "",
            "策略B(协同匹配)": reply_2,
            "策略B评分": "",
            "系统综合结果": reply_3,
            "系统综合评分": "",
            "备注说明": ""
        })

    # 3. 结果保存
    results_dir = TEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    df_details = pd.DataFrame(eval_details)
    save_path = os.path.join(results_dir, "retrieval_logic_human_eval_template.csv")
    df_details.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("对比评测模板已生成完毕")
    print("="*80)
    print(f"文件位置: {save_path}")
    print(f"已包含 2000 条样本，请在 Excel 中打开并开始评分。")

if __name__ == "__main__":
    run_retrieval_logic_comparison()