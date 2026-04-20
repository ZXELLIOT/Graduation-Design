import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.model_engine import SimCSEModelEngine
from tests.local_config import DB_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

def get_bert_embeddings(sentences, model_path, batch_size=32):
    """获取指定路径基准模型的嵌入向量（CLS）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # 如果是相对路径，则尝试在 tests/models 下找
        if not os.path.isabs(model_path) and not model_path.startswith("huggingface/"):
            local_path = os.path.join(TEST_MODELS_DIR, model_path)
            if os.path.exists(local_path):
                model_path = local_path

        load_bar = tqdm(total=2, desc=f"正在加载模型 {os.path.basename(model_path)}", unit="步", leave=False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            load_bar.update(1)
            model = AutoModel.from_pretrained(model_path).to(device)
            load_bar.update(1)
        finally:
            load_bar.close()
    except Exception as e:
        print(f"加载模型 {model_path} 失败: {e}")
        return None
        
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="正在提取基准向量", unit="批", leave=False):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            outputs = model(**inputs)
            # 取 CLS 向量
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
            
    return torch.vstack(all_embeddings)

def run_human_eval_generation():
    # 1. 加载测试数据
    test_csv = os.path.join(DB_DATA_DIR, "lccc_test_neg.csv")
    if not os.path.exists(test_csv):
        print(f"数据文件不存在: {test_csv}")
        return

    csv_bar = tqdm(total=1, desc="正在加载人工评测数据", unit="步", leave=False)
    df = pd.read_csv(test_csv)
    csv_bar.update(1)
    csv_bar.close()
    
    # 随机采样 1000 条作为检索库 (Corpus)
    corpus_df = df.sample(n=min(1000, len(df)), random_state=42)
    corpus_queries = corpus_df["query"].astype(str).tolist()
    corpus_responses = corpus_df["response"].astype(str).tolist()
    
    # 随机采样 100 条测试输入
    remaining_df = df.drop(corpus_df.index)
    query_samples = remaining_df.sample(n=min(100, len(remaining_df)), random_state=24)
        
    test_queries = query_samples["query"].astype(str).tolist()
    expected_responses = query_samples["response"].astype(str).tolist()

    # 2. 初始化模型
    engine = SimCSEModelEngine()
    device = engine.device
    
    # 对比模型列表
    baseline_models_config = {
        "text2vec": "text2vec-base-chinese",
        "bert": "bert-base-chinese",
        "roberta": "chinese-roberta-wwm-ext",
        "mini-lm": "paraphrase-multilingual-MiniLM-L12-v2"
    }

    # 预加载所有基准模型，避免循环内重复加载
    loaded_baselines = {}
    preload_iterator = tqdm(baseline_models_config.items(), total=len(baseline_models_config), desc="正在预加载基准模型", unit="个")
    for name, path in preload_iterator:
        print(f"正在预加载基准模型 {name}...")
        try:
            if not os.path.isabs(path) and not path.startswith("huggingface/"):
                local_path = os.path.join(TEST_MODELS_DIR, path)
                if os.path.exists(local_path):
                    path = local_path
            load_bar = tqdm(total=2, desc=f"正在加载 {name}", unit="步", leave=False)
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                load_bar.update(1)
                model = AutoModel.from_pretrained(path).to(device)
                load_bar.update(1)
            finally:
                load_bar.close()
            model.eval()
            loaded_baselines[name] = (tokenizer, model)
        except Exception as e:
            print(f"预加载模型 {name} 失败: {e}")

    def get_embeddings_fast(sentences, tokenizer, model, batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="正在批量编码", unit="批", leave=False):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
        return torch.vstack(all_embeddings)

    # 3. 编码语料库 - 本模型
    print("正在使用本模型编码语料库...")
    corpus_vecs_ours = []
    for txt in tqdm(corpus_queries, desc="本模型语料编码", unit="条"):
        vec = engine.encode_one(txt, encoder="query")
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)
        corpus_vecs_ours.append(vec)
    corpus_vecs_ours = torch.stack(corpus_vecs_ours).to(device)

    # 编码语料库 - 基准模型
    baseline_corpus_vecs = {}
    for name, (tokenizer, model) in loaded_baselines.items():
        print(f"正在使用 {name} 编码语料库...")
        vecs = get_embeddings_fast(corpus_queries, tokenizer, model)
        baseline_corpus_vecs[name] = vecs.to(device)

    # 4. 执行检索并生成结果
    eval_results = []
    
    print("开始执行检索对比...")
    for i, q_text in enumerate(tqdm(test_queries, desc="正在执行检索", unit="条")):
        result_entry = {
            "测试问句": q_text,
            "期望原句": expected_responses[i],
        }

        # --- 本模型检索 ---
        q_v_ours_raw = engine.encode_one(q_text, encoder="query")
        if isinstance(q_v_ours_raw, np.ndarray):
            q_v_ours = torch.from_numpy(q_v_ours_raw).to(device).unsqueeze(0)
        else:
            q_v_ours = q_v_ours_raw.to(device).unsqueeze(0)
        
        sims_ours = torch.matmul(q_v_ours, corpus_vecs_ours.T).squeeze(0)
        top_idx_ours = int(torch.argmax(sims_ours).item())
        result_entry["本项目输出"] = corpus_responses[top_idx_ours]

        # --- 基准模型检索 ---
        for name, (tokenizer, model) in loaded_baselines.items():
            if name not in baseline_corpus_vecs:
                continue
            
            # 使用预加载好的模型直接编码
            q_v_baseline = get_embeddings_fast([q_text], tokenizer, model).to(device)
            sims_baseline = torch.matmul(q_v_baseline, baseline_corpus_vecs[name].T).squeeze(0)
            top_idx_baseline = int(torch.argmax(sims_baseline).item())
            result_entry[f"{name}输出"] = corpus_responses[top_idx_baseline]
        
        result_entry["本项目评分(1-5)"] = ""
        result_entry["基准最高评分(1-5)"] = ""
        result_entry["备注"] = ""
        
        eval_results.append(result_entry)
        
    # 5. 保存结果
    results_dir = TEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "human_evaluation_comparison.csv")
    pd.DataFrame(eval_results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n人工评测模板已生成: {output_path}")

if __name__ == "__main__":
    run_human_eval_generation()