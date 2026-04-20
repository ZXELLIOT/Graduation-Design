import os
import sys
import torch
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

# 导入本项目模型
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.model_engine import SimCSEModelEngine
from tests.local_config import SIMCSE_MODEL_DIR, TEST_DATA_DIR, TEST_MODELS_DIR, TEST_RESULTS_DIR

# 缓存模型以避免重复加载
_model_data = {}

def get_bert_embeddings(sentences, model_path, batch_size=128):
    """
    加载通用的开源基准模型，并从其输出中提取代表整句含义的向量。
    
    参数:
        sentences: 待处理的句子列表。
        model_path: 基准模型所在的本地目录。
        batch_size: 每一轮并行处理的数量。
    """
    global _model_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 如果该模型之前加载过，直接从内存读取以节省时间
    if model_path not in _model_data:
        load_bar = tqdm(total=2, desc=f"正在加载模型 {os.path.basename(model_path)}", unit="步", leave=False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            load_bar.update(1)
            model = AutoModel.from_pretrained(model_path).to(device)
            load_bar.update(1)
            model.eval()
            _model_data[model_path] = (tokenizer, model)
        finally:
            load_bar.close()
    else:
        tokenizer, model = _model_data[model_path]
    
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"正在使用 {os.path.basename(model_path)} 提取向量", leave=False):
            batch = sentences[i:i+batch_size]
            # 对文本进行预处理和截断
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            outputs = model(**inputs)
            # 提取整句特征：优先取池化后的结果，否则取首位标识的结果
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def load_dataset_file(dataset_name, limit=2000):
    """
    根据给定的数据集名称（如 ATEC、LCQMC 等）加载对应的本地测试文件。
    
    参数:
        dataset_name: 数据集的简称。
        limit: 最大读取行数，用于提高评测速度。
    """
    paths = {
        "ATEC": os.path.join(TEST_DATA_DIR, "atec_test.txt"),
        "BQ": os.path.join(TEST_DATA_DIR, "bq_test.txt"),
        "LCQMC": os.path.join(TEST_DATA_DIR, "lcqmc_test.txt"),
        "STS-B": os.path.join(TEST_DATA_DIR, "stsb_test.txt"),
        "PAWS-X": os.path.join(TEST_DATA_DIR, "pawsx_test.txt"),
        "LCCC": os.path.join(TEST_DATA_DIR, "lccc.txt"),
    }
    path = paths.get(dataset_name)
    if not path or not os.path.exists(path):
        return []
    
    data = []
    # 假设文本文件采用 制表符 分隔，包含三个部分：句子A、句子B、人工打分的相似度等级
    load_bar = tqdm(total=1, desc=f"正在加载 {dataset_name} 文件", unit="步", leave=False)
    df = pd.read_csv(path, sep='\t', header=None, names=['句子1', '句子2', 'label'])
    load_bar.update(1)
    load_bar.close()
    # 截取前 N 条数据进行评测
    if limit > 0:
        df = df.head(limit)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"正在读取 {dataset_name} 测试集", leave=False):
        data.append((row['句子1'], row['句子2'], row['label']))
    return data

def run_multi_comparison(simcse_engine, datasets):
    """
    跨数据集对比评测核心函数：
    依次加载各个基准模型，计算它们在多个公开数据集上的表现评价指标（相关性系数）。
    """
    model_dirs = {
        "本项目模型": None, 
        "对比模型A": os.path.join(TEST_MODELS_DIR, "text2vec-base-chinese"),
        "对比模型B": os.path.join(TEST_MODELS_DIR, "bert-base-chinese"),
        "对比模型C": os.path.join(TEST_MODELS_DIR, "chinese-roberta-wwm-ext"),
        "对比模型D": os.path.join(TEST_MODELS_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
    }

    # 1. 预加载所有评测数据集到内存中
    all_dataset_data = {}
    for ds_name in datasets:
        data = load_dataset_file(ds_name)
        all_dataset_data[ds_name] = {
            "s1": [d[0] for d in data],
            "s2": [d[1] for d in data],
            "labels": [d[2] for d in data],
            "count": len(data)
        }

    # 用于存放各模型在各数据集上的最终得分
    final_results = {ds: [] for ds in all_dataset_data.keys()}

    # 2. 遍历每一个模型进行测试
    print(f"\n开始综合评测流程...")
    print("-" * 60)

    model_iterator = tqdm(model_dirs.items(), total=len(model_dirs), desc="正在评测各模型", unit="个")
    for model_name, path in model_iterator:
        print(f"正在分析模型: {model_name}...")
        dataset_iterator = tqdm(all_dataset_data.items(), total=len(all_dataset_data), desc=f"{model_name} 数据集评测", unit="集", leave=False)
        for ds_name, ds_info in dataset_iterator:
            if path is None:
                # 调用本项目自有的编码引擎
                e1_raw = simcse_engine.encode(ds_info["s1"])
                e2_raw = simcse_engine.encode(ds_info["s2"])
                
                # 统一数据格式
                e1 = e1_raw.cpu().numpy() if torch.is_tensor(e1_raw) else e1_raw
                e2 = e2_raw.cpu().numpy() if torch.is_tensor(e2_raw) else e2_raw
            else:
                if not os.path.exists(path):
                    continue
                # 调用基准模型提取函数
                e1 = get_bert_embeddings(ds_info["s1"], path)
                e2 = get_bert_embeddings(ds_info["s2"], path)
                
            # 计算两组向量的语义重合度得分（余弦值）
            preds = []
            for i in tqdm(range(len(e1)), desc=f"{model_name}-{ds_name} 相似度计算", unit="条", leave=False):
                sim = np.dot(e1[i], e2[i]) / (np.linalg.norm(e1[i]) * np.linalg.norm(e2[i]) + 1e-8)
                preds.append(sim)
                
            # 计算模型打分与人工标注分数之间的相关系数（相关性越高代表模型越准确）
            spearman_res = scipy.stats.spearmanr(ds_info["labels"], preds)
            corr = float(spearman_res[0]) # type: ignore
                
            final_results[ds_name].append({
                "Model": model_name, 
                "Spearman": round(corr, 4)
            })

    # 3. 生成汇总报表
    results_dir = TEST_RESULTS_DIR
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 整理表格形式：每一行为一个模型，每一列为一个测试数据集
    all_models = list(model_dirs.keys())
    matrix_data = []
    
    for model_name in all_models:
        row: dict[str, str | float] = {"模型名称": model_name}
        for ds_name in datasets:
            # 提取该模型对应的得分
            score_found = next((item["Spearman"] for item in final_results[ds_name] if item["Model"] == model_name), 0.0)
            row[ds_name] = float(score_found)
        matrix_data.append(row)
    
    df_matrix = pd.DataFrame(matrix_data)
    
    print("\n跨数据集评测汇总矩阵 (Spearman Correlation)")
    print("=" * 80)
    print(df_matrix.to_string(index=False))
    print("=" * 80)
    
    # 保存为单一的汇总 CSV
    output_path = os.path.join(results_dir, "model_comparison_matrix.csv")
    df_matrix.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n对比矩阵已保存至: {output_path}")

if __name__ == "__main__":
    engine = SimCSEModelEngine(model_dir=SIMCSE_MODEL_DIR)
    target_datasets = ["ATEC", "BQ", "LCQMC", "STS-B", "LCCC"]
    run_multi_comparison(engine, target_datasets)