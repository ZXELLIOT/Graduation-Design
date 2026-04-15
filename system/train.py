import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import argparse
import logging
import pandas as pd

# --- 修正点 1: 修改导入路径 ---
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

# 根据运行时环境返回合适的 autocast 上下文管理器：
def autocast_context(scaler):
    """如果 scaler 非空且有 CUDA，则返回启用混合精度的 autocast，否则返回空上下文。"""
    if scaler is None:
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    # 当可用时优先使用 torch.amp.autocast（在较新 torch 中可用），否则回退到 torch.cuda.amp.autocast
    try:
        from torch import autocast as torch_autocast
        return torch_autocast('cuda', dtype=torch.float16)
    except Exception:
        from torch.cuda.amp import autocast as torch_cuda_autocast
        try:
            return torch_cuda_autocast()
        except Exception:
            return nullcontext()

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 检查 CUDA 是否可用 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# --- 数据集类 ---
class QueryOnlyDataset(Dataset):
    """仅用于问句训练的数据集"""
    def __init__(self, queries):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        # 对于 SimCSE，正例就是文本本身
        return query, query

class QueryResponseDataset(Dataset):
    """用于问答对训练的数据集"""
    def __init__(self, queries, responses):
        if len(queries) != len(responses):
            raise ValueError("Queries and Responses must have the same length.")
        self.queries = queries
        self.responses = responses

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        return query, response

# --- 模型类 ---
class DualEncoderModel(nn.Module):
    def __init__(self, model_name_or_path, pooling_strategy="cls", temperature=0.05):
        super(DualEncoderModel, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.response_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.pooling_strategy = pooling_strategy
        self.temperature = temperature

    def encode(self, encoder, input_ids, attention_mask):
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling_strategy == "cls":
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            masked_output = outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0)
            pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        pooled_output = F.normalize(pooled_output, p=2, dim=1)
        return pooled_output

    def forward(self, query_input_ids=None, query_attention_mask=None, response_input_ids=None, response_attention_mask=None):
        embs = {}
        if query_input_ids is not None and query_attention_mask is not None:
            embs['query'] = self.encode(self.query_encoder, query_input_ids, query_attention_mask)
        if response_input_ids is not None and response_attention_mask is not None:
            embs['response'] = self.encode(self.response_encoder, response_input_ids, response_attention_mask)
        return embs

def run_stage(model, tokenizer, dataset, validation_dataset, stage_name, epochs, batch_size, lr, max_length, is_qr_stage=False, scaler=None):
    """
    执行一个训练阶段
    Args:
        model: 要训练的模型
        tokenizer: 分词器
        dataset: 训练数据集
        validation_dataset: 验证数据集
        stage_name: 阶段名称 ("Stage 1: Query Only" 或 "Stage 2: Query-Response")
        epochs: 该阶段的训练轮数
        batch_size: 批次大小
        lr: 学习率
        max_length: 最大序列长度
        is_qr_stage: 是否是问答对训练阶段 (True for QR, False for Query-only)
        scaler: 混合精度训练的scaler
    """
    # 使用更小的 dataloader，只加载部分数据
    # 在 Windows 上避免多进程 DataLoader 导致的问题，且仅在使用 CUDA 时启用 pin_memory
    pin_memory = True if DEVICE.type == 'cuda' else False
    num_workers = 0 if os.name == 'nt' else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 计算总的训练步数
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * 0.05) # 减少warmup比例
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    model.train()
    step_count = 0

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"{stage_name} - Starting Epoch {epoch + 1}/{epochs}")
        # 为避免在阶段开始时显示空的总体进度条，按 epoch 创建局部进度条
        epoch_bar = tqdm(total=len(dataloader), desc=f"{stage_name} - Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            texts1, texts2 = batch
            
            if is_qr_stage:
                # 问答对阶段
                query_inputs = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                response_inputs = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
                with autocast_context(scaler):
                    encoded_dict = model(
                        query_input_ids=query_inputs['input_ids'],
                        query_attention_mask=query_inputs['attention_mask'],
                        response_input_ids=response_inputs['input_ids'],
                        response_attention_mask=response_inputs['attention_mask']
                    )
                    query_embs = encoded_dict['query']
                    response_embs = encoded_dict['response']
                    # 计算相似度矩阵 (query vs response)
                    similarities = torch.matmul(query_embs, response_embs.T) / model.temperature
            else:
                # 仅问句阶段
                inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
                with autocast_context(scaler):
                    encoded_dict1 = model(query_input_ids=inputs1['input_ids'], query_attention_mask=inputs1['attention_mask'])
                    encoded_dict2 = model(response_input_ids=inputs2['input_ids'], response_attention_mask=inputs2['attention_mask'])
                    emb1 = encoded_dict1.get('query', encoded_dict1.get('response'))
                    emb2 = encoded_dict2.get('response', encoded_dict2.get('query'))
                    # 计算相似度矩阵 (emb1 vs emb2)，它们是相同的文本
                    similarities = torch.matmul(emb1, emb2.T) / model.temperature

            # labels: 对角线位置的索引
            batch_size_current = similarities.size(0)
            labels = torch.arange(batch_size_current).to(DEVICE)
            
            # 计算 InfoNCE loss
            # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
            with autocast_context(scaler):
                loss = F.cross_entropy(similarities, labels)

            # 反向传播 - 使用混合精度
            optimizer.zero_grad()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                # 梯度裁剪，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 修复：将 scheduler.step() 移到 optimizer.step() 之后
            scheduler.step()
            
            total_loss += loss.item()
            step_count += 1

            # 每步更新进度条；每100步更新一次附加信息以减少输出频率
            if step_count % 100 == 0:
                epoch_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                    'epoch': epoch+1
                })
            epoch_bar.update(1)
        
        avg_epoch_loss = total_loss / len(dataloader)
        logger.info(f"{stage_name} - Epoch {epoch + 1} completed. Average Train Loss: {avg_epoch_loss:.4f}")
        epoch_bar.close()
        # --- Validation ---
        if validation_dataset is not None:
            val_loss = evaluate_model(model, tokenizer, validation_dataset, max_length, is_qr_stage, scaler)
            logger.info(f"{stage_name} - Epoch {epoch + 1} completed. Average Val Loss: {val_loss:.4f}")

def evaluate_model(model, tokenizer, validation_dataset, max_length, is_qr_stage, scaler=None):
    """评估模型在验证集上的损失"""
    model.eval()
    # 验证时使用更大的 batch size；在 Windows 上禁用多进程
    pin_memory = True if DEVICE.type == 'cuda' else False
    num_workers = 0 if os.name == 'nt' else 1
    dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            texts1, texts2 = batch
            
            if is_qr_stage:
                # 问答对阶段
                query_inputs = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                response_inputs = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
                with autocast_context(scaler):
                    encoded_dict = model(
                        query_input_ids=query_inputs['input_ids'],
                        query_attention_mask=query_inputs['attention_mask'],
                        response_input_ids=response_inputs['input_ids'],
                        response_attention_mask=response_inputs['attention_mask']
                    )
                    query_embs = encoded_dict['query']
                    response_embs = encoded_dict['response']
                    similarities = torch.matmul(query_embs, response_embs.T) / model.temperature
            else:
                # 仅问句阶段
                inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
                with autocast_context(scaler):
                    encoded_dict1 = model(query_input_ids=inputs1['input_ids'], query_attention_mask=inputs1['attention_mask'])
                    encoded_dict2 = model(response_input_ids=inputs2['input_ids'], response_attention_mask=inputs2['attention_mask'])
                    emb1 = encoded_dict1.get('query', encoded_dict1.get('response'))
                    emb2 = encoded_dict2.get('response', encoded_dict2.get('query'))
                    similarities = torch.matmul(emb1, emb2.T) / model.temperature

            batch_size_current = similarities.size(0)
            labels = torch.arange(batch_size_current).to(DEVICE)
            
            # --- 修正点 2: 使用 autocast（根据 scaler 与 CUDA 可用性选择） ---
            with autocast_context(scaler):
                loss = F.cross_entropy(similarities, labels)
                
            total_val_loss += loss.item()
            num_batches += 1
    
    model.train() # 重新设置为训练模式
    return total_val_loss / num_batches if num_batches > 0 else float('inf')

def main():
    # --- 设置默认参数 (硬编码你的需求) ---
    DEFAULT_ARGS = {
        "model_name_or_path": "shibing624/text2vec-base-chinese",
        "train_data_file": r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data\lccc_train.csv",
        "valid_data_file": r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data\lccc_valid.csv",
        "test_data_file": r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\data\lccc_test.csv",
        "output_dir": r"C:\Users\13713\个人信息\毕业设计\simcse-demo\system\model\mysimcse",
        "stage1_epochs": 1,
        "stage1_batch_size": 8,
        "stage1_learning_rate": 3e-5,
        "stage2_epochs": 1,
        "stage2_batch_size": 8,
        "stage2_learning_rate": 2e-5,
        "max_length": 64,
        "temperature": 0.05,
        "use_fp16": True # 默认开启 FP16
    }

    parser = argparse.ArgumentParser(description="Two-stage training with validation: Stage 1 (Query-only), Stage 2 (Query-Response).")
    
    # 通用参数
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_ARGS["model_name_or_path"])
    parser.add_argument("--train_data_file", type=str, default=DEFAULT_ARGS["train_data_file"])
    parser.add_argument("--valid_data_file", type=str, default=DEFAULT_ARGS["valid_data_file"])
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_ARGS["test_data_file"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_ARGS["temperature"])
    parser.add_argument("--use_fp16", action="store_true", default=DEFAULT_ARGS["use_fp16"]) # 默认开启
    
    # Stage 1 Args
    parser.add_argument("--stage1_epochs", type=int, default=DEFAULT_ARGS["stage1_epochs"])
    parser.add_argument("--stage1_batch_size", type=int, default=DEFAULT_ARGS["stage1_batch_size"])
    parser.add_argument("--stage1_learning_rate", type=float, default=DEFAULT_ARGS["stage1_learning_rate"])
    
    # Stage 2 Args
    parser.add_argument("--stage2_epochs", type=int, default=DEFAULT_ARGS["stage2_epochs"])
    parser.add_argument("--stage2_batch_size", type=int, default=DEFAULT_ARGS["stage2_batch_size"])
    parser.add_argument("--stage2_learning_rate", type=float, default=DEFAULT_ARGS["stage2_learning_rate"])
    
    # Common Args
    parser.add_argument("--max_length", type=int, default=DEFAULT_ARGS["max_length"])
    parser.add_argument("--quick", action="store_true", help="启用快速小规模训练（少量样本、少量轮、较大 batch），便于快速迭代与汇报")
    
    args = parser.parse_args()

    # 如果启用 quick 模式，覆盖部分参数以加速训练/缩短时间
    if args.quick:
        logger.info("Quick mode enabled: limiting data and reducing epochs for fast run")
        # 训练/验证/测试样本数限制
        QUICK_TRAIN_NROWS = 2000
        QUICK_VALID_NROWS = 200
        QUICK_TEST_NROWS = 200
        # 调整训练超参数以加速（可根据需要再调）
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.stage1_batch_size = 32
        args.stage2_batch_size = 32
        args.stage1_learning_rate = 3e-5
        args.stage2_learning_rate = 2e-5
    else:
        QUICK_TRAIN_NROWS = 50000
        QUICK_VALID_NROWS = 1000
        QUICK_TEST_NROWS = 1000

    # 1. 加载分词器和模型
    logger.info(f"Loading tokenizer and model from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
    model = DualEncoderModel(
        model_name_or_path=args.model_name_or_path,
        pooling_strategy="cls",
        temperature=args.temperature
    ).to(DEVICE)

    # 2. 加载训练、验证、测试数据
    logger.info(f"Loading training data from: {args.train_data_file}")
    df_train = pd.read_csv(args.train_data_file, nrows=QUICK_TRAIN_NROWS)
    if 'query' not in df_train.columns or 'response' not in df_train.columns:
        raise ValueError("Training CSV file must contain 'query' and 'response' columns.")
    train_queries = df_train['query'].astype(str).tolist()
    train_responses = df_train['response'].astype(str).tolist()

    logger.info(f"Loading validation data from: {args.valid_data_file}")
    df_valid = pd.read_csv(args.valid_data_file, nrows=QUICK_VALID_NROWS)
    if 'query' not in df_valid.columns or 'response' not in df_valid.columns:
        raise ValueError("Validation CSV file must contain 'query' and 'response' columns.")
    valid_queries = df_valid['query'].astype(str).tolist()
    valid_responses = df_valid['response'].astype(str).tolist()

    # 测试集暂时加载，用于最终评估提示
    logger.info(f"Loading test data from: {args.test_data_file}")
    df_test = pd.read_csv(args.test_data_file, nrows=QUICK_TEST_NROWS)
    if 'query' not in df_test.columns or 'response' not in df_test.columns:
        raise ValueError("Test CSV file must contain 'query' and 'response' columns.")
    test_queries = df_test['query'].astype(str).tolist()
    test_responses = df_test['response'].astype(str).tolist()

    logger.info(f"Loaded {len(train_queries)} train, {len(valid_queries)} valid, {len(test_queries)} test samples.")

    # 创建混合精度训练的scaler
    # --- 修正点 3: 使用 GradScaler ---
    scaler = GradScaler() if args.use_fp16 and torch.cuda.is_available() else None
    
    if scaler is not None:
        logger.info("Mixed precision training (FP16) enabled.")
    else:
        logger.info("Using FP32 training.")

    # --- STAGE 1: Query-only Training ---
    logger.info("--- Starting Stage 1: Query-only Training ---")
    stage1_train_dataset = QueryOnlyDataset(train_queries)
    stage1_valid_dataset = QueryOnlyDataset(valid_queries) # 验证也用问句
    run_stage(
        model=model,
        tokenizer=tokenizer,
        dataset=stage1_train_dataset,
        validation_dataset=stage1_valid_dataset,
        stage_name="Stage 1: Query Only",
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        lr=args.stage1_learning_rate,
        max_length=args.max_length,
        is_qr_stage=False,
        scaler=scaler
    )

    # --- STAGE 2: Query-Response Training ---
    logger.info("--- Starting Stage 2: Query-Response Training ---")
    stage2_train_dataset = QueryResponseDataset(train_queries, train_responses)
    stage2_valid_dataset = QueryResponseDataset(valid_queries, valid_responses) # 验证用问句-回答对
    run_stage(
        model=model,
        tokenizer=tokenizer,
        dataset=stage2_train_dataset,
        validation_dataset=stage2_valid_dataset,
        stage_name="Stage 2: Query-Response",
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        lr=args.stage2_learning_rate,
        max_length=args.max_length,
        is_qr_stage=True,
        scaler=scaler
    )

    # 3. 保存最终模型
    logger.info(f"Saving final fine-tuned model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    # 分别保存 query 与 response encoder，避免覆盖
    query_dir = os.path.join(args.output_dir, "query_encoder")
    response_dir = os.path.join(args.output_dir, "response_encoder")
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)
    model.query_encoder.save_pretrained(query_dir)
    model.response_encoder.save_pretrained(response_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Final model saved successfully.")

    # 4. 提示用户下一步可以进行测试评估
    logger.info("Training completed! You can now load the model from the output directory and perform inference on your test set.")

if __name__ == "__main__":
    main()