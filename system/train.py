import os
import torch
import argparse
import logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from contextlib import nullcontext


def autocast_context(scaler):
    """如果传入的缩放器可用且存在 CUDA，则返回启用混合精度的上下文，否则返回空上下文。

    参数：
    - scaler: 混合精度训练使用的缩放器；若为 None 则不启用混合精度。
    返回：上下文管理器，可用于 with 语句。
    """
    if scaler is None:
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    try:
        from torch import autocast as torch_autocast
        return torch_autocast('cuda', dtype=torch.float16)
    except Exception:
        from torch.cuda.amp import autocast as torch_cuda_autocast
        try:
            return torch_cuda_autocast()
        except Exception:
            return nullcontext()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")

class QueryOnlyDataset(Dataset):
    """仅包含问句的简单数据容器。"""
    def __init__(self, queries):
        """初始化问句列表。

        参数：
        - queries: 问句字符串列表。
        """
        self.queries = queries

    def __len__(self):
        """返回数据条目数（问句数量）。"""
        return len(self.queries)

    def __getitem__(self, idx):
        """根据索引返回一条训练样本，样本以问句自身为正例。

        返回：(问句, 作为对照的文本)
        """
        query = self.queries[idx]
        return query, query

class QueryResponseDataset(Dataset):
    """包含问句与对应回答的简单数据容器。"""
    def __init__(self, queries, responses):
        """初始化问答对列表。

        要求问句列表与回答列表长度一致。
        """
        if len(queries) != len(responses):
            raise ValueError("问句列表与回答列表长度必须一致。")
        self.queries = queries
        self.responses = responses

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        return query, response

class DualEncoderModel(nn.Module):
    """双编码器模型：用于分别编码问句与回答并计算向量相似度。"""
    def __init__(self, model_name_or_path, pooling_strategy="cls", temperature=0.05, local_files_only=False):
        super(DualEncoderModel, self).__init__()
        q_dir = os.path.join(model_name_or_path, 'query_encoder')
        r_dir = os.path.join(model_name_or_path, 'response_encoder')
        if os.path.isdir(q_dir) and os.path.isdir(r_dir):
            self.query_encoder = AutoModel.from_pretrained(q_dir, local_files_only=True)
            self.response_encoder = AutoModel.from_pretrained(r_dir, local_files_only=True)
        else:
            self.query_encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
            self.response_encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.pooling_strategy = pooling_strategy
        self.temperature = temperature

    def encode(self, encoder, input_ids, attention_mask):
        """对一批输入计算句向量并返回归一化后的向量。

        参数：
        - encoder: 具体的编码器对象（用于前向计算）；
        - input_ids, attention_mask: 分词后得到的张量输入；
        返回：归一化后的向量张量，形状为 (批次大小, 向量维度)。
        """
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling_strategy == "cls":
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            masked_output = outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0)
            pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling_strategy}")
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
    """执行一次训练阶段并在每个轮次后进行验证（如提供）。

    参数说明：
    - model: 待训练的模型对象；
    - tokenizer: 文本分词器；
    - dataset / validation_dataset: 训练与验证用的数据容器；
    - stage_name: 阶段标识，用于日志输出；
    - epochs, batch_size, lr, max_length: 基本训练超参；
    - is_qr_stage: 是否为问答对训练阶段（若为否则为问句自监督训练）；
    - scaler: 混合精度训练缩放器，可为 None 表示不使用混合精度。
    """
    pin_memory = True if DEVICE.type == 'cuda' else False
    num_workers = 0 if os.name == 'nt' else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    model.train()
    step_count = 0

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"{stage_name} - Starting Epoch {epoch + 1}/{epochs}")
        epoch_bar = tqdm(total=len(dataloader), desc=f"{stage_name} - Epoch {epoch + 1}/{epochs}")
        
        for batch in enumerate(dataloader):
            texts1, texts2 = batch
            
            if is_qr_stage:
                query_inputs = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                response_inputs = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
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
                inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                with autocast_context(scaler):
                    encoded_dict1 = model(query_input_ids=inputs1['input_ids'], query_attention_mask=inputs1['attention_mask'])
                    encoded_dict2 = model(response_input_ids=inputs2['input_ids'], response_attention_mask=inputs2['attention_mask'])
                    emb1 = encoded_dict1.get('query', encoded_dict1.get('response'))
                    emb2 = encoded_dict2.get('response', encoded_dict2.get('query'))
                    similarities = torch.matmul(emb1, emb2.T) / model.temperature

            batch_size_current = similarities.size(0)
            labels = torch.arange(batch_size_current).to(DEVICE)
            
            with autocast_context(scaler):
                loss = F.cross_entropy(similarities, labels)

            optimizer.zero_grad()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            
            total_loss += loss.item()
            step_count += 1

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
                query_inputs = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                response_inputs = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
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
                inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
                
                with autocast_context(scaler):
                    encoded_dict1 = model(query_input_ids=inputs1['input_ids'], query_attention_mask=inputs1['attention_mask'])
                    encoded_dict2 = model(response_input_ids=inputs2['input_ids'], response_attention_mask=inputs2['attention_mask'])
                    emb1 = encoded_dict1.get('query', encoded_dict1.get('response'))
                    emb2 = encoded_dict2.get('response', encoded_dict2.get('query'))
                    similarities = torch.matmul(emb1, emb2.T) / model.temperature

            batch_size_current = similarities.size(0)
            labels = torch.arange(batch_size_current).to(DEVICE)
            
            with autocast_context(scaler):
                loss = F.cross_entropy(similarities, labels)
                
            total_val_loss += loss.item()
            num_batches += 1
    
    model.train() 
    return total_val_loss / num_batches if num_batches > 0 else float('inf')

def main():
    DEFAULT_ARGS = {
        "model_name_or_path": r"C:\\Users\\13713\\个人信息\\毕业设计\\simcse-demo\\system\\model\\mysimcse",
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
        "use_fp16": True 
    }

    parser = argparse.ArgumentParser(description="Two-stage training with validation: Stage 1 (Query-only), Stage 2 (Query-Response).")
    
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_ARGS["model_name_or_path"])
    parser.add_argument("--train_data_file", type=str, default=DEFAULT_ARGS["train_data_file"])
    parser.add_argument("--valid_data_file", type=str, default=DEFAULT_ARGS["valid_data_file"])
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_ARGS["test_data_file"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_ARGS["temperature"])
    parser.add_argument("--use_fp16", action="store_true", default=DEFAULT_ARGS["use_fp16"])
    parser.add_argument("--stage1_epochs", type=int, default=DEFAULT_ARGS["stage1_epochs"])
    parser.add_argument("--stage1_batch_size", type=int, default=DEFAULT_ARGS["stage1_batch_size"])
    parser.add_argument("--stage1_learning_rate", type=float, default=DEFAULT_ARGS["stage1_learning_rate"])
    parser.add_argument("--stage2_epochs", type=int, default=DEFAULT_ARGS["stage2_epochs"])
    parser.add_argument("--stage2_batch_size", type=int, default=DEFAULT_ARGS["stage2_batch_size"])
    parser.add_argument("--stage2_learning_rate", type=float, default=DEFAULT_ARGS["stage2_learning_rate"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_ARGS["max_length"])
    parser.add_argument("--quick", action="store_true", help="启用快速小规模训练（少量样本、少量轮、较大 batch），便于快速迭代与汇报")
    args = parser.parse_args()

    if args.quick:
        logger.info("Quick mode enabled: limiting data and reducing epochs for fast run")
        QUICK_TRAIN_NROWS = 2000
        QUICK_VALID_NROWS = 200
        QUICK_TEST_NROWS = 200
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

    logger.info(f"Loading tokenizer and model from: {args.model_name_or_path}")
    local_files_only = False
    if os.path.isdir(args.model_name_or_path):
        for cand in ['config.json', 'pytorch_model.bin', 'tf_model.h5', 'flax_model.msgpack']:
            if os.path.exists(os.path.join(args.model_name_or_path, cand)):
                local_files_only = True
                break
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
    model = DualEncoderModel(
        model_name_or_path=args.model_name_or_path,
        pooling_strategy="cls",
        temperature=args.temperature,
        local_files_only=local_files_only
    ).to(DEVICE)

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

    logger.info(f"Loading test data from: {args.test_data_file}")
    df_test = pd.read_csv(args.test_data_file, nrows=QUICK_TEST_NROWS)
    if 'query' not in df_test.columns or 'response' not in df_test.columns:
        raise ValueError("Test CSV file must contain 'query' and 'response' columns.")
    test_queries = df_test['query'].astype(str).tolist()

    logger.info(f"Loaded {len(train_queries)} train, {len(valid_queries)} valid, {len(test_queries)} test samples.")

    scaler = GradScaler() if args.use_fp16 and torch.cuda.is_available() else None
    
    if scaler is not None:
        logger.info("Mixed precision training (FP16) enabled.")
    else:
        logger.info("Using FP32 training.")

    logger.info("--- Starting Stage 1: Query-only Training ---")
    stage1_train_dataset = QueryOnlyDataset(train_queries)
    stage1_valid_dataset = QueryOnlyDataset(valid_queries) 
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

    logger.info(f"Saving final fine-tuned model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    query_dir = os.path.join(args.output_dir, "query_encoder")
    response_dir = os.path.join(args.output_dir, "response_encoder")
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)
    model.query_encoder.save_pretrained(query_dir)
    model.response_encoder.save_pretrained(response_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Final model saved successfully.")
    logger.info("Training completed! You can now load the model from the output directory and perform inference on your test set.")

if __name__ == "__main__":
    main()