"""
train/train.py
文件作用:
    双塔检索模型训练主脚本。
训练策略（两阶段微调）:
    阶段1 — 无监督 SimCSE:
        仅使用问句（query），通过 BERT 内置 Dropout 对同一句子两次前向传播
        产生两个略有差异的向量作为正样本对，让模型学会区分不同句子的语义。
        目标：将通用预训练模型转化为能理解对话语义倾向的基础编码器。
    阶段2 — 有监督匹配（双塔）:
        使用 (问句, 正样本回复, 负样本回复) 三元组，通过对比损失训练模型
        让问句与正确回复的相似度高于错误回复。
        query_encoder 编码问句，response_encoder 编码回复，双塔独立训练。
        目标：让双塔各自具备精确匹配的判别力。
"""

import os
import copy
import torch
import sys
import torch.nn.functional as F
from contextlib import nullcontext
import warnings
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings('ignore', message='Detected call of .lr_scheduler.step')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from train.train_config import (
    TRAIN_NEG_CSV_PATH,
    VALID_NEG_CSV_PATH,
    TEST_NEG_CSV_PATH,
    LOCAL_PRETRAINED_DIR,
    OUTPUT_MODEL_DIR,
)
from train.model_utils import DEVICE, SimCSEEncoder
from train.data_utils import DataManager

def autocast_context():
    return torch.autocast(device_type='cuda', dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

def _tokenize_to_device(tokenizer, texts, max_length):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

# ============================================================
# 阶段1：无监督 SimCSE
# ============================================================
def train_or_eval_stage1(
    encoder, tokenizer, dataloader, max_length, scaler,
    optimizer=None, scheduler=None
):
    """无监督 SimCSE — 同句两次前向，利用 dropout 产生正样本对。"""
    is_train = optimizer is not None
    encoder.model.train() if is_train else encoder.model.eval()
    total_loss = 0.0
    batch_count = 0
    iterator = dataloader
    ctx = nullcontext() if is_train else torch.no_grad()
    with ctx:
        for texts_q in iterator:
            q_in = _tokenize_to_device(tokenizer, texts_q, max_length)
            with autocast_context():
                emb1 = encoder.encode(q_in['input_ids'], q_in['attention_mask'])
                emb2 = encoder.encode(q_in['input_ids'], q_in['attention_mask'])
                sim = torch.matmul(emb1, emb2.T) / 0.05
                labels = torch.arange(sim.size(0), device=DEVICE)
                loss = F.cross_entropy(sim, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
            total_loss += loss.item()
            batch_count += 1
    return float('inf') if batch_count == 0 else total_loss / batch_count

# ============================================================
# 阶段2：有监督匹配（双塔）
# ============================================================
def train_or_eval_stage2(
    query_enc, resp_enc, tokenizer, dataloader, max_length, scaler,
    optimizer=None, scheduler=None
):
    """有监督对比学习 — query_enc 编码问句，resp_enc 编码回复。"""
    is_train = optimizer is not None
    query_enc.model.train() if is_train else query_enc.model.eval()
    resp_enc.model.train() if is_train else resp_enc.model.eval()
    total_loss = 0.0
    batch_count = 0
    iterator = dataloader
    ctx = nullcontext() if is_train else torch.no_grad()
    with ctx:
        for texts_q, texts_pos, texts_neg in iterator:
            q_in = _tokenize_to_device(tokenizer, texts_q, max_length)
            p_in = _tokenize_to_device(tokenizer, texts_pos, max_length)
            n_in = _tokenize_to_device(tokenizer, texts_neg, max_length)
            with autocast_context():
                q_emb = query_enc.encode(q_in['input_ids'], q_in['attention_mask'])
                p_emb = resp_enc.encode(p_in['input_ids'], p_in['attention_mask'])
                n_emb = resp_enc.encode(n_in['input_ids'], n_in['attention_mask'])
                pos_sim = torch.sum(q_emb * p_emb, dim=1)
                neg_sim = torch.sum(q_emb * n_emb, dim=1)
                logits = torch.stack([pos_sim, neg_sim], dim=1) / 0.05
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(query_enc.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(resp_enc.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(query_enc.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(resp_enc.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
            total_loss += loss.item()
            batch_count += 1
    return float('inf') if batch_count == 0 else total_loss / batch_count

# ============================================================
# 训练主流程入口
# ============================================================
def train_model():
    cfg = {
        'local_pretrained_dir': LOCAL_PRETRAINED_DIR,
        'train_data_file': TRAIN_NEG_CSV_PATH,
        'valid_data_file': VALID_NEG_CSV_PATH,
        'test_data_file': TEST_NEG_CSV_PATH,
        'output_dir': OUTPUT_MODEL_DIR,

        'max_length': 64,
        'stage1_train_size': 10,
        'stage1_valid_size': 10,
        'stage1_test_size': 10,
        'stage1_epochs': 2,
        'stage1_batch_size': 64,
        'stage1_learning_rate': 3e-5,

        'stage2_train_size': 10,
        'stage2_valid_size': 10,
        'stage2_test_size': 10,
        'stage2_epochs': 4,
        'stage2_batch_size': 64,
        'stage2_learning_rate': 3e-5,
    }
    for key in ('train_data_file', 'valid_data_file', 'test_data_file', 'output_dir'):
        cfg[key] = os.path.abspath(cfg[key])

    simcse = SimCSEEncoder(os.path.abspath(cfg['local_pretrained_dir']))
    tokenizer = simcse.tokenizer
    scaler = GradScaler(device='cuda', enabled=torch.cuda.is_available())
    print('已启用混合精度训练。' if scaler.is_enabled() else '使用FP32训练。')

    def _nrows(size):
        return None if size <= 0 else size
    s1_train, s2_train = DataManager.load_both(cfg['train_data_file'], nrows=_nrows(cfg['stage1_train_size']))
    s1_valid, s2_valid = DataManager.load_both(cfg['valid_data_file'], nrows=_nrows(cfg['stage1_valid_size']))
    s1_test, s2_test = DataManager.load_both(cfg['test_data_file'], nrows=_nrows(cfg['stage1_test_size']))
    s1_train_loader = DataLoader(s1_train, batch_size=cfg['stage1_batch_size'], shuffle=True)
    s1_valid_loader = DataLoader(s1_valid, batch_size=cfg['stage1_batch_size'], shuffle=False)
    s1_test_loader = DataLoader(s1_test, batch_size=cfg['stage1_batch_size'], shuffle=False)
    s2_train_loader = DataLoader(s2_train, batch_size=cfg['stage2_batch_size'], shuffle=True)
    s2_valid_loader = DataLoader(s2_valid, batch_size=cfg['stage2_batch_size'], shuffle=False)
    s2_test_loader = DataLoader(s2_test, batch_size=cfg['stage2_batch_size'], shuffle=False)

    # 阶段1：无监督 SimCSE — 单个编码器
    print('===== 阶段1：无监督 SimCSE =====')
    opt1 = AdamW(simcse.model.parameters(), lr=cfg['stage1_learning_rate'], weight_decay=0.01)
    total1 = len(s1_train_loader) * cfg['stage1_epochs']
    sch1 = get_linear_schedule_with_warmup(opt1, num_warmup_steps=int(total1 * 0.05), num_training_steps=total1)
    for epoch in range(cfg['stage1_epochs']):
        train_loss = train_or_eval_stage1(
            simcse, tokenizer, s1_train_loader, cfg['max_length'], scaler,
            optimizer=opt1, scheduler=sch1
        )
        valid_loss = train_or_eval_stage1(
            simcse, tokenizer, s1_valid_loader, cfg['max_length'], scaler,
        )
        test_loss = train_or_eval_stage1(
            simcse, tokenizer, s1_test_loader, cfg['max_length'], scaler,
        )
        print(f'Stage1 E{epoch + 1}/{cfg["stage1_epochs"]}  Train: {train_loss:.4f}  Val: {valid_loss:.4f}  Test: {test_loss:.4f}')

    # 阶段2：有监督匹配（双塔独立训练）
    print('===== 阶段2：有监督匹配（双塔）=====')
    # 克隆阶段1训练好的权重作为 response_encoder 的起点
    resp_enc = copy.deepcopy(simcse)
    opt2 = AdamW(
        list(simcse.model.parameters()) + list(resp_enc.model.parameters()),
        lr=cfg['stage2_learning_rate'], weight_decay=0.01,
    )
    total2 = len(s2_train_loader) * cfg['stage2_epochs']
    sch2 = get_linear_schedule_with_warmup(opt2, num_warmup_steps=int(total2 * 0.05), num_training_steps=total2)
    for epoch in range(cfg['stage2_epochs']):
        train_loss = train_or_eval_stage2(
            simcse, resp_enc, tokenizer, s2_train_loader, cfg['max_length'], scaler,
            optimizer=opt2, scheduler=sch2
        )
        valid_loss = train_or_eval_stage2(
            simcse, resp_enc, tokenizer, s2_valid_loader, cfg['max_length'], scaler,
        )
        test_loss = train_or_eval_stage2(
            simcse, resp_enc, tokenizer, s2_test_loader, cfg['max_length'], scaler,
        )
        print(f'Stage2 E{epoch + 1}/{cfg["stage2_epochs"]}  Train: {train_loss:.4f}  Val: {valid_loss:.4f}  Test: {test_loss:.4f}')

    print(f'保存模型到: {cfg["output_dir"]}')
    os.makedirs(cfg['output_dir'], exist_ok=True)
    for sub in ('query_encoder', 'response_encoder'):
        os.makedirs(os.path.join(cfg['output_dir'], sub), exist_ok=True)
    simcse.model.save_pretrained(os.path.join(cfg['output_dir'], 'query_encoder'))
    resp_enc.model.save_pretrained(os.path.join(cfg['output_dir'], 'response_encoder'))
    tokenizer.save_pretrained(cfg['output_dir'])
    print('训练完成，模型已保存。')

if __name__ == '__main__':
    train_model()