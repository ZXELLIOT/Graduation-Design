"""
train/train.py

文件作用:
    双塔检索模型训练主脚本。

训练策略（两阶段微调 Two-Stage Fine-Tuning）:
    阶段1 — 无监督 SimCSE:
        仅使用问句（query），通过 BERT 内置 Dropout 对同一句子两次前向传播
        产生两个略有差异的向量作为正样本对，让模型学会区分不同句子的语义。
        目标：将通用预训练模型转化为能理解对话语义倾向的基础编码器。

    阶段2 — 有监督匹配:
        使用 (问句, 正样本回复, 负样本回复) 三元组，通过对比损失训练模型
        让问句与正确回复的相似度高于错误回复。
        目标：让模型具备精确区分正确/错误回复的判别力。

其他特性:
    - CUDA OOM 自动降 batch size 重试
    - 混合精度 (FP16) + torch.compile 加速
    - 断点可复现的随机种子与数据加载
"""

import os
import argparse
import logging
import multiprocessing as mp
import sys
import gc
from contextlib import nullcontext

# 避免 tokenizer 在主进程预热后被 DataLoader worker fork 时反复告警
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# 降低显存碎片带来的 OOM 概率
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# 关闭 transformers/huggingface 内部进度条，统一使用本项目中文进度条
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 默认在线模型标识
DEFAULT_ONLINE_MODEL_ID = 'shibing624/text2vec-base-chinese'

# 默认本地预训练模型绝对路径（服务器可提前下载到该位置）
DEFAULT_LOCAL_PRETRAINED_DIR = os.path.abspath(LOCAL_PRETRAINED_DIR)


def _resolve_project_path(path_str: str) -> str:
    """将相对路径解析为项目根目录下的绝对路径。"""
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))


def _choose_pretrained_model(model_name_or_path: str, local_pretrained_dir: str):
    """
    预训练模型加载策略：
    1. 若显式传入本地目录（相对/绝对）且存在，优先使用本地；
    2. 否则检查 local_pretrained_dir（绝对路径）是否存在；
    3. 若都不存在，则回退到在线模型下载。
    """
    model_arg = (model_name_or_path or '').strip()

    # 显式传入本地路径优先
    if model_arg:
        model_arg_abs = _resolve_project_path(model_arg)
        if os.path.isdir(model_arg_abs):
            return model_arg_abs, True

    # 默认本地目录检查
    local_abs = os.path.abspath(local_pretrained_dir)
    if os.path.isdir(local_abs):
        return local_abs, True

    # 本地不存在，使用在线模型
    if model_arg and not os.path.isdir(_resolve_project_path(model_arg)):
        return model_arg, False
    return DEFAULT_ONLINE_MODEL_ID, False


def log_device_once():
    """打印设备信息。"""
    if mp.current_process().name == 'MainProcess':
        logger.info(f'使用设备: {DEVICE}')


def configure_cuda_performance():
    """启用 CUDA 高吞吐配置，优化 GPU 训练效率。"""
    if not torch.cuda.is_available():
        return
    # 启用 cudnn benchmark，让编译器针对当前硬件寻找最优算法
    torch.backends.cudnn.benchmark = True
    # 允许在支持的 GPU 上使用 TF32 加速矩阵乘法，牺牲微小精度换取大幅性能提升
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        # 设置 float32 矩阵乘法的精度等级
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def autocast_context(scaler):
    """根据是否启用混合精度返回上下文管理器。
    
    使用混合精度（FP16）可以显著减少显存占用并加快计算速度。
    """
    if scaler is None or not torch.cuda.is_available() or not scaler.is_enabled():
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.float16)


def _tokenize_to_device(tokenizer, texts, max_length):
    """将文本批量分词并转移到指定的计算设备（GPU/CPU）。"""
    inputs = tokenizer(
        texts,
        padding=True,          # 自动补全到 batch 内最长长度
        truncation=True,       # 超出长度限制则截断
        max_length=max_length,
        return_tensors='pt',   # 返回 PyTorch 张量
    )
    # 使用 non_blocking=True 可以稍微提速 host 到 device 的数据传输
    return {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}


def _require_file_exists(path, label):
    """检查文件是否存在。

    参数:
        path: 文件路径。
        label: 文件标签，用于报错提示。
    返回:
        无返回值。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label}文件缺失: {path}')


def _require_min_size(actual_size, expected_size, label):
    """检查样本数量是否达到最小要求。

    参数:
        actual_size: 实际样本数量。
        expected_size: 期望样本数量。
        label: 数据集标签。
    返回:
        无返回值。
    """
    if actual_size < expected_size:
        raise ValueError(f'{label}样本不足，期望至少 {expected_size}，实际 {actual_size}')


def _to_nrows(size_value: int):
    """将样本规模参数转换为 pandas 的 nrows：<=0 表示读取全量。"""
    if size_value is None:
        return None
    return None if int(size_value) <= 0 else int(size_value)


def _apply_quick_cap(size_value: int, cap: int) -> int:
    """quick 模式下统一收敛样本规模。"""
    if size_value is None or int(size_value) <= 0:
        return cap
    return min(int(size_value), cap)


# ============================================================
# 数据集类 (Dataset)
# ============================================================


class PositivePairDataset(Dataset):
    """正样本对数据集：每条数据为 (query, response)。"""

    def __init__(self, queries, responses):
        """初始化正样本对数据集。

        参数:
            queries: 问句列表。
            responses: 回复列表。
        返回:
            无返回值。
        """
        if len(queries) != len(responses):
            raise ValueError('正样本问答数量不一致。')
        self.queries = queries
        self.responses = responses

    def __len__(self):
        """返回数据集大小。"""
        return len(self.queries)

    def __getitem__(self, idx):
        """按索引获取单条样本。

        参数:
            idx: 样本索引。
        返回:
            (query, response) 元组。
        """
        return self.queries[idx], self.responses[idx]


class QueryOnlyDataset(Dataset):
    """仅问句数据集：query。"""

    def __init__(self, queries):
        """初始化问句数据集。

        参数:
            queries: 问句列表。
        返回:
            无返回值。
        """
        self.queries = queries

    def __len__(self):
        """返回数据集大小。"""
        return len(self.queries)

    def __getitem__(self, idx):
        """按索引获取单条问句。

        参数:
            idx: 样本索引。
        返回:
            单条问句字符串。
        """
        return self.queries[idx]


class PositiveNegativeDataset(Dataset):
    """正负样本数据集：query, response, negative_response。"""

    def __init__(self, queries, responses, negative_responses):
        """初始化正负样本数据集。

        参数:
            queries: 问句列表。
            responses: 正样本回复列表。
            negative_responses: 负样本回复列表。
        返回:
            无返回值。
        """
        if len(queries) != len(responses) or len(queries) != len(negative_responses):
            raise ValueError('问句、正样本、负样本数量不一致。')
        self.queries = queries
        self.responses = responses
        self.negative_responses = negative_responses

    def __len__(self):
        """返回数据集大小。"""
        return len(self.queries)

    def __getitem__(self, idx):
        """按索引获取单条三元组样本。

        参数:
            idx: 样本索引。
        返回:
            (query, positive, negative) 元组。
        """
        return self.queries[idx], self.responses[idx], self.negative_responses[idx]


class TrainDataLoader:
    """训练脚本内部专用数据加载器，负责从 CSV 中读取特定列并清理数据。"""

    @staticmethod
    def load_queries(csv_path, nrows=None):
        """加载问句列表，主要用于第一阶段无监督训练。"""
        _require_file_exists(csv_path, '问句CSV')
        progress = tqdm(total=2, desc='正在加载问句数据', unit='步', leave=False, dynamic_ncols=True)
        try:
            df = pd.read_csv(csv_path, nrows=nrows)
            progress.update(1)
            if 'query' not in df.columns:
                raise ValueError(f'CSV列缺失: {csv_path}，需要列: query')
            # 处理空值，并转换为字符串，确保 tokenizer 不报错
            df = df[['query']].fillna('')
            progress.update(1)
            return df['query'].astype(str).tolist()
        finally:
            progress.close()

    @staticmethod
    def load_positive_pairs(csv_path, nrows=None):
        """加载 (问句, 回复) 对，用于基础的正样本对齐。"""
        _require_file_exists(csv_path, '正样本CSV')
        progress = tqdm(total=2, desc='正在加载正样本对', unit='步', leave=False, dynamic_ncols=True)
        try:
            df = pd.read_csv(csv_path, nrows=nrows)
            progress.update(1)
            required_cols = {'query', 'response'}
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f'正样本CSV列缺失: {csv_path}，需要列: {required_cols}')
            df = df[['query', 'response']].fillna('')
            progress.update(1)
            return df['query'].astype(str).tolist(), df['response'].astype(str).tolist()
        finally:
            progress.close()

    @staticmethod
    def load_positive_negative_triplets(csv_path, nrows=None):
        """加载 (问句, 正向回复, 负向回复) 三元组，用于第二阶段硬负样本训练。"""
        _require_file_exists(csv_path, '正负样本CSV')
        progress = tqdm(total=2, desc='正在加载正负样本三元组', unit='步', leave=False, dynamic_ncols=True)
        try:
            df = pd.read_csv(csv_path, nrows=nrows)
            progress.update(1)
            required_cols = {'query', 'response', 'negative_response'}
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f'正负样本CSV列缺失: {csv_path}，需要列: {required_cols}')
            df = df[['query', 'response', 'negative_response']].fillna('')
            progress.update(1)
            return (
                df['query'].astype(str).tolist(),
                df['response'].astype(str).tolist(),
                df['negative_response'].astype(str).tolist(),
            )
        finally:
            progress.close()


# ============================================================
# 双塔模型 (Dual Encoder)
# ============================================================


class DualEncoderModel(nn.Module):
    """
    双编码器模型（Dual Encoder / Two-Tower Model）。

    核心思想:
        问句和回复各自通过独立的编码器转换为向量，计算相似度时使用向量点积。

    两个编码器:
        - query_encoder:    对用户输入的问句编码
        - response_encoder: 对候选回复编码

    为什么用双塔而非单塔？
        单塔（如 BERT 做句对分类）需要在推理时把所有候选对逐对输入模型，
        速度太慢。双塔可以预先计算所有回复的向量存在 FAISS 里，推理时
        只需编码问句一次，在向量空间中做近似最近邻搜索，速度提升巨大。

    池化策略:
        cls:            取 [CLS] 位置向量（BERT 标准做法，单向量代表整句）
        mean:           对所有 token 向量取平均（更平滑，但对齐性略弱）
    """

    def __init__(self, model_name_or_path, pooling_strategy='cls', temperature=0.05, local_files_only=False):
        """初始化双塔模型。

        参数:
            model_name_or_path: 模型目录或模型名称。
            pooling_strategy: 池化策略，支持 cls/mean。
            temperature: 对比学习温度参数。
            local_files_only: 是否仅从本地加载模型。
        返回:
            无返回值。
        """
        super().__init__()
        # 加载两个独立的编码器分支：问句编码器和回复编码器
        q_dir = os.path.join(model_name_or_path, 'query_encoder')
        r_dir = os.path.join(model_name_or_path, 'response_encoder')

        load_bar = tqdm(total=2, desc='正在加载双塔编码器', unit='个', leave=False, dynamic_ncols=True)
        try:
            # 如果对应的目录存在则加载微调后的权重，否则加载预训练的基础模型
            if os.path.isdir(q_dir) and os.path.isdir(r_dir):
                self.query_encoder = AutoModel.from_pretrained(q_dir, local_files_only=True)
                load_bar.update(1)
                self.response_encoder = AutoModel.from_pretrained(r_dir, local_files_only=True)
                load_bar.update(1)
            else:
                self.query_encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
                load_bar.update(1)
                self.response_encoder = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
                load_bar.update(1)
        finally:
            load_bar.close()

        self.pooling_strategy = pooling_strategy # 池化策略：cls 或 mean
        self.temperature = temperature           # 温度参数，控制对比损失的分布平滑度差异

    def encode(self, encoder, input_ids, attention_mask):
        """通用编码方法：文本 -> 向量 -> L2 归一化。"""
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 选择池化方式提取句子级别的表征向量
        if self.pooling_strategy == 'cls':
            # 提取 [CLS] 位置的特征（BERT 类的标准做法）
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == 'mean':
            # 计算所有有效 token（排除 padding）的平均向量
            masked = outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f'不支持的池化方式: {self.pooling_strategy}')
            
        # 核心步骤：执行 L2 归一化，使得后续的点积计算可以直接代表余弦相似度
        return F.normalize(pooled, p=2, dim=1)

    def encode_query(self, input_ids, attention_mask):
        """问句方向的编码操作。"""
        return self.encode(self.query_encoder, input_ids, attention_mask)

    def encode_response(self, input_ids, attention_mask):
        """回复方向的编码操作。"""
        return self.encode(self.response_encoder, input_ids, attention_mask)


def _build_dataloader(dataset, batch_size, shuffle, num_workers=-1, prefetch_factor=4):
    """构建 DataLoader。

    参数:
        dataset: 数据集对象。
        batch_size: 批大小。
        shuffle: 是否打乱。
        num_workers: 工作进程数，<0 表示自动推断。
        prefetch_factor: 预取因子。
    返回:
        配置完成的 DataLoader。
    """
    pin_memory = DEVICE.type == 'cuda'
    # 自动获取最优线程数以拉满 CPU 效率
    if num_workers is None or num_workers < 0:
        cpu_count = os.cpu_count() or 4
        # Linux 服务器可用更高 worker 数；Windows 保守一些
        num_workers = min(12, max(cpu_count - 2, 4))

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'pin_memory': pin_memory,       # 开启内存锁定，加速数据从内存传输到 GPU
        'num_workers': num_workers,     # 开启多 CPU 核心并行预处理数据
        'persistent_workers': True if num_workers > 0 else False, # 训练期间保持进程不销毁
    }

    if num_workers > 0:
        # 预取因子：每个 worker 提前准备 batch 的数量
        loader_kwargs['prefetch_factor'] = max(2, int(prefetch_factor))

    return DataLoader(
        **loader_kwargs,
    )


def _is_cuda_oom_error(error: Exception) -> bool:
    """判断异常是否为 CUDA 显存不足。"""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    msg = str(error).lower()
    return 'out of memory' in msg and 'cuda' in msg


def _cleanup_after_oom():
    """在 OOM 后清理缓存，避免脏状态影响下一次重试。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _batch_size_candidates(initial_batch_size: int, min_batch_size: int, backoff_ratio: float = 0.95):
    """生成从大到小的 batch 重试序列。

    默认使用平滑比例回退（如 0.95），避免一次性减半过于激进。
    """
    initial = max(1, int(initial_batch_size))
    minimum = max(1, int(min_batch_size))
    ratio = float(backoff_ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        ratio = 0.95
    sizes = []
    current = initial
    while current >= minimum:
        if current not in sizes:
            sizes.append(current)
        # 至少下降 1，且默认按比例平滑回退，减少过度降档。
        reduced = int(current * ratio)
        next_size = max(minimum, min(current - 1, reduced))
        if next_size == current:
            break
        current = next_size
    if minimum not in sizes:
        sizes.append(minimum)
    return sizes


def _run_stage_with_auto_batch(
    stage_name,
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    test_dataset,
    init_batch_size,
    min_batch_size,
    batch_backoff_ratio,
    num_workers,
    prefetch_factor,
    epochs,
    learning_rate,
    max_length,
    scaler,
    train_eval_fn,
):
    """按批大小逐级降档重试，直到训练成功或达到最小 batch。"""
    retry_sizes = _batch_size_candidates(init_batch_size, min_batch_size, backoff_ratio=batch_backoff_ratio)
    last_error = None

    retry_iterator = tqdm(retry_sizes, desc=f'{stage_name} 批大小重试', unit='档', dynamic_ncols=True, leave=False)
    for batch_size in retry_iterator:
        logger.info(f'{stage_name} 尝试 batch_size={batch_size}')
        train_loader = _build_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        valid_loader = _build_dataloader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        test_loader = _build_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        try:
            _run_stage(
                stage_name=stage_name,
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                max_length=max_length,
                scaler=scaler,
                train_eval_fn=train_eval_fn,
            )
            test_loss = train_eval_fn(
                model=model,
                tokenizer=tokenizer,
                dataloader=test_loader,
                max_length=max_length,
                scaler=scaler,
                optimizer=None,
                scheduler=None,
                progress_desc=f'{stage_name} 测试',
            )
            logger.info(f'{stage_name} Test Loss: {test_loss:.4f}')
            return batch_size, test_loss
        except Exception as e:
            if not _is_cuda_oom_error(e):
                raise
            last_error = e
            logger.warning(f'{stage_name} 在 batch_size={batch_size} 时发生显存不足，自动降档重试。')
            _cleanup_after_oom()

    raise RuntimeError(
        f'{stage_name} 在最小 batch_size={min_batch_size} 下仍显存不足，请降低 max_length 或关闭 compile。'
    ) from last_error


# ============================================================
# 阶段1：无监督 SimCSE 训练
# ============================================================


def train_or_eval_stage1(
    model,
    tokenizer,
    dataloader,
    max_length,
    scaler,
    optimizer=None,
    scheduler=None,
    progress_desc=None,
):
    """阶段1：执行无监督 SimCSE 训练，构建基础语义空间。
    
    该阶段仅利用问句，通过 BERT 模型内部自带的 Dropout 随机丢弃部分神经元。
    两次输入同个句子，模型会得到两个略有差异的向量（即正样本对），目的是让
    同一个句子的不同视角尽可能接近，不同句子尽可能阔开。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    batch_count = 0

    iterator = dataloader
    progress = None
    if progress_desc is not None:
        progress = tqdm(dataloader, desc=progress_desc, unit='批', dynamic_ncols=True, leave=False)
        iterator = progress

    context_manager = nullcontext() if is_train else torch.no_grad()
    with context_manager:
        for texts_q in iterator:
            # 文本分词编码
            query_inputs = _tokenize_to_device(tokenizer, texts_q, max_length)

            with autocast_context(scaler):
                # 关键：同一批问句前向传播两次，由于内置 dropout 的存在，得到两个视角差异化的向量
                query_emb_1 = model.encode_query(query_inputs['input_ids'], query_inputs['attention_mask'])
                query_emb_2 = model.encode_query(query_inputs['input_ids'], query_inputs['attention_mask'])
                
                # 计算相似度矩阵并除以温度系数，增强损失函数的辨识度
                sim_matrix = torch.matmul(query_emb_1, query_emb_2.T) / model.temperature
                
                # 对角线上的元素互为正样本对，标签即为其索引位置
                labels = torch.arange(sim_matrix.size(0), device=DEVICE)
                # 交叉熵损失函数引导模型学习对比特征
                loss = F.cross_entropy(sim_matrix, labels)

            # 更新梯度
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    # 使用梯度缩放（混合精度）加速计算
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    # 按照学习率计划进行衰减调优
                    scheduler.step()

            total_loss += loss.item()
            batch_count += 1

            if progress is not None and is_train:
                progress.set_postfix({'loss': f'{loss.item():.4f}'})

    if progress is not None:
        progress.close()

    if batch_count == 0:
        return float('inf')
    return total_loss / batch_count


# ============================================================
# 阶段2：有监督匹配训练（三元组对比）
# ============================================================


def train_or_eval_stage2(
    model,
    tokenizer,
    dataloader,
    max_length,
    scaler,
    optimizer=None,
    scheduler=None,
    progress_desc=None,
):
    """阶段2：执行有监督对比学习（三元组/多列匹配训练）。
    
    使用 (问句, 正向回复, 负向回复) 构造训练样本。
    这会教导模型：问句与真实的回复应该更接近，点积相似度更高；而与错误的回复
    相似度应当更低。这是问答匹配系统最为核心的微调阶段。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    batch_count = 0

    iterator = dataloader
    progress = None
    if progress_desc is not None:
        progress = tqdm(dataloader, desc=progress_desc, unit='批', dynamic_ncols=True, leave=False)
        iterator = progress

    context_manager = nullcontext() if is_train else torch.no_grad()
    with context_manager:
        for texts_q, texts_pos, texts_neg in iterator:
            # 环境准备：三列数据分词，准备送向 GPU 计算
            query_inputs = _tokenize_to_device(tokenizer, texts_q, max_length)
            pos_inputs = _tokenize_to_device(tokenizer, texts_pos, max_length)
            neg_inputs = _tokenize_to_device(tokenizer, texts_neg, max_length)

            with autocast_context(scaler):
                # 问句、正向回复、负向回复分别通过对应的编码器生成向量
                q_emb = model.encode_query(query_inputs['input_ids'], query_inputs['attention_mask'])
                pos_emb = model.encode_response(pos_inputs['input_ids'], pos_inputs['attention_mask'])
                neg_emb = model.encode_response(neg_inputs['input_ids'], neg_inputs['attention_mask'])

                # 在 batch 层面计算问句与两者的相似度分值
                pos_sim = torch.sum(q_emb * pos_emb, dim=1)
                neg_sim = torch.sum(q_emb * neg_emb, dim=1)
                
                # 构建二分类或对比损失逻辑（逻辑上让 pos_sim 远大于 neg_sim）
                logits = torch.stack([pos_sim, neg_sim], dim=1) / model.temperature
                # 目标是正确匹配总是位于索引 0，即训练模型识别正向概率
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)

            # 更新权重参数
            if is_train:
                optimizer.zero_grad(set_to_none=True)
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
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            batch_count += 1

            if progress is not None and is_train:
                progress.set_postfix({'loss': f'{loss.item():.4f}'})

    if progress is not None:
        progress.close()

    if batch_count == 0:
        return float('inf')
    return total_loss / batch_count


def _run_stage(
    stage_name,
    model,
    tokenizer,
    train_dataloader,
    valid_dataloader,
    epochs,
    learning_rate,
    max_length,
    scaler,
    train_eval_fn,
):
    """
    执行单个训练阶段（含训练和验证）。

    训练组件:
        - 优化器: AdamW（带 weight decay 的 Adam，防止过拟合）
        - 调度器: 线性预热 + 线性衰减（前 5% 步数从 0 线性增长到目标学习率）
        - 梯度裁剪: max_norm=1.0（防止梯度爆炸）
        tokenizer: 分词器。
        train_dataloader: 训练集加载器。
        valid_dataloader: 验证集加载器。
        epochs: 训练轮数。
        learning_rate: 学习率。
        max_length: 文本最大长度。
        scaler: 混合精度梯度缩放器。
        train_eval_fn: 具体阶段训练函数。
    返回:
        无返回值。
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_iterator = tqdm(range(epochs), desc=f'{stage_name} 训练轮次', unit='轮', dynamic_ncols=True, leave=False)
    for epoch in epoch_iterator:
        logger.info(f'{stage_name} - 开始 Epoch {epoch + 1}/{epochs}')
        avg_train_loss = train_eval_fn(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_dataloader,
            max_length=max_length,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            progress_desc=f'{stage_name} - Epoch {epoch + 1}/{epochs} 训练',
        )
        valid_loss = train_eval_fn(
            model=model,
            tokenizer=tokenizer,
            dataloader=valid_dataloader,
            max_length=max_length,
            scaler=scaler,
            optimizer=None,
            scheduler=None,
            progress_desc=f'{stage_name} - Epoch {epoch + 1}/{epochs} 验证',
        )
        logger.info(f'{stage_name} - Epoch {epoch + 1} 完成，Train Loss: {avg_train_loss:.4f}，Val Loss: {valid_loss:.4f}')


# ============================================================
# 训练主流程入口
# ============================================================


def train_model():
    """
    两阶段微调全流程入口。

    阶段1（无监督 SimCSE）:
        仅用 query 列，同句两次前向产生正样本对 → 构建基础语义空间。
        损失函数: 批内交叉熵（同一句子的两次视角互为正样本）。

    阶段2（有监督匹配）:
        使用 (query, response, negative_response) 三元组 →
        训练模型区分正确/错误回复的能力。
        损失函数: 二分类交叉熵（正样本 logit vs 负样本 logit）。

    输出:
        model/mysimcse/query_encoder/   — 问句编码器
        model/mysimcse/response_encoder/ — 答句编码器
        model/mysimcse/tokenizer 文件    — 分词器
    """
    # 默认超参数配置
    defaults = {
        'model_name_or_path': None,                              # 预训练模型路径或名称，留空则自动选择
        'local_pretrained_dir': LOCAL_PRETRAINED_DIR,            # 本地预训练模型绝对路径
        'train_data_file': TRAIN_NEG_CSV_PATH,                   # 训练集路径
        'valid_data_file': VALID_NEG_CSV_PATH,                   # 验证集路径
        'test_data_file': TEST_NEG_CSV_PATH,                     # 测试集路径
        'output_dir': OUTPUT_MODEL_DIR,                          # 模型保存目录
        'temperature': 0.05,                                     # 对比损失温度
        'max_length': 64,                                        # 最大文本截断长度
        'stage1_train_size': 0,                             
        'stage1_valid_size': 0,
        'stage1_test_size': 0,
        'stage2_train_size': 0,                                  
        'stage2_valid_size': 0,
        'stage2_test_size': 0,
        'stage1_epochs': 2,                                      # 阶段1迭代次数
        'stage1_batch_size': 512,                             
        'stage1_learning_rate': 3e-5,
        'stage2_epochs': 4,                                      # 阶段2通常需要更充分的迭代
        'stage2_batch_size': 512,                              
        'stage2_learning_rate': 3e-5,
        'num_workers': -1,                                       # 自动按机器 CPU 核数压榨并行
        'prefetch_factor': 16,                                  
        'batch_backoff_ratio': 0.95,                             # OOM 后 batch 按比例平滑回退
        'use_fp16': True,                                        # 是否使用自动混合精度加速
        'use_compile': True,                                     # 是否启用 torch.compile
    }

    parser = argparse.ArgumentParser(description='SimCSE 两阶段对比学习微调脚本：统一语义空间与匹配任务。')
    parser.add_argument('--model_name_or_path', type=str, default=defaults['model_name_or_path'], help='预训练模型路径或名称（可留空自动选择）')
    parser.add_argument('--local_pretrained_dir', type=str, default=defaults['local_pretrained_dir'], help='本地预训练模型绝对路径（存在则优先使用）')
    parser.add_argument('--train_data_file', type=str, default=defaults['train_data_file'], help='训练集CSV路径')
    parser.add_argument('--valid_data_file', type=str, default=defaults['valid_data_file'], help='验证集CSV路径')
    parser.add_argument('--test_data_file', type=str, default=defaults['test_data_file'], help='测试集CSV路径')
    parser.add_argument('--output_dir', type=str, default=defaults['output_dir'], help='训练后模型保存的根目录')
    parser.add_argument('--temperature', type=float, default=defaults['temperature'], help='对比学习温度')
    parser.add_argument('--max_length', type=int, default=defaults['max_length'], help='文本最大截断长度')
    parser.add_argument('--stage1_train_size', type=int, default=defaults['stage1_train_size'], help='阶段1训练样本数（<=0 表示全量）')
    parser.add_argument('--stage1_valid_size', type=int, default=defaults['stage1_valid_size'], help='阶段1验证样本数（<=0 表示全量）')
    parser.add_argument('--stage1_test_size', type=int, default=defaults['stage1_test_size'], help='阶段1测试样本数（<=0 表示全量）')
    parser.add_argument('--stage2_train_size', type=int, default=defaults['stage2_train_size'], help='阶段2训练样本数（<=0 表示全量）')
    parser.add_argument('--stage2_valid_size', type=int, default=defaults['stage2_valid_size'], help='阶段2验证样本数（<=0 表示全量）')
    parser.add_argument('--stage2_test_size', type=int, default=defaults['stage2_test_size'], help='阶段2测试样本数（<=0 表示全量）')
    parser.add_argument('--stage1_epochs', type=int, default=defaults['stage1_epochs'], help='阶段1训练轮数')
    parser.add_argument('--stage1_batch_size', type=int, default=defaults['stage1_batch_size'], help='阶段1 Batch Size')
    parser.add_argument('--stage1_learning_rate', type=float, default=defaults['stage1_learning_rate'], help='阶段1学习率')
    parser.add_argument('--stage2_epochs', type=int, default=defaults['stage2_epochs'], help='阶段2训练轮数')
    parser.add_argument('--stage2_batch_size', type=int, default=defaults['stage2_batch_size'], help='阶段2 Batch Size')
    parser.add_argument('--stage2_learning_rate', type=float, default=defaults['stage2_learning_rate'], help='阶段2学习率')
    parser.add_argument('--num_workers', type=int, default=defaults['num_workers'], help='数据加载并行数')
    parser.add_argument('--prefetch_factor', type=int, default=defaults['prefetch_factor'], help='Dataloader预取因子')
    parser.add_argument('--min_batch_size', type=int, default=32, help='OOM 自动降档时允许的最小 batch')
    parser.add_argument('--batch_backoff_ratio', type=float, default=defaults['batch_backoff_ratio'], help='OOM 后 batch 回退比例(0,1)，越接近1越平缓')
    parser.add_argument('--use_fp16', action=argparse.BooleanOptionalAction, default=defaults['use_fp16'], help='是否启用半精度(FP16)以节省显存')
    parser.add_argument('--use_compile', action=argparse.BooleanOptionalAction, default=defaults['use_compile'], help='是否启用 torch.compile 提升吞吐')
    parser.add_argument('--quick', action='store_true', help='快速模式：使用少量数据验证代码流程是否跑通')
  
    args = parser.parse_args()

    # 数据文件使用相对路径输入，运行时统一解析为绝对路径
    args.train_data_file = _resolve_project_path(args.train_data_file)
    args.valid_data_file = _resolve_project_path(args.valid_data_file)
    args.test_data_file = _resolve_project_path(args.test_data_file)

    # 输出目录强制为绝对路径
    args.output_dir = os.path.abspath(_resolve_project_path(args.output_dir))

    # 预训练模型选择：本地绝对路径优先，否则联网下载
    selected_model, local_files_only = _choose_pretrained_model(
        model_name_or_path=args.model_name_or_path,
        local_pretrained_dir=args.local_pretrained_dir,
    )

    if local_files_only:
        logger.info(f'检测到本地预训练模型，使用本地绝对路径: {selected_model}')
    else:
        logger.info(f'未检测到本地预训练模型，改为在线下载: {selected_model}')

    if args.quick:
        args.stage1_train_size = _apply_quick_cap(args.stage1_train_size, 20000)
        args.stage1_valid_size = _apply_quick_cap(args.stage1_valid_size, 500)
        args.stage1_test_size = _apply_quick_cap(args.stage1_test_size, 500)
        args.stage2_train_size = _apply_quick_cap(args.stage2_train_size, 20000)
        args.stage2_valid_size = _apply_quick_cap(args.stage2_valid_size, 500)
        args.stage2_test_size = _apply_quick_cap(args.stage2_test_size, 500)
        logger.info('已启用 quick 模式：仅缩小样本规模，保持高性能 batch 设置。')

    logger.info(
        '样本读取策略: '
        f's1(train/valid/test)={args.stage1_train_size}/{args.stage1_valid_size}/{args.stage1_test_size}, '
        f's2(train/valid/test)={args.stage2_train_size}/{args.stage2_valid_size}/{args.stage2_test_size} '
        '(<=0 表示读取全量)'
    )

    configure_cuda_performance()
    log_device_once()

    _require_file_exists(args.train_data_file, '训练CSV')
    _require_file_exists(args.valid_data_file, '验证CSV')
    _require_file_exists(args.test_data_file, '测试CSV')

    logger.info(f'加载模型与分词器: {selected_model}')
    model_load_bar = tqdm(total=2, desc='正在初始化训练模型', unit='步', dynamic_ncols=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_model, local_files_only=local_files_only)
        model_load_bar.update(1)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

        model = DualEncoderModel(
            model_name_or_path=selected_model,
            pooling_strategy='cls',
            temperature=args.temperature,
            local_files_only=local_files_only,
        ).to(DEVICE)
        model_load_bar.update(1)
    finally:
        model_load_bar.close()

    # 在 PyTorch 2.x + CUDA 环境下可显著提升吞吐
    if args.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune', fullgraph=False)
            logger.info('已启用 torch.compile(mode=max-autotune)。')
        except Exception as e:
            logger.warning(f'torch.compile 启用失败，回退为 eager 模式: {e}')

    logger.info('加载第一阶段数据（仅使用 query 列）...')
    stage1_data_bar = tqdm(total=3, desc='第一阶段数据加载', unit='份', dynamic_ncols=True)
    s1_train_nrows = _to_nrows(args.stage1_train_size)
    s1_valid_nrows = _to_nrows(args.stage1_valid_size)
    s1_test_nrows = _to_nrows(args.stage1_test_size)
    s1_train_q = TrainDataLoader.load_queries(args.train_data_file, nrows=s1_train_nrows)
    stage1_data_bar.update(1)
    s1_valid_q = TrainDataLoader.load_queries(args.valid_data_file, nrows=s1_valid_nrows)
    stage1_data_bar.update(1)
    s1_test_q = TrainDataLoader.load_queries(args.test_data_file, nrows=s1_test_nrows)
    stage1_data_bar.update(1)
    stage1_data_bar.close()
    if s1_train_nrows is not None:
        _require_min_size(len(s1_train_q), s1_train_nrows, '第一阶段训练集')
    if s1_valid_nrows is not None:
        _require_min_size(len(s1_valid_q), s1_valid_nrows, '第一阶段验证集')
    if s1_test_nrows is not None:
        _require_min_size(len(s1_test_q), s1_test_nrows, '第一阶段测试集')

    logger.info('加载第二阶段数据（使用三列 query/response/negative_response）...')
    stage2_data_bar = tqdm(total=3, desc='第二阶段数据加载', unit='份', dynamic_ncols=True)
    s2_train_nrows = _to_nrows(args.stage2_train_size)
    s2_valid_nrows = _to_nrows(args.stage2_valid_size)
    s2_test_nrows = _to_nrows(args.stage2_test_size)
    s2_train_q, s2_train_pos, s2_train_neg = TrainDataLoader.load_positive_negative_triplets(args.train_data_file, nrows=s2_train_nrows)
    stage2_data_bar.update(1)
    s2_valid_q, s2_valid_pos, s2_valid_neg = TrainDataLoader.load_positive_negative_triplets(args.valid_data_file, nrows=s2_valid_nrows)
    stage2_data_bar.update(1)
    s2_test_q, s2_test_pos, s2_test_neg = TrainDataLoader.load_positive_negative_triplets(args.test_data_file, nrows=s2_test_nrows)
    stage2_data_bar.update(1)
    stage2_data_bar.close()
    if s2_train_nrows is not None:
        _require_min_size(len(s2_train_q), s2_train_nrows, '第二阶段训练集')
    if s2_valid_nrows is not None:
        _require_min_size(len(s2_valid_q), s2_valid_nrows, '第二阶段验证集')
    if s2_test_nrows is not None:
        _require_min_size(len(s2_test_q), s2_test_nrows, '第二阶段测试集')

    logger.info(
        '样本规模: '
        f'阶段1 train/valid/test={len(s1_train_q)}/{len(s1_valid_q)}/{len(s1_test_q)}，'
        f'阶段2 train/valid/test={len(s2_train_q)}/{len(s2_valid_q)}/{len(s2_test_q)}'
    )

    scaler = GradScaler(device='cuda', enabled=args.use_fp16 and torch.cuda.is_available())
    logger.info('已启用混合精度训练。' if scaler.is_enabled() else '使用FP32训练。')

    s1_train_dataset = QueryOnlyDataset(s1_train_q)
    s1_valid_dataset = QueryOnlyDataset(s1_valid_q)
    s1_test_dataset = QueryOnlyDataset(s1_test_q)

    logger.info('第一阶段开始：问句无监督语义训练')
    s1_final_batch_size, _ = _run_stage_with_auto_batch(
        stage_name='Stage 1',
        model=model,
        tokenizer=tokenizer,
        train_dataset=s1_train_dataset,
        valid_dataset=s1_valid_dataset,
        test_dataset=s1_test_dataset,
        init_batch_size=args.stage1_batch_size,
        min_batch_size=args.min_batch_size,
        batch_backoff_ratio=args.batch_backoff_ratio,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=args.stage1_epochs,
        learning_rate=args.stage1_learning_rate,
        max_length=args.max_length,
        scaler=scaler,
        train_eval_fn=train_or_eval_stage1,
    )

    s2_train_dataset = PositiveNegativeDataset(s2_train_q, s2_train_pos, s2_train_neg)
    s2_valid_dataset = PositiveNegativeDataset(s2_valid_q, s2_valid_pos, s2_valid_neg)
    s2_test_dataset = PositiveNegativeDataset(s2_test_q, s2_test_pos, s2_test_neg)

    logger.info('第二阶段开始：正负样本匹配训练')
    s2_init_batch_size = max(args.min_batch_size, min(args.stage2_batch_size, s1_final_batch_size))
    _run_stage_with_auto_batch(
        stage_name='Stage 2',
        model=model,
        tokenizer=tokenizer,
        train_dataset=s2_train_dataset,
        valid_dataset=s2_valid_dataset,
        test_dataset=s2_test_dataset,
        init_batch_size=s2_init_batch_size,
        min_batch_size=args.min_batch_size,
        batch_backoff_ratio=args.batch_backoff_ratio,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=args.stage2_epochs,
        learning_rate=args.stage2_learning_rate,
        max_length=args.max_length,
        scaler=scaler,
        train_eval_fn=train_or_eval_stage2,
    )

    logger.info(f'保存模型到: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    query_dir = os.path.join(args.output_dir, 'query_encoder')
    response_dir = os.path.join(args.output_dir, 'response_encoder')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)

    save_bar = tqdm(total=3, desc='正在保存训练产物', unit='项', dynamic_ncols=True)
    model.query_encoder.save_pretrained(query_dir)
    save_bar.update(1)
    model.response_encoder.save_pretrained(response_dir)
    save_bar.update(1)
    tokenizer.save_pretrained(args.output_dir)
    save_bar.update(1)
    save_bar.close()
    logger.info('训练完成，模型已保存。')


if __name__ == '__main__':
    train_model()