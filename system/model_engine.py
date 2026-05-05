"""
system/model_engine.py

文件作用:
    SimCSE 双塔编码引擎。
    负责文本分词、前向推理、池化与向量归一化。
"""

import os
import math
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from contextlib import nullcontext
from system.config import SIMCSE_MODEL_DIR

MAX_LENGTH = 128

class SimCSEModelEngine:
    """
    语义匹配引擎 (Embedding Engine)。

    核心职责:
        加载预训练的双塔模型 (SimCSE Dual-Encoder)，提供"文本 → 语义向量"的转换服务。

    两个编码器:
        query_encoder:    编码用户输入的问句
        response_encoder: 编码知识库中的答句

    使用方式:
        - 批量编码: engine.encode(texts, encoder='query', batch_size=32)
        - 单条编码: engine.encode_one(text, encoder='query', return_numpy=True)
    """

    def __init__(self, model_dir: str = SIMCSE_MODEL_DIR):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        q_dir = os.path.join(self.model_dir, "query_encoder")
        r_dir = os.path.join(self.model_dir, "response_encoder")
        # 硬件加速检测：如果电脑有英伟达显卡且驱动正常，将自动切换至 CUDA 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将两个编码模型部署至指定计算设备 (CUDA 或 CPU)
        self.query_encoder = AutoModel.from_pretrained(q_dir, local_files_only=True).to(self.device)
        self.response_encoder = AutoModel.from_pretrained(r_dir, local_files_only=True).to(self.device)
        # 设为 Eval 模式：推理时禁用 Dropout 以保证结果在此时是确定的
        self.query_encoder.eval()
        self.response_encoder.eval()
        # 如果启用显卡，会自动激活 CUDA 算子优化机制以最大程度提升编码每秒处理条数
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            # Benchmark 会自动找到最适合当前显卡架构的卷积算法
            torch.backends.cudnn.benchmark = True
            # TF32 模式能在极微小精度损失的情况下大幅度提升矩阵乘法速度
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                # 设置全局矩阵乘法精度为 'high' 以在现代 NVIDIA GPU 上获得平衡
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

        # 说明：
        # 1. 本引擎仅负责“文本 -> 向量”，不负责检索与重排。
        # 2. encode / encode_one 默认输出已做 L2 归一化向量。
        self.hidden_size = int(self.query_encoder.config.hidden_size)

    def _l2_normalize(self, embeddings, eps=1e-8):
        """
        L2 范数长度归一化 (Unit Sphere Projection)
        将所有向量投影到单位超球面上。在此基础上，点积运算在数学逻辑上等同于余弦相似度计算。
        
        参数:
            embeddings: 编码后的浮点张量。
            eps: 稳定性修正值，防止输入全为 0 时发生除以零错误。
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
            
        embeddings = embeddings.float()
        # 数据鲁棒性保障：过滤掉可能出现的无效数值 (NaN/Inf)
        embeddings = self._sanitize_tensor_values(embeddings)

        # 归一化核心逻辑：向量 = 向量 / L2长度
        if embeddings.ndim == 1:
            norm = self._safe_l2_norm(embeddings, dim=None, keepdim=False)
            return embeddings / torch.clamp(norm, min=eps)

        norms = self._safe_l2_norm(embeddings, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=eps)

    @staticmethod
    def _sanitize_tensor_values(tensor: torch.Tensor) -> torch.Tensor:
        """手动清洗异常值

        规则：
        1. NaN 替换为 0
        2. +Inf 替换为 1
        3. -Inf 替换为 -1
        """
        finite_mask = torch.isfinite(tensor)
        cleaned = torch.where(finite_mask, tensor, torch.zeros_like(tensor))
        pos_inf_mask = torch.isinf(tensor) & (tensor > 0)
        neg_inf_mask = torch.isinf(tensor) & (tensor < 0)
        cleaned = torch.where(pos_inf_mask, torch.ones_like(cleaned), cleaned)
        cleaned = torch.where(neg_inf_mask, -torch.ones_like(cleaned), cleaned)
        return cleaned

    @staticmethod
    def _safe_l2_norm(tensor: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        """L2 范数：sqrt(sum(x^2))"""
        squared = tensor * tensor
        summed = squared.sum() if dim is None else squared.sum(dim=dim, keepdim=keepdim)
        return torch.sqrt(summed)

    def _sentence_pooling(self, outputs):
        """取出序列第一个位置（[CLS]）的特征向量作为整句语义表示。"""
        return outputs.last_hidden_state[:, 0]

    @staticmethod
    def _batch_slices(total_size: int, batch_size: int):
        """生成批次切片区间。

        例如 total=10, batch=4 -> (0,4), (4,8), (8,10)
        """
        start = 0
        while start < total_size:
            end = min(start + batch_size, total_size)
            yield start, end
            start = end

    def _prepare_inputs(self, batch_texts):
        """完成文本到模型输入张量的准备流程。

        这里保留对 tokenizer 的必要调用（底层词表和切分规则由预训练模型定义），
        但其余步骤（设备迁移、非阻塞传输等）由本类统一处理。
        """
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            pad_to_multiple_of=8 if self.use_amp else None,
        )
        return {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

    def encode(
        self,
        texts,
        batch_size=32,
        encoder: str = "query",
        return_numpy: bool = False,
        show_progress: bool = True,
        normalize: bool = True,
    ):
        """
        批量将文本转换为语义向量。
        
        参数:
            texts: 待转换的文本列表。
            batch_size: 每次处理的样本数量。
            encoder: 指定使用的模型类型，可选 'query' (问句) 或 'response' (答句)。
            return_numpy: 是否将结果转换为通用的数值数组返回。
            show_progress: 是否显示进度条。
            normalize: 是否进行 L2 归一化，默认 True。

        返回:
            - return_numpy=False: torch.Tensor，形状约为 [N, hidden_size]
            - return_numpy=True: np.ndarray，形状约为 [N, hidden_size]

        流程:
            1. 选择编码器（query/response）
            2. 按 batch 切片
            3. tokenizer 构造输入并迁移设备
            4. 前向推理并池化
            5. 可选 L2 归一化
            6. 汇总后按 return_numpy 决定返回格式
        """
        if isinstance(texts, str):
            texts = [texts]
        if encoder not in ("query", "response"):
            raise ValueError("编码器参数只能是 'query' 或 'response'")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        model = self.query_encoder if encoder == "query" else self.response_encoder
        all_embeddings = []

        total_batches = math.ceil(len(texts) / batch_size) if len(texts) > 0 else 0
        progress_bar = tqdm(
            total=total_batches,
            desc=f"语义提取中({encoder})",
            unit="批次",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )

        with torch.inference_mode():
            # 启用显卡自动加速（如果适用）
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_ctx:
                try:
                    for start, end in self._batch_slices(len(texts), batch_size):
                        batch_texts = texts[start:end]
                        inputs = self._prepare_inputs(batch_texts)
                        outputs = model(**inputs)
                        embeddings = self._sentence_pooling(outputs)
                        if normalize:
                            embeddings = self._l2_normalize(embeddings)
                        all_embeddings.append(embeddings.cpu())
                        progress_bar.update(1)
                finally:
                    progress_bar.close()
        # 空输入时返回 [0, hidden_size]，避免下游维度推断出错。
        merged = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, self.hidden_size), dtype=torch.float32)
        return merged.numpy() if return_numpy else merged

    def encode_one(self, text, encoder: str = "query", return_numpy: bool = False, normalize: bool = True):
        """单条文本编码入口。

        说明：
            在线检索场景频繁调用本方法，这里走单条快速路径，
            避免 encode([text]) 的额外列表拼接与合并开销。
        """
        if encoder not in ("query", "response"):
            raise ValueError("编码器参数只能是 'query' 或 'response'")

        model = self.query_encoder if encoder == "query" else self.response_encoder
        safe_text = "" if text is None else str(text)

        with torch.inference_mode():
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_ctx:
                inputs = self._prepare_inputs([safe_text])
                outputs = model(**inputs)
                emb = self._sentence_pooling(outputs)
                if normalize:
                    emb = self._l2_normalize(emb)

        emb_1d = emb[0].detach().cpu().float()
        if return_numpy:
            return emb_1d.numpy().reshape(-1)
        return emb_1d.reshape(-1)