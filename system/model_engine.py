import os
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from contextlib import nullcontext

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.config import SIMCSE_MODEL_DIR


MAX_LENGTH = 128
SENTENCE_POOLING = "cls"


class SimCSEModelEngine:
    """
    语义匹配引擎 (Embedding Engine)
    负责加载预训练的双塔模型 (SimCSE Dual-Encoder)，并提供从原始文本到底层语义向量的转换服务。
    """

    def __init__(self, model_dir: str = SIMCSE_MODEL_DIR, pooling: str = SENTENCE_POOLING):
        """
        初始化模型引擎。
        本类将同时加载问句编码器 (Query Encoder) 和答句编码器 (Response Encoder)。
        
        参数:
            model_dir: 本地模型文件夹的路径。
            pooling: 特征提取策略（默认 'cls'，即取 BERT 的首位特征，其具备全局代表性）。
        """
        self.model_dir = model_dir
        self.pooling = pooling
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
        # 2. encode / encode_one 默认输出已做 L2 归一化向量，

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

    def _masked_mean_pool(self, token_embeddings, attention_mask, eps=1e-8):
        """
        对所有字的特征进行加权平均。
        在提取整句特征时，只考虑实际存在的字词，忽略占位填充部分。
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        # 对序列长度维求和，得到每句“有效 token 特征和”
        summed = torch.sum(masked_embeddings, dim=1)
        # 统计每句有效 token 数，最小值截断到 eps 防止除零
        counts = torch.clamp(mask.sum(dim=1), min=eps)
        return summed / counts

    def _sentence_pooling(self, outputs, attention_mask):
        """
        从模型输出中提取整句的特征向量。
        
        参数:
            outputs: 模型的原始运算结果。
            attention_mask: 用于标识有效字符位置的掩码。
        """
        if self.pooling == "cls":
            # 取出序列第一个位置（通常代表整句含义）的特征
            return outputs.last_hidden_state[:, 0]
        if self.pooling == "mean":
            # 对所有有效位置的特征求平均
            return self._masked_mean_pool(outputs.last_hidden_state, attention_mask)
        if self.pooling == "first_last_avg":
            # 融合模型首层和末层的特征并求平均
            hidden_states = outputs.hidden_states
            first_hidden = hidden_states[1] if len(hidden_states) > 1 else hidden_states[0]
            last_hidden = hidden_states[-1]
            avg_hidden = 0.5 * (first_hidden + last_hidden)
            return self._masked_mean_pool(avg_hidden, attention_mask)
        raise ValueError(f"不支持的提取策略: {self.pooling}")

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

    def encode(self, texts, batch_size=32, encoder: str = "query", return_numpy: bool = False, show_progress: bool = True):
        """
        批量将文本转换为语义向量。
        
        参数:
            texts: 待转换的文本列表。
            batch_size: 每次处理的样本数量。
            encoder: 指定使用的模型类型，可选 'query' (问句) 或 'response' (答句)。
            return_numpy: 是否将结果转换为通用的数值数组返回。
            show_progress: 是否显示进度条。

        返回:
            - return_numpy=False: torch.Tensor，形状约为 [N, hidden_size]
            - return_numpy=True: np.ndarray，形状约为 [N, hidden_size]

        流程:
            1. 选择编码器（query/response）
            2. 按 batch 切片
            3. tokenizer 构造输入并迁移设备
            4. 前向推理并池化
            5. 做 L2 归一化
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

        batch_ranges = list(self._batch_slices(len(texts), batch_size))
        progress_bar = tqdm(
            total=len(batch_ranges),
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
                    for start, end in batch_ranges:
                        batch_texts = texts[start:end]
                        # 将文本切分为模型可理解的数字序列，并迁移到目标设备
                        inputs = self._prepare_inputs(batch_texts)
                        # first_last_avg 池化需要中间层隐藏状态，其余策略不需要，
                        need_hidden_states = self.pooling == "first_last_avg"
                        outputs = model(**inputs, output_hidden_states=need_hidden_states)
                        embeddings = self._sentence_pooling(outputs, inputs["attention_mask"])
                        embeddings = self._l2_normalize(embeddings)
                        all_embeddings.append(embeddings.cpu())
                        progress_bar.update(1)
                finally:
                    progress_bar.close()
        # 空输入时返回空张量
        merged = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0))
        return merged.numpy() if return_numpy else merged

    def encode_one(self, text, encoder: str = "query", return_numpy: bool = False):
        """单条文本编码便捷入口。

        说明：
            复用 encode 的完整流程，保证单条与批量编码逻辑一致。
        """
        emb = self.encode([text], batch_size=1, encoder=encoder, return_numpy=return_numpy, show_progress=False)
        if return_numpy:
            return emb.reshape(-1)
        return emb.reshape(-1)