"""
train/model_utils.py
文件作用:
    预训练模型加载与编码工具。
    训练脚本与评测脚本共享的基础组件。
"""
import multiprocessing as mp
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimCSEEncoder:
    """SimCSE编码器：模型加载、TF32加速、文本到L2归一化向量。"""

    def __init__(self, model_path):
        self._setup_tf32()
        self.device = DEVICE
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path)

    @staticmethod
    def _setup_tf32():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    def _load_model_and_tokenizer(self, model_path):
        if mp.current_process().name == 'MainProcess':
            print(f'使用设备: {self.device}')
        print(f'加载模型与分词器: {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        return model, tokenizer

    def encode(self, input_ids, attention_mask):
        """文本 -> L2 归一化向量。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return F.normalize(pooled, p=2, dim=1)