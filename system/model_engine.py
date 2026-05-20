"""
system/model_engine.py
SimCSE 双塔编码引擎 — 文本 → L2 归一化向量。
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from contextlib import nullcontext
from system.config import SIMCSE_MODEL_DIR,MAX_TEXT_LEN

class SimCSEModelEngine:
    """双塔编码引擎：query_encoder 编码问句，response_encoder 编码答句。"""

    def __init__(self, model_dir: str = SIMCSE_MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        q_dir = os.path.join(model_dir, "query_encoder")
        r_dir = os.path.join(model_dir, "response_encoder")
        self.query_encoder = AutoModel.from_pretrained(q_dir, local_files_only=True).to(self.device).eval()
        self.response_encoder = AutoModel.from_pretrained(r_dir, local_files_only=True).to(self.device).eval()
        self.hidden_size = int(self.query_encoder.config.hidden_size)
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

    def _select_encoder(self, encoder: str):
        if encoder not in ("query", "response"):
            raise ValueError("encoder 必须是 'query' 或 'response'")
        return self.query_encoder if encoder == "query" else self.response_encoder

    def encode(self, texts, batch_size=32, encoder="query",return_numpy=False):
        """批量编码文本 → L2 归一化向量。"""
        if isinstance(texts, str):
            texts = [texts]
        model = self._select_encoder(encoder)
        result = []
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=MAX_TEXT_LEN, return_tensors="pt",
                )
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[:, 0]
                result.append(F.normalize(vec.float(), p=2, dim=1).cpu())

        merged = torch.cat(result, dim=0) if result else torch.empty((0, self.hidden_size))
        return merged.numpy() if return_numpy else merged

    def encode_one(self, text, encoder="query", return_numpy=False):
        """单条编码"""
        arr = self.encode([text], batch_size=1, encoder=encoder,return_numpy=return_numpy)
        return arr[0] if return_numpy else arr[0]