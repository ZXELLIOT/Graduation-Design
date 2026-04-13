import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


@dataclass
class ScratchConfig:
    vocab_size: int
    embed_dim: int = 256
    hidden_size: int = 256
    projection_dim: int = 256
    dropout: float = 0.2


class CharVocab:
    """字符级词表，避免依赖外部分词器。"""

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = [""] * len(stoi)
        for token, idx in stoi.items():
            self.itos[idx] = token

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 2, max_size: int = 0):
        # 统计每个字符出现次数。
        counter = Counter()
        for text in texts:
            counter.update(list(text))

        # 过滤低频字符，减少噪声。
        items = [(ch, freq) for ch, freq in counter.items() if freq >= min_freq]
        # 先按频次降序，再按字符排序，保证结果稳定可复现。
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size > 0:
            items = items[:max_size]

        # 预留两个特殊标记：补齐位和未知字符。
        stoi = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for ch, _ in items:
            stoi[ch] = len(stoi)
        return cls(stoi)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        return cls(stoi)

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    def encode(self, text: str, max_length: int) -> List[int]:
        # 把句子逐字映射为编号。
        ids = [self.stoi.get(ch, UNK_ID) for ch in list(text)]
        # 统一截断长度，控制计算成本。
        ids = ids[:max_length]
        if not ids:
            # 空句子用 UNK 占位，保证后续计算不报错。
            ids = [UNK_ID]
        return ids


class ScratchSimCSE(nn.Module):
    """训练字符级句向量模型。"""

    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.config = config
        # 先把字符编号映射成稠密向量。
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=PAD_ID)
        # 用双向 GRU 编码上下文信息。
        self.encoder = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        # 把编码结果映射到训练目标维度。
        self.projection = nn.Linear(config.hidden_size * 2, config.projection_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [批量, 长度] -> [批量, 长度, 向量维度]
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()

        # 打包后送入 GRU，可跳过补齐位的无效计算。
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )

        # 按有效位置做平均池化，得到句子级向量。
        mask = attention_mask.unsqueeze(-1).type_as(out)
        pooled = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = self.dropout(pooled)

        z = self.projection(pooled)
        # 输出前做归一化，方便后续相似度计算。
        return F.normalize(z, p=2, dim=1)


def collate_texts(
    texts: Sequence[str], vocab: CharVocab, max_length: int, device: torch.device
):
    """把文本批量编码成 input_ids 和 attention_mask。"""
    # 先把每条文本转为编号序列。
    token_ids = [vocab.encode(t, max_length=max_length) for t in texts]
    max_len = max(len(x) for x in token_ids)

    input_ids = torch.full((len(token_ids), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(token_ids), max_len), dtype=torch.long)

    for i, ids in enumerate(token_ids):
        length = len(ids)
        # 前半段写入真实编号，后半段保持 PAD。
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        # attention_mask 中 1 表示真实字符，0 表示补齐位。
        attention_mask[i, :length] = 1

    # 最后统一迁移到目标设备。
    return input_ids.to(device), attention_mask.to(device)
