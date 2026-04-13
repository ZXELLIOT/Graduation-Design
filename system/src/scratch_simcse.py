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
        counter = Counter()
        for text in texts:
            counter.update(list(text))

        items = [(ch, freq) for ch, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size > 0:
            items = items[:max_size]

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
        ids = [self.stoi.get(ch, UNK_ID) for ch in list(text)]
        ids = ids[:max_length]
        if not ids:
            ids = [UNK_ID]
        return ids


class ScratchSimCSE(nn.Module):
    """从0开始训练的字符级句向量模型。"""

    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=PAD_ID)
        self.encoder = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.hidden_size * 2, config.projection_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )

        mask = attention_mask.unsqueeze(-1).type_as(out)
        pooled = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = self.dropout(pooled)

        z = self.projection(pooled)
        return F.normalize(z, p=2, dim=1)


def collate_texts(
    texts: Sequence[str], vocab: CharVocab, max_length: int, device: torch.device
):
    """把文本批量编码成 input_ids 和 attention_mask。"""
    token_ids = [vocab.encode(t, max_length=max_length) for t in texts]
    max_len = max(len(x) for x in token_ids)

    input_ids = torch.full((len(token_ids), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(token_ids), max_len), dtype=torch.long)

    for i, ids in enumerate(token_ids):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1

    return input_ids.to(device), attention_mask.to(device)
