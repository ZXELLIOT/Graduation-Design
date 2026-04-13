import argparse
import glob
import os
import random
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_local_path(path_str: str) -> str:
    """将相对路径解析到当前脚本目录，绝对路径保持不变。"""
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(SCRIPT_DIR, path_str)


def resolve_model_path(model_name: str) -> str:
    """
    解析模型路径。
    - 若传入的是 HuggingFace 本地缓存仓库根目录（含 snapshots），自动选择最新快照目录。
    - 其他情况原样返回（可为在线模型名或本地完整模型目录）。
    """
    if not os.path.isabs(model_name) and not os.path.exists(model_name):
        return model_name

    if os.path.isdir(model_name):
        snapshots_dir = os.path.join(model_name, "snapshots")
        if os.path.isdir(snapshots_dir):
            candidates = [
                p for p in glob.glob(os.path.join(snapshots_dir, "*")) if os.path.isdir(p)
            ]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]

        if os.path.exists(os.path.join(model_name, "config.json")):
            return model_name

    return model_name


def seed_everything(seed: int) -> None:
    """固定随机种子，减少多次运行的波动。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sentences(corpus_csv: str, max_rows: int = 0) -> List[str]:
    """读取 corpus.csv，返回去重后的文本列表。"""
    df = pd.read_csv(corpus_csv, encoding="utf-8")
    if max_rows > 0:
        df = df.head(max_rows)

    texts = []
    for _, row in df.iterrows():
        q = str(row.get("query", "")).strip()
        r = str(row.get("reply", "")).strip()
        if q:
            texts.append(q)
        if r:
            texts.append(r)

    texts = list(dict.fromkeys(texts))
    random.shuffle(texts)
    return texts


def batch_iter(items: List[str], batch_size: int):
    """按 batch_size 分批返回数据。"""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def train_unsup_simcse(
    model_name: str,
    corpus_csv: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    lr: float,
    temperature: float,
    warmup_ratio: float,
    max_rows: int,
    seed: int,
):
    """训练无监督 SimCSE，并保存模型与分词器。"""
    seed_everything(seed)

    corpus_csv = resolve_local_path(corpus_csv)
    output_dir = resolve_local_path(output_dir)
    model_name = resolve_model_path(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    texts = load_sentences(corpus_csv, max_rows=max_rows)
    if len(texts) < batch_size:
        raise ValueError("语料太少，至少需要 >= batch_size 条句子")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train()

    steps_per_epoch = len(texts) // batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(texts)
        epoch_loss = 0.0
        step_count = 0

        for batch_texts in batch_iter(texts, batch_size):
            if len(batch_texts) < batch_size:
                continue

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            out1 = model(**inputs).last_hidden_state[:, 0]
            out2 = model(**inputs).last_hidden_state[:, 0]

            z1 = F.normalize(out1, p=2, dim=1)
            z2 = F.normalize(out2, p=2, dim=1)

            sim_matrix = torch.matmul(z1, z2.T) / temperature
            labels = torch.arange(sim_matrix.size(0), device=device)

            loss1 = F.cross_entropy(sim_matrix, labels)
            loss2 = F.cross_entropy(sim_matrix.T, labels)
            loss = 0.5 * (loss1 + loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[step {global_step}] loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.8f}"
                )

        avg_loss = epoch_loss / max(1, step_count)
        print(f"Epoch {epoch}/{epochs} 完成, avg_loss={avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"训练完成，模型已保存至: {output_dir}")


def parse_args():
    """读取训练参数。"""
    parser = argparse.ArgumentParser(description="训练无监督 SimCSE 中文模型")
    parser.add_argument(
        "--model-name",
        type=str,
        default=r"C:\Users\13713\.cache\huggingface\hub\models--shibing624--text2vec-base-chinese",
    )
    parser.add_argument("--corpus-csv", type=str, default="data/corpus.csv")
    parser.add_argument("--output-dir", type=str, default="modle/my-simcse")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=0, help="0 表示使用全部数据")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_unsup_simcse(
        model_name=args.model_name,
        corpus_csv=args.corpus_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        temperature=args.temperature,
        warmup_ratio=args.warmup_ratio,
        max_rows=args.max_rows,
        seed=args.seed,
    )
