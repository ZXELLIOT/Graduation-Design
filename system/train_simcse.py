import argparse
import json
import os
import random
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.scratch_simcse import CharVocab, ScratchConfig, ScratchSimCSE, collate_texts


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_local_path(path_str: str) -> str:
    """将相对路径解析到当前脚本目录，绝对路径保持不变。"""
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(SCRIPT_DIR, path_str)


def seed_everything(seed: int) -> None:
    """固定随机种子，减少多次运行的波动。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sentences(corpus_csv: str, max_rows: int = 0) -> List[str]:
    """读取 corpus.csv，返回去重后的文本列表。

    训练时把 query/reply 都视为无监督句子样本：
    只需要句子本身，不需要标签。
    """
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


def split_train_eval(texts: List[str], eval_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """按比例划分训练/验证集。"""
    if eval_ratio <= 0:
        return texts, []

    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)

    eval_size = int(len(shuffled) * eval_ratio)
    eval_size = max(1, min(eval_size, len(shuffled) - 1))
    eval_texts = shuffled[:eval_size]
    train_texts = shuffled[eval_size:]
    return train_texts, eval_texts


def evaluate_unsup_top1(
    model: ScratchSimCSE,
    texts: List[str],
    vocab: CharVocab,
    max_length: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> Tuple[float, float]:
    """无监督验证：返回平均 loss 与批内 top1 命中率。"""
    if not texts:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch_texts in batch_iter(texts, batch_size):
            if len(batch_texts) < 2:
                continue

            input_ids, attention_mask = collate_texts(
                batch_texts, vocab=vocab, max_length=max_length, device=device
            )

            z1 = model(input_ids, attention_mask)
            z2 = model(input_ids, attention_mask)
            sim_matrix = torch.matmul(z1, z2.T) / temperature
            labels = torch.arange(sim_matrix.size(0), device=device)

            loss1 = F.cross_entropy(sim_matrix, labels)
            loss2 = F.cross_entropy(sim_matrix.T, labels)
            loss = 0.5 * (loss1 + loss2)

            preds = torch.argmax(sim_matrix, dim=1)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            total_steps += 1

    model.train()
    if total_steps == 0:
        return 0.0, 0.0
    return total_loss / total_steps, total_acc / total_steps


def train_unsup_simcse(
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
    min_freq: int,
    max_vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    projection_dim: int,
    dropout: float,
    eval_ratio: float,
    grad_accum_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    use_amp: bool,
    save_best: bool,
    save_last: bool,
    resume_checkpoint: str,
):
    """从0开始训练字符级 SimCSE，并保存模型与词表。

    训练核心：
    1. 同一批句子前向两次，得到 z1/z2（dropout 产生轻微扰动）
    2. 计算批内相似度矩阵
    3. 用对角线为正样本、非对角线为负样本做对比学习
    """
    seed_everything(seed)

    # 统一把相对路径解析到脚本目录，避免在不同 cwd 下找不到文件。
    corpus_csv = resolve_local_path(corpus_csv)
    output_dir = resolve_local_path(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    texts = load_sentences(corpus_csv, max_rows=max_rows)
    if len(texts) < batch_size:
        raise ValueError("语料太少，至少需要 >= batch_size 条句子")

    train_texts, eval_texts = split_train_eval(texts, eval_ratio=eval_ratio, seed=seed)
    print(f"训练样本: {len(train_texts)} | 验证样本: {len(eval_texts)}")

    # 词表仅由当前训练语料构建，保证“从0开始”不依赖外部词典。
    vocab = CharVocab.build(texts, min_freq=min_freq, max_size=max_vocab_size)
    print(f"词表大小: {len(vocab.stoi)}")

    # 模型结构超参数集中放在 ScratchConfig，便于保存和复现。
    config = ScratchConfig(
        vocab_size=len(vocab.stoi),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        projection_dim=projection_dim,
        dropout=dropout,
    )
    model = ScratchSimCSE(config).to(device)
    print("训练模式: 从0开始（随机初始化参数）")

    model.train()

    steps_per_epoch = max(1, len(train_texts) // batch_size)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(current_step: int) -> float:
        # 线性 warmup + 线性衰减，训练前期更稳定。
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    best_eval_acc = -1.0
    train_log = []

    if resume_checkpoint:
        resume_path = resolve_local_path(resume_checkpoint)
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_eval_acc = float(ckpt.get("best_eval_acc", -1.0))
            print(f"已从断点恢复: {resume_path} (epoch={start_epoch}, step={global_step})")
        else:
            print(f"未找到断点文件，忽略 --resume-checkpoint: {resume_path}")

    for epoch in range(start_epoch, epochs + 1):
        random.shuffle(train_texts)
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for batch_idx, batch_texts in enumerate(batch_iter(train_texts, batch_size), start=1):
            if len(batch_texts) < batch_size:
                continue

            # 文本 -> input_ids/attention_mask（字符级编码）
            input_ids, attention_mask = collate_texts(
                batch_texts, vocab=vocab, max_length=max_length, device=device
            )

            # 同一输入做两次前向，得到 SimCSE 的正样本对。
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                z1 = model(input_ids, attention_mask)
                z2 = model(input_ids, attention_mask)

                # 批内对比：每一行与所有样本比较，对角线位置是正确配对。
                sim_matrix = torch.matmul(z1, z2.T) / temperature
                labels = torch.arange(sim_matrix.size(0), device=device)

                # 双向损失：z1->z2 与 z2->z1 各算一次，取平均更稳。
                loss1 = F.cross_entropy(sim_matrix, labels)
                loss2 = F.cross_entropy(sim_matrix.T, labels)
                loss = 0.5 * (loss1 + loss2)

            scaled_loss = loss / max(1, grad_accum_steps)
            scaler.scale(scaled_loss).backward()

            do_step = (batch_idx % max(1, grad_accum_steps) == 0)
            if do_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            step_count += 1
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[step {global_step}] loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.8f}"
                )

        # 若最后一轮不足 grad_accum_steps，也要完成一次参数更新。
        if step_count > 0 and (step_count % max(1, grad_accum_steps) != 0):
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, step_count)
        eval_loss, eval_acc = evaluate_unsup_top1(
            model=model,
            texts=eval_texts,
            vocab=vocab,
            max_length=max_length,
            batch_size=batch_size,
            temperature=temperature,
            device=device,
        )
        print(
            f"Epoch {epoch}/{epochs} 完成, train_loss={avg_loss:.4f}, "
            f"eval_loss={eval_loss:.4f}, eval_top1={eval_acc:.4f}"
        )

        os.makedirs(output_dir, exist_ok=True)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_loss,
            "eval_loss": eval_loss,
            "eval_top1": eval_acc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        train_log.append(epoch_record)

        if save_last:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_eval_acc": best_eval_acc,
                },
                os.path.join(output_dir, "last.pt"),
            )

        if save_best and eval_acc >= best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_eval_acc": best_eval_acc,
                },
                os.path.join(output_dir, "best.pt"),
            )

    os.makedirs(output_dir, exist_ok=True)
    # 保存 checkpoint：包括权重和训练时模型结构参数。
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": config.vocab_size,
                "embed_dim": config.embed_dim,
                "hidden_size": config.hidden_size,
                "projection_dim": config.projection_dim,
                "dropout": config.dropout,
                "max_length": max_length,
                "temperature": temperature,
            },
        },
        os.path.join(output_dir, "model.pt"),
    )
    # 保存词表和轻量配置文件，供推理端加载。
    vocab.save(os.path.join(output_dir, "vocab.json"))
    with open(os.path.join(output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "scratch_simcse",
                "max_length": max_length,
                "temperature": temperature,
                "eval_ratio": eval_ratio,
                "grad_accum_steps": grad_accum_steps,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "vocab_size": config.vocab_size,
                "embed_dim": config.embed_dim,
                "hidden_size": config.hidden_size,
                "projection_dim": config.projection_dim,
                "dropout": config.dropout,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"训练完成，模型已保存至: {output_dir}")


def parse_args():
    """读取训练参数。"""
    parser = argparse.ArgumentParser(description="从0开始训练无监督 SimCSE 中文模型")
    parser.add_argument("--corpus-csv", type=str, default="data/corpus.csv")
    parser.add_argument("--output-dir", type=str, default="model/my-simcse")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=0, help="0 表示使用全部数据")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=0, help="0 表示不限制词表")
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集占比")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度训练(cuda)")
    parser.add_argument("--save-best", action="store_true", help="保存最佳验证指标模型")
    parser.add_argument("--save-last", action="store_true", help="保存每轮最新断点")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="从断点继续训练")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_unsup_simcse(
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
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        eval_ratio=args.eval_ratio,
        grad_accum_steps=args.grad_accum_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        save_best=args.save_best,
        save_last=args.save_last,
        resume_checkpoint=args.resume_checkpoint,
    )