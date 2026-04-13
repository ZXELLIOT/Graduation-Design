import torch
import time
from tqdm.auto import tqdm
from src.config import MODEL_NAME, MAX_LENGTH, SIMCSE_TEMPERATURE, has_local_model
from src.scratch_simcse import CharVocab, ScratchConfig, ScratchSimCSE, collate_texts


class SimCSEEncoder:
    """推理阶段编码器：负责加载本地模型并输出句向量。"""

    def __init__(self):
        load_start = time.perf_counter()
        print(f"正在加载模型 ({MODEL_NAME})...")
        if not has_local_model():
            raise FileNotFoundError(
                "未找到本地模型文件，请先运行 train_simcse.py 生成 model/my-simcse。"
            )
        # 训练产物由三部分组成：模型权重、词表、配置。
        model_file = f"{MODEL_NAME}/model.pt"
        vocab_file = f"{MODEL_NAME}/vocab.json"
        # 统一先加载到 CPU，再根据设备迁移，兼容更多运行环境。
        checkpoint = torch.load(model_file, map_location="cpu")

        # 1) 词表恢复：保证推理时字符到 id 的映射与训练完全一致。
        self.vocab = CharVocab.load(vocab_file)
        cfg = checkpoint["config"]

        # 2) 按保存的超参数重建同构网络，再加载权重。
        model_cfg = ScratchConfig(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_size=cfg["hidden_size"],
            projection_dim=cfg["projection_dim"],
            dropout=cfg["dropout"],
        )
        self.model = ScratchSimCSE(model_cfg)
        self.model.load_state_dict(checkpoint["state_dict"])
        # 优先使用 checkpoint 中的 max_length，找不到才回退配置默认值。
        self.max_length = int(cfg.get("max_length", MAX_LENGTH))

        # 优先使用 GPU，没有则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 推理模式：只做预测，不更新参数
        self.model.eval()
        load_elapsed = time.perf_counter() - load_start
        print(f"模型加载完毕，系统当前运行在: {self.device}，耗时: {load_elapsed:.2f} 秒")

    def _l2_normalize(self, embeddings, eps=1e-8):
        """手动实现 L2 归一化。"""
        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=eps)

    def encode(self, texts, batch_size=32):
        """
        将文本转换为语义向量。
        使用分批处理，避免一次性占用过多显存。 

        输入：str 或 list[str]
        输出：shape = [N, projection_dim] 的归一化句向量
        """
        if isinstance(texts, str):
            # 允许单条字符串输入，内部统一转成列表处理。
            texts = [texts]

        if not texts:
            output_dim = getattr(self.model.config, "projection_dim", 0)
            return torch.empty((0, output_dim), dtype=torch.float32)
            
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        encode_start = time.perf_counter()
        show_bar = len(texts) >= batch_size * 2

        for i in tqdm(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="编码进度",
            unit="batch",
            dynamic_ncols=True,
            disable=not show_bar,
        ):
            # 分批处理，防止一次性占用过多显存/内存。
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                # 文本批量编码：字符 id + attention mask。
                input_ids, attention_mask = collate_texts(
                    batch_texts,
                    vocab=self.vocab,
                    max_length=self.max_length,
                    device=self.device,
                )
                # 模型前向输出即为 L2 归一化后的句向量。
                embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu())

        # 合并每一批结果，得到 [样本数, 向量维度]。
            merged = torch.cat(all_embeddings, dim=0)
            encode_elapsed = time.perf_counter() - encode_start
            print(f"编码完成，共 {len(texts)} 条文本，耗时: {encode_elapsed:.2f} 秒")
            return merged

    def simcse_similarity(self, query_emb, corpus_emb, temperature=SIMCSE_TEMPERATURE, eps=1e-8):
        """SimCSE 打分：温度缩放后的点积，再映射为 0-1 置信分。

        说明：
        - 输入向量会再次做 L2 归一化，避免数值漂移。
        - 返回值可用于排序或阈值过滤。
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if corpus_emb.dim() == 1:
            corpus_emb = corpus_emb.unsqueeze(0)

        q = self._l2_normalize(query_emb, eps=eps)
        c = self._l2_normalize(corpus_emb, eps=eps)

        # 先算缩放后的相似度，再用 sigmoid 压到 0~1。
        logits = torch.matmul(q, c.transpose(0, 1)) / max(temperature, eps)
        scores = torch.sigmoid(logits)

        if scores.size(0) == 1:
            return scores.squeeze(0)
        return scores