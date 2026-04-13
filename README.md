# 项目简介

本项目实现了一个基于本地训练 SimCSE 句向量模型的中文检索式对话系统。系统流程涵盖数据预处理、模型训练、语料编码、检索匹配与 Web 交互，所有核心环节均可本地复现，无需依赖外部 API。

**主要特性：**
- 支持从零训练字符级 SimCSE 句向量模型，适配中文对话场景
- 采用高效的语料预处理与问答对抽取脚本，支持大规模数据
- 检索模块结合语义相似度与多特征融合，提升匹配准确率
- 提供 Gradio Web 界面，便于交互测试

# 系统流程

1. **数据准备**
   - 运行 `system/process_lccc.py` 从 LCCC-base-split 目录提取原始对话，生成 `data/raw_dialogues_train.txt`、`data/raw_dialogues_test.txt`
   - 运行 `system/src/preprocess.py` 清洗文本，生成标准 `corpus_train.csv`、`corpus_test.csv`

2. **模型训练**
   - 运行 `system/train_simcse.py`，使用本地 CSV 语料从零训练 SimCSE 模型，产出 `model/my-simcse` 目录（含权重、词表、日志、配置）
   - 支持断点续训、混合精度、自动保存最佳模型

3. **语料编码与检索**
   - 运行 `system/src/model.py` 加载本地模型，将语料批量编码为句向量
   - 运行 `system/src/matcher.py`，对用户输入与语料进行多特征融合检索，返回最优回复

4. **Web 交互**
   - 运行 `system/app.py` 启动 Gradio Web 界面，支持输入问句、查看检索结果

5. **微调/增量训练**
   - 可用 `--resume-checkpoint` 参数加载已有模型，配合较小学习率和轮次进行微调

# 目录结构说明

- `system/app.py`：主入口，集成模型加载、检索与 Web 服务
- `system/process_lccc.py`：LCCC 数据抽取脚本
- `system/src/preprocess.py`：文本清洗与标准化
- `system/src/data_loader.py`：CSV 语料加载
- `system/src/model.py`：本地 SimCSE 编码器
- `system/src/matcher.py`：检索与多特征融合
- `system/src/scratch_simcse.py`：模型结构与词表工具
- `system/train_simcse.py`：训练脚本
- `system/data/`：存放中间语料、编码结果
- `model/my-simcse/`：本地模型权重、词表、训练日志

# 运行方式

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 数据处理
   ```bash
   python process_lccc.py
   python src/preprocess.py
   ```

3. 训练模型（正式训练推荐参数，已为默认）
   ```bash
   python train_simcse.py --save-best --save-last --use-amp
   ```

4. 启动 Web 服务
   ```bash
   python app.py
   ```

5. 微调/增量训练（举例）
   ```bash
   python train_simcse.py --resume-checkpoint model/my-simcse/best.pt --lr 1e-4 --epochs 3 --save-best --use-amp
   ```

# 预训练模型参数（默认配置）

| 参数              | 默认值      | 说明                         |
|-------------------|------------|------------------------------|
| epochs            | 10         | 训练轮次                     |
| batch-size        | 128        | 批大小                       |
| max-length        | 64         | 输入句子最大长度             |
| lr                | 5e-4       | 学习率                       |
| warmup-ratio      | 0.05       | 学习率 warmup 比例            |
| temperature       | 0.05       | 对比学习温度                  |
| min-freq          | 2          | 词表最小字符频次             |
| max-vocab-size    | 0          | 词表最大字符数（0为不限制）   |
| embed-dim         | 256        | 字符嵌入维度                  |
| hidden-size       | 256        | GRU 隐层维度                  |
| projection-dim    | 256        | 投影层维度                    |
| dropout           | 0.2        | dropout 概率                  |
| grad-accum-steps  | 2          | 梯度累积步数                  |
| weight-decay      | 0.01       | AdamW 权重衰减                |
| max-grad-norm     | 1.0        | 梯度裁剪阈值                  |
| seed              | 42         | 随机种子                      |

**微调建议：**
- 降低 lr（如 1e-4），epochs 1–3，保持 batch-size、max-length、结构参数一致。
- 使用 `--resume-checkpoint` 指定预训练模型。

如需自定义参数，直接在命令行传入即可覆盖默认值。