# 毕业设计：基于 SimCSE 的检索式中文对话系统

> 最后更新：2026-04-13
> 项目性质：本科毕业设计

## 项目简介
本项目实现了一个中文检索式对话系统。系统以“从0训练”的字符级 SimCSE 句向量模型为核心，结合双模块检索策略，对用户输入与语料库候选问答进行匹配并返回最相关回复。

当前实现不是生成式对话，而是“语义检索 + 阈值兜底”：先把语料编码成向量，再对用户输入做相似度匹配，最后输出最高分对应答句。

## 系统流程
1. 从 `system/data/raw_dialogues.txt` 提取或整理原始问答数据。
2. 通过 `system/src/preprocess.py` 清洗文本并生成 `corpus.csv`。
3. 通过 `system/src/data_loader.py` 读取结构化问答对。
4. 通过 `system/train_simcse.py` 从0训练本地模型（生成 `model/my-simcse`）。
5. 通过 `system/src/model.py` 加载本地模型并生成句向量。
6. 通过 `system/src/matcher.py` 并行计算两种模块分数并选择最高分：
   - 问句-问句匹配（语义 + 字符重叠 + ngram + 长度）
   - 词向量问答匹配（问句词向量与问句/答句词向量联合打分）
7. 通过 `system/app.py` 启动 Gradio Web 界面进行交互。

## 目录说明
- `system/app.py`：程序入口，负责启动预处理、加载模型、构建匹配器和 Web 界面。
- `system/src/config.py`：统一配置，包括路径、阈值、池化策略和温度参数。
- `system/src/preprocess.py`：把原始对话文本清洗成标准 CSV。
- `system/src/data_loader.py`：读取 `corpus.csv` 并返回问句与答句列表。
- `system/src/model.py`：本地模型编码器，负责句向量生成与语义打分。
- `system/src/matcher.py`：检索核心，负责缓存、双模块分数计算和最终决策。
- `system/src/scratch_simcse.py`：字符级词表、模型结构与批处理编码工具。
- `system/train_simcse.py`：从0开始无监督训练脚本（含验证、断点、日志）。
- `system/process_lccc.py`：从 LCCC 解压目录提取对话语料。
- `system/test_two_modules.py`：双模块打分测试脚本。

## 运行方式
建议先进入 `system` 目录再执行命令。

```bash
pip install -r requirements.txt
python app.py
```

如果你需要先生成语料：

```bash
python process_lccc.py
python src/preprocess.py
```

如果你需要重新训练 SimCSE：

```bash
python train_simcse.py
```

## 测试
运行双模块测试：

```bash
python test_two_modules.py
```

## 重要说明
1. 当前项目默认只使用本地模型目录 `system/model/my-simcse`。
2. `system/data/corpus.csv`、`system/data/corpus_embeddings.pt` 和本地模型文件属于生成产物，通常不建议提交到仓库。
3. 推理与测试脚本依赖已训练模型文件：`model.pt` 与 `vocab.json`。