# 毕业设计：基于 SimCSE 的检索式中文对话系统

> 最后更新：2026-04-13
> 项目性质：本科毕业设计

## 项目简介
本项目实现了一个中文检索式对话系统。系统以 SimCSE 句向量表示为核心，结合自定义的加权比对策略，对用户输入与语料库中的候选问句进行相似度检索，并返回最相关的预设回复。

当前实现不是生成式对话，而是“语义检索 + 阈值兜底”的方式：先把语料编码成向量，再对用户输入做语义匹配，最后输出最高分对应的回复。

## 系统流程
1. 从 `system/data/raw_dialogues.txt` 提取或整理原始问答数据。
2. 通过 `system/src/preprocess.py` 清洗文本并生成 `corpus.csv`。
3. 通过 `system/src/data_loader.py` 读取结构化问答对。
4. 通过 `system/src/model.py` 加载本地 SimCSE 模型并生成句向量。
5. 通过 `system/src/matcher.py` 计算匹配分数、比较两种分支结果并选择最高分。
6. 通过 `system/app.py` 启动 Gradio Web 界面进行交互。

## 目录说明
- `system/app.py`：程序入口，负责启动预处理、加载模型、构建匹配器和 Web 界面。
- `system/src/config.py`：统一配置，包括路径、阈值、池化策略和温度参数。
- `system/src/preprocess.py`：把原始对话文本清洗成标准 CSV。
- `system/src/data_loader.py`：读取 `corpus.csv` 并返回问句与答句列表。
- `system/src/model.py`：SimCSE 编码器，负责句向量生成与语义打分。
- `system/src/matcher.py`：检索核心，负责缓存、两路分数计算和最终决策。
- `system/src/comparison.py`：分数比较器，负责选出更高分的分支。
- `system/train_simcse.py`：无监督 SimCSE 训练脚本。
- `system/process_lccc.py`：从 LCCC 压缩包中提取对话语料。
- `system/tests/`：单元测试目录。

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
运行单元测试：

```bash
python -m unittest tests/test_similarity_comparison.py
```

## 重要说明
1. 当前项目默认只使用本地 SimCSE 模型；如果本地模型不存在，会直接报错。
2. `system/data/corpus.csv`、`system/data/corpus_embeddings.pt` 和本地模型文件属于生成产物，通常不建议提交到仓库。
3. 项目中部分目录名沿用了原始命名，例如 `modle`，后续如果重构目录结构需要同步修改配置路径。