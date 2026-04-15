# 项目说明：检索式中文对话系统（简要）

此项目为用于演示的检索式对话系统原型，面向中文语料。项目实现了：

- 文本编码为向量并保存缓存；
- 基于向量的相似度检索与匹配；
- 简易的 Web 界面用于交互演示。

运行前准备

1. 安装 Python（建议 3.8 及以上）。
2. 在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

快速运行

﹣ 启动交互界面：

```bash
python system\app.py
```

﹣ 训练或快速测试模型（生成示例模型）：

```bash
python system\train.py --quick
```

要点说明

- 首次运行若无缓存，系统会从 CSV 加载语料并生成向量缓存（保存在 `system/data`）。该过程可能耗时，控制台会显示进度提示；若已有缓存则会直接加载以加速启动。 
- 若有 GPU，可显著加速向量生成与训练步骤。

主要文件说明

- `system/app.py`：启动入口，包含界面与延迟初始化逻辑；
- `system/train.py`：训练脚本，包含 `--quick` 快速模式；
- `system/src/`：核心实现，包括编码器、数据加载与匹配逻辑；
- `system/data/`：用于存放语料与向量缓存的目录；

如需更详细的使用说明或将缓存导出为可读格式，请告知我需要的输出格式。 
- `system/src/`：算法实现（编码器、匹配器、预处理等）
- `system/data/`：训练与缓存数据（语料、句向量缓存等）

**依赖与安装**
- 详细依赖请参考 `requirements.txt`。

**运行示例**

```bash
cd system
pip install -r ../requirements.txt
python app.py
```

**训练模型**
- 若需要训练 SimCSE，请参考 `system/train.py`（或 repo 中的 `train_simcse.py`），训练完成后模型应放置于 `system/model/` 或按 `system/src/config.py` 中配置的路径。

**数据与缓存**
- 项目会在 `system/data/` 中保存已编码的句向量（例如 `train_query_embeddings.pt`、`train_reply_embeddings.pt`），以加速启动。
