# 基于 SimCSE 的中文检索式对话系统

这是一个用于毕业设计的简单项目，采用本地微调的双编码器模型对中文问句进行检索并返回回答。

主要目录说明：
- `system/`：包含运行与训练脚本、数据与模型目录。
- `system/src/`：项目核心代码，包括编码器封装、匹配逻辑与数据读取。
- `system/data/`：语料文件与缓存向量（pt 文件）。
- `system/model/`：训练或导出的模型文件夹（本项目使用本地模型加载）。

快速开始

1. 安装依赖：

```
pip install -r requirements.txt
```

2. 启动本地演示界面（需先准备好 `system/model/mysimcse` 目录或使用已有模型）：

```
python system/app.py
```

3. 训练模型（如需重新训练，如果使用示例数据，请根据实际路径调整参数）：

```
python system/train.py --quick
```

说明与注意事项

- 请确保 `system/model/mysimcse` 中包含已导出的编码器文件（query_encoder/response_encoder）。
- 项目中可能包含较大的模型与向量文件（*.pt、*.safetensors），建议置于不纳入版本控制的位置。
- 若要清理缓存，可删除 `system/data` 下的 `train_*_embeddings.pt` 与 `train_*s.pt` 文件。