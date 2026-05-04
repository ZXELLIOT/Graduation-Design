# 基于 SimCSE 的中文对话系统研究实现

毕业设计项目。研究并实现了一个基于 SimCSE 双塔语义匹配的中文检索式对话系统：通过无监督 SimCSE + 有监督匹配两阶段微调训练 query/response 双编码器，结合 FAISS 向量检索与加权语义重排完成候选召回，并支持可选的外部大模型增强生成，形成完整的端到端对话链路。

## 系统架构

```
用户输入 → 文本清洗 → 上下文拼接判定 → Query 编码
    → FAISS 粗召回 (HNSW) → 问问/问答加权重排 → Top-K 候选
    → [可选] AI 增强融合 → 最终回复
```

- **双塔模型**: query encoder + response encoder，独立编码问句与答句
- **两阶段训练**: 无监督 SimCSE（语义空间构建）→ 有监督匹配（正负样本对比）
- **FAISS 双索引**: query 索引用于粗召回，response 索引参与重排打分
- **上下文感知**: 关键词触发 + 短文本检测 + 词面/语义多信号判定
- **AI 增强**: 检索候选 + 外部大模型融合润色（可选，默认关闭）

## 目录结构

```
simcse-demo/
├── data_prep/              # 数据预处理
│   ├── data_prep.py            # 清洗、负采样、导出 CSV
│   ├── data_prep_config.py     # 路径与参数配置
│   ├── LCCC-large/             # 原始大语料 JSON（不进入版本控制）
│   └── LCCC-small/             # 原始训练/验证/测试 JSON（不进入版本控制）
├── train/                  # 模型训练
│   ├── train.py                # 两阶段训练主脚本
│   └── text2vec-base-chinese/  # 预训练模型（不进入版本控制）
├── db/                     # 向量库构建
│   ├── encoder_module.py       # 语料编码 → FAISS 索引入库
│   ├── config.py               # 路径配置
│   └── data/                   # CSV 语料 + FAISS 索引文件（不进入版本控制）
├── model/                  # 训练产出的双塔模型（不进入版本控制）
│   └── mysimcse/
│       ├── query_encoder/      # 问句编码器
│       ├── response_encoder/   # 答句编码器
│       └── tokenizer files
├── system/                 # Web 服务主程序
│   ├── app.py                  # FastAPI 入口 + API 路由
│   ├── bootstrap.py            # 启动初始化 + 单例管理
│   ├── chat_service.py         # 聊天主链路（检索/AI增强）
│   ├── comparator.py           # 语义匹配核心（粗召回+加权重排）
│   ├── comparator_settings.py  # 参数热更新
│   ├── runtime_settings.py     # AI 增强运行时开关
│   ├── ai_enhancer.py          # 外部大模型增强回复
│   ├── model_engine.py         # 双塔编码引擎
│   ├── chat_logger.py          # 对话日志读写
│   ├── config.py               # 系统运行配置
│   └── web/                    # 前端页面 (HTML + JS + CSS)
├── tests/                  # 评测脚本
│   ├── test_eval01~04.py       # 模型速度/检索策略/系统基线/AI 消融
│   ├── data/                   # 评测数据集（不进入版本控制）
│   ├── models/                 # 对比模型（不进入版本控制）
│   └── results/                # 评测结果图表（不进入版本控制）
├── runtime/                # 运行时日志（不进入版本控制）
├── requirements.txt
└── README.md
```

## 环境要求

- Python >= 3.10
- CUDA >= 11.8（可选，CPU 模式也可运行但较慢）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备知识库数据

项目使用 [LCCC (Large Chinese Conversational Corpus)](https://github.com/thu-coai/CDial-GPT) 作为知识库。将 `LCCC-large.json` 放入 `data_prep/LCCC-large/` 目录，然后运行数据预处理：

```bash
python data_prep/data_prep.py
```

### 3. 构建 FAISS 向量索引

```bash
# 全量编码入库
python db/encoder_module.py --n_samples 0 --batch_size 128

# 快速测试（仅处理 10000 条）
python db/encoder_module.py --n_samples 10000 --batch_size 128
```

### 4. 启动服务

```bash
python system/app.py
```

服务默认运行在 `http://127.0.0.1:7860`，启动后会自动尝试 cloudflared 内网穿透。

## 模型训练

两阶段微调流程：

```bash
# 完整训练
python train/train.py \
    --stage1_epochs 2 --stage1_batch_size 512 --stage1_learning_rate 3e-5 \
    --stage2_epochs 4 --stage2_batch_size 512 --stage2_learning_rate 3e-5

# 快速验证（--quick 模式使用少量数据跑通全流程）
python train/train.py --quick

# 指定本地预训练模型
python train/train.py --model_name_or_path /path/to/pretrained/model
```

训练关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage1_epochs` | 阶段1（无监督 SimCSE）轮数 | 2 |
| `--stage2_epochs` | 阶段2（有监督匹配）轮数 | 4 |
| `--temperature` | 对比学习温度 | 0.05 |
| `--max_length` | 文本最大截断长度 | 64 |
| `--use_fp16` | 混合精度训练 | True |
| `--use_compile` | torch.compile 加速 | True |
| `--quick` | 快速验证模式 | False |

训练支持 CUDA OOM 自动降 batch size 重试（`--batch_backoff_ratio` 控制回退比例）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/meta` | 服务元信息（知识库来源、隧道地址） |
| POST | `/api/chat` | 聊天接口 |
| GET | `/api/settings/comparator` | 读取当前调参配置 |
| POST | `/api/settings/comparator` | 热更新调参参数 |
| GET | `/api/logs/latest` | 最近 N 条对话日志 |
| POST | `/api/logs/clear` | 清空日志文件 |
| GET | `/api/perf` | 实时性能指标（CPU/内存/P95 时延） |

聊天请求体：

```json
{
  "message": "你好",
  "history": [["之前的问题", "之前的回复"]],
  "conversation_id": "optional-uuid"
}
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AUTO_TUNNEL_ENABLED` | 启动时自动开启 cloudflared 内网穿透 | `1` |
| `TUNNEL_LOCAL_URL` | 隧道目标地址 | `http://127.0.0.1:7860` |
| `AI_ENHANCED_DEFAULT` | 启动时默认开启 AI 增强 | `0` |
| `ARK_API_KEY` | AI 增强使用的 API Key | — |
| `ARK_MODEL_NAME` | AI 增强模型名称 | `Doubao1.5-vision-pro` |
| `ARK_RESPONSES_URL` | AI 增强 API 地址 | `https://ark.cn-beijing.volces.com/api/v3/responses` |
| `ARK_TIMEOUT_SEC` | AI 增强请求超时（秒） | `20` |

## 前端功能

- **对话窗口**: 支持回车发送、Shift+Enter 换行，自动维护多轮上下文
- **调参面板**: 热更新命中阈值、重排权重、粗召回/精排数量、上下文匹配开关、AI 增强开关
- **日志面板**: 实时查看对话日志，含上下文判定理由、检索链路指标、AI 回退原因
- **性能监控**: 每 2 秒刷新 CPU/内存使用率、会话平均时延/P95 时延，折线图可视化
- **会话管理**: 清除当前对话（仅清空窗口，日志保留）、清空日志文件

## 评测

项目包含 4 组评测脚本：

| 脚本 | 评测内容 |
|------|----------|
| `test_eval01` | 多模型相关性 & 编码速度对比 |
| `test_eval02` | 不同检索策略 & 数据库规模精度对比 |
| `test_eval03` | 含/不含系统的基线对比 |
| `test_eval04` | AI 增强模式消融实验 |

运行评测：

```bash
# 设置评审模式（auto 优先 AI 评审，失败回退本地编码器）
export RETRIEVAL_JUDGE_MODE=auto
python tests/test_eval01_model_corr_speed.py
```

结果图表输出至 `tests/results/` 目录。

## 常见问题

### 启动报错模型或索引文件缺失

检查以下资源是否存在：
- `db/data/lccc_large.csv` — 知识库 CSV
- `db/data/large_faiss_db_query.index` — 问句向量索引
- `db/data/large_faiss_db_response.index` — 答句向量索引
- `model/mysimcse/` — 训练好的双塔模型

缺失时按顺序执行：数据预处理 → 构建索引 → 训练或获取模型文件。

### AI 增强开启失败

- 确认已设置 `ARK_API_KEY` 环境变量
- 前端调参面板开启 AI 增强时会自动校验 API Key 可用性

### CUDA 显存不足

- 训练脚本自动支持 OOM 降 batch 重试
- 推理时可设置 `CUDA_VISIBLE_DEVICES=""` 强制使用 CPU 模式
