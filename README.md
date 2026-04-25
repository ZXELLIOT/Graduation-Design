# 基于 SimCSE 的中文检索式对话系统

这是一个用于毕业设计的本地检索式对话系统，核心为双编码器检索（query/response 双塔）+ 可选 AI 增强生成。

系统当前包含以下能力：
- 本地语义检索与问答返回（FAISS 粗召回 + 问问/问答加权精排）
- 上下文拼接开关与关键词触发上下文
- 可选 AI 增强回复（支持本地检索结果融合）
- 前端参数热更新、日志面板、实时性能监控（CPU/内存/对话时延）
- 支持 cloudflared 自动内网穿透（启动即尝试生成公网地址）

## 目录说明

- db/
	- data/: 数据库 CSV 与 FAISS 索引文件
- model/
	- mysimcse/: 本项目训练得到的双塔模型
- system/
	- app.py: FastAPI 入口
	- comparator.py: 检索与上下文判定核心逻辑
	- chat_service.py: 聊天主链路（本地检索/AI增强）
	- web/: 前端页面与交互脚本
- tests/
	- test_eval01~04: 评测脚本（模型相关性、检索策略、基线对比、AI消融）
- train/
	- 数据预处理与训练脚本（两阶段训练）

## 快速开始

1. 安装依赖

pip install -r requirements.txt

2. 启动系统

python system/app.py

默认访问地址：
- 本地页面: http://127.0.0.1:7860
- 元信息接口: http://127.0.0.1:7860/api/meta

若本机已安装 cloudflared，服务启动后会自动尝试创建 quick tunnel，并在控制台打印公网地址。

3. 运行训练（可选）

python train/train.py --quick

## 内网穿透（可选）

默认启用自动穿透，可通过环境变量控制：
- AUTO_TUNNEL_ENABLED=1 或 0：是否自动拉起 cloudflared
- TUNNEL_LOCAL_URL=http://127.0.0.1:7860：穿透映射目标

示例：
- 禁用自动穿透后启动
	- Windows PowerShell: $env:AUTO_TUNNEL_ENABLED="0"; python system/app.py

## 前端功能说明

- 清除对话：仅清空当前会话窗口，不清空日志文件
- 清除日志：清空 runtime/chat_logs/chat_log.jsonl
- 实时性能：每 2 秒刷新设备与系统关键性能指标
- 调参面板：支持上下文开关、粗召回数量、精排数量、AI增强开关等热更新

## 常见问题

1) 启动后提示模型或索引文件缺失
- 检查以下资源是否存在并可读取：
	- db/data/lccc_large.csv
	- db/data/large_faiss_db_query.index
	- db/data/large_faiss_db_response.index
	- model/mysimcse/

2) AI增强开启失败
- 检查环境变量中是否配置有效的 API Key。

3) 内网穿透未生成公网地址
- 确认 cloudflared 已安装且可在终端直接执行。

## 注意事项

- 项目包含大模型与索引文件，建议不要提交到版本库。
- 若要复现实验结果，请固定评测脚本中的随机种子配置。