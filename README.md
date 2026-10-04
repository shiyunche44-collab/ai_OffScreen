# AI OffScreen

个人使用的 AI 电影解说平台：导入一部电影 → 自动分析 → AI 写解说文案 → AI 配音 + 自动选镜 → 人工微调 → 渲染成片 / 导出到剪映、达芬奇精修。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计（项目"宪法"） |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | 实施方案：里程碑、任务拆分、验收标准 |
| [docs/PROVIDERS.md](docs/PROVIDERS.md) | 模型供应商实测说明（MiniMax / DeepSeek / 火山） |
| [docs/PARKING_LOT.md](docs/PARKING_LOT.md) | 想法停车场 |
| [docs/adr/](docs/adr/) | 架构决策记录 |
| [CLAUDE.md](CLAUDE.md) | AI 编码助手守则 |

## 配置

复制 `config.example.yaml` 为 `config.yaml`；密钥通过环境变量提供（见 `.env.example`）。默认使用 MiniMax，DeepSeek 备选。

## 状态

M0 进行中：已完成 M0-01 ~ M0-03、M0-06、M0-07、M0-10（骨架、工具链、依赖规则、公共类型、L0–L4 schema + fixture、CI）。下一步：M0-04 配置系统、M0-08 FFmpeg 封装。
