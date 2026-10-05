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

## 运行

```bash
cp config.example.yaml config.yaml      # 填好 media_roots；密钥只通过环境变量提供（见 .env.example）
make sync web-install                   # 安装后端与前端依赖（首次）
make serve                              # 构建前端并启动：http://127.0.0.1:8000
```

`offscreen serve` 在一个进程里同时提供 API、前端页面和作业 Worker（Ctrl+C 退出，没跑完的作业会放回队列，
下次启动继续）。其他用法：

| 命令 | 说明 |
|------|------|
| `offscreen serve --no-worker` + `offscreen worker` | API 和 Worker 分开两个进程 |
| `offscreen run-all movie.mkv --minutes 3` | 不开网页，命令行直接出片 |
| `cd frontend && npm run dev` | 前端开发服务器（把 `/api` 代理到 8000 端口） |

服务没有登录认证，默认只监听 `127.0.0.1`；不要把它暴露到局域网或公网。

## 配置

复制 `config.example.yaml` 为 `config.yaml`；密钥通过环境变量提供（见 `.env.example`）。默认使用 MiniMax，DeepSeek 备选。

## 状态

M0、M1 已完成；M2（作业系统 + API + 网页骨架）的 11 个任务已实现，等待在浏览器里做完整验收。复盘见 [docs/M1_RETRO.md](docs/M1_RETRO.md)。
