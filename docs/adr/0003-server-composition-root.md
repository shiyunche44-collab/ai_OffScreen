# ADR-0003：新增 `server` 作为 API + Worker 的组装层

- **状态**：已接受
- **日期**：2026-10-05
- **关联**：ARCHITECTURE §3.1、§4.2、任务 M2-11

## 背景

`offscreen serve` 要在一个进程里同时起 API 和 Worker（个人使用，一条命令启动）。
但 §4.2 的依赖契约把 `cli | api | worker` 放在同一层，同层模块互相独立、不能互相 import。
所以没有任何一处合法的代码可以同时持有「FastAPI 应用」和「Worker」。

## 决策

新增模块 `offscreen/server.py`，是唯一的**组装层**（composition root）：它依赖 `api`、`worker`、
`services`，负责把它们接成一个进程（线程里跑 Worker，主线程跑 uvicorn，退出时把没跑完的作业放回队列）。
依赖方向变为：

```
cli  →  server  →  api | worker  →  services  →  stages  →  engine  →  store  →  domain
```

`api` 与 `worker` 仍然互相独立；`cli` 仍然是薄层（解析参数、调用 `server` 或 `services`）。
`server` 同样受 R6 约束：不直接调 `stages` / `store` / `engine`。

## 备选方案

| 方案 | 优点 | 缺点 | 为什么没选 |
|------|------|------|-----------|
| A. 新增 `server` 组装层（本决策） | 依赖方向清晰；组装逻辑可单测；`api` / `worker` 保持独立 | 多一个模块 | — |
| B. 让 `cli.py` 直接组装 | 不加模块 | cli 不再是薄层；组装逻辑（线程、信号、退出顺序）混在命令解析里，难测 | 违背「cli 只做薄层」 |
| C. 把 `cli | api | worker` 改成可互相 import | 改动最小 | 废掉这条契约保护的东西：API 不能直接起作业、Worker 不能依赖 HTTP 层 | 代价太大 |
| D. 只做两个独立进程（`serve` 不含 Worker） | 不触碰依赖 | 违背「一键启动」；每次要开两个终端 | 体验差 |

## 影响

- 修改：`pyproject.toml`（import-linter 的 layers、R6 增加 `offscreen.server`）、ARCHITECTURE §4.1 / §4.2
- 新增依赖：`uvicorn`
- 不破坏数据；不触碰非目标（仍是单机、无新服务进程：Worker 仍是同一进程里的线程，
  也可以用 `offscreen worker` 单独起）

## 落地检查

- [x] ARCHITECTURE.md 已更新
- [x] CLAUDE.md 已更新（`api/` 和 `cli.py` 的依赖说明）
- [x] import-linter 契约已更新
