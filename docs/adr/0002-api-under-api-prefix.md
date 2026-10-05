# ADR-0002：HTTP API 统一挂在 `/api` 下

- **状态**：已接受
- **日期**：2026-10-05
- **关联**：ARCHITECTURE §10、任务 M2-07、M2-11

## 背景

§10 的 API 路径没有前缀：`/assets`、`/projects/{id}`、`/jobs`、`/events`、`/files/{path}`。
前端（M2-07）的页面路由天然也是 `/jobs`（作业中心）、`/projects/:id`（项目页）。
M2-11 要求生产模式由 FastAPI 托管前端静态文件：浏览器直接打开或刷新 `/projects/prj_…` 时，
这个地址既是后端的 JSON 接口，又是前端页面，两者无法共存。

## 决策

所有 HTTP 接口挂在 `/api` 下（`/api/assets`、`/api/jobs/{id}:cancel`、`/api/events`、`/api/files/…`）。
`/api` 之外的路径留给前端：`/` 与所有前端路由返回 `index.html`，静态资源由 FastAPI 托管。
开发时 Vite 把 `/api` 代理到后端，两种模式下前端代码里的路径一致。

## 备选方案

| 方案 | 优点 | 缺点 | 为什么没选 |
|------|------|------|-----------|
| A. 后端统一 `/api` 前缀（本决策） | 业界惯例；同源，无需 CORS；前端路由不受限 | 要改已写的路由测试与 OpenAPI | — |
| B. 前端路由换名避开冲突（`/ui/jobs`…） | 后端零改动 | 地址难看；每新增一个后端资源都要再查一遍冲突；Vite 代理要逐条列路径 | 把冲突问题留给未来 |
| C. 前端用 hash 路由（`/#/jobs`） | 后端零改动 | 链接丑；`/files` 等仍与静态托管有潜在冲突 | 同上 |

## 影响

- 修改：`api/app.py`（路由统一加前缀）、`docs/openapi.json`（路径带 `/api`）、API 相关测试、
  ARCHITECTURE §10（说明前缀）
- 不破坏任何数据；CLI 不经过 HTTP，不受影响
- 不触碰非目标

## 落地检查

- [x] ARCHITECTURE.md 已更新
- [ ] CLAUDE.md 已更新（不涉及）
- [ ] import-linter 契约已更新（不涉及）
