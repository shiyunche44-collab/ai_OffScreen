# ADR-0001：EditPlan 的解说段携带文本快照

- **状态**：已接受
- **日期**：2026-10-04
- **关联**：ARCHITECTURE §2.1 铁律 5、§5.5、§7.4（字幕）、任务 M1-12

## 背景

编译器（T4）要生成字幕轨：§7.4 规定「按配音字级时间戳切行」，这需要这一段的**文案文本**。
但 EditPlan 的 `PlanSegment` 只有 `text_hash`，没有文本；而铁律 5 规定下游「只认上游的产物」，
不允许跨层偷读（编译器的输入是 EditPlan，不能去读 Script）。

## 决策

`PlanSegment` 增加可选字段 `text`：构建计划时，把**配音实际使用的文本**存入。
`text_hash` 仍是该文本的哈希，两者同时存在时必须一致（schema 校验）。
非 stale 的解说段必须带 `text`；stale 段（等待重算）可以不带。编译器只读 EditPlan。

## 备选方案

| 方案 | 优点 | 缺点 | 为什么没选 |
|------|------|------|-----------|
| A. PlanSegment 存 `text` 快照（本决策） | 编译器只读 EditPlan，铁律不变；字幕文本与配音文本必然一致，哪怕 Script 之后又被改了 | EditPlan 多存一份文本 | — |
| B. `compile(plan, script)` 额外读 Script | 不改 schema | 直接违反铁律 5，需要放宽规则，渲染器等也可能被类比放宽 | 破坏分层的代价大于多存一段文本 |
| C. M1 不做字幕轨 | 不触碰任何规则 | M1 退出标准「有字幕」落空，M1-13 无字幕可烧录 | 推迟问题而不是解决 |

## 影响

- 修改：`domain/plan.py`、`tests/fixtures/domain/plan.json`、`docs/schemas/plan.schema.json`、
  `stages/creation/plan.py`（生产者，Stage `version` 1 → 2）、ARCHITECTURE §5.5 示例
- 是否破坏现有数据：不需要迁移，`schema_version` 不变。`text` 是新增字段；
  此前只有 M1 的可重算缓存产物（不是版本化文档库里的数据），Stage `version` +1 后会自动重算
- 对非目标（ARCHITECTURE §1.3）没有触碰

## 落地检查

- [x] ARCHITECTURE.md 已更新（§5.5 示例）
- [x] CLAUDE.md 无需更新（铁律与依赖规则未变）
- [x] import-linter 契约无需更新
