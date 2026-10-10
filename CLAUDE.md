# CLAUDE.md

个人使用的 AI 电影解说平台（导入电影 → 分析 → 文案 → 配音选镜 → 渲染导出）。

- 架构（硬约束）：`docs/ARCHITECTURE.md`
- 实施方案与任务拆分：`docs/IMPLEMENTATION_PLAN.md`
- 想法停车场：`docs/PARKING_LOT.md`
- 架构决策记录：`docs/adr/`
- 模型供应商实测说明：`docs/PROVIDERS.md`（写适配器前必读）

## 工作方式

- 每次只做一个任务 ID（如 `M5-05`），范围以任务卡为准。任务卡之外的改进，写进 `docs/PARKING_LOT.md`，不顺手做
- 涉及结构性改动（新模块、新依赖、新数据层、改依赖方向）前，先读 `docs/ARCHITECTURE.md` 对应章节
- 与架构文档冲突时，停下来说明冲突，不要自行绕过。改规则要先写 ADR
- 提交前必须 `make check` 全绿

## 架构铁律

1. 数据分六层：Source → MovieIndex → Script → EditPlan → Timeline → Output。每个功能都必须落在某一层或某个转换上
2. AI（LLM / ASR / TTS / 视觉）只出现在 分析、写作、编排 三个转换里；编译（compile）和渲染（render）是纯确定性代码，不能调用模型
3. AI 输出必须是通过 Pydantic schema 校验的结构化数据；不让 AI 生成 ffmpeg 命令或可执行代码
4. Timeline 只由编译器生成，任何代码都不手动修改 Timeline
5. 段（Segment）是基本工作单元：配音、选镜、缓存、预览、局部重算都以段为粒度
6. 下游只读上一层的产物，不跨层读取

## 依赖规则（import-linter 强制）

- `domain/` 不 import 任何内部模块，也不 import FastAPI / SQLModel / 模型库
- `algo/` 只依赖 `domain/`，不做 IO
- `stages/` 只通过 `providers/ports.py` 用模型，禁止 import `providers/adapters/`
- 厂商 SDK（anthropic、openai、faster_whisper、insightface、edge_tts 等）只能出现在 `providers/adapters/`
- 只有 `media/` 可以调用 ffmpeg / ffprobe
- `api/` 和 `cli.py` 只调用 `services/`；`server.py` 是唯一把 `api` 与 `worker` 接在一起的组装层（ADR-0003），`cli.py` 通过它启动服务

## 模型供应商与密钥

- 默认供应商 MiniMax（`MINIMAX_API_KEY`，Token Plan 订阅 Key），备选 DeepSeek（`DEEPSEEK_API_KEY`），TTS 备选火山豆包语音（`VOLC_SPEECH_API_KEY`）
- **禁止**在平台代码或配置中使用 `ARK_API_KEY`（火山方舟 Coding Plan Key，仅限编程工具）或任何含 `/api/coding` 的 base_url；火山方舟只接受 `ARK_PAYG_API_KEY`
- 密钥只从环境变量读取；不得写入代码、配置文件、fixture、日志或 `llm_calls` 记录
- 测试默认用假适配器；真实调用的测试标记 `@pytest.mark.heavy`，请求尽量小

## 编码约定

- 源时间用整数毫秒（`*_ms`），输出时间线用整数帧号（`f0` / `f1`，左闭右开）。不用浮点秒
- 帧率用有理数 `{num, den}`
- ID 格式：`前缀_ULID`（`ast_ prj_ ln_ sh_ sc_ ch_ seg_ job_`）
- 改 schema 的顺序：`domain/` → `tests/fixtures/` → 迁移函数 → 生产者 → 消费者；破坏性变更 `schema_version` +1
- 阶段输出语义变化时，Stage 的 `version` +1；改提示词时，模板头部版本号 +1
- 纯算法放 `algo/`，并写单元测试；涉及时长、帧对齐的逻辑写 hypothesis 属性测试
- 文档（Script / EditPlan）版本不可变：编辑即生成新版本，写入需带 `base_version`
- 不新增服务进程、数据库、消息队列；不做多用户；不做浏览器内多轨编辑器（见 ARCHITECTURE §1.3 非目标）

## 常用命令

> 随实现补充；以 `Makefile` 为准。

```bash
make check        # fmt + lint + type + test + import 契约
make api-types    # 由 OpenAPI 生成前端类型
offscreen --help
offscreen run-all movie.mkv --minutes 3          # 电影 -> 解说视频，命中缓存的阶段自动跳过
offscreen stage analysis.shots --asset ast_…     # 只跑某个阶段（--asset 也可直接给电影路径）
offscreen prompt list / render <name> [--vars v.json]   # 看提示词模板及其版本、变量；未给的变量显示为 {{ x }}
offscreen style list / show <id>             # 风格预设：语气、结构、开头钩子、常用句式、禁用词
offscreen report <asset>                         # 分析的耗时、模型用量、数量
offscreen cuts evaluate <asset> [--raw]          # 镜头检测对人工标注切点的 P / R / F1
offscreen select evaluate <asset> [-k 5]         # 选镜排序对人工标注的首选可接受率 / top-k 召回（docs/M5_EVALUATION.md）
```
