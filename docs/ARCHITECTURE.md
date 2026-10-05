# AI OffScreen 架构设计

> 个人使用的 AI 电影解说平台：输入一部电影，在人的把控下产出带 AI 解说配音、字幕、画面剪辑的解说视频，并可导出到剪映 / 达芬奇精修。
>
> **本文档是项目的"宪法"。** 任何与本文档冲突的实现都视为跑偏；要改变这里的规则，先写 ADR（见 `docs/adr/0000-template.md`），再改代码。

---

## 目录

1. [产品定义](#1-产品定义)
2. [核心架构：六层数据 + 五个转换](#2-核心架构六层数据--五个转换)
3. [系统组件与部署](#3-系统组件与部署)
4. [代码结构与依赖规则](#4-代码结构与依赖规则)
5. [数据契约（Schema）](#5-数据契约schema)
6. [流水线引擎](#6-流水线引擎)
7. [各阶段设计](#7-各阶段设计)
8. [模型 Provider 抽象](#8-模型-provider-抽象)
9. [存储设计](#9-存储设计)
10. [API 设计](#10-api-设计)
11. [前端设计](#11-前端设计)
12. [技术选型](#12-技术选型)
13. [关键架构决策（ADR 摘要）](#13-关键架构决策adr-摘要)
14. [风险与对策](#14-风险与对策)
15. [防跑偏机制](#15-防跑偏机制)

---

## 1. 产品定义

### 1.1 核心用户流程

```
① 导入电影（本地路径；可选：外挂字幕、片名）
        │
② 自动分析（后台作业）→ 可浏览：镜头 / 台词 / 场景 / 人物 / 剧情梗概
        │
③ 确认人物命名（可选，人工）
        │
④ 选风格 / 时长 / 视角 → AI 生成文案 → 逐段编辑文案（人工）
        │
⑤ AI 配音 + 自动配画面 → 段落编辑器里调整（换镜头、调入出点、重配音、插原声）
        │
⑥ 低清预览 → 终版渲染 → 导出 mp4 / srt / 剪映草稿 / OTIO
```

### 1.2 目标（Goals）

| ID | 目标 | 可衡量标准 |
|----|------|-----------|
| G1 | 端到端可用 | 一部 2 小时电影 → 5 分钟解说成片，人工操作时间 ≤ 1 小时 |
| G2 | 每一步可看、可改、可重跑 | 所有 AI 产物都是可视化、可编辑的结构化数据 |
| G3 | 局部修改局部重算 | 改一段文案，只重算这一段的配音、选镜、渲染 |
| G4 | 模型可替换 | 换 ASR / LLM / TTS 只改配置，不改业务代码 |
| G5 | 可复现 | 同输入 + 同参数 + 同阶段版本 → 同产物（命中缓存） |

### 1.3 非目标（Non-goals）—— 明确不做

这些不是"以后再说"，而是**架构层面拒绝**。想做其中任何一项，必须先写 ADR 推翻本节。

- ❌ 多用户、登录、权限、计费、SaaS 部署
- ❌ 浏览器里的完整非线编（多轨任意拖拽、关键帧动画、特效、调色）→ 精修走导出到剪映 / 达芬奇
- ❌ 自动发布到抖音 / B 站等平台
- ❌ 规避平台版权或查重检测的"去重"处理（镜像、抽帧、加噪等）
- ❌ 训练 / 微调模型（只做推理 + 提示工程）
- ❌ 微服务、K8s、Redis / Kafka 等消息中间件、分布式部署

> 说明：本工具仅供个人学习研究；成片如需公开发布，版权合规由使用者自行判断。

---

## 2. 核心架构：六层数据 + 五个转换

这是整个系统最重要的一张图。**所有功能都必须能落到某一层数据或某一个转换上，落不上的就是跑偏。**

```
 L0  Source      源素材        电影文件 + 探测信息 + 代理文件 + 音轨
       │
       │  T1  分析 Analysis           【AI】ASR、镜头检测、视觉描述、人脸、剧情理解
       ▼
 L1  MovieIndex  影片索引      台词 / 镜头 / 场景 / 人物 / 剧情（+ 人工修订层）
       │
       │  T2  写作 Writing            【AI】大纲 → 文案 → 校验
       ▼
 L2  Script      解说文案      分段文案，每段引用场景 / 台词（版本化文档）
       │
       │  T3  编排 Planning           【AI】配音 + 候选检索 + 选镜 + 时长适配
       ▼
 L3  EditPlan    剪辑计划      每段：配音音频 + 所选镜头 + 音频策略（版本化文档）
       │
       │  T4  编译 Compile            【确定性代码】排布、对齐帧网格、生成轨道
       ▼
 L4  Timeline    时间线        多轨、绝对时间（帧）—— 渲染的唯一输入
       │
       │  T5  渲染 Render             【确定性代码】FFmpeg
       ▼
 L5  Output      成品          mp4 / srt / ass / OTIO / 剪映草稿 / 封面
```

### 2.1 五条铁律

1. **AI 只出现在 T1–T3。** T4（编译）和 T5（渲染）是纯确定性代码，零模型调用。渲染出问题，一定是数据问题或渲染代码问题，不可能是"AI 抽风"。
2. **每一层都是经 Pydantic Schema 校验的 JSON 文档。** AI 的输出必须先通过 schema 校验才能落盘；不接受自由文本或让 AI 直接生成 FFmpeg 命令。
3. **人工只在 L1、L2、L3 上编辑。** L4 Timeline 永远由编译器生成，不手改；需要精细调整就导出到专业软件。
4. **段（Segment）是系统的基本工作单元。** 一段 = 1–3 句解说（约 3–12 秒）。配音、选镜、缓存、预览渲染、局部重算全部以段为粒度。
5. **下游只认上游的产物，不认上游的过程。** 每个转换的输入是上一层的文档，不允许跨层偷读（例如渲染器不得读取 Script）。

### 2.2 人工编辑点

| 层 | 人能做什么 | 存储方式 |
|----|-----------|---------|
| L1 MovieIndex | 人物命名 / 合并 / 忽略；修正场景边界；修正台词 | **修订层（overrides）**：AI 原始产物不可变，人工修订单独存，读取时合并 |
| L2 Script | 改参数重生成；逐段改文案；单段重写；增删段 | **版本化文档**：每次修改产生新版本 |
| L3 EditPlan | 换镜头、调入出点、锁定镜头、重配音、换音色语速、插原声段、调整段顺序 | **版本化文档** |

### 2.3 局部重算示例

用户改了第 7 段文案：

```
Script v4（seg_07 文本变了）
  → EditPlan 构建：检测到 seg_07 的 text_hash 变化 → 标记为过期（stale）
      → TTS(seg_07)          缓存未命中，重新合成
      → 选镜(seg_07)         重新匹配；被锁定的镜头保留
      → 其余段               缓存全部命中，不动
  → 编译 Timeline            毫秒级，全量重算无所谓
  → 预览渲染                 只重渲 seg_07 的视频片段，其他段复用缓存
```

---

## 3. 系统组件与部署

### 3.1 部署视图（单机）

```
┌──────────────────────────── 你的电脑 ───────────────────────────────┐
│                                                                    │
│  浏览器 ──HTTP/SSE──►  API Server (FastAPI)  ◄──── CLI (Typer)       │
│  React SPA                  │                       │              │
│                             │   两者都只调用 services 层              │
│                             ▼                       ▼              │
│                     ┌──────────────── services ───────────────┐    │
│                     └─────────────┬───────────────────────────┘    │
│                                   │ 提交作业 / 读写文档               │
│                                   ▼                                │
│        SQLite (WAL)  ◄──────  Worker 进程                            │
│        元数据/作业/版本             ├─ gpu 通道（并发 1）               │
│                                  ├─ cpu 通道（并发 N）               │
│        data/ 文件系统  ◄─────     └─ api 通道（并发 M，云端模型调用）     │
│        素材/产物/缓存                     │                          │
│                                         ├─ 进程内模型（懒加载/空闲卸载）│
│                                         ├─ 模型侧车（独立环境, HTTP）   │
│                                         └─ 云端 API（LLM/TTS/…）      │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 职责 | 不做什么 |
|------|------|---------|
| **Web UI** | 浏览分析结果、编辑文案和剪辑计划、预览、发起渲染 | 不做任何媒体处理和 AI 调用 |
| **API Server** | REST + SSE；文件流（支持 Range）；提交作业 | 不执行耗时任务（> 1 秒的都进作业队列） |
| **CLI** | 每个阶段都能单独从命令行跑 | 不包含业务逻辑（和 API 一样只调 services） |
| **Worker** | 从 SQLite 作业表领取并执行阶段；管理 GPU 模型生命周期 | 不提供 HTTP 接口 |
| **模型侧车（sidecar）** | 依赖冲突严重的本地模型（如 CosyVoice、GPT-SoVITS）在独立 venv / 容器里运行，暴露 HTTP | 不含业务逻辑，只做"输入→推理→输出" |
| **SQLite** | 资产、项目、文档版本、作业、产物索引、LLM 调用日志 | 不存大文件 |
| **文件系统 data/** | 媒体、产物、缓存 | — |

### 3.3 资源调度

- **GPU 通道并发恒为 1。** 单卡机器上并发跑两个 GPU 模型只会 OOM 或互相拖慢。
- **模型懒加载 + LRU 卸载**：Worker 维护模型注册表，最多常驻 1–2 个模型，空闲 5 分钟卸载，释放显存。
- **api 通道**：按供应商分别限并发（见 §8.2 `max_concurrency`），瞬时限流指数退避重试最多 3 次；订阅额度耗尽直接失败，不重试（§8.4）。
- 没有 GPU 的机器：在配置里把 ASR / 视觉 / TTS 全切到云端适配器即可，架构不变。

---

## 4. 代码结构与依赖规则

### 4.1 目录结构

```
ai_OffScreen/
├── CLAUDE.md                  # AI 编码助手的守则（防跑偏）
├── README.md
├── Makefile                   # fmt / lint / type / test / check / api-types
├── config.example.yaml        # 配置示例（密钥只写环境变量名）
├── .env.example
├── docs/
│   ├── ARCHITECTURE.md        # 本文档
│   ├── IMPLEMENTATION_PLAN.md # 实施方案与任务拆分
│   ├── PROVIDERS.md           # 模型供应商实测说明
│   ├── PARKING_LOT.md         # 想法停车场
│   ├── adr/                   # 架构决策记录
│   └── schemas/               # 由代码导出的 JSON Schema（生成物，勿手改）
├── backend/
│   ├── pyproject.toml
│   ├── src/offscreen/
│   │   ├── domain/            # 纯数据契约（Pydantic），零 IO、零第三方依赖（pydantic 除外）
│   │   │   ├── common.py      #   ID、TimeMs、TimeRange、Rational、Versioned 基类
│   │   │   ├── asset.py       #   L0
│   │   │   ├── index.py       #   L1：Transcript / Shot / Caption / Face / Scene / Character / Story
│   │   │   ├── script.py      #   L2
│   │   │   ├── plan.py        #   L3
│   │   │   ├── timeline.py    #   L4
│   │   │   └── job.py
│   │   ├── algo/              # 纯算法（无 IO，最易测）：时长适配、选镜打分、字幕断行、混音包络、场景边界启发式
│   │   ├── engine/            # 流水线引擎：Stage 基类、缓存键、ensure()、ArtifactStore
│   │   ├── stages/
│   │   │   ├── analysis/      #   T1：ingest, proxy, shots, keyframes, transcript, stems, captions, faces, embeddings, scenes, story
│   │   │   ├── creation/      #   T2/T3：script, tts, candidates, match, plan_build
│   │   │   └── output/        #   T4/T5：compile, render, export
│   │   ├── providers/
│   │   │   ├── ports.py       #   接口：LLM / ASR / TTS / Embedder / FaceAnalyzer / ShotDetector / Separator
│   │   │   ├── registry.py    #   按配置实例化适配器
│   │   │   └── adapters/      #   anthropic_llm, openai_compat_llm, faster_whisper, edge_tts, cosyvoice_http, …
│   │   ├── media/             # ffmpeg/ffprobe 封装、滤镜图构建、音频读写（唯一允许调用 ffmpeg 的地方）
│   │   ├── prompts/           # *.j2 提示词模板，每个文件头部带版本号
│   │   ├── store/             # SQLite 仓储、版本化文档库、文件布局
│   │   ├── services/          # 用例编排（API 与 CLI 共用）：AnalyzeAsset、GenerateScript、BuildPlan、Render…
│   │   ├── worker/            # 作业队列、通道调度、模型生命周期
│   │   ├── server.py          # 组装层：API + Worker 一个进程，托管前端静态文件（ADR-0003）
│   │   ├── api/               # FastAPI 路由（薄层）
│   │   └── cli.py             # Typer CLI（薄层）
│   └── tests/
│       ├── fixtures/          # 各层 JSON 示例 + 30 秒测试片段
│       ├── unit/
│       └── e2e/
├── frontend/
│   └── src/
│       ├── api/               # 由 OpenAPI 生成的类型和客户端（勿手改）
│       ├── pages/             # Library / Analysis / Script / Plan / Render / Jobs / Settings
│       ├── components/        # Player / ShotStrip / TranscriptPanel / SegmentRow / CandidateDrawer / Trimmer …
│       └── stores/
├── sidecars/                  # 独立环境的本地模型服务（cosyvoice/ 等）
└── data/                      # 运行时数据（gitignore）
```

### 4.2 依赖方向（由 import-linter 在 CI 中强制）

```
        cli
         │
         ▼
      server       ← 组装层：把 api 与 worker 接成一个进程（ADR-0003）
         │
         ▼
   api   /   worker
        │
        ▼
     services
        │
        ▼
     stages ──────────► algo, media, prompts, providers.ports
        │
        ▼
     engine ──► store
        │
        ▼
     domain   ◄── 所有层都可以依赖 domain；domain 不依赖任何内部模块
```

硬性规则：

| 规则 | 说明 |
|------|------|
| R1 | `domain` 不 import 任何内部模块，也不 import FastAPI / SQLModel / ffmpeg / 模型库 |
| R2 | `algo` 只依赖 `domain`，不做 IO |
| R3 | `stages` 只通过 `providers.ports` 使用模型，**禁止** import `providers.adapters` |
| R4 | 厂商 SDK（`anthropic`、`openai`、`faster_whisper`、`insightface`、`edge_tts`…）**只能**出现在 `providers/adapters/` |
| R5 | 只有 `media/` 可以调用 ffmpeg / ffprobe 子进程 |
| R6 | `api/`、`cli.py`、`server.py` 只调用 `services`（与下层的 `api` / `worker`），不直接调 `stages` / `store` / `engine` |
| R7 | 前端类型只能从 OpenAPI 生成，不手写后端数据结构 |

---

## 5. 数据契约（Schema）

### 5.1 通用约定

| 约定 | 规则 |
|------|------|
| **源时间** | L0–L3 中所有指向源片的时间用**整数毫秒**（`*_ms`） |
| **输出时间** | L4 Timeline 中输出位置用**输出帧号**（整数，`f0`/`f1`，左闭右开），帧率在 `output.fps`。编译器负责把一切对齐到帧网格，杜绝累计漂移 |
| **帧率** | 用有理数 `{"num": 24000, "den": 1001}`，不用浮点 23.976 |
| **ID** | `前缀_ULID`：`ast_`资产 `prj_`项目 `ln_`台词 `sh_`镜头 `sc_`场景 `ch_`人物 `seg_`段 `job_`作业 |
| **版本** | 每个文档带 `schema_version`；破坏性变更必须 +1 并提供迁移函数 |
| **JSON 写入** | 规范化序列化（键排序、UTF-8、原子写入：先写临时文件再 rename） |
| **示例** | 每个 schema 在 `tests/fixtures/` 有一份示例，测试保证 round-trip |

### 5.2 L0 MediaAsset

```json
{
  "schema_version": 1,
  "id": "ast_01J8...",
  "title": "Sintel",
  "source_path": "/movies/sintel.mkv",
  "fingerprint": "sha256:…(大小 + 首尾各 16MB 的哈希)",
  "duration_ms": 888000,
  "video": { "width": 1920, "height": 818, "fps": { "num": 24, "den": 1 }, "codec": "h264" },
  "audio": [{ "index": 0, "channels": 2, "sample_rate": 48000, "language": "eng" }],
  "subtitles_external": "/movies/sintel.srt",
  "derived": {
    "proxy": "assets/ast_01J8/proxy_540p.mp4",
    "audio_16k": "assets/ast_01J8/audio_16k.wav",
    "audio_48k": "assets/ast_01J8/audio_48k.wav"
  }
}
```

不复制原片，只记录路径和指纹；文件移动后可按指纹重新关联。

### 5.3 L1 MovieIndex

每个分析阶段输出**独立文件**（便于缓存与重跑），由 `MovieIndex` 读模型把它们合并成一个视图供下游使用。

**transcript.json**
```json
{
  "schema_version": 1, "asset_id": "ast_…", "language": "en",
  "source": "asr:faster-whisper/large-v3",
  "lines": [
    { "id": "ln_0001", "start_ms": 61240, "end_ms": 63900, "speaker": "spk_1",
      "text": "What are you doing here?",
      "words": [{ "w": "What", "start_ms": 61240, "end_ms": 61420 }] }
  ]
}
```

**shots.json**（镜头检测）+ **captions.json**（视觉描述）+ **faces.json**（人脸）以 `shot_id` 关联：
```json
{ "id": "sh_0412", "start_ms": 1234000, "end_ms": 1237100,
  "keyframes": ["kf/sh_0412_a.jpg", "kf/sh_0412_b.jpg", "kf/sh_0412_c.jpg"],
  "quality": { "sharpness": 0.82, "brightness": 0.41 } }

{ "shot_id": "sh_0412",
  "caption": "女孩跪在雪地里，抱着受伤的小龙",
  "shot_size": "close_up", "action": "抱起", "emotion": "悲伤",
  "has_onscreen_text": false, "is_credits": false }

{ "shot_id": "sh_0412",
  "faces": [{ "character_id": "ch_01", "bbox": [0.31, 0.12, 0.58, 0.66], "area_ratio": 0.18,
              "frame": 1, "score": 0.97, "embedding": 0 }] }
```

`faces` 阶段（M3-07）的产物是 `faces.json` 和 `embeddings.npy`：后者每行是一张脸的 L2 归一化特征（float32），`embedding` 是该脸在其中的行号（行序即 faces.json 中人脸的出现顺序）；`character_id` 在聚类（M3-08）之前为空。

`characters` 阶段（M3-08）把特征聚成人物（相似度合并，不依赖 HDBSCAN）：`ch_01` 是出现最多的人；太小或只在一个镜头出现的簇算路人 / 误检，不成为人物。产物：`characters.json`（人数、`centroid_ref` = `centroids.npy#行号`、最多 3 张从关键帧裁出的缩略图 `faces/<id>_<n>.jpg`，名字留空）、填了 `character_id` 的 `faces.json`、`cast.json`（每个镜头出现的人物及其占该镜头人脸面积的比例，下游只读它而不碰人脸）、`centroids.npy`。

`naming` 阶段（M3-09，任务 `character_name`，一次请求）：把每个人物的几张脸部截图、几个代表镜头（时间、描述、附近台词）和全片台词交给模型，问「这是谁」。名字必须附 `evidence`——逐字摘自台词的原句，校验不过的名字有一轮修正机会，仍不过就不保留（人物留空，由人来命名；不靠演员表或外部知识猜）。`name_source = "ai"`。产物是一份自包含的 `characters.json`（缩略图和 `centroids.npy` 一并复制）。

**人工修订**存在 `data/overrides/{asset_id}/characters.overrides.json`，AI 产物不动。每条修订（名字 / 别名 / 忽略 / 合并到）带着当时的人脸簇中心（`centroid`、`merged_into_centroid`）：重聚类会让编号变化，读取和编辑时先按簇中心相似度（余弦 ≥ 0.6，一对一）把修订重新对应到现在的人物，对不上任何人的修订原样留在文件里（`unmatched_edits`），人物回来时还能对上。读取（`GET /assets/{id}/index/characters`）时合并：人工名字优先（`name_source = "human"`），忽略的不显示，合并的并入目标并累加人脸数。

**scenes.json**
```json
{ "id": "sc_040", "start_ms": 1220000, "end_ms": 1302000,
  "shot_ids": ["sh_0405", "…", "sh_0431"], "line_ids": ["ln_0390", "…"],
  "summary": "Sintel 在雪山上找到受伤的小龙并救治它",
  "characters": ["ch_01", "ch_02"], "location": "雪山", "importance": 0.9 }
```

**characters.json**（AI 原始） + **characters.overrides.json**（人工修订）
```json
{ "id": "ch_01", "name": "Sintel", "name_source": "human",
  "aliases": ["小辛"], "role": "主角", "bio": "…",
  "face_cluster": { "size": 312, "centroid_ref": "faces/ch_01_centroid.npy",
                    "thumbnails": ["faces/ch_01_1.jpg"] } }
```

**story.json**
```json
{ "logline": "一句话剧情",
  "synopsis": "300–500 字梗概",
  "acts": [{ "name": "第一幕", "scene_ids": ["sc_001", "…"], "summary": "…" }],
  "turning_points": [{ "scene_id": "sc_040", "what": "…" }],
  "relations": [{ "a": "ch_01", "b": "ch_02", "relation": "伙伴" }],
  "ending": "…", "themes": ["成长", "失去"] }
```

所有关于剧情的陈述都必须带 `scene_id` 引用 —— 这是后续防止文案幻觉的锚点。

### 5.4 L2 Script（版本化文档）

```json
{
  "schema_version": 1, "id": "scr_…", "project_id": "prj_…",
  "version": 4, "parent_version": 3, "author": "human",
  "params": { "style": "suspense", "target_duration_s": 300, "perspective": "third",
              "spoil_ending": true, "language": "zh", "voice_id": "narrator_m1" },
  "outline": [{ "beat": "hook", "scene_refs": ["sc_040"], "target_s": 15 }],
  "segments": [
    { "id": "seg_01", "kind": "narration", "beat": "hook",
      "text": "这个女孩为了一条龙，走遍了整个世界。",
      "scene_refs": ["sc_040"] },
    { "id": "seg_02", "kind": "original", "line_refs": ["ln_0412"],
      "text": "（原声）" }
  ],
  "annotations": [{ "segment_id": "seg_05", "type": "fact_check", "message": "…" }]
}
```

- `kind = narration`：解说段，需要配音 + 选镜
- `kind = original`：原声段，直接播放电影原片该句台词的音画

### 5.5 L3 EditPlan（版本化文档）

```json
{
  "schema_version": 1, "id": "pln_…", "project_id": "prj_…",
  "version": 7, "parent_version": 6, "author": "ai",
  "script_ref": { "id": "scr_…", "version": 4 },
  "segments": [
    { "id": "seg_01", "kind": "narration",
      "text": "这个女孩为了一条龙……", "text_hash": "sha256:…", "stale": false,
      "voice": { "voice_id": "narrator_m1", "speed": 1.1 },
      "audio": { "file": "tts/9f3a….wav", "duration_ms": 5230,
                 "char_timings": [[0, 180], [180, 340]] },
      "clips": [
        { "asset_id": "ast_…", "shot_id": "sh_0412",
          "src_in_ms": 1234400, "src_out_ms": 1236900, "speed": 1.0,
          "locked": false, "score": 0.81 }
      ],
      "source_audio": { "mode": "duck", "stem": "no_vocals", "gain_db": -14 } },
    { "id": "seg_02", "kind": "original",
      "clips": [{ "asset_id": "ast_…", "src_in_ms": 1301200, "src_out_ms": 1304100, "speed": 1.0, "locked": true }],
      "source_audio": { "mode": "full", "stem": "mix", "gain_db": 0 } }
  ],
  "bgm": { "file": "bgm/tense_01.mp3", "gain_db": -22, "duck_under_narration_db": -8 },
  "output_profile": "vertical_1080"
}
```

- `text` 是配音所用文本的快照（编译器生成字幕时只读 EditPlan，见 ADR-0001）；`text_hash` 是它的哈希
- `text_hash` 与 Script 中对应段比对，不一致即 `stale`，触发该段重算
- `locked` 的镜头在自动重匹配时保留

### 5.6 L4 Timeline（编译产物，不手改）

```json
{
  "schema_version": 1,
  "plan_ref": { "id": "pln_…", "version": 7 },
  "output": { "profile": "vertical_1080", "width": 1080, "height": 1920,
              "fps": { "num": 30, "den": 1 }, "layout": "blur_pad" },
  "duration_frames": 9000,
  "video":        [{ "seg": "seg_01", "f0": 0, "f1": 75, "asset_id": "ast_…",
                     "src_in_ms": 1234400, "src_out_ms": 1236900, "speed": 1.0 }],
  "narration":    [{ "seg": "seg_01", "f0": 0, "file": "tts/9f3a….wav", "gain_db": 0 }],
  "source_audio": [{ "seg": "seg_01", "f0": 0, "f1": 75, "asset_id": "ast_…",
                     "stem": "no_vocals", "src_in_ms": 1234400, "gain_db": -14 }],
  "bgm":          [{ "f0": 0, "f1": 9000, "file": "bgm/tense_01.mp3", "gain_db": -22,
                     "duck": { "under": "narration", "depth_db": -8 } }],
  "subtitles":    [{ "f0": 0, "f1": 42, "text": "这个女孩为了一条龙" }],
  "overlays":     [{ "f0": 0, "f1": 60, "type": "title", "text": "…" }]
}
```

### 5.7 Job（作业）

```json
{ "id": "job_…", "stage": "analysis.captions", "scope": { "asset_id": "ast_…" },
  "lane": "api", "status": "running",
  "progress": 0.42, "message": "已描述 630/1500 个镜头",
  "cache_key": "sha256:…", "attempt": 1,
  "error": null, "log_path": "logs/job_….log",
  "created_at": "…", "started_at": "…", "finished_at": null }
```

状态机：`queued → running → (succeeded | failed | canceled)`；`failed` 可手动重试。

---

## 6. 流水线引擎

### 6.1 Stage 定义

```python
class Stage(Protocol):
    name: str              # "analysis.shots"
    version: int           # 输出语义变化时 +1
    lane: Lane             # gpu | cpu | api
    def inputs(self, scope) -> list[ArtifactRef]: ...      # 声明依赖
    def run(self, ctx: StageContext) -> StageOutput: ...   # 纯逻辑：读输入产物 → 写输出产物
```

`StageContext` 提供：输入产物读取、输出目录、`progress(frac, msg)`、`is_canceled()`、provider 获取、结构化日志。

### 6.2 内容寻址缓存

```
cache_key = sha256(
    stage.name, stage.version,
    [每个输入产物的内容哈希],
    规范化后的参数,
    provider 标识 + 模型名 + 提示词版本
)
```

- 产物目录：`data/artifacts/{stage}/{cache_key}/`，内含 `manifest.json`（输出文件列表 + 各自哈希）
- 命中缓存即跳过
- **下游键用上游产物的内容哈希**，而不是上游的 cache_key：上游重跑但输出完全相同时，下游缓存依然命中
- 改提示词 → 提示词版本变 → 该阶段及下游自动失效，无需手动清缓存

### 6.3 执行模型

- **分析（T1）**：静态 DAG，"拉取式"执行：`engine.ensure("analysis.story", asset_id)` 递归解析并补齐缺失的上游
- **创作（T2/T3）**：按段扇出（每段一次 TTS、一次选镜），每次调用同样走缓存
- **输出（T4/T5）**：编译全量（便宜）；渲染按段缓存

```
ingest ─► proxy ─┬─► shots ─► keyframes ─┬─► captions ──┐
                 │                       ├─► faces ─► characters ─┐
                 │                       └─► embeddings │         │
                 ├─► transcript ─────────────────────────┼─► scenes ─► story
                 └─► stems                               │
```

### 6.4 失败、取消、恢复

- api 通道：网络错误 / 429 / 5xx 自动重试，指数退避，最多 3 次
- gpu/cpu 通道：不自动重试（多半是确定性错误），失败后由人决定
- 取消：协作式，阶段在循环中检查 `ctx.is_canceled()`
- Worker 崩溃：重启时把心跳超时的 `running` 作业重置为 `queued`
- 长阶段（如视觉描述 1500 个镜头）内部分批落盘，重跑时跳过已完成批次

### 6.5 并发写冲突（乐观锁）

AI 作业和人工编辑可能同时修改同一文档。规则：写文档必须带 `base_version`；若当前版本 ≠ `base_version`，返回 409，UI 提示"文档已被更新，是否合并/覆盖"。AI 作业生成的版本同样遵守此规则。

---

## 7. 各阶段设计

> 每个阶段：输入 → 输出 → 默认实现 / 备选 → 要点。默认实现以"先能用"为准，备选在后续里程碑替换。

### 7.1 T1 分析

| 阶段 | 输入 → 输出 | 默认实现 / 备选 | 要点 |
|------|------------|----------------|------|
| **ingest** | 文件路径 → MediaAsset | ffprobe | 指纹 = 文件大小 + 首尾各 16MB 的 sha256（大文件全量哈希太慢） |
| **proxy** | 原片 → 540p 代理 + 16k 单声道 wav + 48k 立体声 wav | ffmpeg | 代理用 H.264、GOP=0.5 秒、`faststart`，保证浏览器拖动流畅 |
| **shots** | 代理 → shots.json | PySceneDetect（M1）→ TransNetV2（M3） | 后处理：合并 < 0.5s 的碎镜头；> 8s 的长镜头切成子镜头（便于选镜） |
| **keyframes** | 镜头 → 每镜头 3 帧（10% / 50% / 90%）+ 雪碧图 | ffmpeg | 同时算清晰度（拉普拉斯方差）和亮度，用于过滤糊帧和黑帧 |
| **transcript** | 音频 → transcript.json | 优先导入外挂字幕；否则 faster-whisper large-v3；云端可选 MiniMax `asr-1.0`；中文片可选 FunASR | VAD + 词级时间戳；说话人分离（pyannote）可选；MiniMax ASR 单次 ≤ 500 秒，按静音点切成 ≤ 480 秒的块再拼接 |
| **stems** | 48k 音频 → vocals.wav / no_vocals.wav | Demucs (htdemucs) | 解说时垫 `no_vocals`：保留环境声和配乐，去掉对白，效果远好于整体压低原声 |
| **captions** | 关键帧 + 附近台词 → captions.json | MiniMax-M3 图片理解（已实测）；备选 DeepSeek-V4.1-Flash、本地 Qwen-VL | 一次请求批量描述 N 个镜头；结构化输出；运行前先估算 token 用量；订阅额度有窗口限制，分批落盘可跨窗口续跑 |
| **faces** | 关键帧 → 人脸框 + 特征向量 | InsightFace（SCRFD + ArcFace） | 只在关键帧上跑，不逐帧 |
| **characters** | 人脸特征 → 人物簇 | HDBSCAN / 层次聚类（余弦距离） | 命名：LLM 根据台词上下文提议 + 可选 TMDB 演员表 → 人工确认；人工修订存 overrides，重聚类后按簇中心相似度重新映射 |
| **embeddings** | 关键帧 + 描述 → 向量索引 | 图像：Chinese-CLIP / SigLIP 多语言；文本：bge-m3；存储：LanceDB | 提供 `search_shots(text, filters)` |
| **scenes** | 镜头 + 台词 + 描述 → scenes.json | 启发式候选边界（相邻镜头视觉相似度骤降 + 对白间隙）→ LLM 滑动窗口精修 | 每个场景输出摘要、人物、地点、重要度 |
| **story** | 场景摘要 → story.json | LLM 分层归纳：场景 → 幕 / 转折点 → 梗概 | 不把全部原始台词塞进一个请求；所有陈述带 scene_id |

**上下文规模估算（2 小时电影）**：台词约 1500 行 ≈ 3 万 token；镜头描述约 1500 条 ≈ 6 万 token。长上下文模型能一次装下，但分层归纳更稳、更省钱，也能兼容上下文较短的模型。

### 7.2 T2 写作（Script）

```
参数（风格/时长/视角/是否剧透）+ story.json + 场景摘要
   │
   ├─① 大纲（Outline）：选取节拍与场景，按时长分配篇幅     → 可人工调整
   ├─② 成稿（Write）  ：逐节拍写段落，每段必须带 scene_refs
   ├─③ 规则校验（确定性）：总字数、段长、scene_refs 存在性、禁用词、人名一致 → 不过则自动修正（≤ 2 轮）
   └─④ 事实审查（LLM Critic）：对照 story/scenes 检查事实，只打标注，不自动改写 → UI 中显示
```

- **时长控制**：目标字数 = 目标秒数 × 音色语速（字/秒）。语速按所选音色实测标定（M6），默认 4.5 字/秒
- **风格预设**：YAML 文件（语气、结构模板、开头钩子类型、常用句式、禁用词、人称），不写死在代码里
- **提示词缓存**：MovieIndex 上下文作为稳定前缀，同一部片多次生成 / 单段重写时复用缓存，降低成本和延迟

### 7.3 T3 编排（EditPlan）

**配音（TTS）**
- 每段独立合成，缓存键 = 文本 + 音色 + 语速 + 引擎版本
- 合成前做文本规范化（数字、英文、符号读法）
- 需要字级时间戳：MiniMax TTS 流式模式直接返回逐字时间戳（已实测）；其他不提供时间戳的引擎，对合成音频跑一次 ASR 对齐

**选镜（Matching）—— 全系统最难的部分，分四步，前两步召回，后两步决策：**

```
① 候选生成：scene_refs 内的镜头  ∪  向量检索 top-K  ∪  人名对应人物出镜的镜头
② 打分（可配置权重，纯函数 algo/scoring.py）：
     图文相似度、描述文本相似度、人物匹配、画质（清晰/亮度）、
     景别偏好、复用惩罚、时间顺序一致性、片头片尾/屏幕文字惩罚
③ 选择：默认按分数贪心；可选 LLM 重排（给出段落文本 + top-K 镜头描述，结构化返回有序选择及理由）
④ 时长适配（纯函数 algo/fitting.py）：
     用所选镜头填满配音时长；单镜头 0.8–4 秒；
     裁剪取镜头中段；必要时变速 0.85–1.15；总时长精确到帧
```

- 原声段（`kind=original`）：按 `line_refs` 取台词时间，前后各留 200ms
- 增量：只重算 stale 段；`locked` 镜头不动

### 7.4 T4 编译（Compile）

纯函数：`compile(plan, profile) -> Timeline`

1. 按段顺序排布，所有时间对齐到输出帧网格（先算每段的帧数，再累加，误差不跨段累积）
2. 生成各轨：video / narration / source_audio / bgm / subtitles / overlays
3. 画面适配：16:9 → 9:16 提供 `blur_pad`（模糊背景填充，解说号最常用）/ `center_crop`；`smart_crop`（人脸跟随）进停车场
4. 字幕：按配音字级时间戳切行，行长按输出规格限制（竖屏约 14 字，横屏约 22 字），标点断句

### 7.5 T5 渲染（Render）

```
① 逐段渲染视频（无音频）：每段一个 filter_complex（trim → setpts → 缩放/填充 → fps 统一）
   → 统一编码参数的中间文件，按"段时间线切片 + 输出规格 + 渲染器版本"做缓存
② 音频整体混音（不分段！）：numpy 按采样点摆放配音 / 原声 stem / BGM，
   应用增益与闪避包络、片段边界 10–20ms 淡入淡出 → 一个完整 WAV
   → ffmpeg 两遍 loudnorm（默认 -14 LUFS，真峰值 ≤ -1 dBTP）
③ concat 拼接段视频（-c copy）+ 混音 + 烧录 ASS 字幕 → 终版编码（x264 / NVENC）
```

- **为什么音频不分段**：AAC 等编码有前导采样（priming），分段编码再拼接会在接缝处产生空隙和累积漂移
- 渲染档：`preview`（代理源、360/540p、ultrafast）与 `final`（原片源、1080p、高质量）
- 每次渲染写 `render_manifest.json`：Timeline 版本、各阶段版本、编码参数，用于复现

### 7.6 导出（Export）

| 格式 | 用途 | 实现 |
|------|------|------|
| mp4 | 成片 | 渲染产物 |
| srt / ass | 字幕单独交付 | 由 Timeline.subtitles 生成 |
| OTIO → FCPXML / EDL | 达芬奇 / Premiere / FCP 精修 | OpenTimelineIO，源片路径指向原片以便重新关联 |
| 剪映草稿 | 剪映精修 | pyJianYingDraft（可选） |
| cover.jpg | 封面 | LLM 选镜头 + 标题文字，Pillow 合成 |

---

## 8. 模型 Provider 抽象

> 已验证的接口细节、请求格式和各家坑点见 [`PROVIDERS.md`](./PROVIDERS.md)。

### 8.1 接口（`providers/ports.py`）

| 端口 | 方法（示意） | 默认适配器 | 备选 |
|------|-------------|-----------|------|
| `LLM` | `generate(messages, schema: type[BaseModel], images=None) -> BaseModel` | OpenAI 兼容适配器 → **MiniMax**（MiniMax-M3，文本 + 图片理解） | 同一适配器换 base_url：DeepSeek、火山方舟（按量 Key）、本地 vLLM / Ollama；Anthropic 适配器（可选） |
| `ASR` | `transcribe(wav, language=None) -> Transcript` | faster-whisper（本地，免费，词级时间戳） | MiniMax `asr-1.0`（单次 ≤ 500 秒，需切块）、FunASR |
| `TTS` | `synthesize(text, voice, speed) -> (audio, char_timings?)` | **MiniMax** `speech-2.8-hd`（流式模式直接返回字级时间戳） | 火山豆包语音、Edge-TTS（免费兜底）、CosyVoice 侧车（音色克隆） |
| `Embedder` | `embed_images(paths)`, `embed_texts(texts)` | Chinese-CLIP / SigLIP + bge-m3 | 云端 embedding |
| `FaceAnalyzer` | `detect_and_embed(image) -> list[Face]` | InsightFace | — |
| `ShotDetector` | `detect(video) -> list[TimeRange]` | PySceneDetect | TransNetV2 |
| `Separator` | `separate(wav) -> (vocals, no_vocals)` | Demucs | UVR 系 |

### 8.2 配置：供应商与任务分开配

密钥**只从环境变量读取**，配置文件里只写变量名。完整示例见仓库根目录 `config.example.yaml`。

```yaml
# config.yaml（节选）
providers:
  minimax:                                   # Token Plan 订阅 Key
    kind: openai_compat
    base_url: https://api.minimaxi.com/v1
    api_key_env: MINIMAX_API_KEY
    max_concurrency: 2                       # 订阅套餐有 RPM/TPM 限流
  deepseek:                                  # 按量付费
    kind: openai_compat
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    max_concurrency: 4
  ark:                                       # 火山方舟：只接受「按量付费」Key，见 8.4
    kind: openai_compat
    base_url: https://ark.cn-beijing.volces.com/api/v3
    api_key_env: ARK_PAYG_API_KEY
    enabled: false

tasks:
  scene_segment:   { provider: minimax, model: MiniMax-M3 }
  story:           { provider: minimax, model: MiniMax-M3 }
  script_outline:  { provider: minimax, model: MiniMax-M3 }
  script_write:    { provider: minimax, model: MiniMax-M3 }
  script_critic:   { provider: deepseek, model: deepseek-v4-pro }   # 换一家模型审稿，减少同源偏差
  shot_caption:    { provider: minimax, model: MiniMax-M3 }         # 图片理解
  match_rerank:    { provider: minimax, model: MiniMax-M3 }

asr: { provider: faster_whisper, model: large-v3, device: cuda }
tts: { provider: minimax, model: speech-2.8-hd, default_voice: male-qn-qingse }
```

每个任务独立配置，可以随时把某个任务切到别的供应商或本地模型，业务代码不变。

### 8.3 LLM 调用规范

- **结构化输出，两档实现**：供应商支持 JSON Schema 约束时用原生能力；不支持时（如 MiniMax 的 OpenAI 兼容接口）走 **JSON 降级模式**：提示词内附 schema → 从回复中提取 JSON → Pydantic 校验 → 失败则把校验错误回传重试 1 次 → 仍失败则作业失败。对上层透明，`generate()` 永远返回校验过的对象
- **思考内容与正文分离**：推理模型的思考内容（如 MiniMax `reasoning_split: true` 返回的 `reasoning_content`）单独记日志，不参与解析
- **全量记账**：每次调用记录到 `llm_calls` 表：任务、供应商、模型、提示词版本、输入/输出 token、缓存命中 token、费用（订阅套餐记为 0 但保留 token 数）、耗时、请求/响应文件路径
- **提示词缓存**：同一部片的 MovieIndex 上下文放在提示词稳定前缀，多次生成复用（MiniMax 实测有自动缓存命中）
- **批量**：镜头描述这类大批量任务，一个请求打包多个镜头；供应商有批量接口时再走批量接口
- **预算闸门**：每个作业运行前估算 token 用量，超过配置阈值需要确认

### 8.4 订阅类 Key 的使用边界

当前云环境里的 Key 有两种性质，架构上区别对待：

| Key | 性质 | 平台是否使用 | 依据 |
|-----|------|-------------|------|
| `MINIMAX_API_KEY` | MiniMax Token Plan 订阅 Key（`sk-cp-`） | ✅ 默认使用 | 官方 FAQ：套餐覆盖文本、图像、语音；"面向个人开发者的交互式使用场景"，生产环境建议按量付费；受 5 小时窗口 + 周窗口额度和 RPM/TPM 限流约束 |
| `DEEPSEEK_API_KEY` | 按量付费 Key | ✅ 备选 / 审稿 | 普通开放平台 Key |
| `VOLC_SPEECH_API_KEY` | 火山豆包语音 Key | ✅ TTS 备选 | 普通语音服务 Key |
| `ARK_API_KEY` | 火山方舟 **Coding Plan** Key | ❌ **不接入** | 火山方舟说明：Coding Plan 额度仅限 AI 编程工具中使用，在非指定工具中使用可能被识别为滥用，导致暂停订阅或封号 |

工程约束：

1. **适配器拒绝编程套餐端点**：`base_url` 含 `/api/coding` 时启动即报错，防止误把 Coding Plan 当平台后端
2. 火山方舟如需接入（例如用豆包视觉模型），使用控制台创建的**按量付费 Key**，放在单独的环境变量 `ARK_PAYG_API_KEY`，不复用 `ARK_API_KEY`
3. 订阅额度耗尽（区别于瞬时限流 429）时，作业直接失败并提示"额度耗尽、预计刷新时间"，不做重试风暴；瞬时限流按指数退避重试
4. MiniMax 的音色快速复刻、音色设计不在 Token Plan 内 → 音色克隆走 CosyVoice 侧车或按量 Key

---

## 9. 存储设计

### 9.1 文件布局

```
data/
├── offscreen.db                       # SQLite（WAL 模式）
├── assets/{asset_id}/
│   ├── proxy_540p.mp4
│   ├── audio_16k.wav
│   └── audio_48k.wav
├── artifacts/{stage}/{cache_key}/      # 所有阶段产物（内容寻址，不可变）
│   ├── manifest.json
│   └── …
├── overrides/{asset_id}/               # L1 人工修订层
│   └── characters.overrides.json
├── projects/{project_id}/
│   ├── docs/script/v{n}.json           # 版本化文档（不可变）
│   ├── docs/plan/v{n}.json
│   ├── renders/{render_id}/            # 成片 + render_manifest.json
│   └── exports/
├── library/bgm/                        # 背景音乐库
├── library/voices/                     # 音色参考音频（克隆用）
├── llm/                                # LLM 请求/响应原文（按日期分目录）
└── logs/
```

### 9.2 SQLite 表

| 表 | 主要字段 |
|----|---------|
| `assets` | id, title, source_path, fingerprint, probe_json, created_at |
| `projects` | id, asset_id, name, options_json（时长 / 音色 / 风格 / 是否剧透）, created_at；`current_script_version`、`current_plan_version` 随版本化文档（M4 / M5）加入 |
| `documents` | id, project_id, kind(script/plan), version, parent_version, author(ai/human), path, created_at |
| `artifacts` | cache_key, stage, stage_version, scope, path, content_hash, size, created_at, last_used_at |
| `jobs` | id, stage, scope_json, lane, status, progress, message, cache_key, attempt, error, log_path, cancel_requested, not_before（重试退避）, heartbeat_at, 时间戳 |
| `llm_calls` | id, job_id, stage, asset_id, task, provider, model, prompt_version, in_tokens, out_tokens, cached_tokens, cost_usd, latency_ms, req_path, resp_path；`stage` / `asset_id` 由引擎绑定的运行上下文填入，用于按阶段归集费用 |
| `stage_runs` | id, asset_id, stage, cache_key, job_id, status(ok/error/canceled), error, duration_ms, started_at；每次**真正执行**的阶段一行（缓存命中不记），分析报告的耗时来源 |

- 文档内容存文件，数据库只存索引和指针
- 缓存清理：`offscreen gc` 按 `last_used_at` 和容量上限做 LRU，被当前文档引用的产物不清

---

## 10. API 设计

REST 资源风格；所有耗时操作返回 `job_id`。**所有路径都挂在 `/api` 下**（下表省略前缀，见 ADR-0002）；`/api` 之外留给前端页面与静态资源。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/assets` | 按本地路径导入（限定在配置的媒体根目录内） |
| GET | `/assets`, `/assets/{id}` | 列表 / 详情（含各分析阶段状态） |
| POST | `/assets/{id}/analyze` | 提交分析（可指定目标阶段） |
| GET | `/assets/{id}/report` | 分析报告：各阶段耗时、模型调用与 token / 费用、镜头 / 台词 / 场景 / 人物数量 |
| GET | `/assets/{id}/index/{part}` | 读 transcript / shots / scenes / story（四个独立路由，各有类型；未构建 404）。`shots` 是展示视图：关键帧、雪碧图位置、镜头描述、代理视频路径（相对 data 目录，经 `/files` 取）。`characters` 是已合并人工修订的视图（名字、忽略、合并；见 §7 命名与修订） |
| PATCH | `/assets/{id}/characters/{cid}` | 改名 / 合并 / 忽略（写 overrides） |
| GET | `/assets/{id}/shots/search?q=` | 文本检索镜头 |
| POST | `/projects` | 新建项目（绑定一个资产） |
| POST | `/projects/{id}/script:generate` | 生成文案 → job |
| GET / PUT | `/projects/{id}/script[?version=]` | 读 / 保存新版本（需 `base_version`） |
| POST | `/projects/{id}/script/segments/{sid}:rewrite` | 单段重写（带指令） |
| POST | `/projects/{id}/plan:build` | 构建 / 增量更新剪辑计划 → job |
| GET / PATCH | `/projects/{id}/plan` | 读 / 编辑操作（换镜、裁剪、锁定、重排…，需 `base_version`） |
| GET | `/projects/{id}/plan/segments/{sid}/candidates` | 该段候选镜头及分数 |
| POST | `/projects/{id}/plan/segments/{sid}:preview` | 单段快速预览渲染 → job |
| POST | `/projects/{id}/render` | `profile=preview` 或 `final` → job |
| POST | `/projects/{id}/export` | `format=srt / ass / otio / fcpxml / jianying` → job |
| GET | `/jobs`, `/jobs/{id}`, `/jobs/{id}/log` | 作业查询 |
| POST | `/jobs/{id}:cancel`, `/jobs/{id}:retry` | 作业控制 |
| GET | `/events` | SSE：作业进度、文档版本变化 |
| GET | `/files/{path}` | 静态文件（支持 HTTP Range；限定在 data/ 内） |

---

## 11. 前端设计

### 11.1 页面

| 页面 | 核心功能 |
|------|---------|
| **素材库** | 导入（输入路径 / 浏览媒体根目录）、资产列表、分析进度 |
| **影片分析** | 代理播放器 + 镜头缩略图条（虚拟滚动）+ 台词面板（随播放高亮、点击跳转）+ 场景列表 + 人物面板（命名/合并/忽略） |
| **文案编辑** | 参数表单 → 生成；大纲视图；段落列表（编辑文本、类型、场景引用芯片可点击预览）；审查标注；字数 / 预估时长；版本历史与对比 |
| **剪辑计划（核心页）** | 每段一行：文本 / 配音试听 / 镜头缩略条；候选抽屉（top-K + 分数，点选替换）；入出点裁剪器（代理片上）；锁定；重配音；换音色语速；插入原声段；段落排序；单段预览 |
| **渲染导出** | 选规格、进度与预计剩余时间、历史渲染、下载、导出格式 |
| **作业中心** | 队列、进度、日志、重试、取消 |
| **设置** | Provider 与密钥、按任务的模型配置、音色库、风格预设、BGM 库 |

### 11.2 原则

- **文稿驱动，而不是轨道驱动**：主编辑界面以"段"为行，而不是 NLE 式的时间轨。这是避免陷入"做一个浏览器版剪映"的关键约束
- 预览：单段预览由服务端快速渲染（360p，几秒内完成，带缓存），保证所见即所得；不在浏览器里拼接播放多个片段
- 类型安全：`make api-types` 从后端 OpenAPI 生成 TypeScript 类型

---

## 12. 技术选型

| 领域 | 选择 | 理由 |
|------|------|------|
| 后端语言 | Python 3.11+ | AI / 音视频生态最全 |
| 包管理 | uv | 快；lockfile；可选依赖组（`[gpu]`、`[dev]`） |
| Web 框架 | FastAPI + Pydantic v2 | Schema 即契约，自动 OpenAPI |
| ORM / DB | SQLModel + SQLite（WAL） | 单机零运维 |
| 作业队列 | 自研 SQLite 作业表 + Worker 进程 | 一百来行代码，透明可控，无需 Redis |
| CLI | Typer | 与 FastAPI 同作者风格，类型友好 |
| 媒体 | FFmpeg / ffprobe（子进程）| 事实标准 |
| 镜头检测 | PySceneDetect → TransNetV2 | 先简单后精确 |
| ASR | faster-whisper large-v3；FunASR（中文） | 本地、带词级时间戳 |
| 人声分离 | Demucs htdemucs | 质量好，易用 |
| 人脸 | InsightFace | 检测 + 特征一体 |
| 向量 | LanceDB（嵌入式） | 无服务进程，支持过滤 |
| LLM / VLM | OpenAI 兼容适配器（MiniMax 默认，DeepSeek / 火山方舟按量 / 本地模型同一适配器） | 一个适配器覆盖国内主流供应商；结构化输出由 JSON 降级模式兜底 |
| TTS | MiniMax speech-2.8（默认）→ 火山豆包语音 / Edge-TTS / CosyVoice 侧车 | 现有 Key 可直接用且自带字级时间戳；音色克隆走侧车 |
| 音频处理 | numpy + soundfile；ffmpeg loudnorm | 混音逻辑可测、可控 |
| 时间线交换 | OpenTimelineIO；pyJianYingDraft | 对接专业剪辑软件 |
| 前端 | React + TypeScript + Vite + TanStack Query + Tailwind | 主流、生态全 |
| 音频波形 | wavesurfer.js | 配音试听与裁剪 |
| 质量工具 | ruff、mypy、pytest、hypothesis、import-linter | 依赖规则自动化检查 |

---

## 13. 关键架构决策（ADR 摘要）

| ADR | 决策 | 拒绝的备选 | 理由 |
|-----|------|-----------|------|
| 001 | 单机单体 + 独立 Worker 进程；SQLite + 文件系统 | 微服务、Celery + Redis、Postgres | 个人使用，运维成本必须为零 |
| 002 | 分层文档（Script / EditPlan / Timeline）为唯一事实来源；Timeline 只由编译器生成 | 直接编辑时间轨 | 意图与实现分离，局部重算才可能 |
| 003 | AI 只输出经 schema 校验的结构化数据；T4/T5 零 AI | 让 LLM 直接生成 ffmpeg 命令 | 可调试、可复现、可人工接管 |
| 004 | 内容寻址缓存 + 阶段版本号 + 提示词版本 | 时间戳判断 / 手动清缓存 | 改什么重算什么，结果可复现 |
| 005 | 源时间用整数毫秒；输出时间用帧号 | 浮点秒 | 杜绝浮点误差与音画漂移 |
| 006 | Provider 端口 / 适配器；厂商 SDK 只在 adapters | 业务代码直接调 SDK | 模型可替换 |
| 007 | 不做浏览器 NLE；精修走 OTIO / 剪映导出 | 自研多轨时间线编辑器 | 最大的跑偏风险，投入产出比极低 |
| 008 | 文档版本不可变，每次编辑产生新版本 | 原地修改 | 可回退、可对比 AI 与人工改动 |
| 009 | CLI 先行：每个阶段先有 CLI，再有 UI | UI 先行 | 先验证效果，再做界面 |
| 010 | GPU 通道串行；模型懒加载 + 空闲卸载 | 并发 GPU 任务 | 单卡显存有限 |
| 011 | 依赖冲突的本地模型以侧车进程运行（HTTP） | 全部塞进主环境 | 保持主工程依赖干净 |
| 012 | 音频全片整体混音，不分段编码 | 分段带音频渲染再拼接 | 避免编码前导采样导致的接缝与漂移 |
| 013 | 默认模型供应商 MiniMax（Token Plan），DeepSeek 备选；火山方舟 Coding Plan Key 不接入平台 | 用 Coding Plan Key 作平台后端 | 用现有 Key 快速起步；Coding Plan 条款限定编程工具，存在封号风险（§8.4） |

详细 ADR 以后按需写在 `docs/adr/NNNN-*.md`，模板见 `docs/adr/0000-template.md`。

---

## 14. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 选镜质量差（最大风险） | 成片"画不对词" | 场景引用约束 + 人物约束 + 多信号打分 + LLM 重排；候选抽屉让人工一键替换；评估集持续度量 |
| 文案事实错误 / 幻觉 | 误导观众 | 强制 scene_refs 引用；规则校验；LLM 审查标注；人审 |
| 长片上下文过长 | 质量差、费用高 | 分层归纳；提示词缓存 |
| 本地显存不足 | 跑不动 | 云端适配器兜底；模型按需加载卸载 |
| 音画不同步 | 成片不可用 | 帧网格对齐；音频整体混音；自动化同步测试（合成哔声 + 闪帧素材检测偏移） |
| 渲染慢 | 迭代慢 | 段级缓存；预览档；NVENC |
| 模型费用 / 订阅额度失控 | 钱包、额度窗口被耗尽 | `llm_calls` 记账；运行前估算 token；缓存；多镜头打包请求；额度耗尽快速失败 |
| 订阅 Key 条款变化 | 默认供应商不可用 | 供应商与任务解耦，改配置即可切到 DeepSeek / 按量 Key |
| 依赖地狱（各模型库冲突） | 环境装不上 | 可选依赖组；冲突模型走侧车 |
| 功能蔓延 / 架构跑偏 | 项目烂尾 | 见第 15 节 |

---

## 15. 防跑偏机制

1. **本文档 + `CLAUDE.md` 是硬约束。** 改架构规则必须先写 ADR，再改代码，再更新本文档
2. **Schema 先行。** 改数据结构的顺序固定为：改 `domain/` schema → 更新 `tests/fixtures/` 示例 → 更新迁移函数 → 改生产者 → 改消费者
3. **依赖规则自动化。** import-linter 契约在 CI 里跑，违反第 4.2 节规则直接失败
4. **里程碑退出标准。** 每个里程碑有可验证的退出标准（见实施方案），不达标不开下一个
5. **想法停车场。** 新点子一律先写进 `docs/PARKING_LOT.md`，只在里程碑之间评审是否纳入
6. **落层检验。** 每个新功能必须回答"它属于哪一层数据或哪一个转换"，回答不了就不做
7. **任务卡"不做什么"。** 每个任务都明确写出范围之外的内容
8. **黄金样片回归。** 用 Blender 开源电影（Sintel、Tears of Steel，CC-BY 授权）作为固定测试素材，每个里程碑结束跑一次端到端回归
