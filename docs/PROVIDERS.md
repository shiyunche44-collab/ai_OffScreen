# 模型供应商接入说明

> 本文记录**实测过**的接口事实，供实现适配器时直接参照。架构层面的取舍见 [ARCHITECTURE §8](./ARCHITECTURE.md#8-模型-provider-抽象)。
>
> 实测日期：2026-10-04，在 Claude Code 云环境中用最小请求验证。供应商接口变化较快，实现前若与本文不符，以官方文档为准并更新本文。

---

## 1. 环境变量总览

| 变量 | Key 类型 | 实测结果 | 平台用途 |
|------|---------|---------|---------|
| `MINIMAX_API_KEY` | MiniMax Token Plan 订阅 Key（`sk-cp-` 开头），国内站 | ✅ 文本 / 图片理解 / 语音合成均成功 | **默认**：LLM、VLM、TTS |
| `DEEPSEEK_API_KEY` | DeepSeek 按量付费 Key | ✅ 模型列表可读（未发起生成请求） | 备选 LLM、审稿 |
| `VOLC_SPEECH_API_KEY` | 火山豆包语音 Key | 未测试 | TTS 备选（M6-04） |
| `ARK_API_KEY` | 火山方舟 **Coding Plan** Key（`ark-` 开头） | ⚠️ 通用端点 401；仅编程套餐端点可认证 | ❌ **不接入平台**（见 §4） |
| `ARK_PAYG_API_KEY` | 火山方舟按量付费 Key | 尚未配置 | 将来如需豆包模型时使用 |

规则：密钥只从环境变量读取；配置文件只写变量名；日志与 `llm_calls` 中不得出现密钥。

---

## 2. MiniMax（默认供应商）

### 2.1 基本信息

| 项 | 值 |
|----|----|
| Base URL（OpenAI 兼容） | `https://api.minimaxi.com/v1`（实测可用；官方文档写作 `https://api.minimax.cn/v1`） |
| Base URL（Anthropic 兼容） | `https://api.minimax.cn/anthropic`（未实测） |
| 国际站 | `api.minimax.io` 对该 Key 返回 401，说明 Key 属于国内站 |
| 鉴权 | `Authorization: Bearer $MINIMAX_API_KEY` |

### 2.2 可用文本模型（`GET /v1/models` 实测）

`MiniMax-M3`、`MiniMax-M2.7`、`MiniMax-M2.7-highspeed`、`MiniMax-M2.5`、`MiniMax-M2.5-highspeed`、`MiniMax-M2.1`、`MiniMax-M2.1-highspeed`、`MiniMax-M2`

| 模型 | 上下文 | 说明 |
|------|-------|------|
| `MiniMax-M3` | 1M | 原生多模态（图片理解已实测）；平台默认 |
| `MiniMax-M2.7(-highspeed)` | 204.8K | 纯文本 |
| `MiniMax-M3.1-Flash-Preview` | 1M | 文档列出、支持图片 + **视频**理解；当前 Key 的模型列表里没有，暂不使用 |

### 2.3 对话 `POST /v1/chat/completions`

```json
{
  "model": "MiniMax-M3",
  "max_tokens": 4096,
  "reasoning_split": true,
  "thinking": { "type": "disabled" },
  "messages": [{ "role": "user", "content": "…" }]
}
```

- `reasoning_split: true`：思考内容放进 `reasoning_content`，`content` 保持干净，便于解析 JSON
- `thinking`：`MiniMax-M3` 可 `disabled`；M2.x 无法关闭（传了不报错但不生效）；M3.1-Flash 强制开启，传 `disabled` 报错
- 文档中**没有** `response_format` / JSON Schema 约束 → 适配器使用 JSON 降级模式（ARCHITECTURE §8.3）。实测"只输出 JSON"的指令能被正确遵守
- 响应 `usage.prompt_tokens_details.cached_tokens` 有值 → 存在自动前缀缓存（M1-07 实测：约 240 token 的请求命中 128）
- 错误可能以 HTTP 200 + `base_resp.status_code` 返回。适配器按下列码分类（来自官方错误码表，**未实测触发**）：额度类 `1008`（余额不足）、`2056`（套餐窗口用量耗尽）；鉴权 `1004`、`2049`；瞬时 `1000/1001/1002/1013/1033/1039/1041`

### 2.4 图片理解（同一接口）

```json
{ "role": "user", "content": [
  { "type": "text", "text": "用一句中文描述这张图" },
  { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,…" } }
]}
```

实测：320×180 测试卡图片，`MiniMax-M3` 正确描述为"电视测试卡（彩条、计时器）"。base64 data URL 可用，无需先上传文件。

**一个请求多张图**（M3-06，Sintel 前 8 个镜头，每镜头 3 张 ≤360 px 高的关键帧，一条 user 消息里 24 张图 + 约 700 字提示）：

| 项 | 实测 |
|----|------|
| 输入 / 输出 token | 10 390 / 540（约 1 300 输入 token / 镜头；每张图约 380 token） |
| 延迟 | 42 秒 / 请求（短请求 2 张图约 4 秒）。批次内按顺序请求，210 个镜头约 27 个请求 ≈ 19 分钟 |
| 结构化输出 | JSON 降级模式稳定：8 个 shot_id 全部照抄，枚举值合法，无多余字段 |
| 画面内容 | 片头标题卡被正确标成 `is_credits` 且读出了画面文字（`THE BLENDER FOUNDATION PRESENTS` 等）；景别、动作、情绪合理 |

`algo/shot_captions.py` 的 token 估算按每张图 500 token 计，比实测偏高约 25%（有意偏保守）。

**整片实测**（Sintel 14 分 48 秒、210 个镜头，M3-06 / M3-11，全部 `MiniMax-M3`）：

| 任务 | 请求数 | 输入 / 输出 token | 平均延迟 | 备注 |
|------|-------|------------------|---------|------|
| `shot_caption`（镜头描述） | 28（27 批 + 1 次试探） | 287 772 / 14 330 | 21 秒（并发 1） | 估算 354 698，实际低 19%；全片约 28 分钟（含阶段内顺序等待） |
| `scene_segment`（场景边界 5 个窗口 + 摘要 4 批） | 9 | 13 583 / 1 833 | 5.6 秒 | 纯文本，全程约 50 秒；29 个候选切点 → 15 个场景 |

0 次重试、0 次 schema 修复失败。两小时电影按镜头数线性外推：镜头描述约 2 小时 / 约 2.1M token（所以并发批次和 `confirm_above_tokens` 确认都值得做，见 PARKING_LOT）。

### 2.5 语音合成 `POST /v1/t2a_v2`

```json
{
  "model": "speech-2.8-hd",
  "text": "这个女孩，走遍了世界。",
  "stream": true,
  "voice_setting": { "voice_id": "male-qn-qingse", "speed": 1, "vol": 1, "pitch": 0 },
  "audio_setting": { "sample_rate": 32000, "format": "mp3", "channel": 1 },
  "subtitle_enable": true,
  "subtitle_type": "word"
}
```

- 可选模型：`speech-2.8-hd` / `speech-2.8-turbo` / `speech-2.6-hd` / `speech-2.6-turbo`（及更旧的 02 / 01 系列）
- 音频以 **hex 字符串**返回（`data.audio`），需 `bytes.fromhex()` 解码
- `extra_info.audio_length`：音频时长（毫秒）；`extra_info.usage_characters`：计费字符数
- **必须用流式（`stream: true`）取时间戳**：
  - 流式：SSE `data:` 事件里的 `data.subtitle.timestamped_words` 直接给出逐字时间戳（毫秒，浮点）与字符偏移 `word_begin` / `word_end` —— 正好对应 EditPlan 的 `char_timings`
  - 非流式：时间戳以 `subtitle_file` URL 返回，文件托管在阿里云 OSS（`*.aliyuncs.com`），**云环境网络策略会拦截**，本地机器可访问。统一用流式，避免依赖这个域名
- 支持 `pronunciation_dict` 自定义读音（多音字、英文）→ 文本规范化（M6-02）可利用
- 支持语气词标签，如 `(laughs)`

流式字幕片段示例（实测）：

```json
{ "text": "这个女孩，走遍了世界。", "time_begin": 0, "time_end": 2372.9,
  "timestamped_words": [
    { "word": "这", "time_begin": 42.7,  "time_end": 170.7, "word_begin": 0, "word_end": 1 },
    { "word": "个", "time_begin": 170.7, "time_end": 341.3, "word_begin": 1, "word_end": 2 } ] }
```

流式响应的实测细节（2026-10-04，M1-10 适配器按此实现）：

- 每个 SSE 事件的 `data.status`：`1` 为中间分块，`2` 为最后一个事件
- **最后一个事件（`status: 2`）带有完整音频**（字节数等于 `extra_info.audio_size`）以及 `data.subtitles`；`extra_info` 也只在这个事件里。中间分块的拼接结果与完整音频**字节不同**（少了文件头 / 尾部），所以以最后一个事件的音频为准，仅在它为空时才回退到拼接分块
- `data.subtitle`（单数）出现在中间事件里，`data.subtitles`（复数、数组）出现在最后一个事件里，内容相同；整段文本（实测 30 字）只有一个 subtitle 条目，`word_begin` / `word_end` 是整段输入文本里的全局字符偏移（按 Unicode 码点）
- 时间戳**不是逐字符**的：中日文逐字；英文按词片（`Hello` → `He` + `llo`）；数字按读法重复同一偏移（`2024` 出现 6 次，偏移都是 17–21，时间依次排开）；空格、标签没有条目。因此适配器用 `algo.tts.char_timings_from_spans` 合并同偏移条目、把多字符条目的时间均分，再补齐成「每个输入字符一个区间」
- `extra_info.audio_length`（实测 4969 ms）比最后一个词的 `time_end`（4736 ms）长，含尾部静音；时长以 `audio_length` 为准
- HTTP 200 的流里也可能出现 `base_resp.status_code != 0` 的错误事件，要按错误码分类处理
- 单次文本上限 10000 字符（官方文档）；`speed` 取值 0.5–2.0

### 2.6 语音识别 `POST /v1/speech_to_text`（未实测）

- `multipart/form-data`：`model=asr-1.0`、`file=<音频>`
- 限制：单个文件 **≤ 500 秒、≤ 50MB**，超出返回 400 / 413，不会截断
- 文档称支持流式返回、说话人分离、字幕导出
- 用法：先转 16k 单声道（或 mp3），按静音点切成 ≤ 480 秒的块，逐块识别后按偏移拼接

### 2.7 Token Plan 限制（来自官方 FAQ）

- "面向个人开发者的交互式使用场景……生产环境建议使用按量付费"
- 额度：**5 小时固定窗口 + 周窗口**，两者都要有剩余才能调用；文本、图像、语音共用同一额度池
- 限流：RPM / TPM，超出后约 1 分钟恢复，高峰期可能收紧
- 订阅 Key 与按量 Key **相互独立，不能混用**
- 不覆盖：MiniMax H3 视频生成、音色设计、音色快速复刻

对实现的要求：

- api 通道对 MiniMax 限并发 2（可配）
- 429 → 指数退避重试；额度耗尽类错误 → 作业失败并提示刷新时间，不重试
- 镜头描述务必多镜头打包成一个请求，并分批落盘以便跨额度窗口续跑

---

## 3. DeepSeek（备选）

| 项 | 值 |
|----|----|
| Base URL | `https://api.deepseek.com`（OpenAI 兼容） |
| 鉴权 | `Authorization: Bearer $DEEPSEEK_API_KEY` |

模型（`GET /models` 实测）：

| 模型 ID | 名称 | 输入 | 上下文 |
|--------|------|------|-------|
| `deepseek-flash` | DeepSeek-V4.1-Flash | 文本 + 图片 | 1M |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 文本 | 1M |

实测（M1-07）：`deepseek-flash` 经 `/chat/completions` 返回 JSON 正常；默认就会带 `reasoning_content`（与正文分离），适配器单独记录、不参与解析。余额不足为 HTTP 402（适配器按额度耗尽处理，未实测触发）。

模型列表还声明了 `effort` 档位（low / high / max）和 Anthropic Messages 兼容能力。用途：文案事实审查（换一家模型审稿，减少同源偏差），以及 MiniMax 额度耗尽时的后备。

---

## 4. 火山方舟：为什么当前 Key 不接入

实测：

| 端点 | 结果 |
|------|------|
| `https://ark.cn-beijing.volces.com/api/v3/chat/completions` | 401 AuthenticationError |
| `https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions` | 通过认证，但豆包通用模型返回 "does not support the coding plan feature" |

结论：`ARK_API_KEY` 是 **Coding Plan（编程套餐）Key**。根据火山方舟的说明，Coding Plan 额度仅在 AI 编程工具（Claude Code、Cursor、Cline、TRAE 等）中使用，不能用于直接 API 调用；在非指定工具中使用 Base URL 和 Key，可能被识别为滥用，导致暂停订阅或封号。

因此：

1. 平台不使用 `ARK_API_KEY`，适配器拒绝任何含 `/api/coding` 的 `base_url`
2. 这把 Key 可以继续在编程工具里使用，与本平台无关
3. 如果以后想在平台里用豆包模型（例如 Seed 系列视觉模型），在方舟控制台创建**按量付费 API Key**、开通对应模型，配置到 `ARK_PAYG_API_KEY`，`providers.ark.enabled: true` 即可，代码无需改动

---

## 5. 火山豆包语音（TTS 备选，待接入）

`VOLC_SPEECH_API_KEY` 已在环境中配置，格式为 UUID。尚未实测，接入时（M6-04）先确认：接口版本、鉴权头、可用音色、是否返回字级时间戳；若不返回时间戳，走 ASR 对齐（M6-05）。

---

## 6. InsightFace（人脸检测与特征，本地）

M3-07 实测（云环境 CPU，`insightface` + `onnxruntime`，模型 `buffalo_l`）：

- 首次使用由 InsightFace 从 GitHub releases 下载模型包（约 280 MB，约 3 秒），之后在 `~/.insightface/models/`
- 加载约 13 秒（含下载），之后每张图约 0.33 秒；两小时电影约 4500 张关键帧，约 25 分钟（CPU）
- 检测 + ArcFace 特征一体：每张脸 512 维、已 L2 归一化；检测框是像素坐标，适配器换算为 0..1
- 在 scikit-image 自带的宇航员肖像上：检出 1 张脸（置信度 0.84），同一人镜像后的余弦相似度 > 0.6
- 关键帧是 540p 代理上取的缩略图，小脸（面积 < 画面 0.4%）会被阶段丢弃；动画片的脸检测效果未验证（Sintel 待测）
- 依赖不在默认安装里（和 faster-whisper 一样按需装）：`uv pip install insightface onnxruntime`

## 7. 向量模型（sentence-transformers，本地；M3-10）

- 图像：`clip-ViT-B-32`（关键帧）；查询文本：`clip-ViT-B-32-multilingual-v1`（与图像同空间，支持中文）；描述文本：`BAAI/bge-m3`。都通过 `sentence-transformers` 加载，首次使用从 Hugging Face 下载。
- **未实测**：M3-10 开发所在的云环境网络策略拒绝 huggingface.co（403，镜像站也不通），所以没有跑过真实模型，也没有测过「龙在天上飞」的检索质量；适配器只用替身模型测过，索引与融合用真实的 LanceDB 测过。要验证需要能访问模型下载的环境，装 `sentence-transformers`。
- 依赖：`lancedb` 在默认依赖里（轻量）；`sentence-transformers`（含 torch，约数 GB）按需装。

## 8. 云环境网络备注

- 可访问：`api.minimaxi.com`、`api.deepseek.com`、`ark.cn-beijing.volces.com`
- 被拦截：`*.aliyuncs.com`（MiniMax 非流式字幕文件）、`www.volcengine.com`（文档站）
- 本地运行不受这些限制；云环境中如需放行，在环境的网络策略里添加域名
