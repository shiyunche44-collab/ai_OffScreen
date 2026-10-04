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
- 响应 `usage.prompt_tokens_details.cached_tokens` 有值 → 存在自动前缀缓存

### 2.4 图片理解（同一接口）

```json
{ "role": "user", "content": [
  { "type": "text", "text": "用一句中文描述这张图" },
  { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,…" } }
]}
```

实测：320×180 测试卡图片，`MiniMax-M3` 正确描述为"电视测试卡（彩条、计时器）"。base64 data URL 可用，无需先上传文件。

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

## 6. 云环境网络备注

- 可访问：`api.minimaxi.com`、`api.deepseek.com`、`ark.cn-beijing.volces.com`
- 被拦截：`*.aliyuncs.com`（MiniMax 非流式字幕文件）、`www.volcengine.com`（文档站）
- 本地运行不受这些限制；云环境中如需放行，在环境的网络策略里添加域名
