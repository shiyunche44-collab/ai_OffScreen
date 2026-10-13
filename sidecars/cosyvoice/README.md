# CosyVoice 侧车

本地语音合成与**音色克隆**（MiniMax Token Plan 不含音色复刻）。CosyVoice 依赖 torch 和一批固定版本的库，与主工程冲突，所以单独一个环境、单独一个进程，通过 HTTP 调用（ARCHITECTURE §3、ADR 011）。侧车不含业务逻辑：文本 + 音色进，WAV 出。

平台侧适配器：`backend/src/offscreen/providers/adapters/cosyvoice_tts.py`。

> **状态**：HTTP 协议与适配器已用 `--engine fake` 端到端测试（`backend/tests/unit/test_cosyvoice.py`）。**真实 CosyVoice 引擎尚未实测**：开发所在的云环境拦截了 ModelScope、Hugging Face 和 GitHub 上的 CosyVoice 仓库。第一次在本机跑通后，请把结果补进 `docs/PROVIDERS.md` §9。

## 安装（独立环境）

```bash
# 1. CosyVoice 代码（含子模块 Matcha-TTS）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ~/cosyvoice
cd ~/cosyvoice
conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice
pip install -r requirements.txt          # 按 CosyVoice 自己的说明；Linux 上可能还需 sox / libsox-dev

# 2. 模型（推荐 CosyVoice2-0.5B，约 2 GB）
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

侧车脚本本身只用标准库，不需要额外安装。

## 启动

```bash
conda activate cosyvoice
cd ~/cosyvoice
PYTHONPATH=.:third_party/Matcha-TTS python /path/to/ai_OffScreen/sidecars/cosyvoice/server.py \
    --model-dir pretrained_models/CosyVoice2-0.5B          # 有 GPU 可加 --fp16
# listening on 127.0.0.1:50021
```

默认只监听 `127.0.0.1`。同一时刻只做一次合成，其余请求排队。

平台配置（`config.yaml`）：

```yaml
tts:
  provider: cosyvoice
  model: CosyVoice2-0.5B          # 必须与侧车模型目录名一致，否则适配器拒绝（缓存不能混用不同模型的音频）
  default_voice: narrator_m1
  cosyvoice:
    url: http://127.0.0.1:50021
    timeout_s: 600
```

## 克隆一个音色

参考录音放在 `data/library/voices/`：单人、无背景音乐、3–10 秒、WAV（16 kHz 以上、单声道最稳）。附上录音里说的原话效果更好（`zero_shot`）；不给原话则用 `cross_lingual` 模式，音色像，但语气跟录音的关系弱一些。

```bash
offscreen voice add narrator_m1 --name 我的声音 --provider cosyvoice \
    --reference narrator_m1.wav --reference-text "大家好，今天给大家讲一部电影。"
offscreen voice calibrate narrator_m1      # 实测语速，用于文案时长估算
```

音色库里没有参考录音的音色 id，按模型内置音色（`sft`）处理；CosyVoice2-0.5B 没有内置音色，`CosyVoice-300M-SFT` 有（`中文女`、`中文男` 等，见 `/health` 的 `speakers`）。

更换参考录音或原话后，同一个音色 id 的配音缓存自动失效（缓存键含录音与原话的摘要）。

## 协议（API v1）

`GET /health`

```json
{"status": "ok", "api_version": 1, "model": "CosyVoice2-0.5B", "sample_rate": 24000, "speakers": []}
```

`POST /synthesize`（JSON）

| 字段 | 说明 |
|------|------|
| `text` | 要说的话，1–2000 字（平台已做文本规范化，M6-02） |
| `speed` | 0.5–2.0，默认 1.0 |
| `speaker` | 内置音色名；与 `prompt_audio` 二选一 |
| `prompt_audio` | 参考录音（base64，WAV），克隆用 |
| `prompt_text` | 参考录音里说的话，可选，只能和 `prompt_audio` 一起给 |

成功：`200`，`audio/wav`（16 位单声道，采样率同 `/health`）。没有字级时间戳：平台在计划阶段用 ASR 转写配音后对回原文（M6-05）。
失败：`400`（请求不对，如未知音色）、`500`（推理失败；进程继续服务），正文 `{"error": "..."}`。

`--engine fake`：不加载模型，每 5 个字 1 秒的静音，用于协议测试。
