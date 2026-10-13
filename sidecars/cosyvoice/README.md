# CosyVoice 侧车

本地语音合成 + 音色克隆（参考音频），HTTP 服务。独立环境运行，不装进后端（ARCHITECTURE §13 决策 011：依赖冲突的本地模型以侧车进程运行）。只做「输入 → 推理 → 输出」：不缓存、不读音色库，参考音频随每个请求发来。

MiniMax Token Plan 不含音色复刻，克隆自己的声音走这里（ARCHITECTURE §8）。

## 安装（一次）

需要 NVIDIA GPU；CosyVoice 官方要求 Python 3.10。

```bash
cd sidecars/cosyvoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r vendor/CosyVoice/requirements.txt -r requirements.txt
# 模型（约 2GB）
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='models/CosyVoice2-0.5B')"
```

`vendor/`、`models/`、`.venv/` 不入库。

## 启动

```bash
cd sidecars/cosyvoice && . .venv/bin/activate
PYTHONPATH=vendor/CosyVoice:vendor/CosyVoice/third_party/Matcha-TTS \
  python server.py --model-dir models/CosyVoice2-0.5B --port 50021
```

模型加载完成后才开始监听端口。`curl http://127.0.0.1:50021/health` 返回 `{"model": "CosyVoice2-0.5B", "sample_rate": 24000, "speakers": []}`。

## 在平台里使用

`config.yaml`：

```yaml
tts:
  provider: cosyvoice
  default_voice: narrator_m1
sidecars:
  cosyvoice:
    url: http://127.0.0.1:50021
    model: CosyVoice2-0.5B      # 必须与 /health 的 model 一致
```

登记一个克隆音色（参考音频放在 `data/library/voices/` 下，5–15 秒、干净无背景音的 WAV 最好；配上录音里说的原话，克隆更像）：

```bash
offscreen voice add narrator_m1 --provider cosyvoice \
  --reference narrator_m1.wav --reference-text "这是我的声音，用来给电影解说配音。"
offscreen voice calibrate narrator_m1
```

音色库里没有参考音频的音色 id，按模型自带说话人（SFT 模型，如 CosyVoice-300M-SFT 的「中文女」）合成。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | `{"model", "sample_rate", "speakers"}` |
| POST | `/tts` | JSON：`text`（≤ 10000 字）、`speed`（0.5–2.0）、`speaker` 或 `prompt_audio`（base64 WAV）+ 可选 `prompt_text`；返回 `audio/wav`（16-bit 单声道） |

错误：400 请求无法服务（音频读不了、未知说话人），500 推理失败；正文 `{"detail": "..."}`。同一时间只跑一个合成。

CosyVoice 不返回字级时间戳：计划里的 `char_timings` 为空，字幕对齐等 M6-05（对合成音频跑 ASR 对齐）。

## 测试

`server.py` 的 HTTP 契约由后端测试 `backend/tests/unit/test_cosyvoice.py` 用假引擎覆盖（适配器 ↔ 侧车）。真实模型未在 CI 中运行。
