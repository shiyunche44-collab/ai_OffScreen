from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from offscreen.config import AppConfig
from offscreen.domain.common import TimeRange
from offscreen.providers.adapters.fake import FakeLLM, FakeTTS
from offscreen.services.pipeline import Providers

SRT = """1
00:00:01,000 --> 00:00:03,000
Where are you going?

2
00:00:04,000 --> 00:00:06,000
To the mountain. The dragon is there.

3
00:00:10,000 --> 00:00:12,500
She is hurt. Help me carry her.

4
00:00:17,000 --> 00:00:19,000
We cannot stay here.

5
00:00:22,000 --> 00:00:24,000
Goodbye, my friend.

6
00:00:27,000 --> 00:00:29,000
I will find you again.
"""


@pytest.fixture(scope="session")
def clip30(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 30 s film: moving test pattern with a tone, 320x180 at 23.976 fps."""
    p = tmp_path_factory.mktemp("film") / "clip30.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24000/1001:duration=30",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=30:sample_rate=48000",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p)],
        check=True,
    )  # fmt: skip
    return p


class FixedShots:
    """Four shots of a 30 s film, no detection."""

    id = "fixed-shots@1"

    def detect(
        self, video: Path, *, on_progress: Any = None, should_cancel: Any = None
    ) -> list[TimeRange]:
        return [TimeRange(start_ms=a, end_ms=b) for a, b in
                [(0, 8000), (8000, 16000), (16000, 24000), (24000, 30000)]]  # fmt: skip


def scripted_llm() -> FakeLLM:
    """Answers like a model would, from what the prompts contain: scene ids, length target."""

    def ids(messages: list[Any]) -> list[str]:
        return list(dict.fromkeys(re.findall(r"sc_\d+", messages[0].content)))

    def chunk(_t: str, _m: Any, _s: Any) -> dict[str, Any]:
        return {
            "summary": "两人在雪山相遇，一起照顾受伤的龙。",
            "characters": ["她", "他"],
            "location": "雪山",
            "importance": 0.7,
        }

    def story(_t: str, m: Any, _s: Any) -> dict[str, Any]:
        found = ids(m)
        return {
            "logline": "一个女孩寻找受伤的龙。",
            "synopsis": "女孩在雪山找到受伤的龙，并决定带它离开。",
            "acts": [{"name": "全片", "summary": "寻龙与告别", "scene_ids": found}],
            "turning_points": [{"scene_id": found[0], "what": "找到龙"}],
            "ending": "告别",
            "themes": ["陪伴"],
        }

    def script(_t: str, m: Any, _s: Any) -> dict[str, Any]:
        found = ids(m)
        target = int(re.search(r"全文约 (\d+) 字", m[0].content).group(1))  # type: ignore[union-attr]
        per = target // 3
        return {
            "segments": [
                {"beat": b, "text": "字" * (per - 1) + "。", "scene_refs": [found[i % len(found)]]}
                for i, b in enumerate(["hook", "development", "ending"])
            ]
        }

    return FakeLLM({"story_chunk": chunk, "story": story, "script_write": script})


@pytest.fixture
def fakes() -> Providers:
    return Providers(llm=scripted_llm(), tts=FakeTTS(chars_per_s=4.5), detector=FixedShots())


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path / "data"),
            "providers": {
                "p": {
                    "kind": "openai_compat",
                    "base_url": "https://x.example/v1",
                    "api_key_env": "UNUSED_KEY",
                }
            },
            "tasks": {
                t: {"provider": "p", "model": "fake-model"}
                for t in ("story_chunk", "story", "script_write")
            },
            "asr": {"provider": "faster_whisper"},
            "tts": {"provider": "edge_tts", "default_voice": "voice-x"},
        }
    )


@pytest.fixture
def movie(clip30: Path, tmp_path: Path) -> Path:
    """The film with an external subtitle file beside it (so no ASR is needed)."""
    d = tmp_path / "movies"
    d.mkdir()
    m = d / "Sample.mp4"
    m.write_bytes(clip30.read_bytes())
    m.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    return m
