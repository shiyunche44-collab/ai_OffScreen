"""One real MiniMax synthesis (`pytest -m heavy`): a short sentence, to catch API drift.
Skipped when the key is not in the environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from offscreen.config import AppConfig
from offscreen.providers.adapters.minimax_tts import MiniMaxTTS

pytestmark = pytest.mark.heavy

EXAMPLE = Path(__file__).resolve().parents[3] / "config.example.yaml"


def test_minimax_synthesizes_with_character_timings(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["data_dir"] = str(tmp_path)
    cfg = AppConfig.model_validate(raw)
    if not os.environ.get(cfg.providers[cfg.tts.provider].api_key_env, "").strip():
        pytest.skip("MINIMAX_API_KEY is not set")
    text = "她走进雪山。Hello 2024!"
    tts = MiniMaxTTS(cfg)
    try:
        out = tts.synthesize(text, voice_id=cfg.tts.default_voice)
    finally:
        tts.close()
    assert out.format == "mp3" and out.data[:3] in (b"ID3", b"\xff\xf3", b"\xff\xfb", b"\xff\xf2")
    assert 1500 < out.duration_ms < 8000
    assert len(out.char_timings) == len(text)
    assert all(a <= b <= out.duration_ms for a, b in out.char_timings)
    assert out.billed_chars and out.billed_chars >= len(text)
