"""One real shot-description request: two synthetic shots, three frames each (`pytest -m heavy`).
Skipped when the MiniMax key is not in the environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from offscreen.config import AppConfig
from offscreen.domain.index import Shot
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.adapters.openai_compat import OpenAICompatLLM
from offscreen.stages.analysis.captions import CaptionsStage

pytestmark = pytest.mark.heavy

EXAMPLE = Path(__file__).resolve().parents[3] / "config.example.yaml"


def frame(path: Path, color: str, text: str) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x180",
         "-vf", f"drawtext=text='{text}':fontcolor=white:fontsize=40:x=(w-tw)/2:y=(h-th)/2",
         "-frames:v", "1", str(path)],
        check=True,
    )  # fmt: skip


def test_two_synthetic_shots_come_back_described(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["data_dir"] = str(tmp_path)
    cfg = AppConfig.model_validate(raw)
    if not os.environ.get(cfg.providers["minimax"].api_key_env, "").strip():
        pytest.skip("MINIMAX_API_KEY is not set")
    records: list[LlmCallRecord] = []
    llm = OpenAICompatLLM(cfg, records.append)

    shots = []
    for i, (color, text) in enumerate([("red", "HELLO"), ("blue", "THE END")], 1):
        rels = []
        for c in "abc":
            rel = f"{i}{c}.jpg"
            frame(tmp_path / rel, color, text)
            rels.append(rel)
        shots.append(
            Shot(id=f"sh_{i:04d}", start_ms=(i - 1) * 3000, end_ms=i * 3000, keyframes=rels)
        )

    stage = CaptionsStage(llm, {"shot_caption": "minimax/MiniMax-M3"}, 8)
    out = stage._describe(shots, {s.id: "" for s in shots}, lambda rel: tmp_path / rel)

    assert [c.shot_id for c in out] == ["sh_0001", "sh_0002"]
    assert all(c.caption for c in out)
    assert any(c.has_onscreen_text for c in out)  # the pictures are text on a colour
    rec = records[-1]
    assert rec.status == "ok" and rec.in_tokens > 0
    print("tokens in/out:", rec.in_tokens, rec.out_tokens, "latency", rec.latency_ms, "ms")
    print([c.caption for c in out])
