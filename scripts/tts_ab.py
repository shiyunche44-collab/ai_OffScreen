"""Blind A/B listening test between two TTS engines / voices (M6-04).

Speaks the same narration texts with both, under shuffled labels: each text gives `NN_A.wav` and
`NN_B.wav` (same format and rate for both); which engine is A is drawn per text and written to
`key.json`. Listen, note your
preference per pair in `answers.txt` (one line per text: A, B or =), then score it:

    cd backend
    uv run python ../scripts/tts_ab.py --config ../config.yaml --out ../data/ab \
        --a minimax:speech-2.8-hd:male-qn-qingse \
        --b volc_speech:seed-tts-2.0:zh_male_jieshuoxiaoming_uranus_bigtts
    uv run python ../scripts/tts_ab.py --score ../data/ab

An engine is `provider:model:voice` with `provider` from the config's `providers` (or
`cosyvoice`). Texts are normalized as plans do (algo/textnorm) and synthesized through the TTS
cache under `data_dir`, so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from offscreen.algo.calibrate import STANDARD_TEXTS
from offscreen.algo.textnorm import normalize
from offscreen.config import AppConfig, load_config
from offscreen.media.transcode import extract_audio
from offscreen.services.tts import build_tts

TEXTS: tuple[str, ...] = (
    *STANDARD_TEXTS,
    "1994年，一个叫安迪的银行家被判了两个无期徒刑。19年后，他用一把小锤子，挖穿了监狱的墙。",
    "这部片子豆瓣评分9.2，票房却只有3000万。导演说：“我拍的不是英雄，是普通人。”",
    "注意看，这个男人叫小帅。他刚刚发现，自己的妻子，竟然不是人类。",
)
SAMPLE_RATES = {"minimax": 32000, "volc_tts": 24000}
LISTEN_RATE = 24000
"""Both sides are delivered as mono 16-bit WAV at this rate: neither the file type nor the
sample rate gives an engine away."""


def engine_config(base: AppConfig, spec: str) -> tuple[AppConfig, str]:
    provider, model, voice = spec.split(":", 2)
    kind = "cosyvoice" if provider == "cosyvoice" else base.providers[provider].kind
    rate = SAMPLE_RATES.get("volc_tts" if kind == "volc_tts" else "minimax", 32000)
    tts = base.tts.model_copy(
        update={"provider": provider, "model": model, "default_voice": voice, "sample_rate": rate}
    )
    return base.model_copy(update={"tts": tts}), voice


def generate(cfg: AppConfig, a: str, b: str, out: Path, seed: int | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    engines = {}
    for label, spec in (("a", a), ("b", b)):
        ecfg, voice = engine_config(cfg, spec)
        engines[label] = (spec, build_tts(ecfg), voice)
    key: dict[str, dict[str, str]] = {}
    for n, text in enumerate(TEXTS, 1):
        spoken = normalize(text).text
        sides = ["a", "b"]
        rng.shuffle(sides)
        key[f"{n:02d}"] = {}
        for letter, side in zip("AB", sides, strict=True):
            spec, tts, voice = engines[side]
            audio = tts.synthesize(spoken, voice_id=voice)
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / f"raw.{audio.format}"
                raw.write_bytes(audio.data)
                extract_audio(
                    raw,
                    out / f"{n:02d}_{letter}.wav",
                    stream_index=0,
                    sample_rate=LISTEN_RATE,
                    channels=1,
                    duration_ms=audio.duration_ms,
                )
            key[f"{n:02d}"][letter] = spec
        print(f"{n:02d} {text}")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "texts.txt").write_text(
        "\n".join(f"{n:02d} {t}" for n, t in enumerate(TEXTS, 1)) + "\n", encoding="utf-8"
    )
    print(f"\n{len(TEXTS)} pairs in {out}; key.json holds the answers, don't open it first.")


def score(out: Path) -> None:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    lines = (out / "answers.txt").read_text(encoding="utf-8").split()
    tally: dict[str, int] = {}
    for (n, sides), pick in zip(sorted(key.items()), lines, strict=False):
        winner = "tie" if pick == "=" else sides[pick.upper()]
        tally[winner] = tally.get(winner, 0) + 1
        print(f"{n} {pick} -> {winner}")
    print({k: v for k, v in sorted(tally.items(), key=lambda kv: -kv[1])})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/ab"))
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--score", type=Path, metavar="DIR", help="score answers.txt in DIR")
    args = ap.parse_args()
    if args.score:
        score(args.score)
        return
    if not (args.a and args.b):
        ap.error("--a and --b are required")
    generate(load_config(args.config), args.a, args.b, args.out, args.seed)


if __name__ == "__main__":
    main()
