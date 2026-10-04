"""Disk cache around any TTS engine, keyed by text + voice + speed + engine id.

Layout: `{dir}/{hex}.{format}` (audio) next to `{hex}.json` (duration, timings…). A hit
costs nothing, so it reports `billed_chars=0`."""

from __future__ import annotations

import json
from pathlib import Path

from offscreen.algo.tts import tts_cache_key
from offscreen.providers.ports import TTS, SynthesizedAudio
from offscreen.store.files import atomic_write_bytes


class CachedTTS:
    def __init__(self, inner: TTS, cache_dir: Path) -> None:
        self.inner = inner
        self.dir = cache_dir
        self.hits = 0
        self.misses = 0

    @property
    def id(self) -> str:
        return self.inner.id

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        key = tts_cache_key(text, voice_id, speed, self.inner.id).removeprefix("sha256:")
        meta_path = self.dir / f"{key}.json"
        hit = self._load(key, meta_path)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        result = self.inner.synthesize(text, voice_id=voice_id, speed=speed)
        # Audio first: a metadata file only ever points at a complete audio file.
        atomic_write_bytes(self.dir / f"{key}.{result.format}", result.data)
        meta = {
            "format": result.format,
            "sample_rate": result.sample_rate,
            "duration_ms": result.duration_ms,
            "char_timings": result.char_timings,
            "billed_chars": result.billed_chars,
        }
        atomic_write_bytes(meta_path, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        return result

    def _load(self, key: str, meta_path: Path) -> SynthesizedAudio | None:
        try:
            meta = json.loads(meta_path.read_bytes())
            data = (self.dir / f"{key}.{meta['format']}").read_bytes()
            return SynthesizedAudio(
                data=data,
                format=meta["format"],
                sample_rate=meta["sample_rate"],
                duration_ms=meta["duration_ms"],
                char_timings=[(a, b) for a, b in meta["char_timings"]],
                billed_chars=0,
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None  # absent or damaged: synthesize again and overwrite
