"""Embeddings with sentence-transformers (local). Two kinds of model fit the architecture:

- a joint image-text model (CLIP): `image_model` embeds pictures, `text_model` embeds sentences
  into the same space, so a sentence can be compared with a picture. For Chinese queries use a
  multilingual text side, e.g. `clip-ViT-B-32` for images with `clip-ViT-B-32-multilingual-v1`
  for text;
- a text-only model (e.g. `BAAI/bge-m3`): leave `image_model` empty; images are refused.

The package (and torch) is imported lazily, so everything else works without it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Raise `VERSION` when the settings below change: it is part of the cache key.
VERSION = 1
BATCH = 32


class SentenceTransformerEmbedder:
    def __init__(
        self,
        text_model: str,
        *,
        image_model: str | None = None,
        device: str = "cpu",
        models: dict[str, Any] | None = None,
    ) -> None:
        """`models` maps a model name to a loaded SentenceTransformer-like object (tests inject
        fakes)."""
        self.text_model = text_model
        self.image_model = image_model
        self.device = device
        self._models: dict[str, Any] = dict(models or {})
        self._dim: int | None = None

    @property
    def id(self) -> str:
        images = self.image_model or "-"
        return f"sentence-transformers/{images}+{self.text_model}@{VERSION}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_texts(["."])[0])
        return self._dim

    def _model(self, name: str) -> Any:
        if name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed; install it to embed shots "
                    "(uv pip install sentence-transformers)"
                ) from e
            self._models[name] = SentenceTransformer(name, device=self.device)
        return self._models[name]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._model(self.text_model).encode(
            list(texts), batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in out]

    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        if self.image_model is None:
            raise RuntimeError(f"{self.text_model} embeds text only, not images")
        if not images:
            return []
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover - comes with sentence-transformers
            raise RuntimeError("Pillow is needed to embed images") from e
        opened = []
        try:
            for path in images:
                try:
                    opened.append(Image.open(path).convert("RGB"))
                except OSError as e:
                    raise RuntimeError(f"cannot read the image {path}") from e
            out = self._model(self.image_model).encode(
                opened, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True
            )
        finally:
            for im in opened:
                im.close()
        return [[float(x) for x in row] for row in out]
