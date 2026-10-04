"""Cache key computation (ARCHITECTURE §6.2). Pure: no IO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8. Raises on non-JSON values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_cache_key(
    stage: str,
    version: int,
    input_hashes: Sequence[str],
    params: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> str:
    """`sha256:<hex>` over everything that can change a stage's output.

    `input_hashes` are the *content* hashes of upstream artifacts (not their cache keys),
    in the order the stage declared them. `provider` carries provider id, model name and
    prompt version for stages that call a model.
    """
    payload = canonical_dumps(
        {
            "stage": stage,
            "version": version,
            "inputs": list(input_hashes),
            "params": dict(params),
            "provider": dict(provider),
        }
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
