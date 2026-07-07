"""Deterministic, domain-separated seed derivation for one environment index.

Seeds are derived directly from ``(base_seed, env_idx, stream)`` rather than
drawn from a mutable PRNG.  Consequently, adding random draws to one component
cannot perturb another component's seed (most importantly, rubric scoring).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Self

_DERIVATION_VERSION = "hypotest-determinism-v1"
_MAX_PORTABLE_SEED = (1 << 31) - 1


def derive_seed(base_seed: int, env_idx: int, stream: str) -> int:
    """Derive a stable seed for one component of one indexed environment.

    The 31-bit result is accepted by Python, NumPy's legacy global RNG, R's
    ``set.seed``, ``PYTHONHASHSEED``, and common model-provider APIs.
    Python's process-randomized ``hash()`` is deliberately not used here.
    """
    if not stream:
        raise ValueError("stream must be non-empty")
    payload = f"{_DERIVATION_VERSION}\0{base_seed}\0{env_idx}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_PORTABLE_SEED


@dataclass(frozen=True)
class EnvSeeds:
    """Independent deterministic streams reserved for an environment index."""

    kernel: int
    scheduler: int
    rubric: int

    @classmethod
    def derive(cls, base_seed: int, env_idx: int) -> Self:
        return cls(
            kernel=derive_seed(base_seed, env_idx, "kernel"),
            scheduler=derive_seed(base_seed, env_idx, "scheduler"),
            rubric=derive_seed(base_seed, env_idx, "rubric"),
        )
