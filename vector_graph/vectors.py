from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float32]


def _readonly(vector: NDArray[np.float32]) -> Vector:
    vector.setflags(write=False)
    return vector


def as_vector(values: Iterable[float] | NDArray[np.floating], *, expected_dim: int | None = None) -> Vector:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if expected_dim is not None and vector.shape[0] != expected_dim:
        raise ValueError(f"expected vector dimension {expected_dim}, got {vector.shape[0]}")
    if vector.shape[0] == 0:
        raise ValueError("vectors must not be empty")
    return _readonly(vector.copy())


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    left_array, right_array = _shared_arrays(left, right)
    return float(np.dot(left_array, right_array))


def norm(vector: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def normalize(vector: Sequence[float]) -> Vector:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(array))
    if magnitude == 0.0:
        return _readonly(np.zeros_like(array, dtype=np.float32))
    return _readonly((array / magnitude).astype(np.float32))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_array, right_array = _shared_arrays(left, right)
    if left_array.shape[0] == 0:
        return 0.0
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def cosine01(left: Sequence[float], right: Sequence[float]) -> float:
    return clamp01((cosine(left, right) + 1.0) / 2.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def resize_vector(vector: Sequence[float], dimension: int) -> Vector:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] == dimension:
        return _readonly(array.copy())

    resized = np.zeros(dimension, dtype=np.float32)
    for index, value in enumerate(array):
        resized[index % dimension] += value
    return normalize(resized)


def blend_vectors(vectors: Sequence[Sequence[float]], dimension: int) -> Vector:
    if not vectors:
        return _readonly(np.zeros(dimension, dtype=np.float32))

    blended = np.zeros(dimension, dtype=np.float32)
    for vector in vectors:
        resized = resize_vector(vector, dimension)
        blended += resized
    return normalize(blended)


def canonical_metadata_text(metadata: Mapping[str, Any]) -> str:
    """Return a stable text representation of node metadata.

    Raw vector-valued operational fields are intentionally skipped. They are
    already frame inputs, and serializing them back into metadata text would
    make the derived metadata vector noisy and expensive.
    """

    parts = []
    for raw_key in sorted(metadata.keys(), key=lambda item: str(item)):
        key = str(raw_key)
        value = metadata[raw_key]
        if _is_vector_like(value):
            continue
        normalized = _canonical_metadata_value(value)
        if normalized:
            parts.append(f"{key}={normalized}")
    return " | ".join(parts)


def metadata_vector_from(metadata: Mapping[str, Any], dimension: int = 32) -> Vector:
    return embed_text(canonical_metadata_text(metadata), dimension)


def traversal_vector_from(
    summary_vector: Sequence[float],
    metadata_vector: Sequence[float] | None = None,
    dimension: int = 16,
) -> Vector:
    if metadata_vector is None:
        return resize_vector(summary_vector, dimension)
    return blend_vectors([summary_vector, metadata_vector], dimension)


def effective_summary_vector(
    summary_vector: Sequence[float],
    metadata_vector: Sequence[float] | None = None,
    dimension: int | None = None,
    *,
    summary_weight: float = 0.78,
    metadata_weight: float = 0.22,
) -> Vector:
    target_dimension = dimension or len(np.asarray(summary_vector, dtype=np.float32).reshape(-1))
    summary = resize_vector(summary_vector, target_dimension)
    if metadata_vector is None:
        return summary
    metadata = resize_vector(metadata_vector, target_dimension)
    return normalize(summary * summary_weight + metadata * metadata_weight)


def embed_text(text: str, dimension: int = 64) -> Vector:
    """Deterministic lightweight text vector for demos and tests.

    This is not intended to be semantically strong. It gives the prototype a
    stable embedding source without adding model dependencies.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")

    vector = np.zeros(dimension, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for position, token in enumerate(tokens):
        digest = hashlib.sha256(f"{position}:{token}".encode("utf-8")).digest()
        for byte_index, byte in enumerate(digest):
            target = (byte + byte_index * 31) % dimension
            sign = 1.0 if byte % 2 == 0 else -1.0
            vector[target] += sign * (1.0 + min(len(token), 12) / 12.0)

    return normalize(vector)


def stable_edge_vector(
    src_vector: Sequence[float],
    dst_vector: Sequence[float],
    dimension: int = 32,
) -> Vector:
    """Build a compact deterministic relationship vector from two node vectors."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")

    src = resize_vector(src_vector, dimension)
    dst = resize_vector(dst_vector, dimension)
    values = np.tanh((dst - src) * 0.8 + src * dst * 0.4).astype(np.float32)
    return normalize(values)


def _shared_arrays(left: Sequence[float], right: Sequence[float]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    shared = min(left_array.shape[0], right_array.shape[0])
    return left_array[:shared], right_array[:shared]


def _canonical_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        return str(value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Mapping):
        nested = [
            f"{key}:{_canonical_metadata_value(value[raw_key])}"
            for raw_key in sorted(value.keys(), key=lambda item: str(item))
            for key in [str(raw_key)]
        ]
        return "{" + ",".join(item for item in nested if not item.endswith(":")) + "}"
    if isinstance(value, set | frozenset):
        return "[" + ",".join(sorted(_canonical_metadata_value(item) for item in value)) + "]"
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return "[" + ",".join(_canonical_metadata_value(item) for item in value) + "]"
    return str(value).strip().lower()


def _is_vector_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0 and np.issubdtype(value.dtype, np.number)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not value:
            return False
        return all(isinstance(item, int | float | np.integer | np.floating) for item in value)
    return False
