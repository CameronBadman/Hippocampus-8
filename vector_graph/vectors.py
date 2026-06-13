from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

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
