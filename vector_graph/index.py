from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .frames import NodeFrame
from .vectors import Vector, as_vector, cosine01, resize_vector


@dataclass(frozen=True)
class TraversalIndexConfig:
    dimension: int = 16
    table_count: int = 4
    bits_per_table: int = 8
    seed: int = 17

    def __post_init__(self) -> None:
        for field_name in ("dimension", "table_count", "bits_per_table"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True)
class TraversalIndexHit:
    node_id: str
    score: float
    bucket_matches: int


class TraversalIndex:
    """Deterministic LSH-style seed index over compact traversal vectors."""

    def __init__(
        self,
        *,
        config: TraversalIndexConfig | None = None,
        vector_fn: Callable[[NodeFrame], Sequence[float]] | None = None,
    ) -> None:
        self.config = config or TraversalIndexConfig()
        self.vector_fn = vector_fn or self._default_vector
        self._projections = self._build_projections()
        self._vectors: dict[str, Vector] = {}
        self._node_buckets: dict[str, tuple[tuple[int, int], ...]] = {}
        self._buckets: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add_node(self, node: NodeFrame) -> None:
        if node.node_id in self._vectors:
            self.remove_node(node.node_id)

        vector = self._index_vector(self.vector_fn(node))
        buckets = self._buckets_for(vector)
        self._vectors[node.node_id] = vector
        self._node_buckets[node.node_id] = buckets
        for bucket in buckets:
            self._buckets[bucket].add(node.node_id)

    def add_nodes(self, nodes: Iterable[NodeFrame]) -> None:
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node_id: str) -> None:
        buckets = self._node_buckets.pop(node_id, ())
        self._vectors.pop(node_id, None)
        for bucket in buckets:
            members = self._buckets.get(bucket)
            if members is None:
                continue
            members.discard(node_id)
            if not members:
                del self._buckets[bucket]

    def query(self, vector: Sequence[float], *, limit: int = 8) -> tuple[TraversalIndexHit, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_vector = self._index_vector(vector)
        bucket_matches: dict[str, int] = {}
        for bucket in self._buckets_for(query_vector):
            for node_id in self._buckets.get(bucket, ()):
                bucket_matches[node_id] = bucket_matches.get(node_id, 0) + 1

        hits = [
            TraversalIndexHit(
                node_id=node_id,
                score=self._score(query_vector, self._vectors[node_id], matches),
                bucket_matches=matches,
            )
            for node_id, matches in bucket_matches.items()
        ]
        hits.sort(key=lambda hit: (-hit.score, -hit.bucket_matches, hit.node_id))
        return tuple(hits[:limit])

    def seed_ids(self, vector: Sequence[float], *, limit: int = 8) -> tuple[str, ...]:
        return tuple(hit.node_id for hit in self.query(vector, limit=limit))

    def __len__(self) -> int:
        return len(self._vectors)

    def _default_vector(self, node: NodeFrame) -> Sequence[float]:
        traversal_vector = node.metadata.get("traversal_vector")
        if traversal_vector is not None:
            return as_vector(traversal_vector)
        return node.summary_vector

    def _index_vector(self, vector: Sequence[float]) -> Vector:
        return resize_vector(vector, self.config.dimension)

    def _buckets_for(self, vector: Sequence[float]) -> tuple[tuple[int, int], ...]:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        buckets: list[tuple[int, int]] = []
        for table_index in range(self.config.table_count):
            bucket_id = 0
            for bit_index in range(self.config.bits_per_table):
                projection = self._projections[table_index, bit_index]
                if float(np.dot(values, projection)) >= 0.0:
                    bucket_id |= 1 << bit_index
            buckets.append((table_index, bucket_id))
        return tuple(buckets)

    def _score(self, query_vector: Sequence[float], node_vector: Sequence[float], bucket_matches: int) -> float:
        match_score = bucket_matches / self.config.table_count
        return cosine01(query_vector, node_vector) * 0.75 + match_score * 0.25

    def _build_projections(self) -> NDArray[np.float32]:
        rng = np.random.default_rng(self.config.seed)
        projections = rng.standard_normal(
            (self.config.table_count, self.config.bits_per_table, self.config.dimension),
            dtype=np.float32,
        )
        norms = np.linalg.norm(projections, axis=2, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (projections / norms).astype(np.float32)
