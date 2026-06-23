from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .frames import NodeFrame
from .vectors import Vector, as_vector, resize_vector


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
        self._node_ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._vectors: list[Vector] = []
        self._vector_matrix: NDArray[np.float32] | None = None
        self._matrix_dirty = False
        self._node_buckets: dict[str, tuple[tuple[int, int], ...]] = {}
        self._buckets: dict[tuple[int, int], set[int]] = defaultdict(set)

    def add_node(self, node: NodeFrame) -> None:
        if node.node_id in self._id_to_index:
            self.remove_node(node.node_id)

        vector = self._index_vector(self.vector_fn(node))
        buckets = self._buckets_for(vector)
        node_index = len(self._node_ids)
        self._node_ids.append(node.node_id)
        self._id_to_index[node.node_id] = node_index
        self._vectors.append(vector)
        self._matrix_dirty = True
        self._node_buckets[node.node_id] = buckets
        for bucket in buckets:
            self._buckets[bucket].add(node_index)

    def add_nodes(self, nodes: Iterable[NodeFrame]) -> None:
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node_id: str) -> None:
        node_index = self._id_to_index.pop(node_id, None)
        if node_index is None:
            return

        self._node_buckets.pop(node_id, None)
        del self._node_ids[node_index]
        del self._vectors[node_index]
        self._id_to_index = {active_node_id: index for index, active_node_id in enumerate(self._node_ids)}
        self._buckets = defaultdict(set)
        for active_node_id, buckets in self._node_buckets.items():
            active_index = self._id_to_index[active_node_id]
            for bucket in buckets:
                self._buckets[bucket].add(active_index)
        self._matrix_dirty = True

    def query(self, vector: Sequence[float], *, limit: int = 8) -> tuple[TraversalIndexHit, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_vector = self._index_vector(vector)
        bucket_matches: dict[int, int] = {}
        for bucket in self._buckets_for(query_vector):
            for node_index in self._buckets.get(bucket, ()):
                bucket_matches[node_index] = bucket_matches.get(node_index, 0) + 1

        if not bucket_matches:
            return ()

        candidate_indices = np.fromiter(bucket_matches.keys(), dtype=np.int64)
        matches = np.fromiter(
            (bucket_matches[int(node_index)] for node_index in candidate_indices),
            dtype=np.float32,
            count=len(candidate_indices),
        )
        candidate_vectors = self._matrix()[candidate_indices]
        cosine_scores = (candidate_vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1) + 1.0) / 2.0
        scores = cosine_scores * 0.75 + (matches / self.config.table_count) * 0.25
        ranked = sorted(
            zip(candidate_indices.tolist(), scores.tolist(), matches.astype(np.int32).tolist()),
            key=lambda item: (-item[1], -item[2], self._node_ids[item[0]]),
        )
        return tuple(
            TraversalIndexHit(
                node_id=self._node_ids[node_index],
                score=float(score),
                bucket_matches=int(bucket_matches),
            )
            for node_index, score, bucket_matches in ranked[:limit]
        )

    def seed_ids(self, vector: Sequence[float], *, limit: int = 8) -> tuple[str, ...]:
        return tuple(hit.node_id for hit in self.query(vector, limit=limit))

    def __len__(self) -> int:
        return len(self._id_to_index)

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

    def _build_projections(self) -> NDArray[np.float32]:
        rng = np.random.default_rng(self.config.seed)
        projections = rng.standard_normal(
            (self.config.table_count, self.config.bits_per_table, self.config.dimension),
            dtype=np.float32,
        )
        norms = np.linalg.norm(projections, axis=2, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (projections / norms).astype(np.float32)

    def _matrix(self) -> NDArray[np.float32]:
        if self._vector_matrix is None or self._matrix_dirty:
            if self._vectors:
                self._vector_matrix = np.vstack(self._vectors).astype(np.float32)
            else:
                self._vector_matrix = np.empty((0, self.config.dimension), dtype=np.float32)
            self._matrix_dirty = False
        return self._vector_matrix
