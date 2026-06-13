from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .vectors import Vector, as_vector


@dataclass(frozen=True)
class NodeFrame:
    node_id: str
    summary_vector: Vector
    full_vector: Vector | None = None
    summary_payload: str = ""
    full_payload: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        object.__setattr__(self, "summary_vector", as_vector(self.summary_vector))
        if self.full_vector is not None:
            object.__setattr__(self, "full_vector", as_vector(self.full_vector))


@dataclass(frozen=True)
class EdgeFrame:
    src_id: str
    dst_id: str
    edge_vector: Vector
    confidence: float = 1.0
    created_at: int = 0
    updated_at: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if not self.src_id:
            raise ValueError("src_id is required")
        if not self.dst_id:
            raise ValueError("dst_id is required")
        if self.src_id == self.dst_id:
            raise ValueError("self edges are not supported in this prototype")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "edge_vector", as_vector(self.edge_vector))


@dataclass(frozen=True)
class TraversalScores:
    follow_score: float
    read_full_score: float
    include_score: float
    expand_score: float
    stop_score: float


@dataclass(frozen=True)
class TraversalDecision:
    node_id: str
    parent_id: str | None
    hop: int
    follow_score: float
    read_full_score: float
    include_score: float
    expand_score: float
    stop_score: float
    read_full: bool
    included: bool
    expanded: bool
    path: tuple[str, ...]


@dataclass(frozen=True)
class TraversalResult:
    seed_id: str
    included: tuple[TraversalDecision, ...]
    rejected: tuple[TraversalDecision, ...]
    visited: tuple[TraversalDecision, ...]

