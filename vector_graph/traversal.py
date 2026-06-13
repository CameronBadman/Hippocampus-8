from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .frames import TraversalDecision, TraversalResult
from .scorer import TraversalScorer, path_vector_for
from .store import GraphStore


@dataclass(frozen=True)
class TraversalConfig:
    max_hops: int = 3
    fanout: int = 16
    beam_width: int = 32
    max_visited: int = 512
    max_full_reads: int = 64
    read_full_threshold: float = 0.70
    include_threshold: float = 0.58
    expand_threshold: float = 0.52

    def __post_init__(self) -> None:
        if self.max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        for field_name in ("fanout", "beam_width", "max_visited", "max_full_reads"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


class TraversalController:
    def __init__(
        self,
        *,
        store: GraphStore,
        scorer: TraversalScorer,
        config: TraversalConfig | None = None,
    ) -> None:
        self.store = store
        self.scorer = scorer
        self.config = config or TraversalConfig()

    def traverse(self, *, query_vector: Sequence[float], seed_id: str) -> TraversalResult:
        seed_node = self.store.get_node(seed_id)
        frontier: list[tuple[str, tuple[str, ...]]] = [(seed_id, (seed_id,))]
        seen = {seed_id}
        decisions: dict[str, TraversalDecision] = {}
        full_reads = 0

        for hop in range(self.config.max_hops):
            if not frontier or len(decisions) >= self.config.max_visited:
                break

            candidates: list[tuple[float, float, str, TraversalDecision]] = []
            for current_id, path in sorted(frontier, key=lambda item: item[0]):
                current_node = self.store.get_node(current_id)
                path_nodes = [self.store.get_node(node_id) for node_id in path]
                path_vector = path_vector_for(path_nodes, len(seed_node.summary_vector))
                node_candidates: list[tuple[float, float, str, TraversalDecision]] = []

                for edge in self.store.get_edges(current_id):
                    if len(decisions) + len(candidates) + len(node_candidates) >= self.config.max_visited:
                        break
                    if edge.dst_id in seen:
                        continue

                    dst_node = self.store.get_node(edge.dst_id)
                    scores = self.scorer.score_edge(
                        query_vector=query_vector,
                        current_node=current_node,
                        edge=edge,
                        dst_node=dst_node,
                        path_vector=path_vector,
                        hop=hop,
                    )
                    read_full = (
                        dst_node.full_vector is not None
                        and full_reads < self.config.max_full_reads
                        and scores.read_full_score >= self.config.read_full_threshold
                    )
                    if read_full:
                        full_reads += 1

                    included = scores.include_score >= self.config.include_threshold
                    expanded = (
                        hop < self.config.max_hops
                        and scores.expand_score >= self.config.expand_threshold
                        and scores.stop_score < 0.95
                    )
                    decision = TraversalDecision(
                        node_id=edge.dst_id,
                        parent_id=current_id,
                        hop=hop + 1,
                        follow_score=scores.follow_score,
                        read_full_score=scores.read_full_score,
                        include_score=scores.include_score,
                        expand_score=scores.expand_score,
                        stop_score=scores.stop_score,
                        read_full=read_full,
                        included=included,
                        expanded=expanded,
                        path=path + (edge.dst_id,),
                    )
                    node_candidates.append((scores.follow_score, edge.confidence, edge.dst_id, decision))

                node_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
                candidates.extend(node_candidates[: self.config.fanout])

            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            next_frontier: list[tuple[str, tuple[str, ...]]] = []

            for _, _, node_id, decision in candidates:
                if node_id in seen:
                    continue
                seen.add(node_id)
                decisions[node_id] = decision
                if decision.expanded:
                    next_frontier.append((node_id, decision.path))
                if len(next_frontier) >= self.config.beam_width:
                    break

            frontier = sorted(next_frontier, key=lambda item: item[0])

        visited = tuple(decisions[node_id] for node_id in sorted(decisions))
        included = tuple(decision for decision in visited if decision.included)
        rejected = tuple(decision for decision in visited if not decision.included)
        return TraversalResult(seed_id=seed_id, included=included, rejected=rejected, visited=visited)
