from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, cast

from .frames import EdgeFrame, EdgeScoreContext, NodeFrame, TraversalDecision, TraversalResult, TraversalScores
from .scorer import TraversalScorer, path_vector_for
from .store import GraphStore


@dataclass(frozen=True)
class TraversalConfig:
    max_hops: int = 3
    fanout: int = 16
    beam_width: int = 32
    mode: str = "beam"
    max_visited: int = 512
    max_full_reads: int = 64
    read_full_threshold: float = 0.70
    include_threshold: float = 0.58
    expand_threshold: float = 0.52
    critical_threshold: float = 0.82
    expressway_fanout: int = 128
    expressway_threshold: float = 0.72
    max_expressway_jumps: int = 2

    def __post_init__(self) -> None:
        if self.max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if self.mode not in {"beam", "single_path"}:
            raise ValueError("mode must be 'beam' or 'single_path'")
        for field_name in (
            "fanout",
            "beam_width",
            "max_visited",
            "max_full_reads",
            "expressway_fanout",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_expressway_jumps < 0:
            raise ValueError("max_expressway_jumps must be non-negative")


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

    def traverse(
        self,
        *,
        query_vector: Sequence[float],
        seed_id: str,
        extra_seed_ids: Sequence[str] = (),
    ) -> TraversalResult:
        seed_node = self.store.get_node(seed_id)
        seed_ids = tuple(dict.fromkeys((seed_id, *sorted(extra_seed_ids))))
        frontier: list[tuple[str, tuple[str, ...], int]] = [
            (frontier_seed_id, (frontier_seed_id,), 0)
            for frontier_seed_id in seed_ids
        ]
        seen = set(seed_ids)
        decisions: dict[str, TraversalDecision] = {}
        full_reads = 0

        for hop in range(self.config.max_hops):
            if not frontier or len(decisions) >= self.config.max_visited:
                break

            candidates: list[tuple[float, float, str, TraversalDecision, int]] = []
            candidate_groups: list[tuple[tuple[str, ...], int, int, list[EdgeScoreContext]]] = []
            pending_contexts: list[EdgeScoreContext] = []
            pending_candidate_count = 0
            for current_id, path, expressway_jumps in sorted(frontier, key=lambda item: item[0]):
                current_node = self.store.get_node(current_id)
                path_nodes = [self.store.get_node(node_id) for node_id in path]
                path_vector = tuple(float(value) for value in path_vector_for(path_nodes, len(seed_node.summary_vector)))
                current_is_expressway = self.store.is_expressway(current_id)
                local_fanout = self.config.expressway_fanout if current_is_expressway else self.config.fanout
                group_contexts: list[EdgeScoreContext] = []

                for edge in self.store.get_edges(current_id):
                    if len(decisions) + pending_candidate_count + len(group_contexts) >= self.config.max_visited:
                        break
                    if edge.dst_id in seen:
                        continue
                    if len(group_contexts) >= local_fanout:
                        break

                    dst_node = self.store.get_node(edge.dst_id)
                    group_contexts.append(
                        EdgeScoreContext(
                            current_node=current_node,
                            edge=edge,
                            dst_node=dst_node,
                            path_vector=path_vector,
                            hop=hop,
                        )
                    )

                candidate_groups.append((path, expressway_jumps, local_fanout, group_contexts))
                pending_contexts.extend(group_contexts)
                pending_candidate_count += len(group_contexts)

            scores_by_context = self._score_frontier_contexts(
                query_vector=query_vector,
                candidate_groups=candidate_groups,
                contexts=pending_contexts,
            )
            score_index = 0

            for path, expressway_jumps, local_fanout, group_contexts in candidate_groups:
                node_candidates: list[tuple[float, float, str, TraversalDecision, int]] = []
                for context in group_contexts:
                    scores = scores_by_context[score_index]
                    score_index += 1
                    edge = context.edge
                    dst_node = context.dst_node
                    dst_is_expressway = self.store.is_expressway(edge.dst_id)
                    next_expressway_jumps = expressway_jumps + (1 if dst_is_expressway else 0)
                    can_use_expressway = (
                        dst_is_expressway
                        and next_expressway_jumps <= self.config.max_expressway_jumps
                        and scores.follow_score >= self.config.expressway_threshold
                    )
                    critical_score = max(scores.include_score, scores.read_full_score)
                    critical = critical_score >= self.config.critical_threshold
                    read_full = (
                        dst_node.full_vector is not None
                        and full_reads < self.config.max_full_reads
                        and scores.read_full_score >= self.config.read_full_threshold
                    )
                    if read_full:
                        full_reads += 1

                    included = scores.include_score >= self.config.include_threshold or critical
                    expanded = (
                        hop < self.config.max_hops
                        and (scores.expand_score >= self.config.expand_threshold or can_use_expressway)
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
                        result_score=scores.result_score,
                        critical_score=critical_score,
                        read_full=read_full,
                        included=included,
                        expanded=expanded,
                        critical=critical,
                        expressway=dst_is_expressway,
                        expressway_jumps=next_expressway_jumps,
                        path=path + (edge.dst_id,),
                    )
                    node_candidates.append(
                        (scores.follow_score, edge.confidence, edge.dst_id, decision, next_expressway_jumps)
                    )

                node_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
                candidates.extend(node_candidates[:local_fanout])

            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            next_frontier: list[tuple[str, tuple[str, ...], int]] = []
            effective_beam_width = 1 if self.config.mode == "single_path" else self.config.beam_width

            for _, _, node_id, decision, expressway_jumps in candidates:
                if node_id in seen:
                    continue
                seen.add(node_id)
                decisions[node_id] = decision
                if decision.expanded:
                    next_frontier.append((node_id, decision.path, expressway_jumps))
                if self.config.mode == "single_path":
                    break
                if len(next_frontier) >= effective_beam_width:
                    break

            frontier = sorted(next_frontier, key=lambda item: item[0])

        visited = tuple(decisions[node_id] for node_id in sorted(decisions))
        included = tuple(
            sorted(
                (decision for decision in visited if decision.included),
                key=lambda decision: (
                    -decision.result_score,
                    -decision.include_score,
                    -decision.follow_score,
                    decision.hop,
                    decision.node_id,
                ),
            )
        )
        rejected = tuple(decision for decision in visited if not decision.included)
        return TraversalResult(seed_id=seed_id, included=included, rejected=rejected, visited=visited)

    def _score_frontier_contexts(
        self,
        *,
        query_vector: Sequence[float],
        candidate_groups: Sequence[tuple[tuple[str, ...], int, int, Sequence[EdgeScoreContext]]],
        contexts: Sequence[EdgeScoreContext],
    ) -> tuple[TraversalScores, ...]:
        score_edge_contexts = getattr(self.scorer, "score_edge_contexts", None)
        if score_edge_contexts is not None:
            return tuple(
                cast(Callable[..., Sequence[TraversalScores]], score_edge_contexts)(
                    query_vector=query_vector,
                    contexts=contexts,
                )
            )

        scores: list[TraversalScores] = []
        for _, _, _, group_contexts in candidate_groups:
            if not group_contexts:
                continue
            first = group_contexts[0]
            scores.extend(
                self._score_edge_batch(
                    query_vector=query_vector,
                    current_node=first.current_node,
                    edges=[context.edge for context in group_contexts],
                    dst_nodes=[context.dst_node for context in group_contexts],
                    path_vector=first.path_vector,
                    hop=first.hop,
                )
            )
        return tuple(scores)

    def _score_edge_batch(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edges: Sequence[EdgeFrame],
        dst_nodes: Sequence[NodeFrame],
        path_vector: Sequence[float],
        hop: int,
    ) -> tuple[TraversalScores, ...]:
        score_edges = getattr(self.scorer, "score_edges", None)
        if score_edges is not None:
            return tuple(
                cast(Callable[..., Sequence[TraversalScores]], score_edges)(
                    query_vector=query_vector,
                    current_node=current_node,
                    edges=edges,
                    dst_nodes=dst_nodes,
                    path_vector=path_vector,
                    hop=hop,
                )
            )
        return tuple(
            self.scorer.score_edge(
                query_vector=query_vector,
                current_node=current_node,
                edge=edge,
                dst_node=dst_node,
                path_vector=path_vector,
                hop=hop,
            )
            for edge, dst_node in zip(edges, dst_nodes)
        )
