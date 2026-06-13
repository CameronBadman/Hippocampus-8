from __future__ import annotations

from dataclasses import dataclass

from .frames import EdgeFrame, NodeFrame
from .scorer import TraversalScorer, path_vector_for
from .store import GraphStore
from .traversal import TraversalConfig, TraversalController
from .vectors import stable_edge_vector


@dataclass(frozen=True)
class InsertConfig:
    attach_limit: int = 8
    attach_threshold: float = 0.58
    edge_dimension: int = 32
    traversal: TraversalConfig = TraversalConfig()

    def __post_init__(self) -> None:
        if self.attach_limit <= 0:
            raise ValueError("attach_limit must be positive")
        if self.edge_dimension <= 0:
            raise ValueError("edge_dimension must be positive")


def insert_node(
    *,
    store: GraphStore,
    scorer: TraversalScorer,
    node: NodeFrame,
    seed_id: str | None = None,
    config: InsertConfig | None = None,
) -> tuple[str, ...]:
    config = config or InsertConfig()
    store.add_node(node)

    if seed_id is None:
        nearest = store.find_nearest_summary(node.summary_vector, limit=2)
        seeds = [candidate for candidate in nearest if candidate.node_id != node.node_id]
        if not seeds:
            return ()
        seed_id = seeds[0].node_id

    traversal = TraversalController(store=store, scorer=scorer, config=config.traversal)
    result = traversal.traverse(query_vector=node.summary_vector, seed_id=seed_id)
    candidate_ids = {seed_id}
    candidate_ids.update(decision.node_id for decision in result.visited)
    candidate_ids.discard(node.node_id)

    scored = []
    for candidate_id in sorted(candidate_ids):
        candidate = store.get_node(candidate_id)
        path_nodes = [store.get_node(path_id) for path_id in _path_for(result, candidate_id, seed_id)]
        path_vector = path_vector_for(path_nodes, len(node.summary_vector))
        score = scorer.score_attach(new_node=node, candidate_node=candidate, path_vector=path_vector)
        if score >= config.attach_threshold:
            scored.append((score, candidate_id, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    attached_ids: list[str] = []
    for score, candidate_id, candidate in scored[: config.attach_limit]:
        forward = EdgeFrame(
            src_id=node.node_id,
            dst_id=candidate_id,
            edge_vector=stable_edge_vector(node.summary_vector, candidate.summary_vector, config.edge_dimension),
            confidence=score,
        )
        reverse = EdgeFrame(
            src_id=candidate_id,
            dst_id=node.node_id,
            edge_vector=stable_edge_vector(candidate.summary_vector, node.summary_vector, config.edge_dimension),
            confidence=score,
        )
        store.add_edge(forward)
        store.add_edge(reverse)
        attached_ids.append(candidate_id)

    return tuple(attached_ids)


def _path_for(result, node_id: str, seed_id: str) -> tuple[str, ...]:
    if node_id == seed_id:
        return (seed_id,)
    for decision in result.visited:
        if decision.node_id == node_id:
            return decision.path
    return (seed_id, node_id)

