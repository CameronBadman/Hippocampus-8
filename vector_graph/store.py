from __future__ import annotations

from dataclasses import replace

from .frames import EdgeFrame, NodeFrame


class GraphStore:
    def __init__(self, *, max_outgoing_edges: int = 32, max_expressway_edges: int = 128) -> None:
        if max_outgoing_edges <= 0:
            raise ValueError("max_outgoing_edges must be positive")
        if max_expressway_edges <= 0:
            raise ValueError("max_expressway_edges must be positive")
        self.max_outgoing_edges = max_outgoing_edges
        self.max_expressway_edges = max_expressway_edges
        self._nodes: dict[str, NodeFrame] = {}
        self._outgoing: dict[str, dict[str, EdgeFrame]] = {}

    def add_node(self, node: NodeFrame) -> None:
        self._nodes[node.node_id] = node
        self._outgoing.setdefault(node.node_id, {})

    def get_node(self, node_id: str) -> NodeFrame:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown node {node_id!r}") from exc

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self) -> tuple[NodeFrame, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes))

    def add_edge(self, edge: EdgeFrame) -> None:
        if edge.src_id not in self._nodes:
            raise KeyError(f"unknown edge src node {edge.src_id!r}")
        if edge.dst_id not in self._nodes:
            raise KeyError(f"unknown edge dst node {edge.dst_id!r}")

        outgoing = self._outgoing.setdefault(edge.src_id, {})
        existing = outgoing.get(edge.dst_id)
        if existing is not None:
            edge = replace(edge, created_at=existing.created_at)
        outgoing[edge.dst_id] = edge
        self.prune_edges(edge.src_id)

    def get_edges(self, src_id: str) -> tuple[EdgeFrame, ...]:
        if src_id not in self._nodes:
            raise KeyError(f"unknown node {src_id!r}")
        edges = self._outgoing.get(src_id, {})
        return tuple(
            sorted(
                edges.values(),
                key=lambda edge: (-edge.confidence, edge.created_at, edge.dst_id),
            )
        )

    def prune_edges(self, src_id: str) -> None:
        outgoing = self._outgoing.setdefault(src_id, {})
        limit = self.max_expressway_edges if self.is_expressway(src_id) else self.max_outgoing_edges
        kept = sorted(
            outgoing.values(),
            key=lambda edge: (-edge.confidence, edge.created_at, edge.dst_id),
        )[:limit]
        self._outgoing[src_id] = {edge.dst_id: edge for edge in kept}

    def is_expressway(self, node_id: str) -> bool:
        return bool(self.get_node(node_id).metadata.get("expressway", False))

    def find_nearest_summary(self, query_vector: tuple[float, ...], *, limit: int = 1) -> tuple[NodeFrame, ...]:
        from .vectors import cosine01
        from .scorer import effective_node_summary

        if limit <= 0:
            raise ValueError("limit must be positive")
        scored = [
            (cosine01(query_vector, effective_node_summary(node, dimension=len(query_vector))), node.node_id, node)
            for node in self._nodes.values()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in scored[:limit])
