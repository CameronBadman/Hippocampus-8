from __future__ import annotations

from typing import Protocol, Sequence

from .frames import EdgeFrame, NodeFrame, TraversalScores
from .vectors import blend_vectors, clamp01, cosine01, effective_summary_vector, resize_vector


class TraversalScorer(Protocol):
    def score_edge(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edge: EdgeFrame,
        dst_node: NodeFrame,
        path_vector: Sequence[float],
        hop: int,
    ) -> TraversalScores:
        ...

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        ...


class HeuristicTraversalScorer:
    """Deterministic stand-in for the future small transformer.

    The method signatures are model-shaped: a learned implementation can use
    the same frames and return the same scores.
    """

    def score_edge(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edge: EdgeFrame,
        dst_node: NodeFrame,
        path_vector: Sequence[float],
        hop: int,
    ) -> TraversalScores:
        edge_query = resize_vector(query_vector, len(edge.edge_vector))
        current_effective = effective_node_summary(current_node)
        dst_effective = effective_node_summary(dst_node)

        query_to_dst = cosine01(query_vector, dst_effective)
        current_to_dst = cosine01(current_effective, dst_effective)
        query_to_edge = cosine01(edge_query, edge.edge_vector)
        path_to_dst = cosine01(path_vector, dst_effective)

        follow_score = clamp01(
            query_to_dst * 0.50
            + current_to_dst * 0.15
            + query_to_edge * 0.20
            + path_to_dst * 0.05
            + edge.confidence * 0.10
            - hop * 0.03
        )

        summary_include = clamp01(query_to_dst * 0.80 + query_to_edge * 0.10 + edge.confidence * 0.10)
        if dst_node.full_vector is None:
            include_score = summary_include
            read_full_score = summary_include
        else:
            full_match = cosine01(query_vector, dst_node.full_vector)
            include_score = clamp01(summary_include * 0.55 + full_match * 0.45)
            read_full_score = clamp01(summary_include * 0.75 + follow_score * 0.25)

        expand_score = clamp01(follow_score * 0.80 + current_to_dst * 0.20 - hop * 0.05)
        stop_score = clamp01(1.0 - expand_score)
        result_score = clamp01(include_score * 0.80 + read_full_score * 0.15 + follow_score * 0.05)

        return TraversalScores(
            follow_score=follow_score,
            read_full_score=read_full_score,
            include_score=include_score,
            expand_score=expand_score,
            stop_score=stop_score,
            result_score=result_score,
        )

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        if new_node.full_vector is not None and candidate_node.full_vector is not None:
            full_score = cosine01(new_node.full_vector, candidate_node.full_vector)
        else:
            full_score = cosine01(new_node.summary_vector, candidate_node.summary_vector)

        new_effective = effective_node_summary(new_node)
        candidate_effective = effective_node_summary(candidate_node)
        path_score = cosine01(path_vector, candidate_effective)
        summary_score = cosine01(new_effective, candidate_effective)
        return clamp01(summary_score * 0.60 + full_score * 0.30 + path_score * 0.10)

    def score_attach_batch(
        self,
        *,
        new_node: NodeFrame,
        candidate_nodes: Sequence[NodeFrame],
        path_vectors: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        if len(candidate_nodes) != len(path_vectors):
            raise ValueError("candidate_nodes and path_vectors must have the same length")
        return tuple(
            self.score_attach(
                new_node=new_node,
                candidate_node=candidate,
                path_vector=path_vector,
            )
            for candidate, path_vector in zip(candidate_nodes, path_vectors)
        )


def path_vector_for(nodes: Sequence[NodeFrame], dimension: int) -> tuple[float, ...]:
    return blend_vectors([effective_node_summary(node, dimension=dimension) for node in nodes], dimension)


def effective_node_summary(node: NodeFrame, *, dimension: int | None = None) -> tuple[float, ...]:
    return tuple(
        effective_summary_vector(
            node.summary_vector,
            node.metadata_vector,
            dimension=dimension or len(node.summary_vector),
        )
    )
