from __future__ import annotations

from typing import Protocol, Sequence

from .frames import EdgeFrame, NodeFrame, TraversalScores
from .vectors import blend_vectors, clamp01, cosine01, resize_vector


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

        query_to_dst = cosine01(query_vector, dst_node.summary_vector)
        current_to_dst = cosine01(current_node.summary_vector, dst_node.summary_vector)
        query_to_edge = cosine01(edge_query, edge.edge_vector)
        path_to_dst = cosine01(path_vector, dst_node.summary_vector)

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

        return TraversalScores(
            follow_score=follow_score,
            read_full_score=read_full_score,
            include_score=include_score,
            expand_score=expand_score,
            stop_score=stop_score,
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

        path_score = cosine01(path_vector, candidate_node.summary_vector)
        summary_score = cosine01(new_node.summary_vector, candidate_node.summary_vector)
        return clamp01(summary_score * 0.60 + full_score * 0.30 + path_score * 0.10)


def path_vector_for(nodes: Sequence[NodeFrame], dimension: int) -> tuple[float, ...]:
    return blend_vectors([node.summary_vector for node in nodes], dimension)

