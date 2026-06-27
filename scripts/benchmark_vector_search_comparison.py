#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import (  # noqa: E402
    EdgeFrame,
    GraphStore,
    HeuristicTraversalScorer,
    NodeFrame,
    TraversalConfig,
    TraversalController,
    TraversalIndex,
    TraversalIndexConfig,
)
from vector_graph.vectors import normalize, resize_vector, stable_edge_vector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hippo indexed graph traversal against exact/HNSW vector retrieval."
    )
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--edges-per-node", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-limit", type=int, default=4)
    parser.add_argument("--index-tables", type=int, default=4)
    parser.add_argument("--index-bits", type=int, default=14)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-visited", type=int, default=512)
    parser.add_argument("--vector-noise", type=float, default=0.10)
    parser.add_argument("--query-noise", type=float, default=0.08)
    parser.add_argument("--backend", choices=["auto", "exact", "hnsw"], default="auto")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef", type=int, default=80)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    validate_args(args)

    graph_start = time.perf_counter()
    graph = build_graph(
        node_count=args.nodes,
        cluster_count=args.clusters,
        edges_per_node=args.edges_per_node,
        index_config=TraversalIndexConfig(
            dimension=16,
            table_count=args.index_tables,
            bits_per_table=args.index_bits,
            seed=17,
        ),
        vector_noise=args.vector_noise,
        seed=args.seed,
    )
    graph_build_ms = elapsed_ms(graph_start)

    hnsw = maybe_build_hnsw(
        matrix=graph.matrix,
        backend=args.backend,
        top_k=args.top_k,
        ef=args.hnsw_ef,
        ef_construction=args.hnsw_ef_construction,
        m=args.hnsw_m,
        seed=args.seed,
    )

    scorer = load_scorer(args.checkpoint, device=args.device)
    query_dim = int(getattr(getattr(scorer, "config", None), "query_dim", 32))
    controller = TraversalController(
        store=graph.store,
        scorer=scorer,
        config=TraversalConfig(
            max_hops=args.max_hops,
            fanout=args.fanout,
            beam_width=args.beam_width,
            max_visited=args.max_visited,
            include_threshold=0.0,
            expand_threshold=0.0,
        ),
    )

    query_rng = np.random.default_rng(args.seed + 999)
    warmup_labels = query_rng.integers(0, len(graph.centers), size=args.warmup_queries)
    for label in warmup_labels:
        query = make_query(graph.centers[int(label)], rng=query_rng, noise=args.query_noise)
        _ = exact_search(graph.matrix, query, graph.node_ids, args.top_k)
        if hnsw.index is not None:
            _ = hnsw_search(hnsw.index, args.top_k, query)
        _ = graph.index.query(query, limit=args.top_k)
        _ = run_traversal(
            graph=graph,
            controller=controller,
            query=query,
            query_dim=query_dim,
            seed_limit=args.seed_limit,
            top_k=args.top_k,
        )

    query_labels = query_rng.integers(0, len(graph.centers), size=args.queries)
    rows: dict[str, list[dict[str, float]]] = {
        "exact_vector": [],
        "hippo_seed_index": [],
        "hippo_traversal": [],
    }
    if hnsw.index is not None:
        rows["hnsw_vector"] = []

    for label_value in query_labels:
        label = int(label_value)
        query = make_query(graph.centers[label], rng=query_rng, noise=args.query_noise)

        start = time.perf_counter()
        exact_indices = exact_search(graph.matrix, query, graph.node_ids, args.top_k)
        rows["exact_vector"].append(row_for_indices(exact_indices, graph.labels, label, args.top_k, elapsed_ms(start)))

        if hnsw.index is not None:
            start = time.perf_counter()
            hnsw_indices = hnsw_search(hnsw.index, args.top_k, query)
            rows["hnsw_vector"].append(row_for_indices(hnsw_indices, graph.labels, label, args.top_k, elapsed_ms(start)))

        start = time.perf_counter()
        hippo_hits = graph.index.query(query, limit=args.top_k)
        hippo_seed_indices = [graph.id_to_index[hit.node_id] for hit in hippo_hits]
        rows["hippo_seed_index"].append(
            row_for_indices(hippo_seed_indices, graph.labels, label, args.top_k, elapsed_ms(start))
        )

        start = time.perf_counter()
        traversal = run_traversal(
            graph=graph,
            controller=controller,
            query=query,
            query_dim=query_dim,
            seed_limit=args.seed_limit,
            top_k=args.top_k,
        )
        traversal_row = row_for_indices(
            traversal.indices,
            graph.labels,
            label,
            args.top_k,
            elapsed_ms(start),
        )
        traversal_row["visited"] = float(traversal.visited)
        traversal_row["included"] = float(traversal.included)
        traversal_row["seed_count"] = float(traversal.seed_count)
        rows["hippo_traversal"].append(traversal_row)

    metrics = {name: summarize_rows(method_rows) for name, method_rows in rows.items()}
    report: dict[str, Any] = {
        "benchmark": "vector_search_comparison",
        "notes": [
            "Exact vector search is the brute-force cosine upper-bound baseline.",
            "HNSW is reported only when hnswlib is installed or --backend hnsw is requested successfully.",
            "Hippo traversal uses the deterministic heuristic scorer unless --checkpoint is provided.",
        ],
        "config": {
            "nodes": args.nodes,
            "queries": args.queries,
            "warmup_queries": args.warmup_queries,
            "clusters": args.clusters,
            "edges_per_node": args.edges_per_node,
            "top_k": args.top_k,
            "seed_limit": args.seed_limit,
            "vector_noise": args.vector_noise,
            "query_noise": args.query_noise,
            "index_config": asdict(graph.index.config),
            "traversal_config": {
                "max_hops": args.max_hops,
                "fanout": args.fanout,
                "beam_width": args.beam_width,
                "max_visited": args.max_visited,
            },
            "hnsw": {
                "requested_backend": args.backend,
                "actual_backend": hnsw.backend,
                "available": hnsw.index is not None,
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef": args.hnsw_ef,
            },
            "scorer": {
                "backend": type(scorer).__name__,
                "checkpoint": args.checkpoint,
                "device": str(getattr(scorer, "device", "cpu")),
                "query_dim": query_dim,
            },
            "seed": args.seed,
        },
        "build": {
            "hippo_graph_and_index_ms": round(graph_build_ms, 3),
            "hnsw_build_ms": round(hnsw.build_ms, 3),
        },
        "metrics": metrics,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


class GraphBundle:
    def __init__(
        self,
        *,
        store: GraphStore,
        index: TraversalIndex,
        centers: np.ndarray,
        matrix: np.ndarray,
        labels: np.ndarray,
        node_ids: list[str],
    ) -> None:
        self.store = store
        self.index = index
        self.centers = centers
        self.matrix = matrix
        self.labels = labels
        self.node_ids = node_ids
        self.id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}


class HnswBundle:
    def __init__(self, *, backend: str, build_ms: float, index: Any | None) -> None:
        self.backend = backend
        self.build_ms = build_ms
        self.index = index


class TraversalOutput:
    def __init__(self, *, indices: list[int], visited: int, included: int, seed_count: int) -> None:
        self.indices = indices
        self.visited = visited
        self.included = included
        self.seed_count = seed_count


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "nodes",
        "queries",
        "clusters",
        "edges_per_node",
        "top_k",
        "seed_limit",
        "index_tables",
        "index_bits",
        "max_hops",
        "fanout",
        "beam_width",
        "max_visited",
        "hnsw_m",
        "hnsw_ef_construction",
        "hnsw_ef",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be non-negative")
    if args.nodes < args.clusters:
        raise ValueError("--nodes must be at least --clusters")
    if args.top_k > args.nodes:
        raise ValueError("--top-k must be less than or equal to --nodes")
    if args.vector_noise < 0.0 or args.query_noise < 0.0:
        raise ValueError("noise values must be non-negative")


def build_graph(
    *,
    node_count: int,
    cluster_count: int,
    edges_per_node: int,
    index_config: TraversalIndexConfig,
    vector_noise: float,
    seed: int,
) -> GraphBundle:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(cluster_count, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    store = GraphStore(max_outgoing_edges=edges_per_node)
    index = TraversalIndex(config=index_config)
    labels = np.empty(node_count, dtype=np.int32)
    matrix = np.empty((node_count, 16), dtype=np.float32)
    node_ids: list[str] = []
    cluster_members: list[list[int]] = [[] for _ in range(cluster_count)]

    for node_index in range(node_count):
        cluster = node_index % cluster_count
        vector = make_noisy_unit_vector(centers[cluster], rng=rng, noise=vector_noise)
        node_id = f"n_{node_index:06d}"
        node = NodeFrame(
            node_id=node_id,
            summary_vector=resize_vector(vector, 32),
            full_vector=resize_vector(vector, 64),
            summary_payload=f"cluster {cluster} memory {node_index}",
            metadata={"cluster": int(cluster)},
            traversal_vector=vector,
        )
        store.add_node(node)
        index.add_node(node)
        labels[node_index] = cluster
        matrix[node_index] = vector
        node_ids.append(node_id)
        cluster_members[cluster].append(node_index)

    for members in cluster_members:
        member_count = len(members)
        if member_count <= 1:
            continue
        for position, src_index in enumerate(members):
            src_id = node_ids[src_index]
            src_vector = matrix[src_index]
            edge_count = min(edges_per_node, member_count - 1)
            for offset in range(1, edge_count + 1):
                dst_index = members[(position + offset) % member_count]
                dst_id = node_ids[dst_index]
                confidence = 1.0 - offset / (edge_count + 1)
                store.add_edge(
                    EdgeFrame(
                        src_id=src_id,
                        dst_id=dst_id,
                        edge_vector=stable_edge_vector(src_vector, matrix[dst_index], 16),
                        confidence=confidence,
                    )
                )

    return GraphBundle(store=store, index=index, centers=centers, matrix=matrix, labels=labels, node_ids=node_ids)


def maybe_build_hnsw(
    *,
    matrix: np.ndarray,
    backend: str,
    top_k: int,
    ef: int,
    ef_construction: int,
    m: int,
    seed: int,
) -> HnswBundle:
    if backend == "exact":
        return HnswBundle(backend="exact", build_ms=0.0, index=None)
    try:
        import hnswlib  # type: ignore[import-not-found]
    except ImportError as exc:
        if backend == "hnsw":
            raise RuntimeError("hnswlib is not installed. Install with: .venv/bin/pip install -e '.[hnsw]'") from exc
        return HnswBundle(backend="exact_fallback", build_ms=0.0, index=None)

    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    index.init_index(
        max_elements=matrix.shape[0],
        ef_construction=ef_construction,
        M=m,
        random_seed=seed,
    )
    index.add_items(matrix, np.arange(matrix.shape[0], dtype=np.int64), num_threads=1)
    index.set_ef(max(ef, top_k))
    return HnswBundle(backend="hnsw", build_ms=elapsed_ms(start), index=index)


def load_scorer(checkpoint: str | None, *, device: str):
    if checkpoint is None:
        return HeuristicTraversalScorer()
    from vector_graph.torch_models import TorchTraversalScorer

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TorchTraversalScorer.from_checkpoint(checkpoint, device=device)


def make_query(center: np.ndarray, *, rng: np.random.Generator, noise: float) -> np.ndarray:
    return make_noisy_unit_vector(center, rng=rng, noise=noise)


def make_noisy_unit_vector(center: np.ndarray, *, rng: np.random.Generator, noise: float) -> np.ndarray:
    return np.asarray(normalize(center + rng.normal(scale=noise, size=center.shape).astype(np.float32)), dtype=np.float32)


def exact_search(matrix: np.ndarray, query: np.ndarray, node_ids: Sequence[str], top_k: int) -> list[int]:
    scores = matrix @ query.reshape(-1)
    limit = min(top_k, scores.shape[0])
    candidate_indices = np.argpartition(-scores, kth=limit - 1)[:limit]
    ranked = sorted(
        ((int(index), float(scores[index])) for index in candidate_indices),
        key=lambda item: (-item[1], node_ids[item[0]]),
    )
    return [index for index, _ in ranked]


def hnsw_search(index: Any, top_k: int, query: np.ndarray) -> list[int]:
    labels, _ = index.knn_query(query.reshape(1, -1), k=top_k, num_threads=1)
    return [int(label) for label in labels[0].tolist()]


def run_traversal(
    *,
    graph: GraphBundle,
    controller: TraversalController,
    query: np.ndarray,
    query_dim: int,
    seed_limit: int,
    top_k: int,
) -> TraversalOutput:
    seed_ids = graph.index.seed_ids(query, limit=seed_limit)
    if not seed_ids:
        return TraversalOutput(indices=[], visited=0, included=0, seed_count=0)

    result = controller.traverse(
        query_vector=resize_vector(query, query_dim),
        seed_id=seed_ids[0],
        extra_seed_ids=seed_ids[1:],
    )
    ids = [decision.node_id for decision in result.included[:top_k]]
    return TraversalOutput(
        indices=[graph.id_to_index[node_id] for node_id in ids],
        visited=len(result.visited),
        included=len(result.included),
        seed_count=len(seed_ids),
    )


def row_for_indices(
    indices: Sequence[int],
    labels: np.ndarray,
    target_label: int,
    top_k: int,
    latency_ms: float,
) -> dict[str, float]:
    hits = [1 if int(labels[index]) == target_label else 0 for index in indices[:top_k]]
    hit_count = sum(hits)
    cluster_size = int(np.count_nonzero(labels == target_label))
    first_hit = next((rank + 1 for rank, hit in enumerate(hits) if hit), 0)
    return {
        "latency_ms": latency_ms,
        "precision_at_k": hit_count / top_k,
        "recall_at_k": hit_count / max(cluster_size, 1),
        "hit_at_k": 1.0 if hit_count else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "returned": float(min(len(indices), top_k)),
    }


def summarize_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    metric_names = sorted({name for row in rows for name in row})
    return {
        key: value
        for metric_name in metric_names
        for key, value in summarize_metric(metric_name, [row.get(metric_name, 0.0) for row in rows]).items()
    }


def summarize_metric(name: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{name}_mean": round(float(statistics.fmean(values)), 6),
        f"{name}_p50": round(percentile(values, 0.50), 6),
        f"{name}_p95": round(percentile(values, 0.95), 6),
        f"{name}_p99": round(percentile(values, 0.99), 6),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * probability) - 1))])


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    main()
