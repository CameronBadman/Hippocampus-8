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
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import EdgeFrame, GraphStore, NodeFrame, TraversalConfig, TraversalController, TraversalIndex
from vector_graph.index import TraversalIndexConfig
from vector_graph.torch_models import TorchModelConfig, TorchTraversalScorer
from vector_graph.vectors import normalize, resize_vector, stable_edge_vector


class CountingScorer:
    def __init__(self, scorer: TorchTraversalScorer) -> None:
        self.scorer = scorer
        self.reset()

    def reset(self) -> None:
        self.scored_candidates = 0
        self.score_batches = 0
        self.model_ms = 0.0

    def score_edges(self, **kwargs):
        edges = kwargs["edges"]
        start = time.perf_counter()
        scores = self.scorer.score_edges(**kwargs)
        self.model_ms += (time.perf_counter() - start) * 1000.0
        self.scored_candidates += len(edges)
        self.score_batches += 1 if edges else 0
        return scores

    def score_edge(self, **kwargs):
        return self.score_edges(
            query_vector=kwargs["query_vector"],
            current_node=kwargs["current_node"],
            edges=[kwargs["edge"]],
            dst_nodes=[kwargs["dst_node"]],
            path_vector=kwargs["path_vector"],
            hop=kwargs["hop"],
        )[0]

    def score_attach(self, **kwargs) -> float:
        return self.scorer.score_attach(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark indexed traversal plus torch transformer scoring.")
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--edges-per-node", type=int, default=16)
    parser.add_argument("--seed-limit", type=int, default=4)
    parser.add_argument("--index-tables", type=int, default=4)
    parser.add_argument("--index-bits", type=int, default=14)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-visited", type=int, default=512)
    parser.add_argument("--include-threshold", type=float, default=0.58)
    parser.add_argument("--expand-threshold", type=float, default=0.52)
    parser.add_argument("--read-full-threshold", type=float, default=0.70)
    parser.add_argument("--mode", choices=["beam", "single_path"], default="beam")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    if args.nodes <= 0:
        raise ValueError("--nodes must be positive")
    if args.queries <= 0:
        raise ValueError("--queries must be positive")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be non-negative")

    device = pick_device(args.device)
    graph_start = time.perf_counter()
    store, index, centers = build_graph(
        node_count=args.nodes,
        cluster_count=args.clusters,
        edges_per_node=args.edges_per_node,
        index_config=TraversalIndexConfig(
            dimension=16,
            table_count=args.index_tables,
            bits_per_table=args.index_bits,
            seed=17,
        ),
        seed=args.seed,
    )
    graph_build_ms = (time.perf_counter() - graph_start) * 1000.0

    scorer = load_scorer(args.checkpoint, device=device)
    counting_scorer = CountingScorer(scorer)
    controller = TraversalController(
        store=store,
        scorer=counting_scorer,
        config=TraversalConfig(
            max_hops=args.max_hops,
            fanout=args.fanout,
            beam_width=args.beam_width,
            mode=args.mode,
            max_visited=args.max_visited,
            include_threshold=args.include_threshold,
            expand_threshold=args.expand_threshold,
            read_full_threshold=args.read_full_threshold,
        ),
    )

    query_rng = np.random.default_rng(args.seed + 999)
    query_labels = query_rng.integers(0, len(centers), size=args.queries)
    warmup_labels = query_rng.integers(0, len(centers), size=args.warmup_queries)
    for label in warmup_labels:
        run_query(
            store=store,
            index=index,
            controller=controller,
            scorer=scorer,
            counting_scorer=counting_scorer,
            centers=centers,
            label=int(label),
            query_rng=query_rng,
            seed_limit=args.seed_limit,
        )

    rows = []
    for label in query_labels:
        rows.append(
            run_query(
                store=store,
                index=index,
                controller=controller,
                scorer=scorer,
                counting_scorer=counting_scorer,
                centers=centers,
                label=int(label),
                query_rng=query_rng,
                seed_limit=args.seed_limit,
            )
        )

    report = {
        "nodes": args.nodes,
        "queries": args.queries,
        "warmup_queries": args.warmup_queries,
        "clusters": args.clusters,
        "edges_per_node": args.edges_per_node,
        "seed_limit": args.seed_limit,
        "index_config": asdict(index.config),
        "traversal_config": {
            "max_hops": args.max_hops,
            "fanout": args.fanout,
            "beam_width": args.beam_width,
            "mode": args.mode,
            "max_visited": args.max_visited,
            "include_threshold": args.include_threshold,
            "expand_threshold": args.expand_threshold,
            "read_full_threshold": args.read_full_threshold,
        },
        "checkpoint": args.checkpoint,
        "device": str(device),
        "graph_build_ms": round(graph_build_ms, 3),
        "metrics": summarize_rows(rows),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_graph(
    *,
    node_count: int,
    cluster_count: int,
    edges_per_node: int,
    index_config: TraversalIndexConfig,
    seed: int,
) -> tuple[GraphStore, TraversalIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(cluster_count, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    store = GraphStore(max_outgoing_edges=edges_per_node)
    index = TraversalIndex(config=index_config)
    cluster_members: list[list[str]] = [[] for _ in range(cluster_count)]
    vectors: dict[str, np.ndarray] = {}

    for node_index in range(node_count):
        cluster = node_index % cluster_count
        traversal_vector = np.asarray(
            normalize(centers[cluster] + rng.normal(scale=0.10, size=16).astype(np.float32)),
            dtype=np.float32,
        )
        node_id = f"n_{node_index:06d}"
        node = NodeFrame(
            node_id=node_id,
            summary_vector=resize_vector(traversal_vector, 32),
            full_vector=resize_vector(traversal_vector, 64),
            metadata={"cluster": cluster},
            traversal_vector=traversal_vector,
        )
        store.add_node(node)
        index.add_node(node)
        cluster_members[cluster].append(node_id)
        vectors[node_id] = traversal_vector

    for members in cluster_members:
        member_count = len(members)
        for position, src_id in enumerate(members):
            src_vector = vectors[src_id]
            for offset in range(1, min(edges_per_node, member_count - 1) + 1):
                dst_id = members[(position + offset) % member_count]
                store.add_edge(
                    EdgeFrame(
                        src_id=src_id,
                        dst_id=dst_id,
                        edge_vector=stable_edge_vector(src_vector, vectors[dst_id], 16),
                        confidence=1.0 - offset / (edges_per_node + 1),
                    )
                )

    return store, index, centers


def run_query(
    *,
    store: GraphStore,
    index: TraversalIndex,
    controller: TraversalController,
    scorer: TorchTraversalScorer,
    counting_scorer: CountingScorer,
    centers: np.ndarray,
    label: int,
    query_rng: np.random.Generator,
    seed_limit: int,
) -> dict[str, float | int]:
    traversal_query = np.asarray(
        normalize(centers[label] + query_rng.normal(scale=0.08, size=16).astype(np.float32)),
        dtype=np.float32,
    )
    query_vector = resize_vector(traversal_query, scorer.config.query_dim)

    counting_scorer.reset()
    start = time.perf_counter()
    seed_ids = index.seed_ids(traversal_query, limit=seed_limit)
    index_ms = (time.perf_counter() - start) * 1000.0
    if not seed_ids:
        return empty_row(index_ms=index_ms)

    start = time.perf_counter()
    result = controller.traverse(
        query_vector=query_vector,
        seed_id=seed_ids[0],
        extra_seed_ids=seed_ids[1:],
    )
    traversal_ms = (time.perf_counter() - start) * 1000.0
    visited_clusters = [store.get_node(decision.node_id).metadata["cluster"] for decision in result.visited]
    included_clusters = [store.get_node(decision.node_id).metadata["cluster"] for decision in result.included]
    return {
        "index_ms": index_ms,
        "traversal_ms": traversal_ms,
        "total_ms": index_ms + traversal_ms,
        "model_ms": counting_scorer.model_ms,
        "score_batches": counting_scorer.score_batches,
        "scored_candidates": counting_scorer.scored_candidates,
        "seed_count": len(seed_ids),
        "visited": len(result.visited),
        "included": len(result.included),
        "visited_cluster_purity": purity(visited_clusters, label),
        "included_cluster_purity": purity(included_clusters, label),
    }


def load_scorer(checkpoint: str | None, *, device: str) -> TorchTraversalScorer:
    if checkpoint is not None:
        return TorchTraversalScorer.from_checkpoint(checkpoint, device=device)
    config = TorchModelConfig(
        query_dim=32,
        summary_dim=32,
        edge_dim=16,
        full_dim=64,
        path_dim=32,
        model_kind="transformer",
    )
    return TorchTraversalScorer.initialized(config, seed=17, device=device)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def empty_row(*, index_ms: float) -> dict[str, float | int]:
    return {
        "index_ms": index_ms,
        "traversal_ms": 0.0,
        "total_ms": index_ms,
        "model_ms": 0.0,
        "score_batches": 0,
        "scored_candidates": 0,
        "seed_count": 0,
        "visited": 0,
        "included": 0,
        "visited_cluster_purity": 0.0,
        "included_cluster_purity": 0.0,
    }


def purity(labels: Sequence[int], target: int) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label == target) / len(labels)


def summarize_rows(rows: Sequence[dict[str, float | int]]) -> dict[str, float]:
    return {
        key: value
        for metric in (
            "index_ms",
            "traversal_ms",
            "total_ms",
            "model_ms",
            "score_batches",
            "scored_candidates",
            "seed_count",
            "visited",
            "included",
            "visited_cluster_purity",
            "included_cluster_purity",
        )
        for key, value in summarize_metric(metric, [float(row[metric]) for row in rows]).items()
    }


def summarize_metric(name: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{name}_mean": round(float(statistics.fmean(values)), 4),
        f"{name}_p50": round(percentile(values, 0.50), 4),
        f"{name}_p95": round(percentile(values, 0.95), 4),
        f"{name}_p99": round(percentile(values, 0.99), 4),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * probability) - 1))]


if __name__ == "__main__":
    main()
