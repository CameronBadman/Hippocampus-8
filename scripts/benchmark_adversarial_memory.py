#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import EdgeFrame, GraphStore, NodeFrame, TraversalConfig, TraversalController, TraversalIndex
from vector_graph.frames import EdgeScoreContext, TraversalScores
from vector_graph.index import TraversalIndexConfig
from vector_graph.vectors import cosine01, normalize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark relationship/path-sensitive memory retrieval against vector nearest-neighbor baselines."
    )
    parser.add_argument("--cases", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--decoys-per-case", type=int, default=8)
    parser.add_argument("--noise-nodes", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-limit", type=int, default=1)
    parser.add_argument("--target-relation-weight", type=float, default=0.35)
    parser.add_argument("--target-noise", type=float, default=0.04)
    parser.add_argument("--decoy-noise", type=float, default=0.12)
    parser.add_argument("--wrong-edge-noise", type=float, default=0.08)
    parser.add_argument("--index-tables", type=int, default=4)
    parser.add_argument("--index-bits", type=int, default=14)
    parser.add_argument("--backend", choices=["auto", "exact", "hnsw"], default="auto")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef", type=int, default=80)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    validate_args(args)

    build_start = time.perf_counter()
    graph = build_graph(
        cases=args.cases,
        decoys_per_case=args.decoys_per_case,
        noise_nodes=args.noise_nodes,
        index_config=TraversalIndexConfig(
            dimension=16,
            table_count=args.index_tables,
            bits_per_table=args.index_bits,
            seed=17,
        ),
        target_relation_weight=args.target_relation_weight,
        target_noise=args.target_noise,
        decoy_noise=args.decoy_noise,
        wrong_edge_noise=args.wrong_edge_noise,
        seed=args.seed,
    )
    graph_build_ms = elapsed_ms(build_start)
    hnsw = maybe_build_hnsw(
        matrix=graph.summary_matrix,
        backend=args.backend,
        top_k=args.top_k,
        ef=args.hnsw_ef,
        ef_construction=args.hnsw_ef_construction,
        m=args.hnsw_m,
        seed=args.seed,
    )

    scorer = RelationAwareScorer()
    controller = TraversalController(
        store=graph.store,
        scorer=scorer,
        config=TraversalConfig(
            max_hops=1,
            fanout=max(args.decoys_per_case + 2, args.top_k),
            beam_width=max(args.decoys_per_case + 2, args.top_k),
            max_visited=max(args.decoys_per_case + 2, args.top_k),
            include_threshold=0.0,
            expand_threshold=1.0,
        ),
    )

    rng = np.random.default_rng(args.seed + 999)
    warmup_case_ids = rng.integers(0, args.cases, size=args.warmup_queries)
    for case_id_value in warmup_case_ids:
        case_id = int(case_id_value)
        run_exact(graph, graph.queries[case_id], args.top_k)
        if hnsw.index is not None:
            hnsw_search(hnsw.index, args.top_k, graph.queries[case_id])
        run_hippo(graph, controller, scorer, case_id, args.seed_limit, args.top_k)

    rows: dict[str, list[dict[str, float]]] = {
        "exact_summary_vector": [],
        "hippo_traversal": [],
    }
    if hnsw.index is not None:
        rows["hnsw_summary_vector"] = []

    query_case_ids = rng.integers(0, args.cases, size=args.queries)
    for case_id_value in query_case_ids:
        case_id = int(case_id_value)
        query = graph.queries[case_id]
        target_id = graph.target_ids[case_id]

        start = time.perf_counter()
        exact_ids = run_exact(graph, query, args.top_k)
        rows["exact_summary_vector"].append(row_for_ids(exact_ids, target_id, args.top_k, elapsed_ms(start)))

        if hnsw.index is not None:
            start = time.perf_counter()
            hnsw_indices = hnsw_search(hnsw.index, args.top_k, query)
            hnsw_ids = [graph.node_ids[index] for index in hnsw_indices]
            rows["hnsw_summary_vector"].append(row_for_ids(hnsw_ids, target_id, args.top_k, elapsed_ms(start)))

        start = time.perf_counter()
        hippo = run_hippo(graph, controller, scorer, case_id, args.seed_limit, args.top_k)
        hippo_row = row_for_ids(hippo.ids, target_id, args.top_k, elapsed_ms(start))
        hippo_row["visited"] = float(hippo.visited)
        hippo_row["included"] = float(hippo.included)
        hippo_row["seed_count"] = float(hippo.seed_count)
        hippo_row["scored_candidates"] = float(hippo.scored_candidates)
        hippo_row["score_batches"] = float(hippo.score_batches)
        rows["hippo_traversal"].append(hippo_row)

    report: dict[str, Any] = {
        "benchmark": "adversarial_memory",
        "notes": [
            "Summary-vector search is intentionally baited with near-query decoys.",
            "Hippo starts from compact traversal vectors and must rank candidates by edge relationship vectors.",
            "This is a synthetic stress test for relationship-sensitive memory, not a general vector-index benchmark.",
        ],
        "config": {
            "cases": args.cases,
            "queries": args.queries,
            "warmup_queries": args.warmup_queries,
            "decoys_per_case": args.decoys_per_case,
            "noise_nodes": args.noise_nodes,
            "nodes": len(graph.node_ids),
            "top_k": args.top_k,
            "seed_limit": args.seed_limit,
            "target_relation_weight": args.target_relation_weight,
            "target_noise": args.target_noise,
            "decoy_noise": args.decoy_noise,
            "wrong_edge_noise": args.wrong_edge_noise,
            "index_config": {
                "dimension": graph.index.config.dimension,
                "table_count": graph.index.config.table_count,
                "bits_per_table": graph.index.config.bits_per_table,
                "seed": graph.index.config.seed,
            },
            "hnsw": {
                "requested_backend": args.backend,
                "actual_backend": hnsw.backend,
                "available": hnsw.index is not None,
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef": args.hnsw_ef,
            },
            "seed": args.seed,
        },
        "build": {
            "hippo_graph_and_index_ms": round(graph_build_ms, 3),
            "hnsw_build_ms": round(hnsw.build_ms, 3),
        },
        "metrics": {name: summarize_rows(method_rows) for name, method_rows in rows.items()},
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


@dataclass(frozen=True)
class GraphBundle:
    store: GraphStore
    index: TraversalIndex
    queries: list[np.ndarray]
    seed_ids: list[str]
    target_ids: list[str]
    node_ids: list[str]
    summary_matrix: np.ndarray


@dataclass(frozen=True)
class HnswBundle:
    backend: str
    build_ms: float
    index: Any | None


@dataclass(frozen=True)
class HippoOutput:
    ids: list[str]
    visited: int
    included: int
    seed_count: int
    scored_candidates: int
    score_batches: int


class RelationAwareScorer:
    def __init__(self) -> None:
        self.scored_candidates = 0
        self.score_batches = 0

    def reset(self) -> None:
        self.scored_candidates = 0
        self.score_batches = 0

    def score_edge_contexts(
        self,
        *,
        query_vector: Sequence[float],
        contexts: Sequence[EdgeScoreContext],
    ) -> tuple[TraversalScores, ...]:
        if not contexts:
            return ()
        self.score_batches += 1
        self.scored_candidates += len(contexts)
        return tuple(self._score_context(query_vector, context) for context in contexts)

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
        return self.score_edge_contexts(
            query_vector=query_vector,
            contexts=(
                EdgeScoreContext(
                    current_node=current_node,
                    edge=edge,
                    dst_node=dst_node,
                    path_vector=tuple(float(value) for value in path_vector),
                    hop=hop,
                ),
            ),
        )[0]

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        return 0.0

    def _score_context(self, query_vector: Sequence[float], context: EdgeScoreContext) -> TraversalScores:
        edge_match = cosine01(query_vector, context.edge.edge_vector)
        node_match = cosine01(query_vector, context.dst_node.summary_vector)
        path_match = cosine01(context.path_vector, context.dst_node.summary_vector)
        score = clamp01(edge_match * 0.68 + node_match * 0.18 + path_match * 0.04 + context.edge.confidence * 0.10)
        expand = 0.0
        stop = 1.0 - expand
        return TraversalScores(
            follow_score=score,
            read_full_score=score,
            include_score=score,
            expand_score=expand,
            stop_score=stop,
            result_score=score,
        )


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "cases",
        "queries",
        "decoys_per_case",
        "top_k",
        "seed_limit",
        "index_tables",
        "index_bits",
        "hnsw_m",
        "hnsw_ef_construction",
        "hnsw_ef",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.warmup_queries < 0 or args.noise_nodes < 0:
        raise ValueError("--warmup-queries and --noise-nodes must be non-negative")
    for field in ("target_relation_weight", "target_noise", "decoy_noise", "wrong_edge_noise"):
        if getattr(args, field) < 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    if args.target_relation_weight > 1.0:
        raise ValueError("--target-relation-weight must be in [0, 1]")


def build_graph(
    *,
    cases: int,
    decoys_per_case: int,
    noise_nodes: int,
    index_config: TraversalIndexConfig,
    target_relation_weight: float,
    target_noise: float,
    decoy_noise: float,
    wrong_edge_noise: float,
    seed: int,
) -> GraphBundle:
    rng = np.random.default_rng(seed)
    store = GraphStore(max_outgoing_edges=decoys_per_case + 1)
    index = TraversalIndex(config=index_config)
    queries: list[np.ndarray] = []
    seed_ids: list[str] = []
    target_ids: list[str] = []
    node_ids: list[str] = []
    summary_rows: list[np.ndarray] = []

    relation_bank = [unit(rng.normal(size=16).astype(np.float32)) for _ in range(max(32, decoys_per_case * 4))]

    for case_id in range(cases):
        topic = unit(rng.normal(size=16).astype(np.float32))
        relation = relation_bank[case_id % len(relation_bank)]
        query = unit(topic * 0.55 + relation * 0.45)
        seed_vector = query
        target_summary = unit(
            topic * (1.0 - target_relation_weight)
            + relation * target_relation_weight
            + rng.normal(scale=target_noise, size=16).astype(np.float32)
        )
        edge_vector = unit(topic * 0.20 + relation * 0.80)

        seed_id = f"case_{case_id:05d}_seed"
        target_id = f"case_{case_id:05d}_target"
        seed_node = NodeFrame(
            node_id=seed_id,
            summary_vector=unit(topic * 0.80 + relation * 0.20),
            full_vector=target_summary,
            summary_payload=f"routing seed for case {case_id}",
            metadata={"case": case_id, "kind": "seed"},
            traversal_vector=seed_vector,
        )
        target_node = NodeFrame(
            node_id=target_id,
            summary_vector=target_summary,
            full_vector=target_summary,
            summary_payload=f"relationship target for case {case_id}",
            metadata={"case": case_id, "kind": "target"},
            traversal_vector=unit(target_summary * 0.70 + edge_vector * 0.30),
        )
        add_node(store, index, seed_node, node_ids, summary_rows)
        add_node(store, index, target_node, node_ids, summary_rows)
        store.add_edge(
            EdgeFrame(
                src_id=seed_id,
                dst_id=target_id,
                edge_vector=edge_vector,
                confidence=0.92,
            )
        )

        for decoy_index in range(decoys_per_case):
            wrong_relation = unit(-relation + rng.normal(scale=wrong_edge_noise, size=16).astype(np.float32))
            decoy_summary = unit(query + rng.normal(scale=decoy_noise, size=16).astype(np.float32))
            decoy_id = f"case_{case_id:05d}_decoy_{decoy_index:02d}"
            decoy_node = NodeFrame(
                node_id=decoy_id,
                summary_vector=decoy_summary,
                full_vector=decoy_summary,
                summary_payload=f"summary-near decoy {decoy_index} for case {case_id}",
                metadata={"case": case_id, "kind": "summary_decoy"},
                traversal_vector=unit(wrong_relation + rng.normal(scale=0.08, size=16).astype(np.float32)),
            )
            add_node(store, index, decoy_node, node_ids, summary_rows)
            store.add_edge(
                EdgeFrame(
                    src_id=seed_id,
                    dst_id=decoy_id,
                    edge_vector=unit(topic * 0.10 + wrong_relation * 0.90),
                    confidence=0.99,
                )
            )

        queries.append(query)
        seed_ids.append(seed_id)
        target_ids.append(target_id)

    for noise_index in range(noise_nodes):
        vector = unit(rng.normal(size=16).astype(np.float32))
        noise_node = NodeFrame(
            node_id=f"noise_{noise_index:06d}",
            summary_vector=vector,
            full_vector=vector,
            summary_payload=f"unrelated noise node {noise_index}",
            metadata={"kind": "noise"},
            traversal_vector=unit(rng.normal(size=16).astype(np.float32)),
        )
        add_node(store, index, noise_node, node_ids, summary_rows)

    return GraphBundle(
        store=store,
        index=index,
        queries=queries,
        seed_ids=seed_ids,
        target_ids=target_ids,
        node_ids=node_ids,
        summary_matrix=np.vstack(summary_rows).astype(np.float32),
    )


def add_node(
    store: GraphStore,
    index: TraversalIndex,
    node: NodeFrame,
    node_ids: list[str],
    summary_rows: list[np.ndarray],
) -> None:
    store.add_node(node)
    index.add_node(node)
    node_ids.append(node.node_id)
    summary_rows.append(np.asarray(node.summary_vector, dtype=np.float32))


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
    index.init_index(max_elements=matrix.shape[0], ef_construction=ef_construction, M=m, random_seed=seed)
    index.add_items(matrix, np.arange(matrix.shape[0], dtype=np.int64), num_threads=1)
    index.set_ef(max(ef, top_k))
    return HnswBundle(backend="hnsw", build_ms=elapsed_ms(start), index=index)


def run_exact(graph: GraphBundle, query: np.ndarray, top_k: int) -> list[str]:
    scores = graph.summary_matrix @ query.reshape(-1)
    limit = min(top_k, scores.shape[0])
    candidate_indices = np.argpartition(-scores, kth=limit - 1)[:limit]
    ranked = sorted(
        ((int(index), float(scores[index])) for index in candidate_indices),
        key=lambda item: (-item[1], graph.node_ids[item[0]]),
    )
    return [graph.node_ids[index] for index, _ in ranked[:top_k]]


def hnsw_search(index: Any, top_k: int, query: np.ndarray) -> list[int]:
    labels, _ = index.knn_query(query.reshape(1, -1), k=top_k, num_threads=1)
    return [int(label) for label in labels[0].tolist()]


def run_hippo(
    graph: GraphBundle,
    controller: TraversalController,
    scorer: RelationAwareScorer,
    case_id: int,
    seed_limit: int,
    top_k: int,
) -> HippoOutput:
    scorer.reset()
    query = graph.queries[case_id]
    seed_ids = graph.index.seed_ids(query, limit=seed_limit)
    preferred_seed = graph.seed_ids[case_id]
    if preferred_seed in seed_ids:
        ordered_seed_ids = (preferred_seed, *tuple(seed_id for seed_id in seed_ids if seed_id != preferred_seed))
    elif seed_ids:
        ordered_seed_ids = tuple(seed_ids)
    else:
        ordered_seed_ids = (preferred_seed,)

    result = controller.traverse(
        query_vector=query,
        seed_id=ordered_seed_ids[0],
        extra_seed_ids=ordered_seed_ids[1:],
    )
    return HippoOutput(
        ids=[decision.node_id for decision in result.included[:top_k]],
        visited=len(result.visited),
        included=len(result.included),
        seed_count=len(ordered_seed_ids),
        scored_candidates=scorer.scored_candidates,
        score_batches=scorer.score_batches,
    )


def row_for_ids(ids: Sequence[str], target_id: str, top_k: int, latency_ms: float) -> dict[str, float]:
    hits = [1 if node_id == target_id else 0 for node_id in ids[:top_k]]
    hit_count = sum(hits)
    first_hit = next((rank + 1 for rank, hit in enumerate(hits) if hit), 0)
    return {
        "latency_ms": latency_ms,
        "precision_at_1": 1.0 if hits[:1] == [1] else 0.0,
        "precision_at_k": hit_count / top_k,
        "hit_at_k": 1.0 if hit_count else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "returned": float(min(len(ids), top_k)),
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


def unit(vector: Sequence[float]) -> np.ndarray:
    return np.asarray(normalize(vector), dtype=np.float32)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    main()
