#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vector_graph import EdgeFrame, GraphStore, NodeFrame, TraversalIndex, TraversalIndexConfig, embed_text
from vector_graph.scorer import path_vector_for
from vector_graph.vectors import cosine01, resize_vector, stable_edge_vector


TOPIC_BANK = [
    ("payments", ["invoice", "refund", "settlement", "chargeback", "ledger", "payout", "reconciliation"]),
    ("search", ["index", "query", "ranking", "filter", "document", "snippet", "recall"]),
    ("security", ["token", "policy", "permission", "audit", "risk", "encryption", "identity"]),
    ("training", ["dataset", "negative", "label", "teacher", "student", "distill", "benchmark"]),
    ("storage", ["node", "edge", "graph", "memory", "fanout", "prune", "cache"]),
    ("support", ["ticket", "agent", "handoff", "sla", "customer", "escalation", "resolution"]),
    ("analytics", ["metric", "dashboard", "cohort", "event", "trend", "report", "latency"]),
    ("deployment", ["release", "rollback", "checkpoint", "runtime", "server", "config", "health"]),
    ("routing", ["expressway", "bucket", "seed", "lsh", "path", "hub", "jump"]),
    ("content", ["summary", "full", "context", "chunk", "payload", "section", "annotation"]),
]

BRIDGE_TERMS = [
    "deterministic",
    "prototype",
    "production",
    "failure",
    "tradeoff",
    "workflow",
    "operator",
    "quality",
    "latency",
    "precision",
]


@dataclass(frozen=True)
class GeneratedNode:
    frame: NodeFrame
    topic: str
    terms: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic teacher-label traversal episodes.")
    parser.add_argument("--output-dir", default="data/teacher_episodes")
    parser.add_argument("--domains", type=int, default=12)
    parser.add_argument("--topics-per-domain", type=int, default=6)
    parser.add_argument("--nodes-per-topic", type=int, default=48)
    parser.add_argument("--queries-per-domain", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--seed-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes_000.jsonl"
    rng = random.Random(args.seed)
    total = 0

    with episodes_path.open("w", encoding="utf-8") as handle:
        for domain_index in range(args.domains):
            domain_name = f"domain_{domain_index:03d}"
            store, index, nodes_by_topic = build_domain(
                domain_name=domain_name,
                rng=rng,
                topics_per_domain=args.topics_per_domain,
                nodes_per_topic=args.nodes_per_topic,
            )
            topics = sorted(nodes_by_topic)
            for query_index in range(args.queries_per_domain):
                topic = topics[query_index % len(topics)]
                query = make_query(topic, nodes_by_topic[topic][query_index % len(nodes_by_topic[topic])].terms, rng)
                episode = build_episode(
                    episode_id=f"{domain_name}-q{query_index:04d}",
                    domain_name=domain_name,
                    store=store,
                    index=index,
                    nodes_by_topic=nodes_by_topic,
                    query=query,
                    expected_topic=topic,
                    candidate_limit=args.candidate_limit,
                    seed_limit=args.seed_limit,
                )
                handle.write(json.dumps(episode, separators=(",", ":")) + "\n")
                total += 1

    manifest = {
        "schema_version": 1,
        "kind": "teacher_episodes",
        "seed": args.seed,
        "episodes": total,
        "episode_files": [episodes_path.name],
        "dimensions": {
            "query": 32,
            "summary": 32,
            "edge": 16,
            "full": 64,
            "path": 32,
            "scalars": 2,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {total} teacher episodes to {output_dir}")


def build_domain(
    *,
    domain_name: str,
    rng: random.Random,
    topics_per_domain: int,
    nodes_per_topic: int,
) -> tuple[GraphStore, TraversalIndex, dict[str, list[GeneratedNode]]]:
    selected = rng.sample(TOPIC_BANK, topics_per_domain)
    store = GraphStore(max_outgoing_edges=16)
    index = TraversalIndex(config=TraversalIndexConfig(dimension=16, table_count=4, bits_per_table=14, seed=17))
    nodes_by_topic: dict[str, list[GeneratedNode]] = {}

    for topic_index, (topic, terms) in enumerate(selected):
        topic_name = f"{domain_name}:{topic}"
        nodes_by_topic[topic_name] = []
        for node_index in range(nodes_per_topic):
            node_terms = tuple(rng.sample(terms, min(4, len(terms))))
            distractor_topic, distractor_terms = selected[(topic_index + rng.randrange(1, len(selected))) % len(selected)]
            summary = make_summary(topic, node_index, node_terms, distractor_topic, distractor_terms, rng)
            full = make_full(summary, topic, node_terms, rng)
            node_id = f"{domain_name}:{topic}:{node_index:04d}"
            frame = NodeFrame(
                node_id=node_id,
                summary_vector=embed_text(summary, 32),
                full_vector=embed_text(full, 64),
                summary_payload=summary,
                full_payload=full,
                metadata={
                    "topic": topic_name,
                    "traversal_vector": resize_vector(embed_text(summary, 32), 16),
                },
            )
            store.add_node(frame)
            index.add_node(frame)
            nodes_by_topic[topic_name].append(GeneratedNode(frame=frame, topic=topic_name, terms=node_terms))

    for topic_position, topic in enumerate(sorted(nodes_by_topic)):
        members = nodes_by_topic[topic]
        near_topic = sorted(nodes_by_topic)[(topic_position + 1) % len(nodes_by_topic)]
        near_members = nodes_by_topic[near_topic]
        for position, generated in enumerate(members):
            src = generated.frame
            for offset in range(1, min(12, len(members) - 1) + 1):
                dst = members[(position + offset) % len(members)].frame
                add_edge(store, src, dst, confidence=1.0 - offset / 20.0)
            for offset in range(4):
                dst = near_members[(position * 5 + offset * 13) % len(near_members)].frame
                add_edge(store, src, dst, confidence=0.38 - offset * 0.04)

    return store, index, nodes_by_topic


def build_episode(
    *,
    episode_id: str,
    domain_name: str,
    store: GraphStore,
    index: TraversalIndex,
    nodes_by_topic: dict[str, list[GeneratedNode]],
    query: str,
    expected_topic: str,
    candidate_limit: int,
    seed_limit: int,
) -> dict:
    query_vector = embed_text(query, 32)
    seed_ids = list(index.seed_ids(resize_vector(query_vector, 16), limit=seed_limit))
    if len(seed_ids) < seed_limit:
        for node in store.find_nearest_summary(query_vector, limit=seed_limit):
            if node.node_id not in seed_ids:
                seed_ids.append(node.node_id)
            if len(seed_ids) >= seed_limit:
                break
    candidate_edges: list[tuple[str, EdgeFrame]] = []
    for seed_id in seed_ids:
        for edge in store.get_edges(seed_id):
            candidate_edges.append((seed_id, edge))
    candidate_edges = sorted(candidate_edges, key=lambda item: (-item[1].confidence, item[0], item[1].dst_id))[:candidate_limit]
    seed_nodes = [store.get_node(seed_id) for seed_id in seed_ids]
    path_vector = path_vector_for(seed_nodes[:1], 32) if seed_nodes else resize_vector(query_vector, 32)

    candidates = []
    for parent_id, edge in candidate_edges:
        current = store.get_node(parent_id)
        dst = store.get_node(edge.dst_id)
        teacher = bootstrap_teacher_scores(
            query_vector=query_vector,
            current=current,
            dst=dst,
            edge=edge,
            path_vector=path_vector,
            expected_topic=expected_topic,
        )
        candidates.append(
            {
                "id": f"{episode_id}:{len(candidates):03d}",
                "parent_id": parent_id,
                "dst_id": edge.dst_id,
                "kind": candidate_kind(dst, expected_topic, teacher),
                "edge_summary": edge_summary(current, dst),
                "confidence": round(edge.confidence, 6),
                "hop": 0,
                "node_summary": dst.summary_payload,
                "node_full": dst.full_payload,
                "node_topic": dst.metadata["topic"],
                "bootstrap_teacher": teacher,
            }
        )

    return {
        "schema_version": 1,
        "kind": "teacher_episode",
        "id": episode_id,
        "domain": domain_name,
        "query": query,
        "expected_topic": expected_topic,
        "seed_ids": seed_ids,
        "path": [
            {
                "node_id": node.node_id,
                "summary": node.summary_payload,
                "topic": node.metadata["topic"],
            }
            for node in seed_nodes[:2]
        ],
        "current_node": {
            "node_id": seed_ids[0] if seed_ids else None,
            "summary": store.get_node(seed_ids[0]).summary_payload if seed_ids else "",
            "full": store.get_node(seed_ids[0]).full_payload if seed_ids else "",
            "topic": store.get_node(seed_ids[0]).metadata["topic"] if seed_ids else "",
        },
        "candidates": candidates,
        "teacher_prompt": {
            "instruction": (
                "Label each candidate for graph traversal. Scores are floats from 0 to 1 for "
                "follow, include, expand, read_full, and stop. Prefer candidates that answer "
                "the query and keep the path on task; penalize lexical overlap with wrong intent."
            )
        },
    }


def bootstrap_teacher_scores(
    *,
    query_vector,
    current: NodeFrame,
    dst: NodeFrame,
    edge: EdgeFrame,
    path_vector,
    expected_topic: str,
) -> dict[str, float]:
    same_topic = 1.0 if dst.metadata["topic"] == expected_topic else 0.0
    query_match = cosine01(query_vector, dst.summary_vector)
    edge_match = cosine01(resize_vector(query_vector, len(edge.edge_vector)), edge.edge_vector)
    path_match = cosine01(path_vector, dst.summary_vector)
    current_match = cosine01(current.summary_vector, dst.summary_vector)
    follow = clamp01(same_topic * 0.58 + query_match * 0.22 + edge_match * 0.12 + edge.confidence * 0.08)
    include = clamp01(same_topic * 0.65 + query_match * 0.25 + path_match * 0.10)
    expand = clamp01(same_topic * 0.50 + current_match * 0.20 + edge.confidence * 0.20 + edge_match * 0.10)
    read_full = clamp01((1.0 - abs(query_match - 0.5) * 2.0) * 0.45 + (1.0 - same_topic) * 0.25 + include * 0.30)
    stop = clamp01(1.0 - expand)
    return {
        "follow": round(follow, 6),
        "read_full": round(read_full, 6),
        "include": round(include, 6),
        "expand": round(expand, 6),
        "stop": round(stop, 6),
    }


def candidate_kind(dst: NodeFrame, expected_topic: str, teacher: dict[str, float]) -> str:
    if dst.metadata["topic"] == expected_topic:
        return "positive"
    if teacher["read_full"] >= 0.45:
        return "ambiguous_wrong_topic"
    return "wrong_topic_negative"


def make_query(topic: str, terms: Sequence[str], rng: random.Random) -> str:
    plain_topic = topic.split(":", 1)[1]
    sampled = " ".join(rng.sample(list(terms), min(3, len(terms))))
    style = rng.choice(
        [
            "how do we handle",
            "find the part about",
            "what explains",
            "where is the note for",
            "show me the operational issue with",
        ]
    )
    return f"{style} {plain_topic} {sampled}"


def make_summary(
    topic: str,
    node_index: int,
    terms: Sequence[str],
    distractor_topic: str,
    distractor_terms: Sequence[str],
    rng: random.Random,
) -> str:
    bridge = " ".join(rng.sample(BRIDGE_TERMS, 3))
    overlap = " ".join(rng.sample(list(distractor_terms), 2))
    return f"{topic} chunk {node_index}: {' '.join(terms)}. Related wording: {overlap}. {bridge}."


def make_full(summary: str, topic: str, terms: Sequence[str], rng: random.Random) -> str:
    detail = " ".join(rng.sample(BRIDGE_TERMS, 4))
    return f"{summary} Full detail for {topic}: {' '.join(terms)}. Failure mode and decision notes: {detail}."


def edge_summary(src: NodeFrame, dst: NodeFrame) -> str:
    return f"{src.metadata['topic']} -> {dst.metadata['topic']}"


def add_edge(store: GraphStore, src: NodeFrame, dst: NodeFrame, *, confidence: float) -> None:
    store.add_edge(
        EdgeFrame(
            src_id=src.node_id,
            dst_id=dst.node_id,
            edge_vector=stable_edge_vector(src.summary_vector, dst.summary_vector, 16),
            confidence=max(0.0, min(1.0, confidence)),
        )
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
