#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import EdgeFrame, GraphStore, NodeFrame, TraversalIndex, TraversalIndexConfig, embed_text
from vector_graph.scorer import effective_node_summary, path_vector_for
from vector_graph.vectors import cosine01, metadata_vector_from, resize_vector, stable_edge_vector, traversal_vector_from


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
            metadata = {
                "topic": topic_name,
                "plain_topic": topic,
                "terms": node_terms,
            }
            summary_vector = embed_text(summary, 32)
            metadata_vector = metadata_vector_from(metadata, 32)
            frame = NodeFrame(
                node_id=node_id,
                summary_vector=summary_vector,
                full_vector=embed_text(full, 64),
                summary_payload=summary,
                full_payload=full,
                metadata=metadata,
                metadata_vector=metadata_vector,
                traversal_vector=traversal_vector_from(summary_vector, metadata_vector, 16),
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
    candidate_reasons: dict[tuple[str, str], str] = {}

    def add_candidate(parent_id: str, edge: EdgeFrame, reason: str) -> None:
        key = (parent_id, edge.dst_id)
        if key in candidate_reasons:
            return
        candidate_edges.append((parent_id, edge))
        candidate_reasons[key] = reason

    for seed_id in seed_ids:
        for edge in store.get_edges(seed_id):
            add_candidate(seed_id, edge, "outgoing_seed_edge")
    seed_nodes = [store.get_node(seed_id) for seed_id in seed_ids]
    path_vector = path_vector_for(seed_nodes[:1], 32) if seed_nodes else resize_vector(query_vector, 32)
    current_seed_id = seed_ids[0] if seed_ids else None
    if current_seed_id is not None:
        add_query_target_candidates(
            store=store,
            query_vector=query_vector,
            parent_id=current_seed_id,
            expected_topic=expected_topic,
            nodes_by_topic=nodes_by_topic,
            add_candidate=add_candidate,
        )
        add_lexical_hard_negatives(
            store=store,
            query_vector=query_vector,
            parent_id=current_seed_id,
            expected_topic=expected_topic,
            nodes_by_topic=nodes_by_topic,
            add_candidate=add_candidate,
        )
    candidate_edges = select_balanced_candidate_edges(
        candidate_edges,
        candidate_reasons=candidate_reasons,
        candidate_limit=candidate_limit,
    )

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
        relation = describe_candidate_relation(
            current=current,
            dst=dst,
            edge=edge,
            expected_topic=expected_topic,
            teacher=teacher,
            retrieval_reason=candidate_reasons.get((parent_id, edge.dst_id), "unknown"),
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
                "source_topic": current.metadata["topic"],
                "destination_topic": dst.metadata["topic"],
                "destination_plain_topic": dst.metadata.get("plain_topic", ""),
                "destination_terms": list(dst.metadata.get("terms", ())),
                "relation": relation,
                "retrieval_reason": candidate_reasons.get((parent_id, edge.dst_id), "unknown"),
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
        "query_intent": describe_query_intent(query=query, expected_topic=expected_topic),
        "seed_ids": seed_ids,
        "path": [
            {
                "node_id": node.node_id,
                "summary": node.summary_payload,
                "topic": node.metadata["topic"],
                "plain_topic": node.metadata.get("plain_topic", ""),
                "terms": list(node.metadata.get("terms", ())),
            }
            for node in seed_nodes[:2]
        ],
        "current_node": {
            "node_id": seed_ids[0] if seed_ids else None,
            "summary": store.get_node(seed_ids[0]).summary_payload if seed_ids else "",
            "full": store.get_node(seed_ids[0]).full_payload if seed_ids else "",
            "topic": store.get_node(seed_ids[0]).metadata["topic"] if seed_ids else "",
            "plain_topic": store.get_node(seed_ids[0]).metadata.get("plain_topic", "") if seed_ids else "",
            "terms": list(store.get_node(seed_ids[0]).metadata.get("terms", ())) if seed_ids else [],
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


def add_query_target_candidates(
    *,
    store: GraphStore,
    query_vector,
    parent_id: str,
    expected_topic: str,
    nodes_by_topic: dict[str, list[GeneratedNode]],
    add_candidate,
) -> None:
    parent = store.get_node(parent_id)
    for generated in sorted(
        nodes_by_topic.get(expected_topic, []),
        key=lambda node: (-cosine01(query_vector, node.frame.summary_vector), node.frame.node_id),
    )[:4]:
        if generated.frame.node_id == parent_id:
            continue
        add_candidate(
            parent_id,
            EdgeFrame(
                src_id=parent_id,
                dst_id=generated.frame.node_id,
                edge_vector=stable_edge_vector(
                    effective_node_summary(parent, dimension=16),
                    effective_node_summary(generated.frame, dimension=16),
                    16,
                ),
                confidence=0.82,
            ),
            "nearest_target_topic_candidate",
        )


def select_balanced_candidate_edges(
    candidate_edges: Sequence[tuple[str, EdgeFrame]],
    *,
    candidate_reasons: dict[tuple[str, str], str],
    candidate_limit: int,
) -> list[tuple[str, EdgeFrame]]:
    sorted_edges = sorted(candidate_edges, key=lambda item: (-item[1].confidence, item[0], item[1].dst_id))
    quotas = {
        "nearest_target_topic_candidate": max(2, candidate_limit // 4),
        "lexical_hard_negative_candidate": max(3, candidate_limit // 3),
    }
    selected: list[tuple[str, EdgeFrame]] = []
    selected_keys: set[tuple[str, str]] = set()

    def add(edge_row: tuple[str, EdgeFrame]) -> None:
        key = (edge_row[0], edge_row[1].dst_id)
        if key in selected_keys or len(selected) >= candidate_limit:
            return
        selected.append(edge_row)
        selected_keys.add(key)

    for reason, quota in quotas.items():
        count = 0
        for edge_row in sorted_edges:
            key = (edge_row[0], edge_row[1].dst_id)
            if candidate_reasons.get(key) != reason:
                continue
            add(edge_row)
            count += 1
            if count >= quota:
                break

    for edge_row in sorted_edges:
        add(edge_row)
        if len(selected) >= candidate_limit:
            break
    return selected


def add_lexical_hard_negatives(
    *,
    store: GraphStore,
    query_vector,
    parent_id: str,
    expected_topic: str,
    nodes_by_topic: dict[str, list[GeneratedNode]],
    add_candidate,
) -> None:
    parent = store.get_node(parent_id)
    wrong_topic_nodes = [
        generated
        for topic, members in nodes_by_topic.items()
        if topic != expected_topic
        for generated in members
    ]
    for generated in sorted(
        wrong_topic_nodes,
        key=lambda node: (-cosine01(query_vector, node.frame.summary_vector), node.frame.node_id),
    )[:6]:
        if generated.frame.node_id == parent_id:
            continue
        add_candidate(
            parent_id,
            EdgeFrame(
                src_id=parent_id,
                dst_id=generated.frame.node_id,
                edge_vector=stable_edge_vector(
                    effective_node_summary(parent, dimension=16),
                    effective_node_summary(generated.frame, dimension=16),
                    16,
                ),
                confidence=0.58,
            ),
            "lexical_hard_negative_candidate",
        )


def describe_query_intent(*, query: str, expected_topic: str) -> dict[str, object]:
    plain_topic = expected_topic.split(":", 1)[1] if ":" in expected_topic else expected_topic
    return {
        "text": query,
        "target_topic": expected_topic,
        "target_plain_topic": plain_topic,
        "target_terms": topic_terms(plain_topic),
    }


def describe_candidate_relation(
    *,
    current: NodeFrame,
    dst: NodeFrame,
    edge: EdgeFrame,
    expected_topic: str,
    teacher: dict[str, float],
    retrieval_reason: str,
) -> dict[str, object]:
    same_topic = dst.metadata["topic"] == expected_topic
    source_same_as_destination = current.metadata["topic"] == dst.metadata["topic"]
    if same_topic:
        decision_hint = "target_topic_match"
    elif teacher["read_full"] >= 0.45:
        decision_hint = "lexically_ambiguous_wrong_topic"
    else:
        decision_hint = "off_topic"
    return {
        "source_topic": current.metadata["topic"],
        "destination_topic": dst.metadata["topic"],
        "destination_plain_topic": dst.metadata.get("plain_topic", ""),
        "destination_terms": list(dst.metadata.get("terms", ())),
        "same_as_query_topic": same_topic,
        "same_as_source_topic": source_same_as_destination,
        "edge_confidence": round(edge.confidence, 6),
        "retrieval_reason": retrieval_reason,
        "query_topic_terms": topic_terms(expected_topic.split(":", 1)[1] if ":" in expected_topic else expected_topic),
        "overlap_with_query_topic_terms": sorted(
            set(dst.metadata.get("terms", ()))
            & set(topic_terms(expected_topic.split(":", 1)[1] if ":" in expected_topic else expected_topic))
        ),
        "decision_hint": decision_hint,
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
    dst_effective = effective_node_summary(dst)
    current_effective = effective_node_summary(current)
    query_match = cosine01(query_vector, dst_effective)
    edge_match = cosine01(resize_vector(query_vector, len(edge.edge_vector)), edge.edge_vector)
    path_match = cosine01(path_vector, dst_effective)
    current_match = cosine01(current_effective, dst_effective)
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


def topic_terms(plain_topic: str) -> list[str]:
    for topic, terms in TOPIC_BANK:
        if topic == plain_topic:
            return list(terms)
    return []


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
            edge_vector=stable_edge_vector(
                effective_node_summary(src, dimension=16),
                effective_node_summary(dst, dimension=16),
                16,
            ),
            confidence=max(0.0, min(1.0, confidence)),
        )
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
