#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence


QUERY_DIM = 32
SUMMARY_DIM = 32
EDGE_DIM = 16
FULL_DIM = 64
PATH_DIM = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic vector-frame training data.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--traversal-per-shard", type=int, default=750)
    parser.add_argument("--attach-per-shard", type=int, default=350)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    topics = [random_unit_vector(rng, SUMMARY_DIM) for _ in range(24)]
    full_topics = [resize_vector(topic, FULL_DIM) for topic in topics]

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "dimensions": {
            "query": QUERY_DIM,
            "summary": SUMMARY_DIM,
            "edge": EDGE_DIM,
            "full": FULL_DIM,
            "path": PATH_DIM,
            "scalars": 2,
        },
        "shards": [],
    }

    for shard_index in range(args.shards):
        traversal_path = output_dir / f"traversal_{shard_index:03d}.jsonl"
        attach_path = output_dir / f"attach_{shard_index:03d}.jsonl"
        write_traversal_shard(
            traversal_path,
            rng=rng,
            shard_index=shard_index,
            count=args.traversal_per_shard,
            topics=topics,
        )
        write_attach_shard(
            attach_path,
            rng=rng,
            shard_index=shard_index,
            count=args.attach_per_shard,
            topics=topics,
            full_topics=full_topics,
        )
        manifest["shards"].append(
            {
                "traversal": traversal_path.name,
                "traversal_examples": args.traversal_per_shard,
                "attach": attach_path.name,
                "attach_examples": args.attach_per_shard,
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total_traversal = args.shards * args.traversal_per_shard
    total_attach = args.shards * args.attach_per_shard
    print(f"wrote {total_traversal} traversal examples and {total_attach} attach examples to {output_dir}")


def write_traversal_shard(
    path: Path,
    *,
    rng: random.Random,
    shard_index: int,
    count: int,
    topics: Sequence[Sequence[float]],
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for index in range(count):
            query_topic = rng.randrange(len(topics))
            current_topic = nearby_topic(rng, query_topic, len(topics), near_probability=0.55)
            dst_topic = nearby_topic(rng, query_topic, len(topics), near_probability=0.62)

            query = noisy_vector(rng, topics[query_topic], noise=0.28, dimension=QUERY_DIM)
            current = noisy_vector(rng, topics[current_topic], noise=0.32, dimension=SUMMARY_DIM)
            dst = noisy_vector(rng, topics[dst_topic], noise=0.34, dimension=SUMMARY_DIM)
            path_vector = noisy_vector(
                rng,
                blend_vectors([topics[query_topic], topics[current_topic]]),
                noise=0.40,
                dimension=PATH_DIM,
            )
            edge = stable_edge_vector(current, dst, EDGE_DIM)
            if rng.random() < 0.35:
                edge = noisy_vector(rng, resize_vector(query, EDGE_DIM), noise=0.36, dimension=EDGE_DIM)

            confidence = round(rng.uniform(0.45, 1.0), 6)
            hop = rng.randrange(4)
            scores = traversal_targets(
                query_vector=query,
                current_summary=current,
                edge_vector=edge,
                dst_summary=dst,
                path_vector=path_vector,
                confidence=confidence,
                hop=hop,
            )
            output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "traversal",
                        "id": f"traversal-{shard_index:03d}-{index:06d}",
                        "query": round_vector(query),
                        "current_summary": round_vector(current),
                        "edge": round_vector(edge),
                        "dst_summary": round_vector(dst),
                        "path": round_vector(path_vector),
                        "confidence": confidence,
                        "hop": hop,
                        "target": round_vector(scores),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_attach_shard(
    path: Path,
    *,
    rng: random.Random,
    shard_index: int,
    count: int,
    topics: Sequence[Sequence[float]],
    full_topics: Sequence[Sequence[float]],
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for index in range(count):
            new_topic = rng.randrange(len(topics))
            candidate_topic = nearby_topic(rng, new_topic, len(topics), near_probability=0.58)

            new_summary = noisy_vector(rng, topics[new_topic], noise=0.30, dimension=SUMMARY_DIM)
            candidate_summary = noisy_vector(rng, topics[candidate_topic], noise=0.35, dimension=SUMMARY_DIM)
            new_full = noisy_vector(rng, full_topics[new_topic], noise=0.26, dimension=FULL_DIM)
            candidate_full = noisy_vector(rng, full_topics[candidate_topic], noise=0.31, dimension=FULL_DIM)
            path_vector = noisy_vector(
                rng,
                blend_vectors([topics[new_topic], topics[candidate_topic]]),
                noise=0.48,
                dimension=PATH_DIM,
            )
            target = attach_target(
                new_summary=new_summary,
                candidate_summary=candidate_summary,
                new_full=new_full,
                candidate_full=candidate_full,
                path_vector=path_vector,
            )
            output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "attach",
                        "id": f"attach-{shard_index:03d}-{index:06d}",
                        "new_summary": round_vector(new_summary),
                        "candidate_summary": round_vector(candidate_summary),
                        "new_full": round_vector(new_full),
                        "candidate_full": round_vector(candidate_full),
                        "path": round_vector(path_vector),
                        "target": round(target, 6),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def traversal_targets(
    *,
    query_vector: Sequence[float],
    current_summary: Sequence[float],
    edge_vector: Sequence[float],
    dst_summary: Sequence[float],
    path_vector: Sequence[float],
    confidence: float,
    hop: int,
) -> tuple[float, float, float, float, float]:
    edge_query = resize_vector(query_vector, len(edge_vector))
    query_to_dst = cosine01(query_vector, dst_summary)
    current_to_dst = cosine01(current_summary, dst_summary)
    query_to_edge = cosine01(edge_query, edge_vector)
    path_to_dst = cosine01(path_vector, dst_summary)
    follow = clamp01(
        query_to_dst * 0.50
        + current_to_dst * 0.15
        + query_to_edge * 0.20
        + path_to_dst * 0.05
        + confidence * 0.10
        - hop * 0.03
    )
    include = clamp01(query_to_dst * 0.80 + query_to_edge * 0.10 + confidence * 0.10)
    read_full = clamp01(include * 0.75 + follow * 0.25)
    expand = clamp01(follow * 0.80 + current_to_dst * 0.20 - hop * 0.05)
    stop = clamp01(1.0 - expand)
    return follow, read_full, include, expand, stop


def attach_target(
    *,
    new_summary: Sequence[float],
    candidate_summary: Sequence[float],
    new_full: Sequence[float],
    candidate_full: Sequence[float],
    path_vector: Sequence[float],
) -> float:
    summary = cosine01(new_summary, candidate_summary)
    full = cosine01(new_full, candidate_full)
    path = cosine01(path_vector, candidate_summary)
    return clamp01(summary * 0.60 + full * 0.30 + path * 0.10)


def random_unit_vector(rng: random.Random, dimension: int) -> tuple[float, ...]:
    return normalize([rng.gauss(0.0, 1.0) for _ in range(dimension)])


def nearby_topic(rng: random.Random, topic: int, topic_count: int, *, near_probability: float) -> int:
    if rng.random() > near_probability:
        return rng.randrange(topic_count)
    offset = rng.choice([-2, -1, 0, 1, 2])
    return (topic + offset) % topic_count


def noisy_vector(
    rng: random.Random,
    base: Sequence[float],
    *,
    noise: float,
    dimension: int,
) -> tuple[float, ...]:
    resized = resize_vector(base, dimension)
    values = [value + rng.gauss(0.0, noise) for value in resized]
    return normalize(values)


def stable_edge_vector(src: Sequence[float], dst: Sequence[float], dimension: int) -> tuple[float, ...]:
    src_resized = resize_vector(src, dimension)
    dst_resized = resize_vector(dst, dimension)
    return normalize(
        [
            math.tanh((dst_value - src_value) * 0.8 + src_value * dst_value * 0.4)
            for src_value, dst_value in zip(src_resized, dst_resized)
        ]
    )


def blend_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    dimension = len(vectors[0])
    blended = [0.0] * dimension
    for vector in vectors:
        resized = resize_vector(vector, dimension)
        for index, value in enumerate(resized):
            blended[index] += value
    return normalize(blended)


def resize_vector(vector: Sequence[float], dimension: int) -> tuple[float, ...]:
    if len(vector) == dimension:
        return tuple(vector)
    resized = [0.0] * dimension
    for index, value in enumerate(vector):
        resized[index % dimension] += value
    return normalize(resized)


def normalize(vector: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / magnitude for value in values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    shared = min(len(left), len(right))
    left_values = left[:shared]
    right_values = right[:shared]
    denominator = math.sqrt(sum(value * value for value in left_values)) * math.sqrt(
        sum(value * value for value in right_values)
    )
    if denominator == 0.0:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left_values, right_values)) / denominator


def cosine01(left: Sequence[float], right: Sequence[float]) -> float:
    return clamp01((cosine(left, right) + 1.0) / 2.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def round_vector(vector: Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in vector]


if __name__ == "__main__":
    main()
