#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from generate_synthetic_training_data import (
    EDGE_DIM,
    FULL_DIM,
    PATH_DIM,
    QUERY_DIM,
    SUMMARY_DIM,
    attach_target,
    blend_vectors,
    noisy_vector,
    random_unit_vector,
    resize_vector,
    round_vector,
    stable_edge_vector,
    traversal_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic hard benchmark cases.")
    parser.add_argument("--output-dir", default="data/benchmarks/synthetic")
    parser.add_argument("--traversal-cases", type=int, default=700)
    parser.add_argument("--attach-cases", type=int, default=450)
    parser.add_argument("--seed", type=int, default=9091)
    args = parser.parse_args()

    generate_benchmark(
        output_dir=Path(args.output_dir),
        traversal_cases=args.traversal_cases,
        attach_cases=args.attach_cases,
        seed=args.seed,
    )


def generate_benchmark(
    *,
    output_dir: Path,
    traversal_cases: int,
    attach_cases: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    topics = [random_unit_vector(rng, SUMMARY_DIM) for _ in range(28)]
    full_topics = [resize_vector(topic, FULL_DIM) for topic in topics]

    traversal_path = output_dir / "traversal_ranking.jsonl"
    attach_path = output_dir / "attach_ranking.jsonl"
    write_traversal_cases(traversal_path, rng=rng, topics=topics, count=traversal_cases)
    write_attach_cases(attach_path, rng=rng, topics=topics, full_topics=full_topics, count=attach_cases)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "dimensions": {
            "query": QUERY_DIM,
            "summary": SUMMARY_DIM,
            "edge": EDGE_DIM,
            "full": FULL_DIM,
            "path": PATH_DIM,
            "scalars": 2,
        },
        "files": {
            "traversal": traversal_path.name,
            "attach": attach_path.name,
        },
        "cases": {
            "traversal": traversal_cases,
            "attach": attach_cases,
        },
        "candidate_types": [
            "positive",
            "hard_summary_negative",
            "hard_edge_negative",
            "adversarial_confidence_negative",
            "same_summary_wrong_full_negative",
            "same_full_wrong_summary_negative",
            "path_aligned_wrong_full_negative",
            "near_duplicate_flipped_negative",
            "random_negative",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote benchmark cases to {output_dir}")


def write_traversal_cases(
    path: Path,
    *,
    rng: random.Random,
    topics: Sequence[Sequence[float]],
    count: int,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for case_index in range(count):
            query_topic = rng.randrange(len(topics))
            near_topic = (query_topic + rng.choice([-1, 0, 1])) % len(topics)
            far_topic = (query_topic + rng.randrange(8, len(topics) - 4)) % len(topics)

            query = noisy_vector(rng, topics[query_topic], noise=0.18, dimension=QUERY_DIM)
            current = noisy_vector(rng, topics[near_topic], noise=0.24, dimension=SUMMARY_DIM)
            path_vector = noisy_vector(
                rng,
                blend_vectors([topics[query_topic], topics[near_topic]]),
                noise=0.24,
                dimension=PATH_DIM,
            )

            candidates = []
            candidates.append(
                traversal_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=0,
                    kind="positive",
                    label=1,
                    query=query,
                    current=current,
                    path_vector=path_vector,
                    dst_base=topics[query_topic],
                    edge_base=resize_vector(query, EDGE_DIM),
                    confidence=0.94,
                    hop=0,
                    dst_noise=0.16,
                    edge_noise=0.16,
                )
            )
            candidates.append(
                traversal_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=1,
                    kind="positive",
                    label=1,
                    query=query,
                    current=current,
                    path_vector=path_vector,
                    dst_base=topics[near_topic],
                    edge_base=resize_vector(query, EDGE_DIM),
                    confidence=0.88,
                    hop=1,
                    dst_noise=0.20,
                    edge_noise=0.20,
                )
            )
            candidates.append(
                traversal_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=2,
                    kind="hard_summary_negative",
                    label=0,
                    query=query,
                    current=current,
                    path_vector=path_vector,
                    dst_base=topics[query_topic],
                    edge_base=negate(resize_vector(query, EDGE_DIM)),
                    confidence=0.35,
                    hop=3,
                    dst_noise=0.18,
                    edge_noise=0.10,
                )
            )
            candidates.append(
                traversal_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=3,
                    kind="hard_edge_negative",
                    label=0,
                    query=query,
                    current=current,
                    path_vector=path_vector,
                    dst_base=topics[far_topic],
                    edge_base=resize_vector(query, EDGE_DIM),
                    confidence=0.91,
                    hop=0,
                    dst_noise=0.22,
                    edge_noise=0.12,
                )
            )
            candidates.append(
                traversal_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=4,
                    kind="adversarial_confidence_negative",
                    label=0,
                    query=query,
                    current=current,
                    path_vector=path_vector,
                    dst_base=topics[far_topic],
                    edge_base=stable_edge_vector(current, topics[far_topic], EDGE_DIM),
                    confidence=1.0,
                    hop=0,
                    dst_noise=0.12,
                    edge_noise=0.12,
                )
            )

            for candidate_index in range(5, 12):
                random_topic = rng.randrange(len(topics))
                candidates.append(
                    traversal_candidate(
                        rng,
                        case_index=case_index,
                        candidate_index=candidate_index,
                        kind="random_negative",
                        label=0,
                        query=query,
                        current=current,
                        path_vector=path_vector,
                        dst_base=topics[random_topic],
                        edge_base=stable_edge_vector(current, topics[random_topic], EDGE_DIM),
                        confidence=rng.uniform(0.40, 0.95),
                        hop=rng.randrange(4),
                        dst_noise=0.38,
                        edge_noise=0.38,
                    )
                )

            rng.shuffle(candidates)
            output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "traversal_ranking",
                        "id": f"traversal-benchmark-{case_index:06d}",
                        "query": round_vector(query),
                        "current_summary": round_vector(current),
                        "path": round_vector(path_vector),
                        "candidates": candidates,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def traversal_candidate(
    rng: random.Random,
    *,
    case_index: int,
    candidate_index: int,
    kind: str,
    label: int,
    query: Sequence[float],
    current: Sequence[float],
    path_vector: Sequence[float],
    dst_base: Sequence[float],
    edge_base: Sequence[float],
    confidence: float,
    hop: int,
    dst_noise: float,
    edge_noise: float,
) -> dict:
    dst = noisy_vector(rng, dst_base, noise=dst_noise, dimension=SUMMARY_DIM)
    edge = noisy_vector(rng, edge_base, noise=edge_noise, dimension=EDGE_DIM)
    target = traversal_targets(
        query_vector=query,
        current_summary=current,
        edge_vector=edge,
        dst_summary=dst,
        path_vector=path_vector,
        confidence=confidence,
        hop=hop,
    )
    return {
        "id": f"t{case_index:06d}-c{candidate_index:02d}",
        "kind": kind,
        "label": label,
        "dst_summary": round_vector(dst),
        "edge": round_vector(edge),
        "confidence": round(confidence, 6),
        "hop": hop,
        "oracle": round_vector(target),
    }


def write_attach_cases(
    path: Path,
    *,
    rng: random.Random,
    topics: Sequence[Sequence[float]],
    full_topics: Sequence[Sequence[float]],
    count: int,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for case_index in range(count):
            new_topic = rng.randrange(len(topics))
            near_topic = (new_topic + rng.choice([-1, 0, 1])) % len(topics)
            far_topic = (new_topic + rng.randrange(8, len(topics) - 4)) % len(topics)

            new_summary = noisy_vector(rng, topics[new_topic], noise=0.18, dimension=SUMMARY_DIM)
            new_full = noisy_vector(rng, full_topics[new_topic], noise=0.18, dimension=FULL_DIM)
            path_vector = noisy_vector(
                rng,
                blend_vectors([topics[new_topic], topics[near_topic]]),
                noise=0.25,
                dimension=PATH_DIM,
            )

            candidates = [
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=0,
                    kind="positive",
                    label=1,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[new_topic],
                    full_base=full_topics[new_topic],
                    summary_noise=0.18,
                    full_noise=0.18,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=1,
                    kind="positive",
                    label=1,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[near_topic],
                    full_base=full_topics[near_topic],
                    summary_noise=0.22,
                    full_noise=0.22,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=2,
                    kind="hard_summary_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[new_topic],
                    full_base=full_topics[far_topic],
                    summary_noise=0.16,
                    full_noise=0.14,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=3,
                    kind="hard_full_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[far_topic],
                    full_base=full_topics[new_topic],
                    summary_noise=0.16,
                    full_noise=0.14,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=4,
                    kind="adversarial_path_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[far_topic],
                    full_base=full_topics[far_topic],
                    summary_noise=0.18,
                    full_noise=0.18,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=5,
                    kind="same_summary_wrong_full_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[new_topic],
                    full_base=full_topics[(new_topic + rng.randrange(6, len(topics) - 3)) % len(topics)],
                    summary_noise=0.08,
                    full_noise=0.10,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=6,
                    kind="same_full_wrong_summary_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[(new_topic + rng.randrange(6, len(topics) - 3)) % len(topics)],
                    full_base=full_topics[new_topic],
                    summary_noise=0.10,
                    full_noise=0.08,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=7,
                    kind="path_aligned_wrong_full_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=topics[near_topic],
                    full_base=full_topics[far_topic],
                    summary_noise=0.16,
                    full_noise=0.16,
                ),
                attach_candidate(
                    rng,
                    case_index=case_index,
                    candidate_index=8,
                    kind="near_duplicate_flipped_negative",
                    label=0,
                    new_summary=new_summary,
                    new_full=new_full,
                    path_vector=path_vector,
                    summary_base=flip_alternating(topics[new_topic]),
                    full_base=flip_alternating(full_topics[new_topic]),
                    summary_noise=0.06,
                    full_noise=0.06,
                ),
            ]

            for candidate_index in range(9, 16):
                random_topic = rng.randrange(len(topics))
                candidates.append(
                    attach_candidate(
                        rng,
                        case_index=case_index,
                        candidate_index=candidate_index,
                        kind="random_negative",
                        label=0,
                        new_summary=new_summary,
                        new_full=new_full,
                        path_vector=path_vector,
                        summary_base=topics[random_topic],
                        full_base=full_topics[random_topic],
                        summary_noise=0.40,
                        full_noise=0.40,
                    )
                )

            rng.shuffle(candidates)
            output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "attach_ranking",
                        "id": f"attach-benchmark-{case_index:06d}",
                        "new_summary": round_vector(new_summary),
                        "new_full": round_vector(new_full),
                        "path": round_vector(path_vector),
                        "candidates": candidates,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def attach_candidate(
    rng: random.Random,
    *,
    case_index: int,
    candidate_index: int,
    kind: str,
    label: int,
    new_summary: Sequence[float],
    new_full: Sequence[float],
    path_vector: Sequence[float],
    summary_base: Sequence[float],
    full_base: Sequence[float],
    summary_noise: float,
    full_noise: float,
) -> dict:
    candidate_summary = noisy_vector(rng, summary_base, noise=summary_noise, dimension=SUMMARY_DIM)
    candidate_full = noisy_vector(rng, full_base, noise=full_noise, dimension=FULL_DIM)
    target = attach_target(
        new_summary=new_summary,
        candidate_summary=candidate_summary,
        new_full=new_full,
        candidate_full=candidate_full,
        path_vector=path_vector,
    )
    return {
        "id": f"a{case_index:06d}-c{candidate_index:02d}",
        "kind": kind,
        "label": label,
        "candidate_summary": round_vector(candidate_summary),
        "candidate_full": round_vector(candidate_full),
        "oracle": round(target, 6),
    }


def negate(vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(-value for value in vector)


def flip_alternating(vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(-value if index % 2 == 0 else value for index, value in enumerate(vector))


if __name__ == "__main__":
    main()
