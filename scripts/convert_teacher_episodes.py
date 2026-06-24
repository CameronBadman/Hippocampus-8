#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import embed_text
from vector_graph.vectors import blend_vectors, effective_summary_vector, metadata_vector_from, resize_vector, stable_edge_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert teacher traversal episodes into scorer training JSONL.")
    parser.add_argument("--episodes-dir", default="data/teacher_episodes")
    parser.add_argument("--output-data-dir", default="data/teacher_scorer")
    parser.add_argument("--output-ranking-dir", default="data/teacher_ranked")
    parser.add_argument("--teacher-key", default="qwen_teacher")
    args = parser.parse_args()

    episodes = list(read_episodes(Path(args.episodes_dir)))
    if not episodes:
        raise ValueError(f"no episodes found in {args.episodes_dir}")

    data_dir = Path(args.output_data_dir)
    ranking_dir = Path(args.output_ranking_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir.mkdir(parents=True, exist_ok=True)

    traversal_count, attach_count = write_regression_files(
        episodes,
        data_dir=data_dir,
        teacher_key=args.teacher_key,
    )
    traversal_ranking_count, attach_ranking_count = write_ranking_files(
        episodes,
        ranking_dir=ranking_dir,
        teacher_key=args.teacher_key,
    )
    write_manifest(data_dir, traversal_count=traversal_count, attach_count=attach_count)
    write_ranking_manifest(
        ranking_dir,
        traversal_ranking_count=traversal_ranking_count,
        attach_ranking_count=attach_ranking_count,
    )
    print(
        f"wrote {traversal_count} traversal, {attach_count} attach, "
        f"{traversal_ranking_count} traversal ranking, {attach_ranking_count} attach ranking examples"
    )


def write_regression_files(episodes: Sequence[dict], *, data_dir: Path, teacher_key: str) -> tuple[int, int]:
    traversal_count = 0
    attach_count = 0
    with (data_dir / "traversal_000.jsonl").open("w", encoding="utf-8") as traversal_output, (
        data_dir / "attach_000.jsonl"
    ).open("w", encoding="utf-8") as attach_output:
        for episode in episodes:
            query = effective_text_vector(episode["query"], episode.get("query_intent", {}), 32)
            current_summary = effective_text_vector(
                episode["current_node"]["summary"],
                node_metadata(episode.get("current_node", {})),
                32,
            )
            path_vector = episode_path_vector(episode)
            new_full = embed_text(episode["query"] + " " + episode["expected_topic"], 64)
            for candidate in episode["candidates"]:
                teacher = candidate_teacher(candidate, teacher_key)
                dst_summary = effective_text_vector(
                    candidate["node_summary"],
                    candidate_metadata(candidate),
                    32,
                )
                dst_full = embed_text(candidate.get("node_full") or candidate["node_summary"], 64)
                edge = stable_edge_vector(current_summary, dst_summary, 16)
                target = teacher_target(teacher)
                traversal_output.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "traversal",
                            "id": f"{candidate['id']}:regression",
                            "query": round_vector(query),
                            "current_summary": round_vector(current_summary),
                            "edge": round_vector(edge),
                            "dst_summary": round_vector(dst_summary),
                            "path": round_vector(path_vector),
                            "confidence": candidate["confidence"],
                            "hop": candidate.get("hop", 0),
                            "target": round_vector(target),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                traversal_count += 1
                attach_output.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "attach",
                            "id": f"{candidate['id']}:attach",
                            "new_summary": round_vector(query),
                            "candidate_summary": round_vector(dst_summary),
                            "new_full": round_vector(new_full),
                            "candidate_full": round_vector(dst_full),
                            "path": round_vector(path_vector),
                            "target": round(float(teacher["include"]), 6),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                attach_count += 1
    return traversal_count, attach_count


def write_ranking_files(episodes: Sequence[dict], *, ranking_dir: Path, teacher_key: str) -> tuple[int, int]:
    traversal_count = 0
    attach_count = 0
    with (ranking_dir / "traversal_ranking.jsonl").open("w", encoding="utf-8") as traversal_output, (
        ranking_dir / "attach_ranking.jsonl"
    ).open("w", encoding="utf-8") as attach_output:
        for episode in episodes:
            query = effective_text_vector(episode["query"], episode.get("query_intent", {}), 32)
            current_summary = effective_text_vector(
                episode["current_node"]["summary"],
                node_metadata(episode.get("current_node", {})),
                32,
            )
            path_vector = episode_path_vector(episode)
            new_full = embed_text(episode["query"] + " " + episode["expected_topic"], 64)
            traversal_candidates = []
            attach_candidates = []
            for candidate in episode["candidates"]:
                teacher = candidate_teacher(candidate, teacher_key)
                dst_summary = effective_text_vector(
                    candidate["node_summary"],
                    candidate_metadata(candidate),
                    32,
                )
                dst_full = embed_text(candidate.get("node_full") or candidate["node_summary"], 64)
                traversal_target = float(teacher["follow"])
                result_target = float(teacher.get("result", teacher["include"]))
                attach_target = float(teacher["include"])
                traversal_candidates.append(
                    {
                        "id": candidate["id"],
                        "kind": candidate["kind"],
                        "label": hard_label(traversal_target),
                        "rank_target": round(traversal_target, 6),
                        "result_label": hard_label(result_target),
                        "result_rank_target": round(result_target, 6),
                        "weight": rank_weight(result_target),
                        "dst_summary": round_vector(dst_summary),
                        "edge": round_vector(stable_edge_vector(current_summary, dst_summary, 16)),
                        "confidence": candidate["confidence"],
                        "hop": candidate.get("hop", 0),
                        "oracle": round_vector(teacher_target(teacher)),
                    }
                )
                attach_candidates.append(
                    {
                        "id": candidate["id"],
                        "kind": candidate["kind"],
                        "label": hard_label(attach_target),
                        "rank_target": round(attach_target, 6),
                        "weight": rank_weight(attach_target),
                        "candidate_summary": round_vector(dst_summary),
                        "candidate_full": round_vector(dst_full),
                        "oracle": round(attach_target, 6),
                    }
                )
            traversal_output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "traversal_ranking",
                        "id": f"{episode['id']}:traversal-ranking",
                        "query": round_vector(query),
                        "current_summary": round_vector(current_summary),
                        "path": round_vector(path_vector),
                        "candidates": traversal_candidates,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            attach_output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "attach_ranking",
                        "id": f"{episode['id']}:attach-ranking",
                        "new_summary": round_vector(query),
                        "new_full": round_vector(new_full),
                        "path": round_vector(path_vector),
                        "candidates": attach_candidates,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            traversal_count += 1
            attach_count += 1
    return traversal_count, attach_count


def candidate_teacher(candidate: dict, teacher_key: str) -> dict[str, float]:
    teacher = candidate.get(teacher_key) or candidate.get("teacher") or candidate.get("bootstrap_teacher")
    if teacher is None:
        raise ValueError(f"candidate {candidate.get('id')} has no teacher labels")
    return {
        "follow": float(teacher["follow"]),
        "read_full": float(teacher["read_full"]),
        "include": float(teacher["include"]),
        "expand": float(teacher["expand"]),
        "stop": float(teacher["stop"]),
        "result": float(teacher.get("result", teacher["include"])),
    }


def teacher_target(teacher: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        teacher["follow"],
        teacher["read_full"],
        teacher["include"],
        teacher["expand"],
        teacher["stop"],
        teacher.get("result", teacher["include"]),
    )


def hard_label(score: float) -> int:
    return 1 if score >= 0.65 else 0


def rank_weight(score: float) -> float:
    return round(0.5 + abs(float(score) - 0.5) * 2.0, 6)


def episode_path_vector(episode: dict):
    summaries = [
        effective_text_vector(node["summary"], node_metadata(node), 32)
        for node in episode.get("path", [])
    ]
    if not summaries:
        return effective_text_vector(episode["query"], episode.get("query_intent", {}), 32)
    return blend_vectors(summaries, 32)


def effective_text_vector(text: str, metadata: dict, dimension: int):
    summary = embed_text(text, dimension)
    metadata_vector = metadata_vector_from(metadata, dimension)
    return effective_summary_vector(summary, metadata_vector, dimension=dimension)


def node_metadata(node: dict) -> dict:
    return {
        "topic": node.get("topic", ""),
        "plain_topic": node.get("plain_topic", ""),
        "terms": node.get("terms", []),
    }


def candidate_metadata(candidate: dict) -> dict:
    return {
        "topic": candidate.get("node_topic") or candidate.get("destination_topic", ""),
        "plain_topic": candidate.get("destination_plain_topic", ""),
        "terms": candidate.get("destination_terms", []),
        "kind": candidate.get("kind", ""),
        "retrieval_reason": candidate.get("retrieval_reason", ""),
        "relation": candidate.get("relation", {}),
    }


def read_episodes(episodes_dir: Path) -> Iterable[dict]:
    for path in sorted(episodes_dir.glob("episodes_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def write_manifest(data_dir: Path, *, traversal_count: int, attach_count: int) -> None:
    manifest = {
        "schema_version": 1,
        "dimensions": {
            "query": 32,
            "summary": 32,
            "edge": 16,
            "full": 64,
            "path": 32,
            "metadata": 32,
            "traversal": 16,
            "scalars": 2,
        },
        "shards": [
            {
                "traversal": "traversal_000.jsonl",
                "traversal_examples": traversal_count,
                "attach": "attach_000.jsonl",
                "attach_examples": attach_count,
            }
        ],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ranking_manifest(ranking_dir: Path, *, traversal_ranking_count: int, attach_ranking_count: int) -> None:
    manifest = {
        "schema_version": 1,
        "dimensions": {
            "metadata": 32,
            "traversal": 16,
        },
        "traversal_ranking": "traversal_ranking.jsonl",
        "traversal_ranking_cases": traversal_ranking_count,
        "attach_ranking": "attach_ranking.jsonl",
        "attach_ranking_cases": attach_ranking_count,
    }
    (ranking_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_vector(vector: Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in resize_vector(vector, len(vector))]


if __name__ == "__main__":
    main()
