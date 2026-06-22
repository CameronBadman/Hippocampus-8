#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


SCORE_KEYS = ("follow", "read_full", "include", "expand", "stop")
DEFAULT_BASE_URLS = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
NON_RETRYABLE_ERROR_MARKERS = (
    "AllocationQuota.FreeTierOnly",
    "free tier of the model has been exhausted",
    "InvalidApiKey",
    "Invalid API-key",
    "Unauthorized",
)


class NonRetryableTeacherError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Label teacher traversal episodes with Qwen.")
    parser.add_argument("--episodes-dir", default="data/teacher_episodes")
    parser.add_argument("--output-dir", default="data/qwen_teacher_episodes")
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen-plus"))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL"))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("QWEN_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError(f"missing API key in {args.api_key_env} or QWEN_API_KEY")

    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_shard_args(args.shard_index, args.shard_count)
    output_name = args.output_file or default_output_name(args.shard_index)
    output_path = output_dir / output_name
    if args.overwrite and output_path.exists():
        output_path.unlink()

    done_ids = read_done_ids(output_path)
    base_urls = [args.base_url] if args.base_url else list(DEFAULT_BASE_URLS)
    selected = 0
    labeled = 0

    with output_path.open("a", encoding="utf-8") as output:
        for episode_index, episode in enumerate(read_episodes(episodes_dir)):
            if not episode_in_shard(episode_index, shard_index=args.shard_index, shard_count=args.shard_count):
                continue
            if args.max_episodes is not None and selected >= args.max_episodes:
                break
            selected += 1
            if episode["id"] in done_ids:
                continue
            if args.dry_run:
                print(build_prompt(episode))
                labeled += 1
                continue
            labels = label_episode(
                episode,
                api_key=api_key or "",
                base_urls=base_urls,
                model=args.model,
                timeout=args.request_timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
                json_mode=not args.no_json_mode,
            )
            labeled_episode = apply_labels(episode, labels)
            output.write(json.dumps(labeled_episode, separators=(",", ":")) + "\n")
            output.flush()
            done_ids.add(episode["id"])
            labeled += 1
            print(f"labeled {episode['id']} ({len(episode['candidates'])} candidates)")

    write_manifest(output_dir, source_dir=episodes_dir, model=args.model)
    print(f"selected {selected} episodes, labeled {labeled} new episodes into {output_path}")


def validate_shard_args(shard_index: int | None, shard_count: int | None) -> None:
    if shard_index is None and shard_count is None:
        return
    if shard_index is None or shard_count is None:
        raise ValueError("--shard-index and --shard-count must be provided together")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")


def default_output_name(shard_index: int | None) -> str:
    if shard_index is None:
        return "episodes_000.jsonl"
    return f"episodes_{shard_index:03d}.jsonl"


def episode_in_shard(episode_index: int, *, shard_index: int | None, shard_count: int | None) -> bool:
    if shard_index is None or shard_count is None:
        return True
    return episode_index % shard_count == shard_index


def label_episode(
    episode: dict,
    *,
    api_key: str,
    base_urls: list[str],
    model: str,
    timeout: float,
    retries: int,
    retry_delay: float,
    json_mode: bool,
) -> dict[str, dict[str, float]]:
    prompt = build_prompt(episode)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        for base_url in base_urls:
            try:
                content = call_chat_completion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    timeout=timeout,
                    json_mode=json_mode,
                )
                return parse_labels(content, expected_ids=[candidate["id"] for candidate in episode["candidates"]])
            except NonRetryableTeacherError:
                raise
            except Exception as exc:  # noqa: BLE001 - script should retry API and parse failures.
                last_error = exc
        if attempt < retries:
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"failed to label episode {episode['id']}: {last_error}") from last_error


def call_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    json_mode: bool,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic graph traversal teacher. Label candidates for a small student "
                    "model. Use all provided relationship metadata, not just lexical overlap. Return only "
                    "valid JSON. Scores must be calibrated floats from 0 to 1."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"{url} returned HTTP {exc.code}: {body[:800]}"
        if is_non_retryable_http_error(exc.code, body):
            raise NonRetryableTeacherError(message) from exc
        raise RuntimeError(message) from exc
    data = json.loads(body)
    return data["choices"][0]["message"]["content"]


def is_non_retryable_http_error(status_code: int, body: str) -> bool:
    if status_code == 401:
        return True
    return status_code == 403 and any(marker in body for marker in NON_RETRYABLE_ERROR_MARKERS)


def build_prompt(episode: dict) -> str:
    candidates = []
    for candidate in episode["candidates"]:
        candidates.append(
            {
                "id": candidate["id"],
                "parent_id": candidate["parent_id"],
                "dst_id": candidate["dst_id"],
                "candidate_kind": candidate.get("kind", ""),
                "source_topic": candidate.get("source_topic", ""),
                "destination_topic": candidate.get("destination_topic", candidate.get("node_topic", "")),
                "destination_plain_topic": candidate.get("destination_plain_topic", ""),
                "destination_terms": candidate.get("destination_terms", []),
                "edge_summary": candidate.get("edge_summary", ""),
                "edge_confidence": candidate.get("confidence", 0.0),
                "hop": candidate.get("hop", 0),
                "retrieval_reason": candidate.get("retrieval_reason", ""),
                "relation": candidate.get("relation", {}),
                "node_summary": candidate.get("node_summary", ""),
                "node_full": candidate.get("node_full", ""),
            }
        )
    task = {
        "episode_id": episode["id"],
        "query": episode["query"],
        "query_intent": episode.get("query_intent", {}),
        "expected_topic": episode.get("expected_topic", ""),
        "path": [
            {
                "node_id": node.get("node_id"),
                "summary": node.get("summary", ""),
                "topic": node.get("topic", ""),
                "plain_topic": node.get("plain_topic", ""),
                "terms": node.get("terms", []),
            }
            for node in episode.get("path", [])
        ],
        "current_node": {
            "node_id": episode.get("current_node", {}).get("node_id"),
            "summary": episode.get("current_node", {}).get("summary", ""),
            "full": episode.get("current_node", {}).get("full", ""),
            "topic": episode.get("current_node", {}).get("topic", ""),
            "plain_topic": episode.get("current_node", {}).get("plain_topic", ""),
            "terms": episode.get("current_node", {}).get("terms", []),
        },
        "candidates": candidates,
    }
    return (
        "Label each candidate for whether a deterministic graph traversal student should follow it.\n"
        "Use these score meanings:\n"
        "- follow: traverse this edge next.\n"
        "- read_full: inspect the full destination node before deciding.\n"
        "- include: include the destination node in the answer set.\n"
        "- expand: continue exploring from the destination node.\n"
        "- stop: stop this branch after this candidate.\n"
        "Decision policy:\n"
        "- Use query_intent/expected_topic as the target for this synthetic teacher task.\n"
        "- A destination that matches the target topic and terms should usually receive high follow/include/expand.\n"
        "- A candidate with similar words but the wrong destination topic is a hard negative: prefer read_full over follow/include.\n"
        "- A bridge or expressway-style edge should get follow only when it moves toward the target topic; otherwise stop it.\n"
        "- Keep follow and include separate: follow means useful next traversal step, include means useful answer node.\n"
        "- Calibrate aggressively: strongest target candidates near 0.9-1.0, clear off-topic negatives near 0.0-0.1.\n"
        'Return exactly: {"candidates":[{"id":"...","follow":0.0,"read_full":0.0,"include":0.0,"expand":0.0,"stop":0.0}]}.\n'
        "Do not include markdown, prose, or extra keys.\n\n"
        f"Task JSON:\n{json.dumps(task, ensure_ascii=True, separators=(',', ':'))}"
    )


def parse_labels(content: str, *, expected_ids: list[str]) -> dict[str, dict[str, float]]:
    data = json.loads(strip_json_fence(content))
    rows = data.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("teacher response must contain a candidates list")
    labels: dict[str, dict[str, float]] = {}
    for row in rows:
        candidate_id = row.get("id")
        if candidate_id not in expected_ids:
            raise ValueError(f"unexpected candidate id in teacher response: {candidate_id}")
        labels[candidate_id] = {key: clamp01(float(row[key])) for key in SCORE_KEYS}
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in labels]
    if missing:
        raise ValueError(f"teacher response missing candidate ids: {missing[:5]}")
    return labels


def apply_labels(episode: dict, labels: dict[str, dict[str, float]]) -> dict:
    labeled = json.loads(json.dumps(episode))
    for candidate in labeled["candidates"]:
        candidate["qwen_teacher"] = {key: round(value, 6) for key, value in labels[candidate["id"]].items()}
    return labeled


def strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def read_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {episode["id"] for episode in read_jsonl(output_path)}


def read_episodes(episodes_dir: Path) -> Iterable[dict]:
    for path in sorted(episodes_dir.glob("episodes_*.jsonl")):
        yield from read_jsonl(path)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_manifest(output_dir: Path, *, source_dir: Path, model: str) -> None:
    episode_files = sorted(path.name for path in output_dir.glob("episodes_*.jsonl"))
    labeled_count = count_unique_episode_ids(output_dir)
    manifest = {
        "schema_version": 1,
        "kind": "qwen_teacher_episodes",
        "source_dir": str(source_dir),
        "teacher_model": model,
        "episodes": labeled_count,
        "episode_files": episode_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_unique_episode_ids(output_dir: Path) -> int:
    episode_ids = set()
    for path in sorted(output_dir.glob("episodes_*.jsonl")):
        episode_ids.update(episode["id"] for episode in read_jsonl(path))
    return len(episode_ids)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
