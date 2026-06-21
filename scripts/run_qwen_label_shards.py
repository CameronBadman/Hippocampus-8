#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


LABELER_PATH = Path(__file__).resolve().parent / "label_teacher_episodes_qwen.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen teacher labeling in timeout-bounded shards.")
    parser.add_argument("--episodes-dir", default="data/teacher_episodes")
    parser.add_argument("--output-dir", default="data/qwen_teacher_episodes")
    parser.add_argument("--shard-count", type=int, default=64)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--expected-per-shard", type=int, default=None)
    parser.add_argument("--shard-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--progress-file", default=None)
    args = parser.parse_args()

    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.start_shard < 0 or args.start_shard >= args.shard_count:
        raise ValueError("--start-shard must be in [0, shard-count)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_file) if args.progress_file else output_dir / "shard_progress.json"
    selected_shards = list(range(args.start_shard, args.shard_count))
    if args.max_shards is not None:
        selected_shards = selected_shards[: args.max_shards]

    results = []
    started_all = time.time()
    for shard in selected_shards:
        shard_path = output_dir / f"episodes_{shard:03d}.jsonl"
        line_count = count_lines(shard_path)
        if args.expected_per_shard is not None and line_count >= args.expected_per_shard:
            result = shard_result(shard, "complete", line_count=line_count, seconds=0.0, detail="already complete")
            results.append(result)
            write_progress(progress_path, args=args, results=results, started_all=started_all)
            print(format_result(result), flush=True)
            continue

        command = build_command(args, shard)
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.shard_timeout,
            )
            seconds = time.time() - started
            line_count = count_lines(shard_path)
            status = classify_shard_status(
                returncode=completed.returncode,
                line_count=line_count,
                expected_per_shard=args.expected_per_shard,
            )
            detail = completed.stdout[-3000:]
            result = shard_result(
                shard,
                status,
                line_count=line_count,
                seconds=seconds,
                returncode=completed.returncode,
                detail=detail,
            )
        except subprocess.TimeoutExpired as exc:
            seconds = time.time() - started
            line_count = count_lines(shard_path)
            detail = timeout_output(exc)
            result = shard_result(
                shard,
                "timeout",
                line_count=line_count,
                seconds=seconds,
                returncode=None,
                detail=detail,
            )

        results.append(result)
        write_progress(progress_path, args=args, results=results, started_all=started_all)
        print(format_result(result), flush=True)
        if result["status"] != "complete" and not args.continue_on_failure:
            raise SystemExit(1)

    write_progress(progress_path, args=args, results=results, started_all=started_all)
    failures = [result for result in results if result["status"] != "complete"]
    if failures:
        raise SystemExit(1)


def build_command(args: argparse.Namespace, shard: int) -> list[str]:
    command = [
        sys.executable,
        str(LABELER_PATH),
        "--episodes-dir",
        args.episodes_dir,
        "--output-dir",
        args.output_dir,
        "--shard-index",
        str(shard),
        "--shard-count",
        str(args.shard_count),
        "--request-timeout",
        str(args.request_timeout),
        "--retries",
        str(args.retries),
        "--retry-delay",
        str(args.retry_delay),
    ]
    if args.model is not None:
        command.extend(["--model", args.model])
    if args.base_url is not None:
        command.extend(["--base-url", args.base_url])
    return command


def classify_shard_status(*, returncode: int, line_count: int, expected_per_shard: int | None) -> str:
    if returncode != 0:
        return "error"
    if expected_per_shard is not None and line_count < expected_per_shard:
        return "partial"
    return "complete"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def shard_result(
    shard: int,
    status: str,
    *,
    line_count: int,
    seconds: float,
    returncode: int | None = None,
    detail: str = "",
) -> dict:
    return {
        "shard": shard,
        "status": status,
        "line_count": line_count,
        "seconds": round(seconds, 3),
        "returncode": returncode,
        "detail_tail": detail,
    }


def format_result(result: dict) -> str:
    return (
        f"shard {result['shard']:03d}: {result['status']} "
        f"lines={result['line_count']} seconds={result['seconds']} returncode={result['returncode']}"
    )


def timeout_output(exc: subprocess.TimeoutExpired) -> str:
    output = exc.output or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return str(output)[-3000:]


def write_progress(
    progress_path: Path,
    *,
    args: argparse.Namespace,
    results: list[dict],
    started_all: float,
) -> None:
    progress = {
        "schema_version": 1,
        "episodes_dir": args.episodes_dir,
        "output_dir": args.output_dir,
        "shard_count": args.shard_count,
        "expected_per_shard": args.expected_per_shard,
        "elapsed_seconds": round(time.time() - started_all, 3),
        "results": results,
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
