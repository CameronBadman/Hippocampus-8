#!/usr/bin/env python3
"""Progressively mirror a local run directory to Google Drive with rclone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or sync a local Hippo/Qwen run directory to Google Drive."
    )
    parser.add_argument("--local-dir", required=True, type=Path, help="Local run directory to mirror.")
    parser.add_argument(
        "--remote",
        required=True,
        help="rclone destination, for example gdrive:hippo-qwen-runs/all_12288.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "sync"),
        default="copy",
        help="Use copy for safe append/update mirroring. Sync may delete remote files.",
    )
    parser.add_argument("--interval-seconds", type=float, default=180.0)
    parser.add_argument("--max-cycles", type=int, default=0, help="Use 0 to run until interrupted.")
    parser.add_argument("--rclone-bin", default="rclone")
    parser.add_argument("--include", action="append", default=[], help="rclone include pattern.")
    parser.add_argument("--exclude", action="append", default=[], help="rclone exclude pattern.")
    parser.add_argument("--transfers", type=int, default=8)
    parser.add_argument("--checkers", type=int, default=16)
    parser.add_argument("--stats", default="30s")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--create-local-dir", action="store_true")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path(".drive_sync_status.json"),
        help="Local JSON status file updated after every sync cycle.",
    )
    return parser.parse_args()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def tail(text: str, *, lines: int = 40) -> list[str]:
    return text.splitlines()[-lines:]


def rclone_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"Could not find {name!r}. Install rclone, then run `rclone config` "
            "and create a Google Drive remote such as `gdrive`."
        )
    return path


def build_command(args: argparse.Namespace, rclone: str) -> list[str]:
    command = [
        rclone,
        args.mode,
        str(args.local_dir),
        args.remote,
        "--transfers",
        str(args.transfers),
        "--checkers",
        str(args.checkers),
        "--stats",
        args.stats,
        "--create-empty-src-dirs",
    ]
    for pattern in args.include:
        command.extend(["--include", pattern])
    for pattern in args.exclude:
        command.extend(["--exclude", pattern])
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_cycle(args: argparse.Namespace, rclone: str, cycle: int) -> dict[str, Any]:
    command = build_command(args, rclone)
    started_at = time.time()
    process = subprocess.run(command, capture_output=True, text=True)
    finished_at = time.time()
    return {
        "cycle": cycle,
        "state": "ok" if process.returncode == 0 else "error",
        "returncode": process.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(finished_at - started_at, 3),
        "command": command,
        "stdout_tail": tail(process.stdout),
        "stderr_tail": tail(process.stderr),
    }


def main() -> int:
    args = parse_args()
    if args.create_local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)
    if not args.local_dir.exists():
        raise SystemExit(f"Local directory does not exist: {args.local_dir}")
    if not args.local_dir.is_dir():
        raise SystemExit(f"Local path is not a directory: {args.local_dir}")

    rclone = rclone_path(args.rclone_bin)
    cycle = 0
    while args.max_cycles <= 0 or cycle < args.max_cycles:
        cycle += 1
        payload = run_cycle(args, rclone, cycle)
        write_status(args.status_file, payload)
        print(
            f"cycle {cycle}: {payload['state']} "
            f"returncode={payload['returncode']} elapsed={payload['elapsed_seconds']}s",
            flush=True,
        )
        if payload["returncode"] != 0:
            for line in payload["stderr_tail"]:
                print(line, flush=True)
            if args.fail_fast:
                return int(payload["returncode"])
        if args.max_cycles > 0 and cycle >= args.max_cycles:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
