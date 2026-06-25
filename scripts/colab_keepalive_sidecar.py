#!/usr/bin/env python3
"""Local Colab keepalive sidecar for long Hippo/Qwen runs.

This script owns a Colab Codex adapter session directly, so it can keep a
connected Colab notebook active without Codex manually polling it. It prints a
connection URL, waits for the browser-side Colab MCP bridge to connect, then
periodically adds a small Python cell, runs it, records status, and deletes the
cell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_REPO = Path("/home/cameron/projects/google-collab-codex-con")
DEFAULT_REMOTE_STATUS_PATHS = (
    "/content/qwen_all_12288_background_status.json",
    "/content/qwen_domain_labeler_background_status.json",
)
CELL_MARKER = "hippo-qwen keepalive sidecar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporary Colab heartbeat/status cells from a local sidecar."
    )
    parser.add_argument(
        "--adapter-repo",
        type=Path,
        default=DEFAULT_ADAPTER_REPO if DEFAULT_ADAPTER_REPO.exists() else None,
        help=(
            "Path to google-collab-codex-con if colab_codex_adapter is not "
            "installed in this Python environment."
        ),
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=180.0,
        help="Delay between heartbeat cells after each cycle finishes.",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum time to wait for browser connection. Use 0 to wait forever.",
    )
    parser.add_argument(
        "--connect-poll-seconds",
        type=float,
        default=5.0,
        help="How often to check for a browser connection while waiting.",
    )
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for each temporary cell to run.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after this many cycles. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--mode",
        choices=("keepalive", "status", "both"),
        default="both",
        help="Whether the temporary cell only heartbeats, only reports status, or both.",
    )
    parser.add_argument(
        "--remote-status-path",
        action="append",
        default=list(DEFAULT_REMOTE_STATUS_PATHS),
        help=(
            "JSON status file to summarize inside Colab. Can be passed more "
            "than once. Defaults target current Qwen background runs."
        ),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path(".colab_keepalive_status.json"),
        help="Local JSON status file updated after every sidecar cycle.",
    )
    parser.add_argument(
        "--keep-cells",
        action="store_true",
        help="Leave temporary sidecar cells in the notebook instead of deleting them.",
    )
    parser.add_argument(
        "--cleanup-existing",
        action="store_true",
        help="Delete old sidecar cells before the first cycle.",
    )
    parser.add_argument(
        "--no-publish-url",
        action="store_true",
        help="Do not write the Colab connection URL to the adapter open-url file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print connection and error lines.",
    )
    return parser.parse_args()


def load_adapter(adapter_repo: Path | None) -> tuple[type[Any], Any]:
    if adapter_repo is not None:
        src_root = adapter_repo.resolve()
        if (src_root / "colab_codex_adapter").exists():
            sys.path.insert(0, str(src_root))

    try:
        from colab_codex_adapter.jobs import result_data
        from colab_codex_adapter.session import ColabSessionManager
    except ModuleNotFoundError as exc:
        adapter_python = (
            adapter_repo.resolve() / ".venv" / "bin" / "python"
            if adapter_repo is not None
            else None
        )
        adapter_hint = (
            f" Try running with {adapter_python}."
            if adapter_python is not None and adapter_python.exists()
            else ""
        )
        raise SystemExit(
            "Could not import colab_codex_adapter. Install google-collab-codex-con "
            "or pass --adapter-repo /path/to/google-collab-codex-con. "
            f"Missing module: {exc.name}.{adapter_hint}"
        ) from exc

    return ColabSessionManager, result_data


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_cell_code(args: argparse.Namespace, cycle: int) -> str:
    remote_paths = [str(path) for path in args.remote_status_path or []]
    return f'''# {CELL_MARKER}
import json
import os
import pathlib
import time

CYCLE = {cycle}
MODE = {args.mode!r}
REMOTE_STATUS_PATHS = {remote_paths!r}


def _read_json(path):
    target = pathlib.Path(path)
    if not target.exists():
        return {{"path": str(target), "exists": False}}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {{"path": str(target), "exists": True, "error": f"{{type(exc).__name__}}: {{exc}}"}}
    if not isinstance(data, dict):
        return {{"path": str(target), "exists": True, "type": type(data).__name__}}
    payload = {{"path": str(target), "exists": True}}
    for key in (
        "state",
        "started_at",
        "finished_at",
        "running",
        "total_shards",
        "complete_shards",
        "failed_shards",
        "partial_shards",
        "episodes",
        "output_dir",
        "progress_file",
        "log_file",
    ):
        if key in data:
            payload[key] = data[key]
    for key in ("workers", "processes", "pids"):
        value = data.get(key)
        if isinstance(value, list):
            payload[key] = value
            payload[f"{{key}}_alive"] = _alive_count(value)
    return payload


def _alive_count(values):
    alive = 0
    for value in values:
        pid = value.get("pid") if isinstance(value, dict) else value
        try:
            os.kill(int(pid), 0)
        except Exception:
            continue
        alive += 1
    return alive


def _heartbeat():
    payload = {{
        "kind": "hippo_qwen_keepalive",
        "cycle": CYCLE,
        "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "timestamp": time.time(),
    }}
    pathlib.Path("/content/hippo_qwen_keepalive.json").write_text(
        json.dumps(payload, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    return payload


payload = {{
    "kind": "hippo_qwen_sidecar_cycle",
    "cycle": CYCLE,
    "mode": MODE,
    "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "timestamp": time.time(),
}}
if MODE in ("keepalive", "both"):
    payload["heartbeat"] = _heartbeat()
if MODE in ("status", "both"):
    payload["remote_status"] = [_read_json(path) for path in REMOTE_STATUS_PATHS]

print("HIPPO_SIDECAR " + json.dumps(payload, sort_keys=True))
'''


def extract_output_lines(outputs: Any) -> list[str]:
    if not isinstance(outputs, list):
        return []
    lines: list[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        text = output.get("text")
        if isinstance(text, str):
            lines.extend(text.splitlines())
        elif isinstance(text, list):
            for item in text:
                lines.extend(str(item).splitlines())
        if output.get("output_type") == "error":
            ename = output.get("ename") or "error"
            evalue = output.get("evalue") or ""
            lines.append(f"{ename}: {evalue}".strip())
    return [line for line in lines if line]


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    code = cell.get("code")
    return code if isinstance(code, str) else ""


async def remote_tool_names(session: Any) -> set[str]:
    return {tool.name for tool in await session.list_tools()}


async def get_cells(session: Any, result_data: Any, include_outputs: bool = False) -> list[dict[str, Any]]:
    result = await session.call_tool("get_cells", {"includeOutputs": include_outputs})
    cells = result_data(result).get("cells", [])
    return cells if isinstance(cells, list) else []


async def delete_cell(session: Any, cell_id: str) -> None:
    await session.call_tool("delete_cell", {"cellId": cell_id})


async def cleanup_sidecar_cells(session: Any, result_data: Any, *, quiet: bool) -> int:
    cells = await get_cells(session, result_data, include_outputs=False)
    deleted = 0
    for cell in cells:
        cell_id = cell.get("id")
        if isinstance(cell_id, str) and CELL_MARKER in cell_source(cell):
            await delete_cell(session, cell_id)
            deleted += 1
    if deleted and not quiet:
        print(f"deleted {deleted} old sidecar cell(s)", flush=True)
    return deleted


async def run_temp_cell(
    session: Any,
    result_data: Any,
    *,
    code: str,
    delete_after: bool,
    timeout: float,
) -> dict[str, Any]:
    cells = await get_cells(session, result_data, include_outputs=False)
    add_result = await session.call_tool(
        "add_code_cell",
        {"cellIndex": len(cells), "language": "python", "code": code},
        timeout=timeout,
    )
    cell_id = result_data(add_result).get("newCellId")
    if not isinstance(cell_id, str):
        raise RuntimeError("Colab did not return a newCellId from add_code_cell")

    try:
        run_result = await session.call_tool("run_code_cell", {"cellId": cell_id}, timeout=timeout)
        run_data = result_data(run_result)
        outputs = run_data.get("outputs", [])
        return {
            "cell_id": cell_id,
            "outputs": outputs if isinstance(outputs, list) else [],
            "output_lines": extract_output_lines(outputs),
        }
    finally:
        if delete_after:
            try:
                await delete_cell(session, cell_id)
            except Exception as exc:
                print(f"warning: failed to delete sidecar cell {cell_id}: {exc}", flush=True)


async def wait_for_colab(session: Any, args: argparse.Namespace) -> dict[str, Any]:
    await session.start()
    connection = await session.connection_url()
    print("Colab sidecar URL:")
    print(connection["url"], flush=True)

    started = time.monotonic()
    first = True
    while True:
        status = await session.connect(
            wait_seconds=args.connect_poll_seconds,
            open_browser=first and not args.no_publish_url,
        )
        first = False
        status_payload = status.__dict__
        if status.connected:
            print(
                f"connected: port={status.port} tools={status.remote_tool_count}",
                flush=True,
            )
            return status_payload

        elapsed = time.monotonic() - started
        if args.connect_timeout_seconds and elapsed >= args.connect_timeout_seconds:
            raise TimeoutError(f"Colab browser did not connect after {elapsed:.1f}s")
        if not args.quiet:
            error = f" last_error={status.last_error}" if status.last_error else ""
            print(f"waiting for Colab browser... elapsed={elapsed:.1f}s{error}", flush=True)
        await asyncio.sleep(args.connect_poll_seconds)


async def main_async() -> int:
    args = parse_args()
    ColabSessionManager, result_data = load_adapter(args.adapter_repo)

    session = ColabSessionManager()
    cycle = 0
    last_payload: dict[str, Any] = {}
    try:
        connection_status = await wait_for_colab(session, args)
        names = await remote_tool_names(session)
        required = {"get_cells", "add_code_cell", "run_code_cell"}
        if not args.keep_cells:
            required.add("delete_cell")
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Colab frontend is missing required tools: {', '.join(missing)}")

        if args.cleanup_existing:
            await cleanup_sidecar_cells(session, result_data, quiet=args.quiet)

        while args.max_cycles <= 0 or cycle < args.max_cycles:
            cycle += 1
            started_at = time.time()
            code = build_cell_code(args, cycle)
            try:
                result = await run_temp_cell(
                    session,
                    result_data,
                    code=code,
                    delete_after=not args.keep_cells,
                    timeout=args.cell_timeout_seconds,
                )
                last_payload = {
                    "state": "ok",
                    "cycle": cycle,
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "connection": connection_status,
                    "cell_id": result["cell_id"],
                    "output_lines": result["output_lines"],
                }
                write_status(args.status_file, last_payload)
                if not args.quiet:
                    print(f"cycle {cycle}: ok", flush=True)
                    for line in result["output_lines"]:
                        print(line, flush=True)
            except Exception as exc:
                last_payload = {
                    "state": "error",
                    "cycle": cycle,
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "connection": connection_status,
                }
                write_status(args.status_file, last_payload)
                print(f"cycle {cycle}: {last_payload['error']}", flush=True)

            if args.max_cycles > 0 and cycle >= args.max_cycles:
                break
            await asyncio.sleep(args.interval_seconds)
    finally:
        if last_payload:
            write_status(args.status_file, {**last_payload, "sidecar_stopped_at": time.time()})
        await session.close()

    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("interrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
