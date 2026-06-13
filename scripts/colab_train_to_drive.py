#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab helper: train, benchmark, and save artifacts to Google Drive.")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/Hippocampus-8")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ranking-batch-size", type=int, default=192)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.35)
    parser.add_argument("--ranking-margin", type=float, default=0.08)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--attach-regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--attach-listwise-loss-weight", type=float, default=None)
    parser.add_argument("--hard-summary-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-full-negative-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--model-kind", choices=("mlp", "transformer"), default="mlp")
    parser.add_argument("--attach-head-kind", choices=("transformer", "hybrid"), default="transformer")
    parser.add_argument("--skip-mount", action="store_true")
    args = parser.parse_args()

    if not args.skip_mount:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive")

    drive_dir = Path(args.drive_dir)
    checkpoint_dir = drive_dir / "checkpoints"
    report_dir = drive_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_name = args.checkpoint_name or f"synthetic_scorer_{args.model_kind}_ranked_{stamp}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    report_path = report_dir / checkpoint_name.replace(".pt", "_benchmark.json")

    train_command = [
        sys.executable,
        "scripts/train_scorer.py",
        "--data-dir",
        "data/synthetic",
        "--ranking-data-dir",
        "data/synthetic_ranked",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--ranking-batch-size",
        str(args.ranking_batch_size),
        "--ranking-loss-weight",
        str(args.ranking_loss_weight),
        "--ranking-margin",
        str(args.ranking_margin),
        "--listwise-loss-weight",
        str(args.listwise_loss_weight),
        "--attach-regression-loss-weight",
        str(args.attach_regression_loss_weight),
        "--hard-summary-negative-weight",
        str(args.hard_summary_negative_weight),
        "--hard-full-negative-weight",
        str(args.hard_full_negative_weight),
        "--lr",
        str(args.lr),
        "--model-kind",
        args.model_kind,
        "--attach-head-kind",
        args.attach_head_kind,
        "--output",
        str(checkpoint_path),
    ]
    if args.attach_listwise_loss_weight is not None:
        train_command.extend(["--attach-listwise-loss-weight", str(args.attach_listwise_loss_weight)])
    run(train_command)

    benchmark_command = [
        sys.executable,
        "scripts/benchmark_scorer.py",
        "--checkpoint",
        str(checkpoint_path),
        "--benchmark-dir",
        "data/benchmarks/synthetic",
        "--json-output",
        str(report_path),
    ]
    run(benchmark_command)

    latest_checkpoint = checkpoint_dir / "latest.pt"
    latest_report = report_dir / "latest_benchmark.json"
    shutil.copy2(checkpoint_path, latest_checkpoint)
    shutil.copy2(report_path, latest_report)
    print(f"checkpoint: {checkpoint_path}")
    print(f"report:     {report_path}")
    print(f"latest:     {latest_checkpoint}")


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
