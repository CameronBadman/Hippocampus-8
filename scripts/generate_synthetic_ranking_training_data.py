#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from generate_synthetic_benchmark import main as benchmark_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ranked synthetic training cases.")
    parser.add_argument("--output-dir", default="data/synthetic_ranked")
    parser.add_argument("--traversal-cases", type=int, default=3000)
    parser.add_argument("--attach-cases", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    # Reuse the hard ranking-case generator, but with a different seed and
    # output directory so benchmark cases never leak into training.
    import sys

    sys.argv = [
        "generate_synthetic_benchmark.py",
        "--output-dir",
        str(Path(args.output_dir)),
        "--traversal-cases",
        str(args.traversal_cases),
        "--attach-cases",
        str(args.attach_cases),
        "--seed",
        str(args.seed),
    ]
    benchmark_main()


if __name__ == "__main__":
    main()

