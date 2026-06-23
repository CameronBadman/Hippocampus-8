#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from generate_synthetic_benchmark import generate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ranked synthetic training cases.")
    parser.add_argument("--output-dir", default="data/synthetic_ranked")
    parser.add_argument("--traversal-cases", type=int, default=3000)
    parser.add_argument("--attach-cases", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    # Reuse the hard ranking-case generator with a different seed and output
    # directory so benchmark cases never leak into training.
    generate_benchmark(
        output_dir=Path(args.output_dir),
        traversal_cases=args.traversal_cases,
        attach_cases=args.attach_cases,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
