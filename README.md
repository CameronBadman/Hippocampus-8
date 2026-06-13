# Hippo Qwen 2 Vector Graph Prototype

A small Python prototype for deterministic traversal over a graph where both
nodes and edges have vector frames.

The important separation is:

- node vectors hold information content
- edge vectors hold relationship geometry
- the scorer assigns traversal scores
- the controller makes deterministic decisions

Run the demo:

```bash
python3 demo.py
```

Run tests:

```bash
python3 -m unittest
```

Install locally:

```bash
python3 -m pip install -e .
```

Install with the PyTorch backend:

```bash
python3 -m pip install -e ".[torch]"
```

The core graph/controller uses NumPy. The PyTorch scorer is optional and lives
in `vector_graph.torch_models`.

Generate synthetic training data:

```bash
python3 scripts/generate_synthetic_training_data.py
```

Train the PyTorch scorer:

```bash
python3 scripts/train_scorer.py --data-dir data/synthetic --epochs 12
```

Generate hard synthetic benchmarks:

```bash
python3 scripts/generate_synthetic_benchmark.py
```

Benchmark a trained checkpoint:

```bash
python3 scripts/benchmark_scorer.py \
  --checkpoint models/synthetic_scorer.pt \
  --benchmark-dir data/benchmarks/synthetic
```

# Hippocampus-8
