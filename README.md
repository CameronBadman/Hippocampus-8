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

# Hippocampus-8
