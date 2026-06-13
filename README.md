# Hippo Qwen 2 Vector Graph Prototype

A small dependency-free Python prototype for deterministic traversal over a graph
where both nodes and edges have vector frames.

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

# Hippocampus-8
