# Hippocampus-8

Relationship-aware memory for AI agents.

Hippocampus-8 is an experimental memory engine that stores memories as a
bounded graph of vector frames. Nodes store memory content and metadata. Edges
store compact relationship vectors, so an agent can retrieve by context and
relationship instead of only nearest-vector similarity.

The goal is simple: make agent memory more deterministic, more inspectable, and
better at following the right context.

## Why It Matters

Standard vector search is good at finding things that look similar. That breaks
down when two memories are semantically close but only one has the right role in
the current situation. Hippocampus-8 adds relationship vectors and deterministic
graph traversal so the memory system can ask: "Which memory is connected in the
right way?"

## What Works

- deterministic node and edge graph storage
- first-class metadata, summary, full, and traversal vectors
- compact edge vectors that encode relationships without `relation_type` labels
- deterministic beam and single-path traversal
- deterministic result ranking through `result_score`
- compact traversal-vector seed index
- expressway nodes for long-range routing
- batched traversal and attach scoring
- PyTorch scorer backend with a small transformer option
- Qwen-teacher synthetic data pipeline and benchmark scripts

## Current Results

Source artifact: `rich_1536/reports/benchmark_result_model_exact_holdout.json`.

Synthetic Qwen-teacher exact holdout:

| Metric | Result |
| --- | ---: |
| Attach top-1 accuracy | 0.9870 |
| Attach precision at 90% recall | 0.9912 |
| Attach hard-negative pairwise accuracy | 0.9953 |
| Cases | 230 |

Adversarial relationship-retrieval benchmark:

| Method | Top-1 |
| --- | ---: |
| HNSW summary-vector search | 0.676 |
| Hippo relationship traversal | 1.000 |

This benchmark baits nearest-vector search with semantically close decoys. Hippo
wins by following the relationship path, not only the closest summary vector.

These are synthetic benchmarks. They show the architecture is working, but they
are not yet production or customer-data claims.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest
.venv/bin/python demo.py
```

Optional extras:

```bash
.venv/bin/pip install -e ".[torch]"
.venv/bin/pip install -e ".[hnsw]"
```

## Minimal API

```python
from vector_graph import GraphStore, NodeFrame, EdgeFrame
from vector_graph import TraversalConfig, TraversalController
from vector_graph import HeuristicTraversalScorer, embed_text
from vector_graph.vectors import stable_edge_vector

store = GraphStore(max_outgoing_edges=16)

store.add_node(NodeFrame("root", summary_vector=embed_text("project memory", 32)))
store.add_node(NodeFrame("answer", summary_vector=embed_text("relationship traversal", 32)))

root = store.get_node("root")
answer = store.get_node("answer")

store.add_edge(
    EdgeFrame(
        src_id="root",
        dst_id="answer",
        edge_vector=stable_edge_vector(root.summary_vector, answer.summary_vector, 16),
    )
)

controller = TraversalController(
    store=store,
    scorer=HeuristicTraversalScorer(),
    config=TraversalConfig(max_hops=2, fanout=8, beam_width=8),
)

result = controller.traverse(
    query_vector=embed_text("find traversal memory", 32),
    seed_id="root",
)

print([decision.node_id for decision in result.included])
```

## Status

This is an engineer-demo prototype, not production infrastructure yet.

Next work:

- real-data or customer-style validation
- domain-level holdouts
- end-to-end 10k, 50k, and 100k node benchmarks
- persistent storage
- stable server API
- frontend demo
