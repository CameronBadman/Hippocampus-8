# Hippocampus-8

Deterministic, relationship-aware memory for AI agents.

Hippocampus-8 is a prototype memory engine that stores memories as a bounded
graph of vector frames. Nodes hold memory content, metadata, summary vectors,
full vectors, and compact traversal vectors. Edges hold compact relationship
vectors, allowing retrieval to depend on how memories are connected instead of
only how close their embeddings are.

The project is built around a simple goal: make agent memory more deterministic,
inspectable, and reliable when context depends on relationships between facts.

## Why It Exists

Nearest-vector search is useful, but agent memory often has harder retrieval
conditions:

- many memories share vocabulary or topic structure
- multiple memories are semantically close to the query
- the correct answer depends on role, workflow, or relationship path
- repeated agent interactions create near-duplicate context

Hippocampus-8 adds a deterministic graph traversal layer on top of vector
representations. The system can rank memories by relationship context, not only
embedding similarity.

## Core Design

- **Node vector frames** for summary, full, metadata, and traversal information
- **Edge vector frames** that encode relationships without symbolic
  `relation_type` labels
- **Deterministic traversal** with bounded fanout, beam search, and single-path
  modes
- **Result ranking** through a dedicated `result_score`
- **Compact seed indexing** over traversal vectors
- **Expressway nodes** for long-range routing
- **Batch scoring** for traversal and attach decisions
- **PyTorch scorer backend** with a small transformer option
- **Synthetic teacher pipeline** for Qwen-labeled training and evaluation data

## Current Evidence

The current benchmark results are synthetic and teacher-generated. They are
useful engineering evidence, not production or customer-data claims.

### Rich 1536 Exact Holdout

Source artifacts:

- `rich_1536/reports/benchmark_result_model_exact_holdout.json`
- `rich_1536/reports/benchmark_metadata_model_exact_holdout.json`

| Model | Head | Cases | Top-1 | Avg precision | Precision @ 90% recall | Hard-neg pairwise | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Result model | Attach | 230 | 0.9870 | 0.9919 | 0.9912 | 0.9953 | 0.9935 |
| Result model | Traversal | 230 | 1.0000 | 0.8902 | 0.6036 | 0.9388 | 1.0000 |
| Metadata model | Attach | 230 | 1.0000 | 0.9975 | 0.9943 | 0.9987 | 1.0000 |
| Metadata model | Traversal | 230 | 1.0000 | 0.9919 | 0.9798 | 0.9955 | 1.0000 |

The attach head measures whether a new memory links to the correct existing
context. The traversal head measures whether the graph walk ranks the right
candidate during retrieval. The metadata model includes first-class metadata
vectors and is the stronger result on this holdout.

### Adversarial Relationship Retrieval

This benchmark is intentionally adversarial. It creates a more realistic memory
failure mode than clean nearest-neighbor lookup: many candidate memories are
semantically close, share vocabulary, and look plausible, but only one is
connected through the correct relationship path.

| Method | Top-1 |
| --- | ---: |
| HNSW summary-vector search | 0.676 |
| Hippo relationship traversal | 1.000 |

The result shows the intended advantage of the graph design: when summary-vector
similarity ranks plausible decoys too highly, relationship traversal can recover
the correct memory.

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
from vector_graph import EdgeFrame, GraphStore, NodeFrame
from vector_graph import HeuristicTraversalScorer, TraversalConfig
from vector_graph import TraversalController, embed_text
from vector_graph.vectors import stable_edge_vector

store = GraphStore(max_outgoing_edges=16)

store.add_node(
    NodeFrame("root", summary_vector=embed_text("project memory", 32))
)
store.add_node(
    NodeFrame("answer", summary_vector=embed_text("relationship traversal", 32))
)

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

Hippocampus-8 is an engineer-demo prototype. The graph engine, deterministic
traversal, metadata vectorization, synthetic data pipeline, and benchmark scripts
are implemented. The next phase is validation on more realistic workloads and a
stable service layer.

Planned work:

- real-data or customer-style validation
- stronger domain-level holdouts
- end-to-end 10k, 50k, and 100k node benchmarks
- persistent storage
- stable server API
- frontend demo
