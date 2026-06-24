# Hippocampus-8

Deterministic vector-frame graph traversal prototype.

This repo explores a memory graph where nodes and edges carry learned vector
frames. The graph stores dense node information, compact traversal vectors, and
edge vectors that encode relationship geometry without symbolic `relation_type`
labels. A small scorer ranks traversal decisions, and the controller applies
deterministic ordering so identical inputs produce identical output order.

## Current Status

This is an engineer-demo prototype, not a production memory system yet.

What works:

- bounded node/edge graph storage
- first-class deterministic metadata and traversal vectors
- deterministic beam and single-path traversal
- expressway nodes for long-range routing
- compact deterministic seed index over traversal vectors
- PyTorch scorer backend with a transformer option
- traversal result ranking via `result_score`
- Qwen-teacher synthetic data pipeline
- benchmark scripts for ranking, hard negatives, calibration, and latency

Current best saved run:

```text
/content/drive/MyDrive/hippo-qwen-runs/rich_1536/rich_1536_transformer_result_a100_e128_full.pt
sha256: 67da16189456c0a347d0c781ae095db39b14d237d614a8ce2101ad6914501d93
```

Exact trainer-holdout results from that run:

| Head | Cases | Top-1 | Avg precision | Precision @ recall 90 | Hard-neg pairwise | ms/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traversal result rank | 230 | 1.0000 | 0.8902 | 0.6036 | 0.9388 | 0.193 |
| Attach | 230 | 0.9870 | 0.9919 | 0.9912 | 0.9953 | 0.256 |

Honest caveat: these are synthetic/Qwen-teacher benchmarks. The prototype is
promising, but it still needs real-data evaluation and end-to-end graph-scale
retrieval tests before production claims.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Install the optional PyTorch backend:

```bash
.venv/bin/pip install -e ".[torch]"
```

Run tests:

```bash
.venv/bin/python -m unittest
```

Run the local deterministic demo:

```bash
.venv/bin/python demo.py
```

## Core Model

The prototype separates graph state from model scoring:

- `NodeFrame`: summary vector, optional full vector, metadata vector, traversal vector, payload, metadata
- `EdgeFrame`: compact relationship vector plus confidence
- `TraversalScores`: `follow`, `read_full`, `include`, `expand`, `stop`, `result`
- `TraversalController`: deterministic graph walk and result ordering
- `TraversalIndex`: deterministic compact seed lookup
- `insert_node`: traverses first, then attaches a new node to ranked candidates

`result_score` is the score used to order returned included nodes. It is
separate from `follow_score` so the traversal can use bridge nodes without
ranking those bridge nodes above final answers.

Node metadata is vectorized automatically. Raw metadata stays available as a
dict, but `metadata_vector` and compact `traversal_vector` are first-class frame
fields and are used by index lookup and scorer inputs.

## Minimal Example

```python
from vector_graph import (
    EdgeFrame,
    GraphStore,
    HeuristicTraversalScorer,
    NodeFrame,
    TraversalConfig,
    TraversalController,
    embed_text,
)
from vector_graph.vectors import stable_edge_vector

store = GraphStore(max_outgoing_edges=16)
store.add_node(
    NodeFrame(
        "root",
        summary_vector=embed_text("graph memory", 32),
        metadata={"project": "hippo", "kind": "seed"},
    )
)
store.add_node(
    NodeFrame(
        "answer",
        summary_vector=embed_text("vector frame traversal", 32),
        metadata={"project": "hippo", "kind": "design"},
    )
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
result = controller.traverse(query_vector=embed_text("traverse vector frame graph", 32), seed_id="root")

print([decision.node_id for decision in result.included])
```

## Traversal Modes

```python
TraversalConfig(mode="beam", fanout=16, beam_width=32)
TraversalConfig(mode="single_path", fanout=32, max_hops=12)
```

Expressway nodes keep relationship semantics implicit in vector frames, while
allowing a larger bounded edge set for long-range routing:

```python
store = GraphStore(max_outgoing_edges=32, max_expressway_edges=128)
hub = NodeFrame(
    node_id="topic_hub",
    summary_vector=summary,
    metadata={"expressway": True},
)
```

Deterministic indexed starts use compact traversal vectors:

```python
index = TraversalIndex(config=TraversalIndexConfig(dimension=16, seed=17))
index.add_nodes(store.nodes())
seed_ids = index.seed_ids(query_traversal_vector, limit=8)
result = controller.traverse(
    query_vector=query_vector,
    seed_id=seed_id,
    extra_seed_ids=seed_ids,
)
```

## Data And Training

Generated datasets, Qwen teacher runs, and checkpoints are intentionally not
tracked. Regenerate them locally or in Colab when needed.

Synthetic smoke data:

```bash
.venv/bin/python scripts/generate_synthetic_training_data.py
.venv/bin/python scripts/generate_synthetic_ranking_training_data.py
.venv/bin/python scripts/generate_synthetic_benchmark.py
```

Qwen-teacher episode path:

```bash
.venv/bin/python scripts/generate_teacher_graph_episodes.py \
  --output-dir data/teacher_episodes

.venv/bin/python scripts/label_teacher_episodes_qwen.py \
  --episodes-dir data/teacher_episodes \
  --output-dir data/qwen_teacher_episodes

.venv/bin/python scripts/convert_teacher_episodes.py \
  --episodes-dir data/qwen_teacher_episodes \
  --output-data-dir data/teacher_scorer \
  --output-ranking-dir data/teacher_ranked
```

Domain-diverse Qwen-teacher episode path:

```bash
.venv/bin/python scripts/generate_domain_teacher_episodes.py \
  --domain-set all \
  --episodes 4096 \
  --candidate-limit 16 \
  --output-dir data/domain_teacher_episodes

.venv/bin/python scripts/run_qwen_label_shards.py \
  --episodes-dir data/domain_teacher_episodes \
  --output-dir data/qwen_domain_teacher_episodes \
  --shard-count 256 \
  --expected-per-shard 16 \
  --request-timeout 60 \
  --retries 2 \
  --continue-on-failure
```

This generator is designed to stress metadata scope: same-domain wrong workflow,
cross-domain distractors, bridge nodes that should be followed but not included,
compliance negatives, tenant/entity mismatches, and realistic operational
phrasing.

For a larger paid teacher run, use the broad/all domain sets instead of only
raising the episode count on the curated pack:

```bash
.venv/bin/python scripts/generate_domain_teacher_episodes.py \
  --domain-set all \
  --episodes 12288 \
  --candidate-limit 16 \
  --output-dir data/domain_teacher_episodes_all_12288
```

`curated` contains the original five hand-authored business domains. `broad`
adds 17 more verticals covering legal, security, insurance, HR, retail,
education, energy, travel, media, manufacturing, banking, construction,
telecom, food safety, property management, biotech labs, and nonprofit grants.
`all` combines both sets.

Train the transformer scorer:

```bash
.venv/bin/python scripts/train_scorer.py \
  --model-kind transformer \
  --attach-head-kind hybrid \
  --data-dir data/teacher_scorer \
  --ranking-data-dir data/teacher_ranked \
  --epochs 128 \
  --batch-size 2048 \
  --ranking-batch-size 384 \
  --ranking-loss-weight 0.5 \
  --listwise-loss-weight 0.25 \
  --checkpoint-selection ranking_loss \
  --output models/scorer.pt
```

Benchmark a checkpoint:

```bash
.venv/bin/python scripts/benchmark_scorer.py \
  --checkpoint models/scorer.pt \
  --benchmark-dir data/teacher_ranked \
  --batch-size 4096 \
  --json-output reports/benchmark.json
```

The scorer benchmark reports top-k ranking, precision/recall curves,
hard-negative pairwise accuracy, calibration (`brier_score`, `ece_10`), scoring
latency, and per-action traversal quality for `follow`, `read_full`, `include`,
`expand`, `stop`, and `result` when teacher oracle vectors are present.

Curated domain validation without Qwen-teacher targets:

```bash
.venv/bin/python scripts/generate_domain_validation.py \
  --output-dir data/domain_validation_curated

.venv/bin/python scripts/benchmark_scorer.py \
  --checkpoint models/scorer.pt \
  --benchmark-dir data/domain_validation_curated \
  --batch-size 4096 \
  --json-output reports/domain_validation.json
```

This pack is hand-authored across SaaS billing, fintech risk, healthcare ops,
supply chain, and DevOps incident domains. It is still not customer data, but it
is a stricter validation than the teacher holdout because labels are manually
assigned and include same-domain hard negatives.

Index plus traversal benchmark:

```bash
.venv/bin/python scripts/benchmark_indexed_traversal.py \
  --nodes 50000 \
  --queries 100 \
  --checkpoint models/scorer.pt
```

## Colab Notes

The heavier training path is intended for Colab/GPU. The saved A100 run above
used Google Drive paths under:

```text
/content/drive/MyDrive/hippo-qwen-runs/rich_1536
```

The current repo includes `scripts/colab_train_to_drive.py` for the older
synthetic path. For serious runs, prefer the explicit commands in the training
section so the data directory, report path, checkpoint name, commit SHA, and
split semantics are captured in a manifest.

## Remaining Work

Before this should be presented as production-ready:

- run real-data benchmarks, not only synthetic/Qwen-teacher labels
- add domain-level or generator-seed-level holdouts
- benchmark end-to-end retrieval at 10k, 50k, and 100k nodes
- improve traversal precision at high recall
- make Colab training/report manifest generation a first-class script
- add error-analysis reports for failed traversal and attach cases
