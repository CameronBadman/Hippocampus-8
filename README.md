# Hippocampus-8

Deterministic vector-frame graph memory prototype.

Hippocampus-8 stores memory as a bounded graph. Nodes carry summary, full,
metadata, and traversal vectors. Edges carry compact relationship vectors
instead of symbolic `relation_type` labels. Traversal is deterministic: the
same graph, query, scorer, and config produce the same visited set and result
ordering.

This is an engineer-demo prototype, not production infrastructure yet.

## What Works

- bounded node and edge graph storage
- first-class node metadata vectors and compact traversal vectors
- edge vectors that implicitly encode relationship geometry
- deterministic beam and single-path traversal
- deterministic result ranking via `result_score`
- expressway nodes for long-range routing
- compact deterministic seed index over traversal vectors
- PyTorch scorer backend with a small transformer option
- batch traversal and attach scoring
- Qwen-teacher synthetic data pipeline
- benchmark scripts for ranking, hard negatives, calibration, latency, and
  HNSW/vector-search comparison

## Latest Saved Run

Current best checkpoint from the broad Qwen-teacher run:

```text
/content/drive/MyDrive/hippo-qwen-runs/all_12288/training_runs/a100_384_e128_20260625_022008/all_12288_384_transformer_a100_e128.pt
sha256: 8e975ffd270a8a79f4812bf899f98b3da34c345f27c05a6cc50fd35af4e7fcbe
```

Training data for that run:

```text
384 labeled shards
6,144 Qwen-labeled episodes
98,304 traversal examples
98,304 attach examples
6,144 traversal ranking cases
6,144 attach ranking cases
```

Benchmark on the same teacher-ranked distribution:

| Head | Cases | Top-1 | Avg precision | Precision @ recall 90 | ms/case |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traversal | 6,144 | 1.0000 | 0.99996 | 0.99994 | 0.232 |
| Attach | 6,144 | 0.9097 | 0.9083 | 0.2819 | 0.255 |

Honest caveat: this proves the student model can imitate the generated teacher
distribution. It does not yet prove customer-data generalization or production
retrieval quality at 50k to 100k nodes.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Optional extras:

```bash
.venv/bin/pip install -e ".[torch]"
.venv/bin/pip install -e ".[hnsw]"
```

Run tests:

```bash
.venv/bin/python -m unittest
```

Run the local deterministic demo:

```bash
.venv/bin/python demo.py
```

## Core API

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
result = controller.traverse(
    query_vector=embed_text("traverse vector frame graph", 32),
    seed_id="root",
)

print([decision.node_id for decision in result.included])
```

## Design Notes

`NodeFrame`:

- `summary_vector`: compact semantic summary
- `full_vector`: optional richer detail vector
- `metadata_vector`: deterministic vectorization of raw metadata
- `traversal_vector`: compact routing vector, currently 16 dimensions by default
- `metadata`: raw stable dict for inspection and filtering

`EdgeFrame`:

- `edge_vector`: compact learned relationship frame
- `confidence`: deterministic pruning and ordering signal

`TraversalScores`:

- `follow`: should traversal move through this edge
- `read_full`: should the full node be inspected
- `include`: should the node be included
- `expand`: should traversal continue from this node
- `stop`: should this route stop
- `result`: returned-result rank score

`result_score` is separate from `follow_score` so bridge or expressway nodes can
be useful for routing without outranking final answers.

## Benchmarking

Scorer benchmark against ranked teacher files:

```bash
.venv/bin/python scripts/benchmark_scorer.py \
  --checkpoint models/scorer.pt \
  --benchmark-dir data/teacher_ranked \
  --batch-size 4096 \
  --json-output reports/scorer_benchmark.json
```

The scorer benchmark reports top-k ranking, precision/recall curves,
hard-negative pairwise accuracy, calibration, latency, and per-action traversal
quality for `follow`, `read_full`, `include`, `expand`, `stop`, and `result`.

HNSW/vector-search comparison:

```bash
.venv/bin/python scripts/benchmark_vector_search_comparison.py \
  --nodes 50000 \
  --queries 100 \
  --backend auto \
  --json-output reports/vector_search_comparison.json
```

With the trained transformer checkpoint in Colab:

```bash
python scripts/benchmark_vector_search_comparison.py \
  --nodes 50000 \
  --queries 100 \
  --backend hnsw \
  --checkpoint /content/drive/MyDrive/hippo-qwen-runs/all_12288/training_runs/a100_384_e128_20260625_022008/all_12288_384_transformer_a100_e128.pt \
  --device cuda \
  --json-output /content/drive/MyDrive/hippo-qwen-runs/all_12288/reports/vector_search_comparison_50k.json
```

The comparison reports:

- `exact_vector`: brute-force cosine upper-bound baseline
- `hnsw_vector`: HNSW cosine retrieval when `hnswlib` is installed
- `hippo_seed_index`: deterministic compact traversal-vector seed lookup
- `hippo_traversal`: graph walk plus scorer from the indexed seeds

Primary metrics are `precision_at_k`, `hit_at_k`, `mrr`, `latency_ms`, and for
Hippo traversal also `visited`, `included`, and `seed_count`.

Transformer traversal scale benchmark:

```bash
.venv/bin/python scripts/benchmark_indexed_traversal.py \
  --nodes 50000 \
  --queries 100 \
  --checkpoint models/scorer.pt
```

## Data And Training

Generated datasets, teacher runs, reports, and checkpoints are intentionally not
tracked in git. Use Drive or local `runs/` for artifacts.

Generate broad synthetic teacher episodes:

```bash
.venv/bin/python scripts/generate_domain_teacher_episodes.py \
  --domain-set all \
  --episodes 12288 \
  --candidate-limit 16 \
  --output-dir data/domain_teacher_episodes
```

Label with Qwen:

```bash
.venv/bin/python scripts/run_qwen_label_shards.py \
  --episodes-dir data/domain_teacher_episodes \
  --output-dir data/qwen_teacher_episodes \
  --shard-count 768 \
  --expected-per-shard 16 \
  --request-timeout 60 \
  --retries 2 \
  --continue-on-failure
```

Convert labels into scorer/ranking data:

```bash
.venv/bin/python scripts/convert_teacher_episodes.py \
  --episodes-dir data/qwen_teacher_episodes \
  --output-data-dir data/teacher_scorer \
  --output-ranking-dir data/teacher_ranked
```

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

## Local Drive Handoff

Qwen labeling is API-bound, so it can run locally while training runs in Colab.
The local runner uses `rclone` to pull existing Drive artifacts, resume only
incomplete shards, convert labels, and push progress after each worker round.

```bash
scripts/local_qwen_drive_run.sh \
  --run-name all_12288 \
  --remote gdrive:hippo-qwen-runs/all_12288 \
  --workers 6 \
  --target-complete-shards 384
```

It prompts for a Qwen/DashScope key if `DASHSCOPE_API_KEY` or `QWEN_API_KEY` is
not already set.

## Colab Workflow

In Colab:

```bash
git clone https://github.com/CameronBadman/Hippocampus-8.git /content/hippo-qwen-2
cd /content/hippo-qwen-2
python -m pip install -e ".[torch,hnsw]"
```

Mount Drive and use artifacts under:

```text
/content/drive/MyDrive/hippo-qwen-runs/all_12288
```

The helper `scripts/colab_keepalive_sidecar.py` can keep a separate Colab bridge
warm and delete temporary status cells, but normal training and benchmarks
should be reproducible from explicit shell commands.

## Remaining Work

- run real-data or customer-style validation, not only synthetic/Qwen-teacher
  labels
- add domain-level or generator-seed-level holdouts
- benchmark end-to-end retrieval at 10k, 50k, and 100k nodes
- compare against HNSW with the same query workload and saved JSON reports
- improve attach behavior when high recall is required
- add persistent storage and a stable server API
- add error-analysis reports for failed traversal and attach cases
