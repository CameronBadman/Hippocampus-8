#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from vector_graph.torch_models import SmallAttachNet, SmallTraversalNet, TorchModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained vector-frame scorer checkpoint.")
    parser.add_argument("--benchmark-dir", default="data/benchmarks/synthetic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    traversal_model, attach_model, _ = load_models(Path(args.checkpoint), device=device)
    benchmark_dir = Path(args.benchmark_dir)

    traversal_cases = list(read_jsonl(benchmark_dir / "traversal_ranking.jsonl"))
    attach_cases = list(read_jsonl(benchmark_dir / "attach_ranking.jsonl"))

    traversal_metrics = evaluate_traversal(traversal_model, traversal_cases, device=device)
    attach_metrics = evaluate_attach(attach_model, attach_cases, device=device)
    report = {
        "checkpoint": args.checkpoint,
        "benchmark_dir": str(benchmark_dir),
        "device": str(device),
        "traversal": traversal_metrics,
        "attach": attach_metrics,
    }

    print_report(report)
    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_models(checkpoint_path: Path, *, device: torch.device) -> tuple[SmallTraversalNet, SmallAttachNet, TorchModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TorchModelConfig(**checkpoint["config"])
    traversal_model = SmallTraversalNet(
        query_dim=config.query_dim,
        summary_dim=config.summary_dim,
        edge_dim=config.edge_dim,
        path_dim=config.path_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    attach_model = SmallAttachNet(
        summary_dim=config.summary_dim,
        full_dim=config.full_dim,
        path_dim=config.path_dim,
        hidden_dim=config.attach_hidden_dim,
    ).to(device)
    traversal_model.load_state_dict(checkpoint["traversal_model"])
    attach_model.load_state_dict(checkpoint["attach_model"])
    traversal_model.eval()
    attach_model.eval()
    return traversal_model, attach_model, config


def evaluate_traversal(model: SmallTraversalNet, cases: list[dict], *, device: torch.device) -> dict[str, float | int]:
    rows = []
    spans = []
    labels = []
    kinds = []
    oracle_scores = []
    for case in cases:
        start = len(rows)
        for candidate in case["candidates"]:
            rows.append(
                case["query"]
                + case["current_summary"]
                + candidate["edge"]
                + candidate["dst_summary"]
                + case["path"]
            )
            labels.append(candidate["label"])
            kinds.append(candidate["kind"])
            oracle_scores.append(candidate["oracle"][0])
        spans.append((start, len(rows)))

    predictions = score_tensor(model, rows, device=device)[:, 0].tolist()
    return ranking_metrics(
        cases=cases,
        spans=spans,
        predictions=predictions,
        labels=labels,
        kinds=kinds,
        oracle_scores=oracle_scores,
    )


def evaluate_attach(model: SmallAttachNet, cases: list[dict], *, device: torch.device) -> dict[str, float | int]:
    rows = []
    spans = []
    labels = []
    kinds = []
    oracle_scores = []
    for case in cases:
        start = len(rows)
        for candidate in case["candidates"]:
            rows.append(
                case["new_summary"]
                + candidate["candidate_summary"]
                + case["new_full"]
                + candidate["candidate_full"]
                + case["path"]
            )
            labels.append(candidate["label"])
            kinds.append(candidate["kind"])
            oracle_scores.append(candidate["oracle"])
        spans.append((start, len(rows)))

    predictions = score_tensor(model, rows, device=device).reshape(-1).tolist()
    return ranking_metrics(
        cases=cases,
        spans=spans,
        predictions=predictions,
        labels=labels,
        kinds=kinds,
        oracle_scores=oracle_scores,
    )


def ranking_metrics(
    *,
    cases: list[dict],
    spans: list[tuple[int, int]],
    predictions: list[float],
    labels: list[int],
    kinds: list[str],
    oracle_scores: list[float],
) -> dict[str, float | int]:
    precision_sums = {1: 0.0, 3: 0.0, 5: 0.0}
    recall_sums = {1: 0.0, 3: 0.0, 5: 0.0}
    ndcg_sums = {3: 0.0, 5: 0.0}
    top1_correct = 0
    oracle_top1_correct = 0
    mrr_sum = 0.0
    oracle_mrr_sum = 0.0
    adversarial_correct = 0
    adversarial_total = 0
    oracle_pair_correct = 0
    oracle_pair_total = 0
    total_positives = 0
    positive_scores = []
    negative_scores = []

    for start, end in spans:
        indexes = list(range(start, end))
        ranked = sorted(indexes, key=lambda index: (-predictions[index], index))
        oracle_ranked = sorted(indexes, key=lambda index: (-oracle_scores[index], index))
        positives = [index for index in indexes if labels[index] == 1]
        total_positives += len(positives)
        if labels[ranked[0]] == 1:
            top1_correct += 1
        if labels[oracle_ranked[0]] == 1:
            oracle_top1_correct += 1

        for rank, index in enumerate(ranked, start=1):
            if labels[index] == 1:
                mrr_sum += 1.0 / rank
                break
        for rank, index in enumerate(oracle_ranked, start=1):
            if labels[index] == 1:
                oracle_mrr_sum += 1.0 / rank
                break

        for k in precision_sums:
            selected = ranked[:k]
            hits = sum(labels[index] for index in selected)
            precision_sums[k] += hits / min(k, len(selected))
            recall_sums[k] += hits / max(len(positives), 1)

        for k in ndcg_sums:
            ndcg_sums[k] += ndcg_at_k([labels[index] for index in ranked], k)

        for pos in positives:
            positive_scores.append(predictions[pos])
            for neg in indexes:
                if labels[neg] == 0:
                    negative_scores.append(predictions[neg])
                    if kinds[neg].startswith("hard") or kinds[neg].startswith("adversarial"):
                        adversarial_total += 1
                        if predictions[pos] > predictions[neg]:
                            adversarial_correct += 1
                    if oracle_scores[pos] > oracle_scores[neg]:
                        oracle_pair_total += 1
                        if predictions[pos] > predictions[neg]:
                            oracle_pair_correct += 1

    threshold_05 = threshold_metrics(predictions, labels, threshold=0.5)
    best = best_f1(predictions, labels)
    case_count = len(cases)
    kind_counts = Counter(kinds)
    return {
        "cases": case_count,
        "candidates": len(labels),
        "positives": total_positives,
        "top1_accuracy": top1_correct / max(case_count, 1),
        "oracle_top1_accuracy": oracle_top1_correct / max(case_count, 1),
        "mrr": mrr_sum / max(case_count, 1),
        "oracle_mrr": oracle_mrr_sum / max(case_count, 1),
        "precision_at_1": precision_sums[1] / max(case_count, 1),
        "precision_at_3": precision_sums[3] / max(case_count, 1),
        "precision_at_5": precision_sums[5] / max(case_count, 1),
        "recall_at_1": recall_sums[1] / max(case_count, 1),
        "recall_at_3": recall_sums[3] / max(case_count, 1),
        "recall_at_5": recall_sums[5] / max(case_count, 1),
        "ndcg_at_3": ndcg_sums[3] / max(case_count, 1),
        "ndcg_at_5": ndcg_sums[5] / max(case_count, 1),
        "adversarial_pairwise_accuracy": adversarial_correct / max(adversarial_total, 1),
        "oracle_pairwise_accuracy": oracle_pair_correct / max(oracle_pair_total, 1),
        "threshold_0_5_precision": threshold_05["precision"],
        "threshold_0_5_recall": threshold_05["recall"],
        "threshold_0_5_f1": threshold_05["f1"],
        "best_f1": best["f1"],
        "best_f1_threshold": best["threshold"],
        "mean_positive_score": mean(positive_scores),
        "mean_negative_score": mean(negative_scores),
        "kind_counts": dict(sorted(kind_counts.items())),
    }


def score_tensor(model: torch.nn.Module, rows: list[list[float]], *, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        tensor = torch.tensor(rows, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu()


def threshold_metrics(scores: list[float], labels: list[int], *, threshold: float) -> dict[str, float]:
    predicted = [1 if score >= threshold else 0 for score in scores]
    true_positive = sum(1 for prediction, label in zip(predicted, labels) if prediction == 1 and label == 1)
    false_positive = sum(1 for prediction, label in zip(predicted, labels) if prediction == 1 and label == 0)
    false_negative = sum(1 for prediction, label in zip(predicted, labels) if prediction == 0 and label == 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall)}


def best_f1(scores: list[float], labels: list[int]) -> dict[str, float]:
    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for step in range(101):
        threshold = step / 100.0
        metrics = threshold_metrics(scores, labels, threshold=threshold)
        if metrics["f1"] > best["f1"]:
            best = {"threshold": threshold, **metrics}
    return best


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def ndcg_at_k(labels: list[int], k: int) -> float:
    dcg = sum(label / log2(rank + 1) for rank, label in enumerate(labels[:k], start=1))
    ideal = sorted(labels, reverse=True)
    idcg = sum(label / log2(rank + 1) for rank, label in enumerate(ideal[:k], start=1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def log2(value: int) -> float:
    return math.log2(value)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def print_report(report: dict) -> None:
    print(f"checkpoint: {report['checkpoint']}")
    print(f"benchmark:  {report['benchmark_dir']}")
    print(f"device:     {report['device']}")
    for section in ("traversal", "attach"):
        metrics = report[section]
        print()
        print(section)
        for key in (
            "cases",
            "candidates",
            "top1_accuracy",
            "oracle_top1_accuracy",
            "mrr",
            "oracle_mrr",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "ndcg_at_3",
            "ndcg_at_5",
            "adversarial_pairwise_accuracy",
            "oracle_pairwise_accuracy",
            "threshold_0_5_precision",
            "threshold_0_5_recall",
            "threshold_0_5_f1",
            "best_f1",
            "best_f1_threshold",
            "mean_positive_score",
            "mean_negative_score",
        ):
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
