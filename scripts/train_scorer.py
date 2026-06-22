#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from vector_graph.torch_models import TorchModelConfig, create_model_pair, traversal_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vector-frame scorer models from JSONL shards.")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--ranking-data-dir", default=None)
    parser.add_argument("--output", default="models/synthetic_scorer.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ranking-batch-size", type=int, default=128)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.35)
    parser.add_argument("--ranking-margin", type=float, default=0.15)
    parser.add_argument("--ranking-min-delta", type=float, default=0.10)
    parser.add_argument("--listwise-temperature", type=float, default=0.15)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--traversal-regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--attach-regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--traversal-listwise-loss-weight", type=float, default=None)
    parser.add_argument("--attach-listwise-loss-weight", type=float, default=None)
    parser.add_argument("--hard-summary-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-full-negative-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-kind", choices=("mlp", "transformer"), default="mlp")
    parser.add_argument("--attach-head-kind", choices=("transformer", "hybrid"), default="transformer")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("combined_loss", "ranking_loss", "validation_loss", "final"),
        default="combined_loss",
        help="Metric used to choose the saved checkpoint. Ranking-aware modes require --ranking-data-dir.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)

    config = load_config(data_dir, model_kind=args.model_kind, attach_head_kind=args.attach_head_kind)
    traversal_x, traversal_y = load_traversal_examples(data_dir)
    attach_x, attach_y = load_attach_examples(data_dir)
    traversal_ranking = None
    attach_ranking = None
    if args.ranking_data_dir is not None:
        ranking_data_dir = Path(args.ranking_data_dir)
        traversal_ranking = load_traversal_ranking_examples(ranking_data_dir)
        attach_ranking = load_attach_ranking_examples(
            ranking_data_dir,
            hard_summary_negative_weight=args.hard_summary_negative_weight,
            hard_full_negative_weight=args.hard_full_negative_weight,
        )

    traversal_model, attach_model = create_model_pair(config)
    traversal_model = traversal_model.to(device)
    attach_model = attach_model.to(device)

    print(f"device: {device}")
    print(f"model kind: {config.model_kind}")
    print(f"traversal examples: {len(traversal_x)}")
    print(f"attach examples: {len(attach_x)}")
    if traversal_ranking is not None and attach_ranking is not None:
        print(f"traversal ranking cases: {len(traversal_ranking[0])}")
        print(f"attach ranking cases: {len(attach_ranking[0])}")

    traversal_history, traversal_best_state = train_regressor(
        name="traversal",
        model=traversal_model,
        x=traversal_x,
        y=traversal_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        regression_loss_weight=args.traversal_regression_loss_weight,
        ranking=traversal_ranking,
        ranking_batch_size=args.ranking_batch_size,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        ranking_min_delta=args.ranking_min_delta,
        listwise_loss_weight=coalesce(args.traversal_listwise_loss_weight, args.listwise_loss_weight),
        listwise_temperature=args.listwise_temperature,
        ranking_score_index=5 if config.traversal_output_dim > 5 else 0,
        checkpoint_selection=args.checkpoint_selection,
    )
    attach_history, attach_best_state = train_regressor(
        name="attach",
        model=attach_model,
        x=attach_x,
        y=attach_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 1,
        device=device,
        regression_loss_weight=args.attach_regression_loss_weight,
        ranking=attach_ranking,
        ranking_batch_size=args.ranking_batch_size,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        ranking_min_delta=args.ranking_min_delta,
        listwise_loss_weight=coalesce(args.attach_listwise_loss_weight, args.listwise_loss_weight),
        listwise_temperature=args.listwise_temperature,
        ranking_score_index=None,
        checkpoint_selection=args.checkpoint_selection,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "traversal_model": traversal_best_state,
            "attach_model": attach_best_state,
            "traversal_history": traversal_history,
            "attach_history": attach_history,
            "selection": args.checkpoint_selection,
        },
        output,
    )
    print(f"saved checkpoint: {output}")


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def coalesce(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def load_config(data_dir: Path, *, model_kind: str, attach_head_kind: str) -> TorchModelConfig:
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dimensions = manifest["dimensions"]
        return TorchModelConfig(
            query_dim=dimensions["query"],
            summary_dim=dimensions["summary"],
            edge_dim=dimensions["edge"],
            full_dim=dimensions["full"],
            path_dim=dimensions["path"],
            scalar_dim=dimensions.get("scalars", 2),
            model_kind=model_kind,
            attach_head_kind=attach_head_kind,
        )
    return TorchModelConfig(
        query_dim=32,
        summary_dim=32,
        edge_dim=16,
        full_dim=64,
        path_dim=32,
        model_kind=model_kind,
        attach_head_kind=attach_head_kind,
    )


def load_traversal_examples(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for path in sorted(data_dir.glob("traversal_*.jsonl")):
        for example in read_jsonl(path):
            rows.append(
                example["query"]
                + example["current_summary"]
                + example["edge"]
                + example["dst_summary"]
                + example["path"]
                + list(traversal_scalars(example["confidence"], example["hop"]))
            )
            targets.append(normalize_traversal_target(example["target"]))
    if not rows:
        raise ValueError(f"no traversal_*.jsonl files found in {data_dir}")
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def load_attach_examples(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for path in sorted(data_dir.glob("attach_*.jsonl")):
        for example in read_jsonl(path):
            rows.append(
                example["new_summary"]
                + example["candidate_summary"]
                + example["new_full"]
                + example["candidate_full"]
                + example["path"]
            )
            targets.append([example["target"]])
    if not rows:
        raise ValueError(f"no attach_*.jsonl files found in {data_dir}")
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def load_traversal_ranking_examples(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = []
    labels = []
    weights = []
    for path in sorted(data_dir.glob("traversal_ranking*.jsonl")):
        for case in read_jsonl(path):
            group = []
            group_labels = []
            group_weights = []
            for candidate in case["candidates"]:
                group.append(
                    case["query"]
                    + case["current_summary"]
                    + candidate["edge"]
                    + candidate["dst_summary"]
                    + case["path"]
                    + list(traversal_scalars(candidate["confidence"], candidate["hop"]))
                )
                group_labels.append(
                    float(
                        candidate.get(
                            "result_rank_target",
                            candidate.get("rank_target", candidate["label"]),
                        )
                    )
                )
                group_weights.append(float(candidate.get("weight", 1.0)))
            groups.append(group)
            labels.append(group_labels)
            weights.append(group_weights)
    if not groups:
        raise ValueError(f"no traversal_ranking*.jsonl files found in {data_dir}")
    return (
        torch.tensor(groups, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def load_attach_ranking_examples(
    data_dir: Path,
    *,
    hard_summary_negative_weight: float,
    hard_full_negative_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = []
    labels = []
    weights = []
    for path in sorted(data_dir.glob("attach_ranking*.jsonl")):
        for case in read_jsonl(path):
            group = []
            group_labels = []
            group_weights = []
            for candidate in case["candidates"]:
                group.append(
                    case["new_summary"]
                    + candidate["candidate_summary"]
                    + case["new_full"]
                    + candidate["candidate_full"]
                    + case["path"]
                )
                group_labels.append(float(candidate.get("rank_target", candidate["label"])))
                group_weights.append(
                    float(candidate.get("weight", 1.0))
                    * attach_candidate_weight(
                        candidate["kind"],
                        hard_summary_negative_weight=hard_summary_negative_weight,
                        hard_full_negative_weight=hard_full_negative_weight,
                    )
                )
            groups.append(group)
            labels.append(group_labels)
            weights.append(group_weights)
    if not groups:
        raise ValueError(f"no attach_ranking*.jsonl files found in {data_dir}")
    return (
        torch.tensor(groups, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def attach_candidate_weight(
    kind: str,
    *,
    hard_summary_negative_weight: float,
    hard_full_negative_weight: float,
) -> float:
    if kind in {"hard_summary_negative", "same_full_wrong_summary_negative"}:
        return hard_summary_negative_weight
    if kind in {"hard_full_negative", "same_summary_wrong_full_negative", "path_aligned_wrong_full_negative"}:
        return hard_full_negative_weight
    return 1.0


def normalize_traversal_target(target: list[float]) -> list[float]:
    values = [float(value) for value in target]
    if len(values) == 5:
        values.append(values[2])
    if len(values) != 6:
        raise ValueError(f"traversal target must have 5 or 6 values, got {len(values)}")
    return values


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def train_regressor(
    *,
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    regression_loss_weight: float = 1.0,
    ranking: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ranking_batch_size: int = 128,
    ranking_loss_weight: float = 0.35,
    ranking_margin: float = 0.08,
    ranking_min_delta: float = 0.10,
    listwise_loss_weight: float = 0.0,
    listwise_temperature: float = 0.15,
    ranking_score_index: int | None = None,
    checkpoint_selection: str = "combined_loss",
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    dataset = TensorDataset(x, y)
    validation_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - validation_size
    train_data, validation_data = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    ranking_loader = None
    ranking_validation_loader = None
    if ranking is not None:
        ranking_dataset = TensorDataset(*ranking)
        ranking_validation_size = max(1, int(len(ranking_dataset) * 0.15))
        ranking_train_size = len(ranking_dataset) - ranking_validation_size
        ranking_train_data, ranking_validation_data = random_split(
            ranking_dataset,
            [ranking_train_size, ranking_validation_size],
            generator=torch.Generator().manual_seed(seed + 17),
        )
        ranking_loader = DataLoader(
            ranking_train_data,
            batch_size=ranking_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 101),
        )
        ranking_validation_loader = DataLoader(ranking_validation_data, batch_size=ranking_batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []
    best_selection_loss = float("inf")
    best_state = clone_state_dict(model)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        if regression_loss_weight > 0.0:
            train_loss = run_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer=optimizer,
                device=device,
                loss_weight=regression_loss_weight,
            )
        ranking_loss = 0.0
        if ranking_loader is not None:
            ranking_loss = run_ranking_epoch(
                model,
                ranking_loader,
                optimizer=optimizer,
                device=device,
                loss_weight=ranking_loss_weight,
                margin=ranking_margin,
                min_delta=ranking_min_delta,
                listwise_loss_weight=listwise_loss_weight,
                listwise_temperature=listwise_temperature,
                score_index=ranking_score_index,
            )
        model.eval()
        with torch.inference_mode():
            validation_loss = run_epoch(model, validation_loader, loss_fn, optimizer=None, device=device)
            validation_ranking_loss = 0.0
            if ranking_validation_loader is not None:
                validation_ranking_loss = run_ranking_epoch(
                    model,
                    ranking_validation_loader,
                    optimizer=None,
                    device=device,
                    loss_weight=ranking_loss_weight,
                    margin=ranking_margin,
                    min_delta=ranking_min_delta,
                    listwise_loss_weight=listwise_loss_weight,
                    listwise_temperature=listwise_temperature,
                    score_index=ranking_score_index,
                )
        validation_score_mean, validation_score_std = prediction_stats(model, validation_loader, device=device)
        selection_loss = checkpoint_selection_loss(
            checkpoint_selection,
            validation_loss=validation_loss,
            validation_ranking_loss=validation_ranking_loss,
            has_ranking=ranking_validation_loader is not None,
            epoch=epoch,
        )
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_state = clone_state_dict(model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "ranking_loss": ranking_loss,
                "validation_loss": validation_loss,
                "validation_ranking_loss": validation_ranking_loss,
                "validation_score_mean": validation_score_mean,
                "validation_score_std": validation_score_std,
                "selection_loss": selection_loss,
                "best_selection_loss": best_selection_loss,
            }
        )
        print(
            f"{name} epoch {epoch:02d}: train={train_loss:.6f} rank={ranking_loss:.6f} "
            f"val={validation_loss:.6f} val_rank={validation_ranking_loss:.6f} "
            f"select={selection_loss:.6f} score_mean={validation_score_mean:.4f} score_std={validation_score_std:.4f}"
        )

    return history, best_state


def checkpoint_selection_loss(
    selection: str,
    *,
    validation_loss: float,
    validation_ranking_loss: float,
    has_ranking: bool,
    epoch: int,
) -> float:
    if selection == "validation_loss" or not has_ranking:
        return validation_loss
    if selection == "ranking_loss":
        return validation_ranking_loss
    if selection == "combined_loss":
        return validation_loss + validation_ranking_loss
    if selection == "final":
        return -float(epoch)
    raise ValueError(f"unknown checkpoint selection {selection!r}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_weight: float = 1.0,
) -> float:
    total_loss = 0.0
    total_examples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y) * loss_weight
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch_x)
        total_examples += len(batch_x)
    return total_loss / max(total_examples, 1)


def prediction_stats(model: nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[float, float]:
    outputs = []
    with torch.inference_mode():
        for batch_x, _ in loader:
            outputs.append(model(batch_x.to(device)).detach().cpu().reshape(-1))
    if not outputs:
        return 0.0, 0.0
    values = torch.cat(outputs)
    return float(values.mean()), float(values.std(unbiased=False))


def run_ranking_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_weight: float,
    margin: float,
    min_delta: float,
    listwise_loss_weight: float,
    listwise_temperature: float,
    score_index: int | None,
) -> float:
    total_loss = 0.0
    total_examples = 0
    for batch_x, batch_labels, batch_weights in loader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)
        flat_x = batch_x.reshape(-1, batch_x.shape[-1])
        prediction = model(flat_x).reshape(batch_x.shape[0], batch_x.shape[1], -1)
        if score_index is None:
            scores = prediction.squeeze(-1)
        else:
            scores = prediction[..., score_index]
        pairwise_loss = pairwise_margin_loss(scores, batch_labels, weights=batch_weights, margin=margin, min_delta=min_delta)
        listwise_loss = listwise_softmax_loss(scores, batch_labels, temperature=listwise_temperature)
        loss = pairwise_loss * loss_weight + listwise_loss * listwise_loss_weight
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch_x)
        total_examples += len(batch_x)
    return total_loss / max(total_examples, 1)


def pairwise_margin_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    margin: float,
    min_delta: float = 0.0,
) -> torch.Tensor:
    losses = []
    if weights is None:
        weights = torch.ones_like(labels)
    for case_scores, case_labels, case_weights in zip(scores, labels, weights):
        label_diff = case_labels[:, None] - case_labels[None, :]
        mask = label_diff >= min_delta
        if not torch.any(mask):
            continue
        score_diff = case_scores[:, None] - case_scores[None, :]
        pair_weights = torch.sqrt(case_weights[:, None].clamp_min(0.0) * case_weights[None, :].clamp_min(0.0))
        pair_losses = torch.relu(margin * label_diff.clamp_min(0.0) - score_diff)
        weighted_losses = pair_losses * pair_weights * mask
        losses.append(weighted_losses.sum() / (pair_weights * mask).sum().clamp_min(1e-12))
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def listwise_softmax_loss(scores: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.15) -> torch.Tensor:
    label_mass = labels.sum(dim=1, keepdim=True)
    valid = label_mass.squeeze(1) > 0.0
    if not torch.any(valid):
        return scores.new_tensor(0.0)
    centered = labels - labels.max(dim=1, keepdim=True).values
    target = torch.softmax(centered / max(temperature, 1e-6), dim=1)
    log_probs = torch.log_softmax(scores, dim=1)
    losses = -(target * log_probs).sum(dim=1)
    return losses[valid].mean()


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


if __name__ == "__main__":
    main()
