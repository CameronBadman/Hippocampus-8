#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from vector_graph.torch_models import SmallAttachNet, SmallTraversalNet, TorchModelConfig, traversal_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vector-frame scorer models from JSONL shards.")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--ranking-data-dir", default=None)
    parser.add_argument("--output", default="models/synthetic_scorer.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ranking-batch-size", type=int, default=128)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.35)
    parser.add_argument("--ranking-margin", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)

    config = load_config(data_dir)
    traversal_x, traversal_y = load_traversal_examples(data_dir)
    attach_x, attach_y = load_attach_examples(data_dir)
    traversal_ranking = None
    attach_ranking = None
    if args.ranking_data_dir is not None:
        ranking_data_dir = Path(args.ranking_data_dir)
        traversal_ranking = load_traversal_ranking_examples(ranking_data_dir)
        attach_ranking = load_attach_ranking_examples(ranking_data_dir)

    traversal_model = SmallTraversalNet(
        query_dim=config.query_dim,
        summary_dim=config.summary_dim,
        edge_dim=config.edge_dim,
        path_dim=config.path_dim,
        scalar_dim=config.scalar_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    attach_model = SmallAttachNet(
        summary_dim=config.summary_dim,
        full_dim=config.full_dim,
        path_dim=config.path_dim,
        hidden_dim=config.attach_hidden_dim,
    ).to(device)

    print(f"device: {device}")
    print(f"traversal examples: {len(traversal_x)}")
    print(f"attach examples: {len(attach_x)}")
    if traversal_ranking is not None and attach_ranking is not None:
        print(f"traversal ranking cases: {len(traversal_ranking[0])}")
        print(f"attach ranking cases: {len(attach_ranking[0])}")

    traversal_history = train_regressor(
        name="traversal",
        model=traversal_model,
        x=traversal_x,
        y=traversal_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        ranking=traversal_ranking,
        ranking_batch_size=args.ranking_batch_size,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        ranking_score_index=0,
    )
    attach_history = train_regressor(
        name="attach",
        model=attach_model,
        x=attach_x,
        y=attach_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 1,
        device=device,
        ranking=attach_ranking,
        ranking_batch_size=args.ranking_batch_size,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        ranking_score_index=None,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "traversal_model": traversal_model.state_dict(),
            "attach_model": attach_model.state_dict(),
            "traversal_history": traversal_history,
            "attach_history": attach_history,
        },
        output,
    )
    print(f"saved checkpoint: {output}")


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_config(data_dir: Path) -> TorchModelConfig:
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
        )
    return TorchModelConfig(query_dim=32, summary_dim=32, edge_dim=16, full_dim=64, path_dim=32)


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
            targets.append(example["target"])
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


def load_traversal_ranking_examples(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    groups = []
    labels = []
    for path in sorted(data_dir.glob("traversal_ranking*.jsonl")):
        for case in read_jsonl(path):
            group = []
            group_labels = []
            for candidate in case["candidates"]:
                group.append(
                    case["query"]
                    + case["current_summary"]
                    + candidate["edge"]
                    + candidate["dst_summary"]
                    + case["path"]
                    + list(traversal_scalars(candidate["confidence"], candidate["hop"]))
                )
                group_labels.append(candidate["label"])
            groups.append(group)
            labels.append(group_labels)
    if not groups:
        raise ValueError(f"no traversal_ranking*.jsonl files found in {data_dir}")
    return torch.tensor(groups, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def load_attach_ranking_examples(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    groups = []
    labels = []
    for path in sorted(data_dir.glob("attach_ranking*.jsonl")):
        for case in read_jsonl(path):
            group = []
            group_labels = []
            for candidate in case["candidates"]:
                group.append(
                    case["new_summary"]
                    + candidate["candidate_summary"]
                    + case["new_full"]
                    + candidate["candidate_full"]
                    + case["path"]
                )
                group_labels.append(candidate["label"])
            groups.append(group)
            labels.append(group_labels)
    if not groups:
        raise ValueError(f"no attach_ranking*.jsonl files found in {data_dir}")
    return torch.tensor(groups, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


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
    ranking: tuple[torch.Tensor, torch.Tensor] | None = None,
    ranking_batch_size: int = 128,
    ranking_loss_weight: float = 0.35,
    ranking_margin: float = 0.08,
    ranking_score_index: int | None = None,
) -> list[dict[str, float]]:
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
    if ranking is not None:
        ranking_dataset = TensorDataset(*ranking)
        ranking_loader = DataLoader(
            ranking_dataset,
            batch_size=ranking_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 101),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer, device=device)
        ranking_loss = 0.0
        if ranking_loader is not None:
            ranking_loss = run_ranking_epoch(
                model,
                ranking_loader,
                optimizer=optimizer,
                device=device,
                loss_weight=ranking_loss_weight,
                margin=ranking_margin,
                score_index=ranking_score_index,
            )
        model.eval()
        with torch.inference_mode():
            validation_loss = run_epoch(model, validation_loader, loss_fn, optimizer=None, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "ranking_loss": ranking_loss,
                "validation_loss": validation_loss,
            }
        )
        print(f"{name} epoch {epoch:02d}: train={train_loss:.6f} rank={ranking_loss:.6f} val={validation_loss:.6f}")

    return history


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_examples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch_x)
        total_examples += len(batch_x)
    return total_loss / max(total_examples, 1)


def run_ranking_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: float,
    margin: float,
    score_index: int | None,
) -> float:
    total_loss = 0.0
    total_examples = 0
    for batch_x, batch_labels in loader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)
        flat_x = batch_x.reshape(-1, batch_x.shape[-1])
        prediction = model(flat_x).reshape(batch_x.shape[0], batch_x.shape[1], -1)
        if score_index is None:
            scores = prediction.squeeze(-1)
        else:
            scores = prediction[..., score_index]
        loss = pairwise_margin_loss(scores, batch_labels, margin=margin) * loss_weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch_x)
        total_examples += len(batch_x)
    return total_loss / max(total_examples, 1)


def pairwise_margin_loss(scores: torch.Tensor, labels: torch.Tensor, *, margin: float) -> torch.Tensor:
    losses = []
    for case_scores, case_labels in zip(scores, labels):
        positive = case_scores[case_labels > 0.5]
        negative = case_scores[case_labels <= 0.5]
        if len(positive) == 0 or len(negative) == 0:
            continue
        losses.append(torch.relu(margin - positive[:, None] + negative[None, :]).mean())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


if __name__ == "__main__":
    main()
