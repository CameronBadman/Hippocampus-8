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
    parser.add_argument("--output", default="models/synthetic_scorer.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer, device=device)
        model.eval()
        with torch.inference_mode():
            validation_loss = run_epoch(model, validation_loader, loss_fn, optimizer=None, device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"{name} epoch {epoch:02d}: train={train_loss:.6f} val={validation_loss:.6f}")

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


if __name__ == "__main__":
    main()
