from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ImportError(
        "vector_graph.torch_models requires PyTorch. Install the 'torch' extra or run in Colab."
    ) from exc

from .frames import EdgeFrame, NodeFrame, TraversalScores
from .scorer import TraversalScorer
from .vectors import resize_vector


class SmallTraversalNet(nn.Module):
    """Small MLP scorer for vector-frame traversal.

    It is intentionally simple for the prototype. A transformer can replace
    this class without changing the graph/controller API.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        summary_dim: int,
        edge_dim: int,
        path_dim: int,
        scalar_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.summary_dim = summary_dim
        self.edge_dim = edge_dim
        self.path_dim = path_dim
        self.scalar_dim = scalar_dim
        raw_dim = query_dim + summary_dim + edge_dim + summary_dim + path_dim + scalar_dim
        interaction_dim = summary_dim * 6 + edge_dim * 2
        input_dim = raw_dim + interaction_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        query, current, edge, dst, path, scalars = split_traversal_inputs(
            inputs,
            query_dim=self.query_dim,
            summary_dim=self.summary_dim,
            edge_dim=self.edge_dim,
            path_dim=self.path_dim,
            scalar_dim=self.scalar_dim,
        )
        edge_query = fold_resize(query, self.edge_dim)
        features = torch.cat(
            [
                query,
                current,
                edge,
                dst,
                path,
                scalars,
                query * dst,
                torch.abs(query - dst),
                current * dst,
                torch.abs(current - dst),
                path * dst,
                torch.abs(path - dst),
                edge_query * edge,
                torch.abs(edge_query - edge),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.layers(features))


class SmallAttachNet(nn.Module):
    def __init__(self, *, summary_dim: int, full_dim: int, path_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.full_dim = full_dim
        self.path_dim = path_dim
        raw_dim = summary_dim + summary_dim + full_dim + full_dim + path_dim
        interaction_dim = summary_dim * 4 + full_dim * 2
        input_dim = raw_dim + interaction_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        new_summary, candidate_summary, new_full, candidate_full, path = split_attach_inputs(
            inputs,
            summary_dim=self.summary_dim,
            full_dim=self.full_dim,
            path_dim=self.path_dim,
        )
        features = torch.cat(
            [
                new_summary,
                candidate_summary,
                new_full,
                candidate_full,
                path,
                new_summary * candidate_summary,
                torch.abs(new_summary - candidate_summary),
                new_full * candidate_full,
                torch.abs(new_full - candidate_full),
                path * candidate_summary,
                torch.abs(path - candidate_summary),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.layers(features))


class SmallTraversalTransformerNet(nn.Module):
    """Small BERT-style token mixer over vector frames."""

    def __init__(
        self,
        *,
        query_dim: int,
        summary_dim: int,
        edge_dim: int,
        path_dim: int,
        scalar_dim: int = 2,
        hidden_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.summary_dim = summary_dim
        self.edge_dim = edge_dim
        self.path_dim = path_dim
        self.scalar_dim = scalar_dim
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.summary_projection = nn.Linear(summary_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        self.path_projection = nn.Linear(path_dim, hidden_dim)
        self.scalar_projection = nn.Linear(scalar_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_embedding = nn.Embedding(15, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 5), nn.Sigmoid())

    def forward(self, inputs: Tensor) -> Tensor:
        query, current, edge, dst, path, scalars = split_traversal_inputs(
            inputs,
            query_dim=self.query_dim,
            summary_dim=self.summary_dim,
            edge_dim=self.edge_dim,
            path_dim=self.path_dim,
            scalar_dim=self.scalar_dim,
        )
        batch_size = inputs.shape[0]
        query_summary = fold_resize(query, self.summary_dim)
        path_summary = fold_resize(path, self.summary_dim)
        edge_query = fold_resize(query, self.edge_dim)
        tokens = torch.stack(
            [
                self.query_projection(query),
                self.summary_projection(current),
                self.edge_projection(edge),
                self.summary_projection(dst),
                self.path_projection(path),
                self.scalar_projection(scalars),
                self.summary_projection(query_summary * dst),
                self.summary_projection(torch.abs(query_summary - dst)),
                self.summary_projection(current * dst),
                self.summary_projection(torch.abs(current - dst)),
                self.summary_projection(path_summary * dst),
                self.summary_projection(torch.abs(path_summary - dst)),
                self.edge_projection(edge_query * edge),
                self.edge_projection(torch.abs(edge_query - edge)),
            ],
            dim=1,
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        type_ids = torch.arange(tokens.shape[1], device=inputs.device).unsqueeze(0)
        encoded = self.encoder(tokens + self.type_embedding(type_ids))
        return self.output(encoded[:, 0])


class SmallAttachTransformerNet(nn.Module):
    def __init__(
        self,
        *,
        summary_dim: int,
        full_dim: int,
        path_dim: int,
        hidden_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.full_dim = full_dim
        self.path_dim = path_dim
        self.summary_projection = nn.Linear(summary_dim, hidden_dim)
        self.full_projection = nn.Linear(full_dim, hidden_dim)
        self.path_projection = nn.Linear(path_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_embedding = nn.Embedding(12, hidden_dim)
        raw_dim = summary_dim + summary_dim + full_dim + full_dim + path_dim
        interaction_dim = summary_dim * 4 + full_dim * 2
        self.feature_projection = nn.Sequential(
            nn.Linear(raw_dim + interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        new_summary, candidate_summary, new_full, candidate_full, path = split_attach_inputs(
            inputs,
            summary_dim=self.summary_dim,
            full_dim=self.full_dim,
            path_dim=self.path_dim,
        )
        batch_size = inputs.shape[0]
        path_summary = fold_resize(path, self.summary_dim)
        explicit_features = torch.cat(
            [
                new_summary,
                candidate_summary,
                new_full,
                candidate_full,
                path,
                new_summary * candidate_summary,
                torch.abs(new_summary - candidate_summary),
                new_full * candidate_full,
                torch.abs(new_full - candidate_full),
                path_summary * candidate_summary,
                torch.abs(path_summary - candidate_summary),
            ],
            dim=-1,
        )
        tokens = torch.stack(
            [
                self.summary_projection(new_summary),
                self.summary_projection(candidate_summary),
                self.full_projection(new_full),
                self.full_projection(candidate_full),
                self.path_projection(path),
                self.summary_projection(new_summary * candidate_summary),
                self.summary_projection(torch.abs(new_summary - candidate_summary)),
                self.full_projection(new_full * candidate_full),
                self.full_projection(torch.abs(new_full - candidate_full)),
                self.summary_projection(path_summary * candidate_summary),
                self.summary_projection(torch.abs(path_summary - candidate_summary)),
            ],
            dim=1,
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        type_ids = torch.arange(tokens.shape[1], device=inputs.device).unsqueeze(0)
        encoded = self.encoder(tokens + self.type_embedding(type_ids))
        explicit = self.feature_projection(explicit_features)
        return self.output(torch.cat([encoded[:, 0], explicit], dim=-1))


@dataclass(frozen=True)
class TorchModelConfig:
    query_dim: int
    summary_dim: int
    edge_dim: int
    full_dim: int
    path_dim: int
    scalar_dim: int = 2
    model_kind: str = "mlp"
    hidden_dim: int = 128
    attach_hidden_dim: int = 96
    transformer_layers: int = 2
    transformer_heads: int = 4


class TorchTraversalScorer(TraversalScorer):
    def __init__(
        self,
        *,
        traversal_model: nn.Module,
        attach_model: nn.Module,
        config: TorchModelConfig,
        device: str = "cpu",
    ) -> None:
        self.traversal_model = traversal_model.to(device).eval()
        self.attach_model = attach_model.to(device).eval()
        self.config = config
        self.device = torch.device(device)

    @classmethod
    def initialized(cls, config: TorchModelConfig, *, seed: int = 0, device: str = "cpu") -> "TorchTraversalScorer":
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        traversal_model, attach_model = create_model_pair(config)
        return cls(traversal_model=traversal_model, attach_model=attach_model, config=config, device=device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> "TorchTraversalScorer":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = TorchModelConfig(**checkpoint["config"])
        traversal_model, attach_model = create_model_pair(config)
        traversal_model.load_state_dict(checkpoint["traversal_model"])
        attach_model.load_state_dict(checkpoint["attach_model"])
        return cls(traversal_model=traversal_model, attach_model=attach_model, config=config, device=device)

    def score_edge(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edge: EdgeFrame,
        dst_node: NodeFrame,
        path_vector: Sequence[float],
        hop: int,
    ) -> TraversalScores:
        return self.score_edges(
            query_vector=query_vector,
            current_node=current_node,
            edges=[edge],
            dst_nodes=[dst_node],
            path_vector=path_vector,
            hop=hop,
        )[0]

    def score_edges(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edges: Sequence[EdgeFrame],
        dst_nodes: Sequence[NodeFrame],
        path_vector: Sequence[float],
        hop: int,
    ) -> tuple[TraversalScores, ...]:
        if len(edges) != len(dst_nodes):
            raise ValueError("edges and dst_nodes must have the same length")
        if not edges:
            return ()

        rows = []
        query = resize_vector(query_vector, self.config.query_dim)
        current_summary = resize_vector(current_node.summary_vector, self.config.summary_dim)
        path = resize_vector(path_vector, self.config.path_dim)
        for edge, dst_node in zip(edges, dst_nodes):
            rows.append(
                [
                    query,
                    current_summary,
                    resize_vector(edge.edge_vector, self.config.edge_dim),
                    resize_vector(dst_node.summary_vector, self.config.summary_dim),
                    path,
                    traversal_scalars(edge.confidence, hop, self.config.scalar_dim),
                ]
            )

        inputs = self._batch_tensor(rows)
        with torch.inference_mode():
            batch_scores = self.traversal_model(inputs).detach().cpu().tolist()
        return tuple(_scores_from_values(scores) for scores in batch_scores)

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        new_full = new_node.full_vector if new_node.full_vector is not None else new_node.summary_vector
        candidate_full = candidate_node.full_vector if candidate_node.full_vector is not None else candidate_node.summary_vector
        inputs = self._tensor(
            [
                resize_vector(new_node.summary_vector, self.config.summary_dim),
                resize_vector(candidate_node.summary_vector, self.config.summary_dim),
                resize_vector(new_full, self.config.full_dim),
                resize_vector(candidate_full, self.config.full_dim),
                resize_vector(path_vector, self.config.path_dim),
            ]
        )
        with torch.inference_mode():
            score = self.attach_model(inputs).detach().cpu().item()
        return float(score)

    def _tensor(self, vectors: Sequence[Sequence[float]]) -> Tensor:
        values = torch.tensor([value for vector in vectors for value in vector], dtype=torch.float32, device=self.device)
        return values.unsqueeze(0)

    def _batch_tensor(self, rows: Sequence[Sequence[Sequence[float]]]) -> Tensor:
        values = [[value for vector in row for value in vector] for row in rows]
        return torch.tensor(values, dtype=torch.float32, device=self.device)


def _scores_from_values(values: Sequence[float]) -> TraversalScores:
    return TraversalScores(
        follow_score=float(values[0]),
        read_full_score=float(values[1]),
        include_score=float(values[2]),
        expand_score=float(values[3]),
        stop_score=float(values[4]),
    )


def create_model_pair(config: TorchModelConfig) -> tuple[nn.Module, nn.Module]:
    if config.model_kind == "mlp":
        return (
            SmallTraversalNet(
                query_dim=config.query_dim,
                summary_dim=config.summary_dim,
                edge_dim=config.edge_dim,
                path_dim=config.path_dim,
                scalar_dim=config.scalar_dim,
                hidden_dim=config.hidden_dim,
            ),
            SmallAttachNet(
                summary_dim=config.summary_dim,
                full_dim=config.full_dim,
                path_dim=config.path_dim,
                hidden_dim=config.attach_hidden_dim,
            ),
        )
    if config.model_kind == "transformer":
        return (
            SmallTraversalTransformerNet(
                query_dim=config.query_dim,
                summary_dim=config.summary_dim,
                edge_dim=config.edge_dim,
                path_dim=config.path_dim,
                scalar_dim=config.scalar_dim,
                hidden_dim=config.hidden_dim,
                layers=config.transformer_layers,
                heads=config.transformer_heads,
            ),
            SmallAttachTransformerNet(
                summary_dim=config.summary_dim,
                full_dim=config.full_dim,
                path_dim=config.path_dim,
                hidden_dim=config.attach_hidden_dim,
                layers=config.transformer_layers,
                heads=config.transformer_heads,
            ),
        )
    raise ValueError(f"unknown model_kind {config.model_kind!r}")


def traversal_scalars(confidence: float, hop: int, scalar_dim: int = 2) -> tuple[float, ...]:
    values = [max(0.0, min(1.0, float(confidence))), max(0.0, min(1.0, float(hop) / 3.0))]
    if scalar_dim <= len(values):
        return tuple(values[:scalar_dim])
    return tuple(values + [0.0] * (scalar_dim - len(values)))


def split_traversal_inputs(
    inputs: Tensor,
    *,
    query_dim: int,
    summary_dim: int,
    edge_dim: int,
    path_dim: int,
    scalar_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    sizes = [query_dim, summary_dim, edge_dim, summary_dim, path_dim, scalar_dim]
    return torch.split(inputs, sizes, dim=-1)


def split_attach_inputs(
    inputs: Tensor,
    *,
    summary_dim: int,
    full_dim: int,
    path_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    sizes = [summary_dim, summary_dim, full_dim, full_dim, path_dim]
    return torch.split(inputs, sizes, dim=-1)


def fold_resize(values: Tensor, dimension: int) -> Tensor:
    if values.shape[-1] == dimension:
        return values
    output = values.new_zeros((*values.shape[:-1], dimension))
    for index in range(values.shape[-1]):
        output[..., index % dimension] += values[..., index]
    norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
    return torch.where(norm > 0.0, output / norm.clamp_min(1e-12), output)
