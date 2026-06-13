from __future__ import annotations

from dataclasses import dataclass
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
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = query_dim + summary_dim + edge_dim + summary_dim + path_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self.layers(inputs))


class SmallAttachNet(nn.Module):
    def __init__(self, *, summary_dim: int, full_dim: int, path_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        input_dim = summary_dim + summary_dim + full_dim + full_dim + path_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self.layers(inputs)).squeeze(-1)


@dataclass(frozen=True)
class TorchModelConfig:
    query_dim: int
    summary_dim: int
    edge_dim: int
    full_dim: int
    path_dim: int
    hidden_dim: int = 128
    attach_hidden_dim: int = 96


class TorchTraversalScorer(TraversalScorer):
    def __init__(
        self,
        *,
        traversal_model: SmallTraversalNet,
        attach_model: SmallAttachNet,
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
        traversal_model = SmallTraversalNet(
            query_dim=config.query_dim,
            summary_dim=config.summary_dim,
            edge_dim=config.edge_dim,
            path_dim=config.path_dim,
            hidden_dim=config.hidden_dim,
        )
        attach_model = SmallAttachNet(
            summary_dim=config.summary_dim,
            full_dim=config.full_dim,
            path_dim=config.path_dim,
            hidden_dim=config.attach_hidden_dim,
        )
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
        del hop
        inputs = self._tensor(
            [
                resize_vector(query_vector, self.config.query_dim),
                resize_vector(current_node.summary_vector, self.config.summary_dim),
                resize_vector(edge.edge_vector, self.config.edge_dim),
                resize_vector(dst_node.summary_vector, self.config.summary_dim),
                resize_vector(path_vector, self.config.path_dim),
            ]
        )
        with torch.inference_mode():
            scores = self.traversal_model(inputs).detach().cpu().reshape(-1).tolist()
        return TraversalScores(
            follow_score=float(scores[0]),
            read_full_score=float(scores[1]),
            include_score=float(scores[2]),
            expand_score=float(scores[3]),
            stop_score=float(scores[4]),
        )

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
