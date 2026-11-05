"""
Reusable Group Lasso / Adaptive Group Lasso utilities for neural feature selection.

The original alg-net scripts apply proximal shrinkage to rows of a hand-written
first-layer matrix shaped [n_features, hidden]. In idiomatic PyTorch, most input
weights are shaped [hidden, n_features], so this module treats each input feature
as one column group by default.

Works naturally with:
  - MLP / Transformer input projections: nn.Linear(in_features, hidden)
  - GRU / LSTM first recurrent input weights: weight_ih_l0, and reverse weights
    for bidirectional models

Typical use:
    model = GRURegressor(input_size=20, hidden_size=64)
    selector = FeatureGroupProx.from_rnn(model.rnn, input_size=20)
    train_with_prox(model, train_loader, loss_fn, optimizer, selector, lam=0.01)
    gl_norms = selector.group_norms()
    selector.set_adaptive_weights(gl_norms, gamma=2.0)
    train_with_prox(model, train_loader, loss_fn, optimizer, selector, lam=0.01)
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


Batch = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class GroupedParameter:
    parameter: torch.nn.Parameter
    feature_dim: int = 1

    def moved(self) -> torch.Tensor:
        if self.feature_dim == 0:
            return self.parameter
        return self.parameter.transpose(0, self.feature_dim)


class FeatureGroupProx:
    """Apply GL/AGL proximal updates to groups of input-feature weights."""

    def __init__(
        self,
        grouped_parameters: Sequence[GroupedParameter],
        input_size: int,
        adaptive_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
    ):
        self.grouped_parameters = list(grouped_parameters)
        self.input_size = input_size
        self.eps = eps
        self.adaptive_weights = None
        if adaptive_weights is not None:
            self.adaptive_weights = adaptive_weights.detach().clone()
        self._validate()

    @classmethod
    def from_linear(cls, layer: nn.Linear, adaptive_weights: Optional[torch.Tensor] = None):
        return cls(
            [GroupedParameter(layer.weight, feature_dim=1)],
            input_size=layer.in_features,
            adaptive_weights=adaptive_weights,
        )

    @classmethod
    def from_rnn(
        cls,
        rnn: nn.RNNBase,
        input_size: Optional[int] = None,
        adaptive_weights: Optional[torch.Tensor] = None,
    ):
        if input_size is None:
            input_size = rnn.input_size

        grouped = [GroupedParameter(rnn.weight_ih_l0, feature_dim=1)]
        reverse = getattr(rnn, "weight_ih_l0_reverse", None)
        if reverse is not None:
            grouped.append(GroupedParameter(reverse, feature_dim=1))
        return cls(grouped, input_size=input_size, adaptive_weights=adaptive_weights)

    def _validate(self):
        if not self.grouped_parameters:
            raise ValueError("At least one grouped parameter is required.")
        for grouped in self.grouped_parameters:
            if grouped.parameter.shape[grouped.feature_dim] != self.input_size:
                raise ValueError(
                    "Grouped parameter feature dimension has size {}, expected {}.".format(
                        grouped.parameter.shape[grouped.feature_dim], self.input_size
                    )
                )

    @property
    def device(self):
        return self.grouped_parameters[0].parameter.device

    def group_norms(self) -> torch.Tensor:
        norms_sq = torch.zeros(self.input_size, device=self.device)
        for grouped in self.grouped_parameters:
            weights = grouped.moved().reshape(self.input_size, -1)
            norms_sq = norms_sq + weights.pow(2).sum(dim=1)
        return torch.sqrt(norms_sq)

    def selected_features(self, threshold: float = 1e-8) -> torch.Tensor:
        return self.group_norms() > threshold

    def set_adaptive_weights(self, base_group_norms: torch.Tensor, gamma: float = 2.0):
        base_group_norms = base_group_norms.detach().to(self.device)
        self.adaptive_weights = 1.0 / base_group_norms.clamp_min(self.eps).pow(gamma)
        return self

    def clear_adaptive_weights(self):
        self.adaptive_weights = None
        return self

    @torch.no_grad()
    def step(self, lam: float, lr: float):
        group_norms = self.group_norms()
        penalty = torch.full_like(group_norms, float(lam))
        if self.adaptive_weights is not None:
            penalty = penalty * self.adaptive_weights.to(self.device)

        scale = (group_norms - lr * penalty).clamp_min(0.0)
        scale = scale / group_norms.clamp_min(self.eps)

        for grouped in self.grouped_parameters:
            view = grouped.moved()
            view.mul_(scale.view([self.input_size] + [1] * (view.dim() - 1)))


def train_with_prox(
    model: nn.Module,
    loader: Iterable[Batch],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    selector: FeatureGroupProx,
    lam: float,
    device: Optional[torch.device] = None,
    epochs: int = 1,
    input_getter: Callable[[Batch], torch.Tensor] = lambda batch: batch[0],
    target_getter: Callable[[Batch], torch.Tensor] = lambda batch: batch[1],
) -> List[float]:
    """Train a model and apply one proximal step after every optimizer step."""

    if device is None:
        device = next(model.parameters()).device

    losses = []
    model.train()
    for _ in range(epochs):
        for batch in loader:
            x = input_getter(batch).to(device)
            y = target_getter(batch).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            selector.step(lam=lam, lr=lr)
            losses.append(float(loss.detach().cpu()))
    return losses


class MLPRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (64, 64, 64), output_size: int = 1):
        super().__init__()
        layers = []
        prev = input_size
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        layers.extend([self.input_layer, nn.Tanh()])
        prev = hidden_sizes[0]
        for hidden in hidden_sizes[1:]:
            layers.extend([nn.Linear(prev, hidden), nn.Tanh()])
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def feature_selector(self) -> FeatureGroupProx:
        return FeatureGroupProx.from_linear(self.input_layer)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

    def feature_selector(self) -> FeatureGroupProx:
        return FeatureGroupProx.from_rnn(self.rnn)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

    def feature_selector(self) -> FeatureGroupProx:
        return FeatureGroupProx.from_rnn(self.rnn)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1])

    def feature_selector(self) -> FeatureGroupProx:
        return FeatureGroupProx.from_linear(self.input_projection)
