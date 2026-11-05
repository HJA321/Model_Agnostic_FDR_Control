"""
Clean PyTorch AFS feature-selection components for tabular and sequence models.

The original AFS code learns a sample-wise attention matrix A and multiplies the
input by A before the prediction model. This file keeps that idea but separates
the feature gate from the downstream model, so it can be reused with MLP, GRU,
LSTM, and Transformer backbones.

For sequence inputs shaped [batch, time, features], the default behavior is
variable-level feature selection: one gate is learned for each input feature and
broadcast to every time step. Use context_mode="flatten" if the gate should look
at the whole sequence instead of a temporal average when computing attention.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Batch = Tuple[torch.Tensor, torch.Tensor]


def _init_linear(linear: nn.Linear):
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.1)
    nn.init.constant_(linear.bias, 0.1)


class AFSFeatureGate(nn.Module):
    """AFS attention module that generates one gate per input feature."""

    def __init__(
        self,
        input_size: int,
        extract_size: int = 32,
        attention_size: int = 2,
        context_mode: str = "mean",
        sequence_length: Optional[int] = None,
    ):
        super().__init__()
        if attention_size != 2:
            raise ValueError("AFS uses binary select/unselect attention; attention_size must be 2.")
        if context_mode not in {"mean", "last", "flatten"}:
            raise ValueError("context_mode must be one of: mean, last, flatten.")
        if context_mode == "flatten" and sequence_length is None:
            raise ValueError("sequence_length is required when context_mode='flatten'.")

        self.input_size = input_size
        self.extract_size = extract_size
        self.attention_size = attention_size
        self.context_mode = context_mode
        self.sequence_length = sequence_length
        context_size = input_size * sequence_length if context_mode == "flatten" else input_size

        self.extract = nn.Linear(context_size, extract_size)
        self.attention_weight = nn.Parameter(torch.empty(input_size, extract_size, attention_size))
        self.attention_bias = nn.Parameter(torch.empty(input_size, attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        _init_linear(self.extract)
        nn.init.trunc_normal_(self.attention_weight, mean=0.0, std=0.1)
        nn.init.constant_(self.attention_bias, 0.1)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() != 3:
            raise ValueError("Expected tabular [batch, features] or sequence [batch, time, features] input.")
        if x.shape[-1] != self.input_size:
            raise ValueError("Expected input feature size {}, got {}.".format(self.input_size, x.shape[-1]))
        if self.context_mode == "mean":
            return x.mean(dim=1)
        if self.context_mode == "last":
            return x[:, -1]
        return x.reshape(x.shape[0], -1)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        context = self.context(x)
        e = torch.tanh(self.extract(context))
        logits = torch.einsum("be,fea->bfa", e, self.attention_weight)
        logits = logits + self.attention_bias.unsqueeze(0)
        return F.softmax(logits, dim=2)[:, :, 1]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attention = self.attention(x)
        if x.dim() == 2:
            gated = x * attention
        else:
            gated = x * attention.unsqueeze(1)
        if return_attention:
            return gated, attention
        return gated


class AFSFeatureSelectionModel(nn.Module):
    """Wrap any backbone with an AFS input gate."""

    def __init__(self, gate: AFSFeatureGate, backbone: nn.Module):
        super().__init__()
        self.gate = gate
        self.backbone = backbone

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate.attention(x)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        gated, attention = self.gate(x, return_attention=True)
        output = self.backbone(gated)
        if return_attention:
            return output, attention
        return output


class MLPBackbone(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (64, 64, 64), output_size: int = 1):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUBackbone(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class LSTMBackbone(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class TransformerBackbone(nn.Module):
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
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1])


def build_afs_mlp(
    input_size: int,
    hidden_sizes: Sequence[int] = (64, 64, 64),
    output_size: int = 1,
    extract_size: int = 32,
) -> AFSFeatureSelectionModel:
    gate = AFSFeatureGate(input_size=input_size, extract_size=extract_size)
    backbone = MLPBackbone(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    return AFSFeatureSelectionModel(gate, backbone)


def build_afs_gru(
    input_size: int,
    hidden_size: int = 64,
    output_size: int = 1,
    extract_size: int = 32,
    context_mode: str = "mean",
    bidirectional: bool = False,
) -> AFSFeatureSelectionModel:
    gate = AFSFeatureGate(input_size=input_size, extract_size=extract_size, context_mode=context_mode)
    backbone = GRUBackbone(input_size=input_size, hidden_size=hidden_size, output_size=output_size, bidirectional=bidirectional)
    return AFSFeatureSelectionModel(gate, backbone)


def build_afs_lstm(
    input_size: int,
    hidden_size: int = 64,
    output_size: int = 1,
    extract_size: int = 32,
    context_mode: str = "mean",
    bidirectional: bool = False,
) -> AFSFeatureSelectionModel:
    gate = AFSFeatureGate(input_size=input_size, extract_size=extract_size, context_mode=context_mode)
    backbone = LSTMBackbone(input_size=input_size, hidden_size=hidden_size, output_size=output_size, bidirectional=bidirectional)
    return AFSFeatureSelectionModel(gate, backbone)


def build_afs_transformer(
    input_size: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    output_size: int = 1,
    extract_size: int = 32,
    context_mode: str = "mean",
) -> AFSFeatureSelectionModel:
    gate = AFSFeatureGate(input_size=input_size, extract_size=extract_size, context_mode=context_mode)
    backbone = TransformerBackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_size=output_size,
    )
    return AFSFeatureSelectionModel(gate, backbone)


def train_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    input_getter: Callable[[Batch], torch.Tensor] = lambda batch: batch[0],
    target_getter: Callable[[Batch], torch.Tensor] = lambda batch: batch[1],
) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = input_getter(batch).to(device)
        y = target_getter(batch).to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.shape[0]
        total_count += x.shape[0]
    return total_loss / max(total_count, 1)


@torch.no_grad()
def feature_scores(
    model: AFSFeatureSelectionModel,
    loader: Iterable[Batch],
    device: Optional[torch.device] = None,
    input_getter: Callable[[Batch], torch.Tensor] = lambda batch: batch[0],
) -> torch.Tensor:
    """Average AFS attention over a dataset, matching the paper's feature score."""

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total = None
    count = 0

    for batch in loader:
        x = input_getter(batch).to(device)
        attention = model.attention(x)
        batch_sum = attention.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += attention.shape[0]
    return (total / max(count, 1)).detach().cpu()


def selected_features(scores: torch.Tensor, k: Optional[int] = None, threshold: Optional[float] = None) -> torch.Tensor:
    if k is not None:
        return torch.argsort(scores, descending=True)[:k]
    if threshold is None:
        threshold = float(scores.mean())
    return torch.nonzero(scores >= threshold, as_tuple=False).flatten()
