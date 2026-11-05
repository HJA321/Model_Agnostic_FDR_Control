"""Clean PyTorch implementation of mirror-statistic deep feature selection.

This module keeps the proposed method separate from the original experiment
scripts.  It trains two independent models on two data splits, computes input
gradient scores, builds mirror statistics, and selects variables by the
mirror-FDR threshold.

The selector is model-agnostic: pass a ``build_model(input_shape)`` function
returning any differentiable PyTorch model.  For sequence inputs shaped
``[batch, time, features]``, the default feature axis is the last dimension and
input gradients are summed over batch and time to produce one score per
variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Array = np.ndarray
LossName = Literal["mse", "binary", "ce"]
PsiName = Literal["min", "mean", "geomean", "harmmean"]
OptimizerName = Literal["sgd", "adam", "adamw"]
GradientEvalName = Literal["train_split", "cross_fit"]


@dataclass
class MirrorPathPoint:
    step: int
    tau: float
    r_plus: int
    r_minus: int
    fdp_hat: float
    selected_size: int
    loss1: float
    loss2: float


@dataclass
class MirrorFeatureSelectionResult:
    selected: Array
    tau: float
    mirror_scores: Array
    xi1: Array
    xi2: Array
    fdp_hat: float
    r_plus: int
    r_minus: int
    history: list[MirrorPathPoint] = field(default_factory=list)
    model1: Optional[nn.Module] = None
    model2: Optional[nn.Module] = None
    mu: Optional[Array] = None
    sigma: Optional[Array] = None


def _positive_axis(ndim: int, axis: int) -> int:
    axis = axis if axis >= 0 else ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"feature_axis={axis} is out of range for ndim={ndim}")
    return axis


def standardize_by_feature(
    X: Array,
    feature_axis: int = -1,
    eps: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """Standardize each feature across all non-feature axes.

    For tabular ``X`` shaped ``[N, F]`` this standardizes across samples.  For
    sequences ``[N, T, F]`` it standardizes each variable across samples and
    time.  The returned ``mu`` and ``sigma`` are one-dimensional arrays of
    length ``F`` for chain-rule rescaling of input gradients.
    """

    X = np.asarray(X, dtype=np.float32)
    axis = _positive_axis(X.ndim, feature_axis)
    if axis == 0:
        raise ValueError("feature_axis cannot be the sample/batch axis")

    reduce_axes = tuple(i for i in range(X.ndim) if i != axis)
    mu_keep = X.mean(axis=reduce_axes, keepdims=True)
    sigma_keep = X.std(axis=reduce_axes, keepdims=True) + eps
    Xn = (X - mu_keep) / sigma_keep

    mu = np.moveaxis(mu_keep, axis, -1).reshape(-1).astype(np.float32)
    sigma = np.moveaxis(sigma_keep, axis, -1).reshape(-1).astype(np.float32)
    return Xn.astype(np.float32), mu, sigma


def psi_aggregate(a: Array, b: Array, method: PsiName = "min", eps: float = 1e-12) -> Array:
    if method == "min":
        return np.minimum(a, b)
    if method == "mean":
        return 0.5 * (a + b)
    if method == "geomean":
        return np.sqrt(np.maximum(a * b, 0.0) + eps)
    if method == "harmmean":
        return (2.0 * a * b) / (a + b + eps)
    raise ValueError(f"Unknown psi method: {method}")


def mirror_statistic(xi1: Array, xi2: Array, psi: PsiName = "min") -> Array:
    xi1 = np.asarray(xi1, dtype=np.float64)
    xi2 = np.asarray(xi2, dtype=np.float64)
    sign = np.sign(xi1 * xi2)
    mag = psi_aggregate(np.abs(xi1), np.abs(xi2), method=psi)
    return sign * mag


def _threshold_candidates(M: Array) -> Array:
    vals = np.asarray(M, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    abs_vals = np.unique(np.abs(vals))
    abs_vals = abs_vals[abs_vals > 0.0]
    if abs_vals.size == 0:
        return np.array([], dtype=np.float64)
    abs_vals.sort()
    grid = np.concatenate([[0.0], abs_vals])
    return 0.5 * (grid[:-1] + grid[1:])


def mirror_fdr_threshold(M: Array, alpha: float) -> tuple[float, int, int, float]:
    """Return tau for #{M < -tau} / (#{M > tau} or 1) <= alpha."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    for tau in _threshold_candidates(M):
        r_plus = int((M > tau).sum())
        r_minus = int((M < -tau).sum())
        fdp_hat = r_minus / max(r_plus, 1)
        if fdp_hat <= alpha:
            return float(tau), r_plus, r_minus, float(fdp_hat)
    return np.inf, 0, 0, np.inf


def _set_requires_grad(model: nn.Module, flag: bool) -> list[bool]:
    old = []
    for param in model.parameters():
        old.append(param.requires_grad)
        param.requires_grad_(flag)
    return old


def _restore_requires_grad(model: nn.Module, old: list[bool]) -> None:
    for param, flag in zip(model.parameters(), old):
        param.requires_grad_(flag)


def _scalarize_output(
    output: torch.Tensor,
    target_index: Optional[int] = None,
    multi_output: Literal["logsumexp", "mean", "max"] = "logsumexp",
) -> torch.Tensor:
    """Convert model output to one scalar score per sample for input gradients."""

    if output.ndim == 1:
        return output

    flat = output.reshape(output.shape[0], -1)
    if flat.shape[1] == 1:
        return flat[:, 0]
    if target_index is not None:
        return flat[:, int(target_index)]
    if multi_output == "logsumexp":
        return torch.logsumexp(flat, dim=1)
    if multi_output == "mean":
        return flat.mean(dim=1)
    if multi_output == "max":
        return flat.max(dim=1).values
    raise ValueError(f"Unknown multi_output mode: {multi_output}")


def _contains_rnn(model: nn.Module) -> bool:
    return any(isinstance(module, (nn.RNN, nn.GRU, nn.LSTM, nn.RNNBase)) for module in model.modules())


def input_gradient_scores(
    model: nn.Module,
    X: torch.Tensor,
    *,
    feature_axis: int = -1,
    batch_size: int = 2048,
    score_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    target_index: Optional[int] = None,
    multi_output: Literal["logsumexp", "mean", "max"] = "logsumexp",
    force_train_for_cuda_rnn: bool = True,
) -> Array:
    """Sum d f(x_i) / d x_ij over all non-feature axes.

    For ``X`` shaped ``[N, T, F]`` and ``feature_axis=-1``, this returns ``[F]``.
    """

    axis = _positive_axis(X.ndim, feature_axis)
    if axis == 0:
        raise ValueError("feature_axis cannot be the sample/batch axis")

    was_training = model.training
    use_train_mode = force_train_for_cuda_rnn and X.is_cuda and _contains_rnn(model)
    model.train(use_train_mode)
    if not use_train_mode:
        model.eval()

    old_requires_grad = _set_requires_grad(model, False)
    scores = torch.zeros(X.shape[axis], dtype=X.dtype, device=X.device)
    reduce_dims = tuple(i for i in range(X.ndim) if i != axis)

    try:
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            Xb = X[start:end].detach().clone().requires_grad_(True)
            if score_fn is None:
                scalar = _scalarize_output(
                    model(Xb),
                    target_index=target_index,
                    multi_output=multi_output,
                )
            else:
                scalar = score_fn(model, Xb).reshape(Xb.shape[0])
            grads = torch.autograd.grad(
                scalar.sum(),
                Xb,
                retain_graph=False,
                create_graph=False,
            )[0]
            scores += grads.sum(dim=reduce_dims)
    finally:
        _restore_requires_grad(model, old_requires_grad)
        model.train(was_training)

    return scores.detach().cpu().numpy()


def _prepare_target(y: Array, loss: LossName, device: torch.device) -> torch.Tensor:
    if loss == "ce":
        return torch.as_tensor(y, dtype=torch.long, device=device)
    return torch.as_tensor(y, dtype=torch.float32, device=device)


def _loss_value(pred: torch.Tensor, target: torch.Tensor, loss: LossName) -> torch.Tensor:
    if loss == "mse":
        return 0.5 * F.mse_loss(pred.reshape_as(target.float()), target.float())
    if loss == "binary":
        return F.binary_cross_entropy_with_logits(pred.reshape_as(target.float()), target.float())
    if loss == "ce":
        return F.cross_entropy(pred, target.long())
    raise ValueError(f"Unknown loss: {loss}")


def _make_optimizer(
    model: nn.Module,
    optimizer: OptimizerName,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _train_one_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    loss: LossName,
    batch_size: int,
    generator: torch.Generator,
    clip_grad_norm: Optional[float],
) -> float:
    n = X.shape[0]
    idx = torch.randint(
        low=0,
        high=n,
        size=(min(batch_size, n),),
        generator=generator,
        device=X.device,
    )
    Xb = X.index_select(0, idx)
    yb = y.index_select(0, idx)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = _loss_value(model(Xb), yb, loss)
    loss_value.backward()
    if clip_grad_norm is not None and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    return float(loss_value.detach().cpu())


def fit_mirror_feature_selector(
    X: Array,
    y: Array,
    build_model: Callable[[tuple[int, ...]], nn.Module],
    *,
    alpha: float = 0.1,
    steps: int = 1000,
    loss: LossName = "mse",
    feature_axis: int = -1,
    batch_size: int = 128,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    optimizer: OptimizerName = "sgd",
    psi: PsiName = "min",
    compute_every: Optional[int] = None,
    xi_batch_size: int = 2048,
    gradient_eval: GradientEvalName = "train_split",
    standardize: bool = True,
    seed_split: int = 2025,
    seed_model1: int = 11,
    seed_model2: int = 22,
    device: Optional[torch.device | str] = None,
    score_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    target_index: Optional[int] = None,
    multi_output: Literal["logsumexp", "mean", "max"] = "logsumexp",
    clip_grad_norm: Optional[float] = None,
    return_models: bool = False,
) -> MirrorFeatureSelectionResult:
    """Run mirror-statistic feature selection with an arbitrary PyTorch model.

    Args:
        X: Input array. Use ``[N, F]`` for tabular data or ``[N, T, F]`` for
            sequence data.
        y: Targets.
        build_model: Callable receiving ``input_shape=X.shape[1:]`` and
            returning an ``nn.Module``.
        gradient_eval: ``"train_split"`` is the paper-style default.  Use
            ``"cross_fit"`` to match the swapped-gradient variant in the
            original proposed code.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    axis = _positive_axis(X.ndim, feature_axis)
    if axis == 0:
        raise ValueError("feature_axis cannot be the sample/batch axis")

    if standardize:
        Xn, mu, sigma = standardize_by_feature(X, feature_axis=axis)
    else:
        Xn = X.astype(np.float32)
        n_features = X.shape[axis]
        mu = np.zeros(n_features, dtype=np.float32)
        sigma = np.ones(n_features, dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    rng = np.random.default_rng(seed_split)
    perm = rng.permutation(X.shape[0])
    cut = X.shape[0] // 2
    idx1, idx2 = perm[:cut], perm[cut:]
    if len(idx1) == 0 or len(idx2) == 0:
        raise ValueError("Need at least two samples to create two splits")

    X1 = torch.as_tensor(Xn[idx1], dtype=torch.float32, device=device)
    X2 = torch.as_tensor(Xn[idx2], dtype=torch.float32, device=device)
    y1 = _prepare_target(y[idx1], loss, device)
    y2 = _prepare_target(y[idx2], loss, device)

    torch.manual_seed(seed_model1)
    model1 = build_model(tuple(X.shape[1:])).to(device)
    torch.manual_seed(seed_model2)
    model2 = build_model(tuple(X.shape[1:])).to(device)

    opt1 = _make_optimizer(model1, optimizer, lr, weight_decay)
    opt2 = _make_optimizer(model2, optimizer, lr, weight_decay)
    gen1 = _make_generator(device, seed_model1 + 1)
    gen2 = _make_generator(device, seed_model2 + 1)

    if compute_every is None:
        compute_every = steps

    history: list[MirrorPathPoint] = []
    last_loss1 = last_loss2 = float("nan")
    final_xi1 = final_xi2 = final_M = None
    final_tau = final_fdp = np.inf
    final_r_plus = final_r_minus = 0
    final_selected = np.array([], dtype=int)

    for step in range(1, steps + 1):
        last_loss1 = _train_one_step(
            model1,
            X1,
            y1,
            opt1,
            loss=loss,
            batch_size=batch_size,
            generator=gen1,
            clip_grad_norm=clip_grad_norm,
        )
        last_loss2 = _train_one_step(
            model2,
            X2,
            y2,
            opt2,
            loss=loss,
            batch_size=batch_size,
            generator=gen2,
            clip_grad_norm=clip_grad_norm,
        )

        if step % compute_every != 0 and step != steps:
            continue

        if gradient_eval == "train_split":
            xi1_std = input_gradient_scores(
                model1,
                X1,
                feature_axis=axis,
                batch_size=xi_batch_size,
                score_fn=score_fn,
                target_index=target_index,
                multi_output=multi_output,
            )
            xi2_std = input_gradient_scores(
                model2,
                X2,
                feature_axis=axis,
                batch_size=xi_batch_size,
                score_fn=score_fn,
                target_index=target_index,
                multi_output=multi_output,
            )
        elif gradient_eval == "cross_fit":
            xi1_std = input_gradient_scores(
                model2,
                X1,
                feature_axis=axis,
                batch_size=xi_batch_size,
                score_fn=score_fn,
                target_index=target_index,
                multi_output=multi_output,
            )
            xi2_std = input_gradient_scores(
                model1,
                X2,
                feature_axis=axis,
                batch_size=xi_batch_size,
                score_fn=score_fn,
                target_index=target_index,
                multi_output=multi_output,
            )
        else:
            raise ValueError("gradient_eval must be 'train_split' or 'cross_fit'")

        xi1 = xi1_std / sigma
        xi2 = xi2_std / sigma
        M = mirror_statistic(xi1, xi2, psi=psi)
        tau, r_plus, r_minus, fdp_hat = mirror_fdr_threshold(M, alpha=alpha)
        selected = np.where(M > tau)[0] if np.isfinite(tau) else np.array([], dtype=int)

        final_xi1, final_xi2, final_M = xi1, xi2, M
        final_tau, final_fdp = tau, fdp_hat
        final_r_plus, final_r_minus = r_plus, r_minus
        final_selected = selected
        history.append(
            MirrorPathPoint(
                step=step,
                tau=float(tau),
                r_plus=int(r_plus),
                r_minus=int(r_minus),
                fdp_hat=float(fdp_hat),
                selected_size=int(selected.size),
                loss1=float(last_loss1),
                loss2=float(last_loss2),
            )
        )

    assert final_xi1 is not None and final_xi2 is not None and final_M is not None
    return MirrorFeatureSelectionResult(
        selected=final_selected,
        tau=float(final_tau),
        mirror_scores=final_M,
        xi1=final_xi1,
        xi2=final_xi2,
        fdp_hat=float(final_fdp),
        r_plus=int(final_r_plus),
        r_minus=int(final_r_minus),
        history=history,
        model1=model1 if return_models else None,
        model2=model2 if return_models else None,
        mu=mu,
        sigma=sigma,
    )


class MLPRegressor(nn.Module):
    """Small tabular baseline model for quick experiments."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: tuple[int, ...] = (64, 64),
        output_dim: int = 1,
    ):
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("MLPRegressor expects tabular input_shape=(features,)")
        dims = (input_shape[0],) + hidden_dims + (output_dim,)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRURegressor(nn.Module):
    """Reference GRU for sequence input ``[batch, time, features]``."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("GRURegressor expects input_shape=(time, features)")
        _, n_features = input_shape
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        if self.rnn.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            last = h[-1]
        return self.head(last)


class LSTMRegressor(nn.Module):
    """Reference LSTM for sequence input ``[batch, time, features]``."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("LSTMRegressor expects input_shape=(time, features)")
        _, n_features = input_shape
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.rnn(x)
        if self.rnn.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            last = h[-1]
        return self.head(last)


class TransformerRegressor(nn.Module):
    """Reference Transformer encoder for ``[batch, time, features]`` inputs."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("TransformerRegressor expects input_shape=(time, features)")
        time_steps, n_features = input_shape
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, time_steps, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


__all__ = [
    "MirrorFeatureSelectionResult",
    "MirrorPathPoint",
    "fit_mirror_feature_selector",
    "input_gradient_scores",
    "mirror_fdr_threshold",
    "mirror_statistic",
    "psi_aggregate",
    "standardize_by_feature",
    "MLPRegressor",
    "GRURegressor",
    "LSTMRegressor",
    "TransformerRegressor",
]
