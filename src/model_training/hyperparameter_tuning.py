import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.model_training.model_factory import build_sequence_model


def _as_1d_float_tensor(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    return y.to(device=device, dtype=torch.float32).reshape(-1)


def _time_series_cv_splits(
    n_samples: int,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    min_train_size: int = 64,
    gap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Expanding-window CV splits for sequence samples.
    Returns list of tuples: (train_start, train_end, val_start, val_end)
    """
    n_folds = max(1, int(n_folds))
    val_ratio = float(np.clip(val_ratio, 0.05, 0.4))
    min_train_size = max(8, int(min_train_size))
    gap = max(0, int(gap))

    effective_n = int(n_samples) - int(gap)
    if effective_n <= min_train_size:
        raise ValueError(
            "Not enough samples for time-series CV after enforcing gap: "
            f"n_samples={n_samples}, gap={gap}, min_train_size={min_train_size}"
        )

    val_size_raw = int(math.ceil(n_samples * val_ratio))
    max_val_allowed = max(1, effective_n - min_train_size)
    val_size = max(1, min(max(8, val_size_raw), max_val_allowed))
    splits: List[Tuple[int, int, int, int]] = []

    for fold_idx in range(n_folds):
        train_end = effective_n - (n_folds - fold_idx) * val_size
        val_start = train_end + gap
        val_end = val_start + val_size
        if train_end < min_train_size:
            continue
        if val_end > n_samples:
            continue
        splits.append((0, train_end, val_start, val_end))

    if len(splits) == 0:
        fallback_val_raw = int(math.ceil(n_samples * 0.2))
        fallback_val = max(1, min(max(8, fallback_val_raw), max_val_allowed))
        train_end = effective_n - fallback_val
        train_end = max(min_train_size, train_end)
        val_start = train_end + gap
        val_end = n_samples
        if val_end - val_start < 1:
            raise ValueError("Not enough samples to build CV splits.")
        splits = [(0, train_end, val_start, val_end)]

    return splits


def _fit_one_split_with_early_stopping(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model_type: str,
    model_kwargs: Dict,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int = 200,
    patience: int = 12,
    random_state: int = 42,
) -> float:
    """
    Train one model on one split and return best validation MSE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(random_state))

    X_train = X_train.to(device=device, dtype=torch.float32)
    X_val = X_val.to(device=device, dtype=torch.float32)
    y_train = _as_1d_float_tensor(y_train, device=device)
    y_val = _as_1d_float_tensor(y_val, device=device)

    input_dim = int(X_train.shape[2])
    model = build_sequence_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        output_dim=1,
        dropout=float(dropout),
        **model_kwargs,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    max_epochs = max(1, int(max_epochs))
    patience = max(1, int(patience))
    best_val = float("inf")
    stale = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train).reshape(-1)
        loss = criterion(pred_train, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val).reshape(-1)
            val_loss = float(criterion(pred_val, y_val).item())

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(best_val)


# Deprecated / currently unused:
# def hyperparameter_optimize(
#     params,
#     param_names,
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     model_type="lstm",
#     model_kwargs=None,
#     hidden_dim: int = 32,
#     num_layers: int = 2,
#     max_epochs: int = 200,
#     patience: int = 12,
# ):
#     """
#     Backward-compatible objective helper on a single split.
#     """
#     params = dict(zip(param_names, params))
#     model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
#     return _fit_one_split_with_early_stopping(
#         X_train=X_train,
#         y_train=y_train,
#         X_val=X_val,
#         y_val=y_val,
#         model_type=model_type,
#         model_kwargs=model_kwargs,
#         hidden_dim=int(hidden_dim),
#         num_layers=int(num_layers),
#         dropout=float(params["dropout"]),
#         lr=float(params["lr"]),
#         weight_decay=float(params["weight_decay"]),
#         max_epochs=max_epochs,
#         patience=patience,
#     )


def find_optimal_params(
    X_train_torch: torch.Tensor,
    y_train_torch: torch.Tensor,
    X_val_torch: torch.Tensor = None,
    y_val_torch: torch.Tensor = None,
    model_type: str = "lstm",
    model_kwargs: Dict = None,
    fixed_hidden_dim: int = 32,
    fixed_num_layers: int = 2,
    n_trials: int = 24,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    min_train_size: int = 64,
    max_epochs: int = 200,
    patience: int = 12,
    lr_min: float = 1e-4,
    lr_max: float = 5e-3,
    weight_decay_min: float = 1e-3,
    weight_decay_max: float = 3e-1,
    dropout_min: float = 0.0,
    dropout_max: float = 0.2,
    wd_target: float = 2e-2,
    lambda_wd: float = 5e-4,
    lambda_dropout: float = 1e-3,
    cv_gap: int = 0,
    random_state: int = 42,
):
    """
    Tune only lr / weight_decay / dropout with time-series CV.

    Notes:
      - X_val_torch / y_val_torch are accepted for backward compatibility and ignored.
      - Hidden_dim and num_layers stay fixed.
      - Single-GPU friendly: trials run sequentially.
    """
    del X_val_torch, y_val_torch

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_type = str(model_type).lower()
    fixed_hidden_dim = int(fixed_hidden_dim)
    fixed_num_layers = int(fixed_num_layers)

    if model_type == "transformer":
        nhead = int(model_kwargs.get("nhead", 4))
        if nhead <= 0:
            nhead = 1
        if fixed_hidden_dim % nhead != 0:
            fixed_hidden_dim = int(math.ceil(fixed_hidden_dim / nhead) * nhead)

    X_all = X_train_torch.detach().float()
    y_all = y_train_torch.detach().float().reshape(-1)
    n_samples = int(X_all.shape[0])
    if n_samples < 20:
        raise ValueError(f"Too few sequence samples for tuning: n={n_samples}")

    splits = _time_series_cv_splits(
        n_samples=n_samples,
        n_folds=int(n_folds),
        val_ratio=float(val_ratio),
        min_train_size=int(min_train_size),
        gap=int(cv_gap),
    )

    rng = np.random.default_rng(int(random_state))
    n_trials = max(4, int(n_trials))
    lr_min = float(max(1e-6, lr_min))
    lr_max = float(max(lr_min * 1.01, lr_max))
    weight_decay_min = float(max(1e-8, weight_decay_min))
    weight_decay_max = float(max(weight_decay_min * 1.01, weight_decay_max))
    dropout_min = float(np.clip(dropout_min, 0.0, 0.8))
    dropout_max = float(np.clip(dropout_max, dropout_min + 1e-4, 0.9))
    wd_target = float(max(weight_decay_min, wd_target))

    # Include one deterministic anchor trial.
    candidates = [
        {
            "lr": float(np.sqrt(lr_min * lr_max)),
            "weight_decay": float(min(weight_decay_max, max(weight_decay_min, wd_target))),
            "dropout": float(min(dropout_max, max(dropout_min, 0.05))),
        }
    ]
    for _ in range(n_trials - 1):
        candidates.append(
            {
                "lr": float(np.exp(rng.uniform(np.log(lr_min), np.log(lr_max)))),
                "weight_decay": float(np.exp(rng.uniform(np.log(weight_decay_min), np.log(weight_decay_max)))),
                "dropout": float(rng.uniform(dropout_min, dropout_max)),
            }
        )

    best = None
    trial_records = []
    for trial_idx, params in enumerate(candidates):
        fold_losses = []
        for tr0, tr1, va0, va1 in splits:
            val_loss = _fit_one_split_with_early_stopping(
                X_train=X_all[tr0:tr1],
                y_train=y_all[tr0:tr1],
                X_val=X_all[va0:va1],
                y_val=y_all[va0:va1],
                model_type=model_type,
                model_kwargs=model_kwargs,
                hidden_dim=fixed_hidden_dim,
                num_layers=fixed_num_layers,
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                max_epochs=max_epochs,
                patience=patience,
                random_state=int(random_state),
            )
            fold_losses.append(float(val_loss))

        mean_cv_mse = float(np.mean(fold_losses))
        wd_penalty = float(lambda_wd) * max(0.0, wd_target - float(params["weight_decay"])) ** 2
        dropout_penalty = float(lambda_dropout) * (float(params["dropout"]) ** 2)
        objective = mean_cv_mse + wd_penalty + dropout_penalty

        trial = {
            "trial_index": int(trial_idx),
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "dropout": float(params["dropout"]),
            "hidden_dim": int(fixed_hidden_dim),
            "num_layers": int(fixed_num_layers),
            "cv_mse": mean_cv_mse,
            "fold_mse": [float(v) for v in fold_losses],
            "wd_penalty": float(wd_penalty),
            "dropout_penalty": float(dropout_penalty),
            "objective": float(objective),
        }
        trial_records.append(trial)
        if (best is None) or (trial["objective"] < best["objective"]):
            best = trial

    best["search_param_names"] = ["lr", "weight_decay", "dropout"]
    best["search_space"] = {
        "lr": [float(lr_min), float(lr_max)],
        "weight_decay": [float(weight_decay_min), float(weight_decay_max)],
        "dropout": [float(dropout_min), float(dropout_max)],
    }
    best["search_config"] = {
        "n_trials": int(n_trials),
        "n_folds": int(n_folds),
        "val_ratio": float(val_ratio),
        "cv_gap": int(cv_gap),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "wd_target": float(wd_target),
        "lambda_wd": float(lambda_wd),
        "lambda_dropout": float(lambda_dropout),
    }
    best["trial_results"] = trial_records
    return best
