#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import math
import os
import random
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_generation_sample import simulate_deeplink_t_dataset
from src.model_training.hyperparameter_tuning import find_optimal_params
from src.model_training.model_factory import build_sequence_model as build_shared_sequence_model


# =========================
# Main parameters (edit here)
# =========================
SEED = 2
NUM_SIMULATIONS = 50

# Data size
N_SAMPLES = 2500
N_RELEVANT = 15
N_IRRELEVANT = 285
LOOKBACK = 5

# DeepLINK-T feature generation
FACTOR_MODEL = "logistic"  # linear | logistic
LATENT_DIM = 10

# y generation
Y_MODE = "lag_weighted"  # lag_weighted | current_x
Y_LINK_FUNCTION = "tanh"  # linear | nonlinear/sinexp | tanh | sin
SIGNAL_AMPLITUDE = 10.0
LABEL_NOISE = 1.0
LAG_DECAY = 0.9
LAG_WEIGHT_NORM = "l2"  # l1 | l2

# Shared model shape / optimizer controls
MODEL_TYPE = "lstm"  # rnn | gru | lstm | transformer
HIDDEN_DIM_MULTIPLIER = 20.0
NUM_LAYERS = 2
DROPOUT = 0.0
OPTIM_LR = 0.001
OPTIM_WEIGHT_DECAY = 0.01
TRANSFORMER_NHEAD = 2
TRANSFORMER_DIM_FEEDFORWARD = None
CATNET_TRAIN_EPOCHS = 1000

# Same automatic tuning controls as the non-baseline sequence simulation.
AUTO_TUNE_HPARAMS = True
TUNE_N_TRIALS = 50
TUNE_N_FOLDS = 3
TUNE_VAL_RATIO = 0.2
TUNE_MAX_EPOCHS = 1000
TUNE_PATIENCE = 12
TUNE_WEIGHT_DECAY_TARGET = 2e-2
TUNE_LAMBDA_WD = 5e-4
TUNE_LAMBDA_DROPOUT = 1e-3

# Shared evaluation controls
FDR_Q = 0.1
STANDARDIZE_INPUT = True
SHOW_PROGRESS = False

# Baselines to run
RUN_AFS = True
RUN_PROX_GL = True
RUN_PROX_AGL = True
RUN_MIRROR = True

# AFS controls. Selection is the cleaned AFS module's native rule:
# average attention score >= mean average attention score.
AFS_TRAIN_EPOCHS = CATNET_TRAIN_EPOCHS
AFS_BATCH_SIZE = None  # None means full-batch, matching the current simulation style
AFS_EXTRACT_SIZE = 32
AFS_CONTEXT_MODE = "mean"  # mean | last | flatten
AFS_SELECTION_MODE = "mean_plus_std"  # mean | mean_plus_std
AFS_STD_MULTIPLIER = 1.0

# Prox / ALGNet controls. Lambda is chosen by validation loss, then feature
# selection is the original nonzero group-norm rule.
PROX_TRAIN_EPOCHS = CATNET_TRAIN_EPOCHS
PROX_BATCH_SIZE = None  # None means full-batch
PROX_LAMBDA_GRID = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], dtype=float)
PROX_VALIDATION_RATIO = 0.2
PROX_TIME_ORDERED_VALIDATION = True
PROX_ZERO_THRESHOLD = 1e-8
PROX_ADAPTIVE_GAMMA = 2.0

# Mirror controls. Selection is the mirror module's native FDR rule.
MIRROR_STEPS = 1000
MIRROR_BATCH_SIZE = 128
MIRROR_OPTIMIZER = "sgd"  # sgd | adam | adamw
MIRROR_PSI = "min"  # min | mean | geomean | harmmean
MIRROR_COMPUTE_EVERY = None
MIRROR_XI_BATCH_SIZE = 2048
MIRROR_GRADIENT_EVAL = "train_split"  # train_split | cross_fit
MIRROR_CLIP_GRAD_NORM = None

# Output controls
RESULTS_DIR = Path("baseline_simulation_runs")
RESULT_CSV_PATH = None
SAVE_PER_SIM_DETAILS = False


ROOT = Path(__file__).resolve().parent
AFS_PATH = ROOT / "baselines" / "AAAI-2019-AFS-master" / "afs_feature_selection.py"
PROX_PATH = ROOT / "baselines" / "alg-net-main" / "feature_selection_prox.py"
MIRROR_PATH = ROOT / "baselines" / "deep-feature-selection-main" / "proposed" / "feature_selection_mirror.py"


def _load_module(module_name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.9 dataclasses need the module registered while class decorators run.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FlattenMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(-1)
        target_flat = target.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        if pred_flat.numel() != target_flat.numel():
            raise ValueError(
                f"Prediction/target size mismatch: pred={tuple(pred_flat.shape)}, target={tuple(target_flat.shape)}"
            )
        return torch.mean((pred_flat - target_flat).pow(2))


@dataclass
class BaselineResult:
    method: str
    selection_policy: str
    scores: Dict[str, float]
    selected_features: Dict[str, float]
    threshold: float | None = None
    train_loss_final: float | None = None
    extra: Dict[str, Any] | None = None


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_pid{os.getpid()}_{uuid4().hex[:8]}"


def _ensure_non_conflicting_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix or ".csv"
    stem = path.stem
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _create_run_output_path(run_id: str) -> Path:
    if RESULT_CSV_PATH is not None:
        out_path = Path(RESULT_CSV_PATH)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return _ensure_non_conflicting_path(out_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _ensure_non_conflicting_path(RESULTS_DIR / f"baseline_simulation_results_{run_id}.csv")


def _write_header(path: Path, settings: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Baseline Simulation Settings\n")
        for k, v in settings.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            f.write(f"# {k}: {v}\n")
        f.write("# ---\n")
        f.flush()
        os.fsync(f.fileno())


def _append_row(path: Path, row: Dict[str, Any], wrote_header: bool) -> bool:
    with path.open("a", encoding="utf-8") as f:
        pd.DataFrame([row]).to_csv(f, index=False, header=not wrote_header)
        f.flush()
        os.fsync(f.fileno())
    return True


def _set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def _maybe_suppress_output(enable_output: bool):
    if bool(enable_output):
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def _standardize_feature_df(data: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    data_f = data.astype(np.float64)
    col_mean = data_f.mean(axis=0)
    col_std = data_f.std(axis=0, ddof=0)
    col_std = col_std.mask(col_std <= eps, 1.0)
    return ((data_f - col_mean) / col_std).astype(np.float32)


def _build_sequence_tensors(
    data_df: pd.DataFrame,
    y: Iterable[float],
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for i in range(int(lookback), len(data_df)):
        rows.append(data_df.iloc[i - int(lookback):i].values)
    X_np = np.asarray(rows, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1)[int(lookback):]
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"Sequence length mismatch: X_seq={X_np.shape[0]}, y_seq={y_np.shape[0]}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


def _make_hidden_dim(input_dim: int, model_type: str) -> int:
    raw = max(8, int(float(HIDDEN_DIM_MULTIPLIER) * np.log(max(2, int(input_dim)))))
    if str(model_type).lower() != "transformer":
        return raw
    nhead = max(1, int(TRANSFORMER_NHEAD))
    return int(math.ceil(raw / nhead) * nhead)


def _build_sequence_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    model_kwargs = _model_kwargs(model_type)
    return build_shared_sequence_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout,
        **model_kwargs,
    )


def _model_kwargs(model_type: str) -> Dict[str, Any]:
    if str(model_type).lower() != "transformer":
        return {}
    kwargs: Dict[str, Any] = {"nhead": int(TRANSFORMER_NHEAD)}
    if TRANSFORMER_DIM_FEEDFORWARD is not None:
        kwargs["dim_feedforward"] = int(TRANSFORMER_DIM_FEEDFORWARD)
    return kwargs


def _tune_hparams(
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    random_state: int,
) -> Dict[str, Any] | None:
    if not bool(AUTO_TUNE_HPARAMS):
        return None
    try:
        return find_optimal_params(
            X_train_torch=X_seq,
            y_train_torch=y_seq,
            model_type=model_type,
            model_kwargs=_model_kwargs(model_type),
            fixed_hidden_dim=int(hidden_dim),
            fixed_num_layers=int(num_layers),
            n_trials=int(TUNE_N_TRIALS),
            n_folds=int(TUNE_N_FOLDS),
            val_ratio=float(TUNE_VAL_RATIO),
            cv_gap=int(LOOKBACK),
            max_epochs=int(TUNE_MAX_EPOCHS),
            patience=int(TUNE_PATIENCE),
            wd_target=float(TUNE_WEIGHT_DECAY_TARGET),
            lambda_wd=float(TUNE_LAMBDA_WD),
            lambda_dropout=float(TUNE_LAMBDA_DROPOUT),
            random_state=int(random_state),
        )
    except Exception:
        return None


def _resolve_optim_params(tuned: Dict[str, Any] | None) -> Tuple[float, float, float]:
    if tuned is None:
        return float(OPTIM_LR), float(OPTIM_WEIGHT_DECAY), float(DROPOUT)
    return float(tuned["lr"]), float(tuned["weight_decay"]), float(tuned["dropout"])


def _make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int | None,
    shuffle: bool,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    n = int(X.shape[0])
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= n:
        return [(X, y)]
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    if bool(shuffle):
        rng.shuffle(idx)
    out = []
    bs = int(batch_size)
    for start in range(0, n, bs):
        batch_idx = torch.as_tensor(idx[start:start + bs], dtype=torch.long)
        out.append((X.index_select(0, batch_idx), y.index_select(0, batch_idx)))
    return out


def _eval_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    loss_fn = FlattenMSELoss()
    with torch.no_grad():
        pred = model(X.to(device=device, dtype=torch.float32))
        loss = loss_fn(pred, y.to(device=device, dtype=torch.float32))
    return float(loss.detach().cpu())


def _train_plain(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int | None,
    seed: int,
    device: torch.device,
) -> float:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = FlattenMSELoss()
    final_loss = float("nan")
    for epoch in range(max(1, int(epochs))):
        model.train()
        loader = _make_loader(X, y, batch_size=batch_size, shuffle=True, seed=int(seed) + epoch)
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


def _split_train_val(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    validation_ratio: float,
    time_ordered: bool,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(X.shape[0])
    n_val = max(1, int(round(n * float(validation_ratio))))
    n_train = max(1, n - n_val)
    if n_train >= n:
        n_train = n - 1
    if bool(time_ordered):
        train_idx = torch.arange(0, n_train, dtype=torch.long)
        val_idx = torch.arange(n_train, n, dtype=torch.long)
    else:
        rng = np.random.default_rng(int(seed))
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
    return (
        X.index_select(0, train_idx),
        y.index_select(0, train_idx),
        X.index_select(0, val_idx),
        y.index_select(0, val_idx),
    )


def _selection_metrics(selected_features: Dict[str, float], relevant_set: set, irrelevant_set: set) -> Dict[str, float]:
    selected_names = set(selected_features.keys())
    n_total = len(selected_names)
    n_rel = len(selected_names & relevant_set)
    n_irr = len(selected_names & irrelevant_set)
    power = n_rel / len(relevant_set) if len(relevant_set) > 0 else 0.0
    fdr = n_irr / n_total if n_total > 0 else 0.0
    return {
        "selected_total_features": float(n_total),
        "selected_relevant_features": float(n_rel),
        "selected_irrelevant_features": float(n_irr),
        "power": float(power),
        "fdr": float(fdr),
    }


def _score_summary(scores: Dict[str, float]) -> Dict[str, float]:
    arr = np.asarray(list(scores.values()), dtype=np.float64)
    if arr.size == 0:
        return {"score_min": np.nan, "score_max": np.nan, "score_mean": np.nan, "score_std": np.nan}
    return {
        "score_min": float(np.min(arr)),
        "score_max": float(np.max(arr)),
        "score_mean": float(np.mean(arr)),
        "score_std": float(np.std(arr)),
    }


def _indices_to_selected(indices: Sequence[int], feature_names: Sequence[str], scores: Dict[str, float]) -> Dict[str, float]:
    return {str(feature_names[int(i)]): float(scores[str(feature_names[int(i)])]) for i in indices}


def _save_feature_details(
    run_dir: Path,
    result_key: str,
    result: BaselineResult,
    relevant_set: set,
) -> None:
    rows = []
    selected = set(result.selected_features.keys())
    sorted_scores = sorted(result.scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    rank = {name: i + 1 for i, (name, _) in enumerate(sorted_scores)}
    for feature, score in result.scores.items():
        rows.append(
            {
                "result_key": result_key,
                "method": result.method,
                "selection_policy": result.selection_policy,
                "feature": str(feature),
                "is_relevant": str(feature) in relevant_set,
                "score": float(score),
                "selected": str(feature) in selected,
                "rank_descending": int(rank[str(feature)]),
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / f"{result_key}_feature_details.csv", index=False)


def _selector_for_model(prox_module, model: nn.Module, model_type: str):
    mt = str(model_type).lower()
    if mt == "rnn":
        return prox_module.FeatureGroupProx.from_rnn(model.rnn)
    if mt == "gru":
        return prox_module.FeatureGroupProx.from_rnn(model.gru)
    if mt == "lstm":
        return prox_module.FeatureGroupProx.from_rnn(model.lstm)
    if mt == "transformer":
        return prox_module.FeatureGroupProx.from_linear(model.input_proj)
    raise ValueError(f"Unsupported MODEL_TYPE for prox selector: {model_type}")


def _train_prox_once(
    prox_module,
    model: nn.Module,
    selector,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lam: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int | None,
    seed: int,
    device: torch.device,
) -> float:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = FlattenMSELoss()
    losses: List[float] = []
    for epoch in range(max(1, int(epochs))):
        loader = _make_loader(X, y, batch_size=batch_size, shuffle=True, seed=int(seed) + epoch)
        losses.extend(
            prox_module.train_with_prox(
                model=model,
                loader=loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                selector=selector,
                lam=float(lam),
                device=device,
                epochs=1,
            )
        )
    return float(losses[-1]) if losses else float("nan")


def _choose_prox_lambda(
    prox_module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    adaptive_weights: torch.Tensor | None,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[float, float, List[Dict[str, float]]]:
    X_train, y_train, X_val, y_val = _split_train_val(
        X,
        y,
        validation_ratio=float(PROX_VALIDATION_RATIO),
        time_ordered=bool(PROX_TIME_ORDERED_VALIDATION),
        seed=int(seed),
    )
    best_lam = float(PROX_LAMBDA_GRID[0])
    best_val = float("inf")
    logs: List[Dict[str, float]] = []
    for lam in np.asarray(PROX_LAMBDA_GRID, dtype=float):
        _set_all_seeds(int(seed) + int(abs(np.log10(max(float(lam), 1e-12))) * 1000))
        model = _build_sequence_model(MODEL_TYPE, int(X.shape[-1]), hidden_dim, int(NUM_LAYERS), 1, float(dropout))
        selector = _selector_for_model(prox_module, model, MODEL_TYPE)
        if adaptive_weights is not None:
            selector.adaptive_weights = adaptive_weights.detach().clone().to(selector.device)
        train_loss = _train_prox_once(
            prox_module,
            model,
            selector,
            X_train,
            y_train,
            lam=float(lam),
            epochs=int(PROX_TRAIN_EPOCHS),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=PROX_BATCH_SIZE,
            seed=int(seed),
            device=device,
        )
        val_loss = _eval_loss(model, X_val, y_val, device)
        row = {"lambda": float(lam), "train_loss": float(train_loss), "val_loss": float(val_loss)}
        logs.append(row)
        if val_loss < best_val:
            best_val = float(val_loss)
            best_lam = float(lam)
    return best_lam, best_val, logs


def run_afs_baseline(
    afs_module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: Sequence[str],
    *,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> BaselineResult:
    _set_all_seeds(seed)
    backbone = _build_sequence_model(MODEL_TYPE, int(X.shape[-1]), hidden_dim, int(NUM_LAYERS), 1, float(dropout))
    if str(AFS_CONTEXT_MODE).lower() == "flatten":
        gate = afs_module.AFSFeatureGate(
            input_size=int(X.shape[-1]),
            extract_size=int(AFS_EXTRACT_SIZE),
            context_mode="flatten",
            sequence_length=int(X.shape[1]),
        )
    else:
        gate = afs_module.AFSFeatureGate(
            input_size=int(X.shape[-1]),
            extract_size=int(AFS_EXTRACT_SIZE),
            context_mode=str(AFS_CONTEXT_MODE),
        )
    model = afs_module.AFSFeatureSelectionModel(gate, backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = FlattenMSELoss()
    final_loss = float("nan")
    for epoch in range(max(1, int(AFS_TRAIN_EPOCHS))):
        loader = _make_loader(X, y, batch_size=AFS_BATCH_SIZE, shuffle=True, seed=int(seed) + epoch)
        final_loss = afs_module.train_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
    score_tensor = afs_module.feature_scores(
        model=model,
        loader=[(X, y)],
        device=device,
    )
    scores = {str(name): float(score_tensor[j].detach().cpu()) for j, name in enumerate(feature_names)}
    afs_mode = str(AFS_SELECTION_MODE).lower().strip()
    if afs_mode == "mean":
        threshold = float(score_tensor.mean().detach().cpu())
        selection_policy = "afs_mean_attention_threshold"
    elif afs_mode == "mean_plus_std":
        threshold = float((score_tensor.mean() + float(AFS_STD_MULTIPLIER) * score_tensor.std(unbiased=False)).detach().cpu())
        selection_policy = "afs_mean_plus_std_attention_threshold"
    else:
        raise ValueError("AFS_SELECTION_MODE must be one of: mean, mean_plus_std.")
    selected_idx = afs_module.selected_features(score_tensor, threshold=threshold)
    selected = _indices_to_selected(selected_idx.detach().cpu().numpy().tolist(), feature_names, scores)
    return BaselineResult(
        method="afs",
        selection_policy=selection_policy,
        scores=scores,
        selected_features=selected,
        threshold=threshold,
        train_loss_final=float(final_loss),
        extra={
            "afs_context_mode": str(AFS_CONTEXT_MODE),
            "afs_extract_size": int(AFS_EXTRACT_SIZE),
            "afs_selection_mode": str(AFS_SELECTION_MODE),
            "afs_std_multiplier": float(AFS_STD_MULTIPLIER),
            "used_lr": float(lr),
            "used_weight_decay": float(weight_decay),
            "used_dropout": float(dropout),
        },
    )


def run_prox_baseline(
    prox_module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: Sequence[str],
    *,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
    adaptive: bool,
) -> BaselineResult:
    _set_all_seeds(seed)
    adaptive_weights = None
    first_stage_lam = np.nan
    first_stage_val_loss = np.nan

    if bool(adaptive):
        first_stage_lam, first_stage_val_loss, _ = _choose_prox_lambda(
            prox_module,
            X,
            y,
            adaptive_weights=None,
            hidden_dim=hidden_dim,
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            seed=int(seed),
            device=device,
        )
        first_model = _build_sequence_model(MODEL_TYPE, int(X.shape[-1]), hidden_dim, int(NUM_LAYERS), 1, float(dropout))
        first_selector = _selector_for_model(prox_module, first_model, MODEL_TYPE)
        _train_prox_once(
            prox_module,
            first_model,
            first_selector,
            X,
            y,
            lam=float(first_stage_lam),
            epochs=int(PROX_TRAIN_EPOCHS),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=PROX_BATCH_SIZE,
            seed=int(seed) + 101,
            device=device,
        )
        base_norms = first_selector.group_norms().detach().cpu()
        adaptive_weights = 1.0 / base_norms.clamp_min(float(first_selector.eps)).pow(float(PROX_ADAPTIVE_GAMMA))

    chosen_lam, val_loss, lambda_logs = _choose_prox_lambda(
        prox_module,
        X,
        y,
        adaptive_weights=adaptive_weights,
        hidden_dim=hidden_dim,
        dropout=float(dropout),
        lr=float(lr),
        weight_decay=float(weight_decay),
        seed=int(seed) + (1009 if adaptive else 0),
        device=device,
    )

    _set_all_seeds(seed + (2003 if adaptive else 0))
    model = _build_sequence_model(MODEL_TYPE, int(X.shape[-1]), hidden_dim, int(NUM_LAYERS), 1, float(dropout))
    selector = _selector_for_model(prox_module, model, MODEL_TYPE)
    if adaptive_weights is not None:
        selector.adaptive_weights = adaptive_weights.detach().clone().to(selector.device)
    final_loss = _train_prox_once(
        prox_module,
        model,
        selector,
        X,
        y,
        lam=float(chosen_lam),
        epochs=int(PROX_TRAIN_EPOCHS),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=PROX_BATCH_SIZE,
        seed=int(seed) + 3001,
        device=device,
    )

    norms = selector.group_norms().detach().cpu()
    scores = {str(name): float(norms[j]) for j, name in enumerate(feature_names)}
    mask = selector.selected_features(threshold=float(PROX_ZERO_THRESHOLD)).detach().cpu().numpy().astype(bool)
    selected_idx = np.nonzero(mask)[0].tolist()
    selected = _indices_to_selected(selected_idx, feature_names, scores)
    method = "prox_agl" if adaptive else "prox_gl"
    return BaselineResult(
        method=method,
        selection_policy="prox_nonzero_group_norm",
        scores=scores,
        selected_features=selected,
        threshold=float(PROX_ZERO_THRESHOLD),
        train_loss_final=float(final_loss),
        extra={
            "prox_lambda": float(chosen_lam),
            "prox_validation_loss": float(val_loss),
            "prox_lambda_grid": np.asarray(PROX_LAMBDA_GRID, dtype=float).tolist(),
            "prox_lambda_log": lambda_logs,
            "prox_first_stage_lambda": float(first_stage_lam),
            "prox_first_stage_validation_loss": float(first_stage_val_loss),
            "prox_adaptive_gamma": float(PROX_ADAPTIVE_GAMMA) if adaptive else np.nan,
            "used_lr": float(lr),
            "used_weight_decay": float(weight_decay),
            "used_dropout": float(dropout),
        },
    )


def run_mirror_baseline(
    mirror_module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: Sequence[str],
    *,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> BaselineResult:
    def build_model(input_shape: tuple[int, ...]) -> nn.Module:
        return _build_sequence_model(
            MODEL_TYPE,
            int(input_shape[-1]),
            hidden_dim,
            int(NUM_LAYERS),
            1,
            float(dropout),
        )

    result = mirror_module.fit_mirror_feature_selector(
        X=X.detach().cpu().numpy().astype(np.float32),
        y=y.detach().cpu().numpy().astype(np.float32),
        build_model=build_model,
        alpha=float(FDR_Q),
        steps=int(MIRROR_STEPS),
        loss="mse",
        feature_axis=-1,
        batch_size=int(MIRROR_BATCH_SIZE),
        lr=float(lr),
        weight_decay=float(weight_decay),
        optimizer=str(MIRROR_OPTIMIZER),
        psi=str(MIRROR_PSI),
        compute_every=MIRROR_COMPUTE_EVERY,
        xi_batch_size=int(MIRROR_XI_BATCH_SIZE),
        gradient_eval=str(MIRROR_GRADIENT_EVAL),
        # Data are standardized once in this runner, exactly before sequence construction.
        standardize=False,
        seed_split=int(seed) + 1,
        seed_model1=int(seed) + 11,
        seed_model2=int(seed) + 22,
        device=device,
        clip_grad_norm=MIRROR_CLIP_GRAD_NORM,
        return_models=False,
    )
    scores = {str(name): float(result.mirror_scores[j]) for j, name in enumerate(feature_names)}
    selected = _indices_to_selected(np.asarray(result.selected, dtype=int).tolist(), feature_names, scores)
    loss_final = np.nan
    if len(result.history) > 0:
        loss_final = float(0.5 * (result.history[-1].loss1 + result.history[-1].loss2))
    return BaselineResult(
        method="mirror",
        selection_policy="mirror_native_fdr_threshold",
        scores=scores,
        selected_features=selected,
        threshold=float(result.tau),
        train_loss_final=float(loss_final),
        extra={
            "mirror_tau": float(result.tau),
            "mirror_fdp_hat": float(result.fdp_hat),
            "mirror_r_plus": int(result.r_plus),
            "mirror_r_minus": int(result.r_minus),
            "mirror_history_len": int(len(result.history)),
            "mirror_optimizer": str(MIRROR_OPTIMIZER),
            "mirror_psi": str(MIRROR_PSI),
            "mirror_gradient_eval": str(MIRROR_GRADIENT_EVAL),
            "used_lr": float(lr),
            "used_weight_decay": float(weight_decay),
            "used_dropout": float(dropout),
        },
    )


def _result_to_row(
    result: BaselineResult,
    *,
    run_id: str,
    result_key: str,
    result_csv_path: Path,
    simulation_id: int,
    seed: int,
    relevant_set: set,
    irrelevant_set: set,
) -> Dict[str, Any]:
    metrics = _selection_metrics(result.selected_features, relevant_set, irrelevant_set)
    row: Dict[str, Any] = {
        "run_id": run_id,
        "result_key": result_key,
        "result_file": str(result_csv_path.resolve()),
        "simulation_id": int(simulation_id),
        "seed": int(seed),
        "method": str(result.method),
        "selection_policy": str(result.selection_policy),
        "model_type": str(MODEL_TYPE),
        "fdr_q": float(FDR_Q),
        "threshold": np.nan if result.threshold is None else float(result.threshold),
        "fdr": float(metrics["fdr"]),
        "power": float(metrics["power"]),
        "n_samples": int(N_SAMPLES),
        "effective_train_samples": max(int(N_SAMPLES) - int(LOOKBACK), 0),
        "n_relevant_features": int(len(relevant_set)),
        "n_irrelevant_features": int(len(irrelevant_set)),
        "selected_total_features": int(metrics["selected_total_features"]),
        "selected_relevant_features": int(metrics["selected_relevant_features"]),
        "selected_irrelevant_features": int(metrics["selected_irrelevant_features"]),
        "train_loss_final": np.nan if result.train_loss_final is None else float(result.train_loss_final),
    }
    row.update(_score_summary(result.scores))
    method_specific_defaults = {
        "afs_context_mode": np.nan,
        "afs_extract_size": np.nan,
        "afs_selection_mode": np.nan,
        "afs_std_multiplier": np.nan,
        "used_lr": np.nan,
        "used_weight_decay": np.nan,
        "used_dropout": np.nan,
        "tuned_found": np.nan,
        "tuned_cv_mse": np.nan,
        "tuned_objective": np.nan,
        "prox_lambda": np.nan,
        "prox_validation_loss": np.nan,
        "prox_lambda_grid": np.nan,
        "prox_lambda_log": np.nan,
        "prox_first_stage_lambda": np.nan,
        "prox_first_stage_validation_loss": np.nan,
        "prox_adaptive_gamma": np.nan,
        "mirror_tau": np.nan,
        "mirror_fdp_hat": np.nan,
        "mirror_r_plus": np.nan,
        "mirror_r_minus": np.nan,
        "mirror_history_len": np.nan,
        "mirror_optimizer": np.nan,
        "mirror_psi": np.nan,
        "mirror_gradient_eval": np.nan,
    }
    row.update(method_specific_defaults)
    if result.extra:
        for key, value in result.extra.items():
            if isinstance(value, (dict, list, tuple)):
                row[key] = repr(value)
            else:
                row[key] = value
    return row


def _settings(run_id: str, result_csv_path: Path) -> Dict[str, Any]:
    return {
        "RUN_ID": run_id,
        "RESULT_CSV_PATH": str(result_csv_path.resolve()),
        "SEED": int(SEED),
        "NUM_SIMULATIONS": int(NUM_SIMULATIONS),
        "N_SAMPLES": int(N_SAMPLES),
        "N_RELEVANT": int(N_RELEVANT),
        "N_IRRELEVANT": int(N_IRRELEVANT),
        "LOOKBACK": int(LOOKBACK),
        "FACTOR_MODEL": str(FACTOR_MODEL),
        "LATENT_DIM": int(LATENT_DIM),
        "Y_MODE": str(Y_MODE),
        "Y_LINK_FUNCTION": str(Y_LINK_FUNCTION),
        "SIGNAL_AMPLITUDE": float(SIGNAL_AMPLITUDE),
        "LABEL_NOISE": float(LABEL_NOISE),
        "LAG_DECAY": float(LAG_DECAY),
        "LAG_WEIGHT_NORM": str(LAG_WEIGHT_NORM),
        "MODEL_TYPE": str(MODEL_TYPE),
        "HIDDEN_DIM_MULTIPLIER": float(HIDDEN_DIM_MULTIPLIER),
        "NUM_LAYERS": int(NUM_LAYERS),
        "DROPOUT": float(DROPOUT),
        "OPTIM_LR": float(OPTIM_LR),
        "OPTIM_WEIGHT_DECAY": float(OPTIM_WEIGHT_DECAY),
        "TRANSFORMER_NHEAD": int(TRANSFORMER_NHEAD),
        "TRANSFORMER_DIM_FEEDFORWARD": TRANSFORMER_DIM_FEEDFORWARD,
        "CATNET_TRAIN_EPOCHS": int(CATNET_TRAIN_EPOCHS),
        "AUTO_TUNE_HPARAMS": bool(AUTO_TUNE_HPARAMS),
        "TUNE_N_TRIALS": int(TUNE_N_TRIALS),
        "TUNE_N_FOLDS": int(TUNE_N_FOLDS),
        "TUNE_VAL_RATIO": float(TUNE_VAL_RATIO),
        "TUNE_MAX_EPOCHS": int(TUNE_MAX_EPOCHS),
        "TUNE_PATIENCE": int(TUNE_PATIENCE),
        "TUNE_WEIGHT_DECAY_TARGET": float(TUNE_WEIGHT_DECAY_TARGET),
        "TUNE_LAMBDA_WD": float(TUNE_LAMBDA_WD),
        "TUNE_LAMBDA_DROPOUT": float(TUNE_LAMBDA_DROPOUT),
        "FDR_Q": float(FDR_Q),
        "STANDARDIZE_INPUT": bool(STANDARDIZE_INPUT),
        "RUN_AFS": bool(RUN_AFS),
        "RUN_PROX_GL": bool(RUN_PROX_GL),
        "RUN_PROX_AGL": bool(RUN_PROX_AGL),
        "RUN_MIRROR": bool(RUN_MIRROR),
        "AFS_TRAIN_EPOCHS": int(AFS_TRAIN_EPOCHS),
        "AFS_BATCH_SIZE": AFS_BATCH_SIZE,
        "AFS_EXTRACT_SIZE": int(AFS_EXTRACT_SIZE),
        "AFS_CONTEXT_MODE": str(AFS_CONTEXT_MODE),
        "AFS_SELECTION_MODE": str(AFS_SELECTION_MODE),
        "AFS_STD_MULTIPLIER": float(AFS_STD_MULTIPLIER),
        "PROX_TRAIN_EPOCHS": int(PROX_TRAIN_EPOCHS),
        "PROX_BATCH_SIZE": PROX_BATCH_SIZE,
        "PROX_LAMBDA_GRID": np.asarray(PROX_LAMBDA_GRID, dtype=float).tolist(),
        "PROX_VALIDATION_RATIO": float(PROX_VALIDATION_RATIO),
        "PROX_TIME_ORDERED_VALIDATION": bool(PROX_TIME_ORDERED_VALIDATION),
        "PROX_ZERO_THRESHOLD": float(PROX_ZERO_THRESHOLD),
        "PROX_ADAPTIVE_GAMMA": float(PROX_ADAPTIVE_GAMMA),
        "MIRROR_STEPS": int(MIRROR_STEPS),
        "MIRROR_BATCH_SIZE": int(MIRROR_BATCH_SIZE),
        "MIRROR_OPTIMIZER": str(MIRROR_OPTIMIZER),
        "MIRROR_PSI": str(MIRROR_PSI),
        "MIRROR_COMPUTE_EVERY": MIRROR_COMPUTE_EVERY,
        "MIRROR_XI_BATCH_SIZE": int(MIRROR_XI_BATCH_SIZE),
        "MIRROR_GRADIENT_EVAL": str(MIRROR_GRADIENT_EVAL),
        "MIRROR_CLIP_GRAD_NORM": MIRROR_CLIP_GRAD_NORM,
        "SAVE_PER_SIM_DETAILS": bool(SAVE_PER_SIM_DETAILS),
    }


def main() -> None:
    if not any([RUN_AFS, RUN_PROX_GL, RUN_PROX_AGL, RUN_MIRROR]):
        raise ValueError("At least one baseline must be enabled.")

    afs_module = _load_module("baseline_afs_feature_selection", AFS_PATH) if RUN_AFS else None
    prox_module = _load_module("baseline_feature_selection_prox", PROX_PATH) if (RUN_PROX_GL or RUN_PROX_AGL) else None
    mirror_module = _load_module("baseline_feature_selection_mirror", MIRROR_PATH) if RUN_MIRROR else None

    run_id = _make_run_id()
    result_csv_path = _create_run_output_path(run_id)
    run_dir = result_csv_path.parent / f"{result_csv_path.stem}_artifacts"
    if SAVE_PER_SIM_DETAILS:
        run_dir.mkdir(parents=True, exist_ok=True)
    _write_header(result_csv_path, _settings(run_id, result_csv_path))
    wrote_results_header = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sim_idx in range(int(NUM_SIMULATIONS)):
        run_seed = int(SEED) + 5 * int(sim_idx)
        _set_all_seeds(run_seed)
        data, y, relevant_columns, irrelevant_columns = simulate_deeplink_t_dataset(
            n_samples=int(N_SAMPLES),
            n_relevant=int(N_RELEVANT),
            n_irrelevant=int(N_IRRELEVANT),
            latent_dim=int(LATENT_DIM),
            factor_model=FACTOR_MODEL,
            y_mode=Y_MODE,
            lookback=int(LOOKBACK),
            lag_decay=float(LAG_DECAY),
            lag_weight_norm=LAG_WEIGHT_NORM,
            signal_amplitude=float(SIGNAL_AMPLITUDE),
            link_function=Y_LINK_FUNCTION,
            noise_level=float(LABEL_NOISE),
            random_state=run_seed,
        )
        relevant_set = set(str(c) for c in relevant_columns)
        irrelevant_set = set(str(c) for c in irrelevant_columns)
        if bool(STANDARDIZE_INPUT):
            data = _standardize_feature_df(data)
        feature_names = [str(c) for c in data.columns]
        X_seq, y_seq = _build_sequence_tensors(data, y, LOOKBACK)
        hidden_dim = _make_hidden_dim(int(X_seq.shape[-1]), MODEL_TYPE)
        tuned = _tune_hparams(
            X_seq,
            y_seq,
            model_type=MODEL_TYPE,
            hidden_dim=hidden_dim,
            num_layers=int(NUM_LAYERS),
            random_state=run_seed,
        )
        used_lr, used_weight_decay, used_dropout = _resolve_optim_params(tuned)

        results: List[BaselineResult] = []
        with _maybe_suppress_output(enable_output=bool(SHOW_PROGRESS)):
            if RUN_AFS:
                results.append(
                    run_afs_baseline(
                        afs_module,
                        X_seq,
                        y_seq,
                        feature_names,
                        hidden_dim=hidden_dim,
                        lr=float(used_lr),
                        weight_decay=float(used_weight_decay),
                        dropout=float(used_dropout),
                        seed=run_seed + 101,
                        device=device,
                    )
                )
            if RUN_PROX_GL:
                results.append(
                    run_prox_baseline(
                        prox_module,
                        X_seq,
                        y_seq,
                        feature_names,
                        hidden_dim=hidden_dim,
                        lr=float(used_lr),
                        weight_decay=float(used_weight_decay),
                        dropout=float(used_dropout),
                        seed=run_seed + 202,
                        device=device,
                        adaptive=False,
                    )
                )
            if RUN_PROX_AGL:
                results.append(
                    run_prox_baseline(
                        prox_module,
                        X_seq,
                        y_seq,
                        feature_names,
                        hidden_dim=hidden_dim,
                        lr=float(used_lr),
                        weight_decay=float(used_weight_decay),
                        dropout=float(used_dropout),
                        seed=run_seed + 303,
                        device=device,
                        adaptive=True,
                    )
                )
            if RUN_MIRROR:
                results.append(
                    run_mirror_baseline(
                        mirror_module,
                        X_seq,
                        y_seq,
                        feature_names,
                        hidden_dim=hidden_dim,
                        lr=float(used_lr),
                        weight_decay=float(used_weight_decay),
                        dropout=float(used_dropout),
                        seed=run_seed + 404,
                        device=device,
                    )
                )

        for result in results:
            if result.extra is None:
                result.extra = {}
            result.extra.update(
                {
                    "tuned_found": tuned is not None,
                    "tuned_cv_mse": float(tuned.get("cv_mse", np.nan)) if tuned is not None else np.nan,
                    "tuned_objective": float(tuned.get("objective", np.nan)) if tuned is not None else np.nan,
                }
            )
            result_key = f"{run_id}_sim{sim_idx + 1}_{result.method}"
            row = _result_to_row(
                result,
                run_id=run_id,
                result_key=result_key,
                result_csv_path=result_csv_path,
                simulation_id=sim_idx + 1,
                seed=run_seed,
                relevant_set=relevant_set,
                irrelevant_set=irrelevant_set,
            )
            wrote_results_header = _append_row(result_csv_path, row, wrote_results_header)
            if SAVE_PER_SIM_DETAILS:
                _save_feature_details(run_dir, result_key, result, relevant_set)

        if bool(SHOW_PROGRESS):
            print(f"[baseline] finished simulation {sim_idx + 1}/{int(NUM_SIMULATIONS)}", flush=True)

    if bool(SHOW_PROGRESS):
        print(f"Saved baseline results to: {result_csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
