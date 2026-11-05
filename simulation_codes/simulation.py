#!/usr/bin/env python3
from __future__ import annotations

import inspect
import io
import math
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from catnet import (
    S_CatNet_with_details,
    _mirror_balance_and_smoothness_factors,
    _mirror_magnitude_from_vectors,
    _select_cutoff_from_mirror_stats,
    _select_features_by_cutoff,
    compute_group_scalar_importance_from_model,
    compute_sequence_fit_diagnostics,
)
from data_generation_sample import simulate_deeplink_t_dataset
from src.model_training.hyperparameter_tuning import find_optimal_params
from src.model_training.model_factory import build_sequence_model

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
DATA_GENERATION_MODE = "forecast"  # forecast | NLP
FACTOR_MODEL = "logistic"  # linear | logistic
LATENT_DIM = 10
LATENT_AR_PHI = 0.8

# y generation
Y_MODE = "lag_weighted"  # lag_weighted | current_x
Y_LINK_FUNCTION = "tanh"  # linear | nonlinear/sinexp | tanh/scaled_tanh | sin
SIGNAL_AMPLITUDE = 10.0
NLP_SIGNAL_AMPLITUDE = 4.0
LABEL_NOISE = 1.0
LAG_DECAY = 0.9
LAG_WEIGHT_NORM = "l2"  # l1 | l2
LAG_WEIGHT_SCHEME = None  # None uses forecast=decay, NLP=chunk_uniform
LAG_CHUNK_SIZE = 5
LAG_CHUNK_WEIGHT_LOW = 0.8
LAG_CHUNK_WEIGHT_HIGH = 1.2
STANDARDIZE_SIGNAL = None  # None uses forecast=False, NLP=True
LINK_SCALE = 2.0

# Model
MODEL_TYPE = "lstm"  # lstm | rnn | gru | transformer
HIDDEN_DIM_MULTIPLIER = 20.0
NUM_LAYERS = 2
DROPOUT = 0.0
OPTIM_LR = 0.001
OPTIM_WEIGHT_DECAY = 0.01
TRANSFORMER_NHEAD = 2
TRANSFORMER_DIM_FEEDFORWARD = None
CATNET_VARIANT = "s_catnet"  # fixed: s_catnet

# Group Neural GM
FDR_Q = 0.1
GROUP_METHOD = "scheme_c_free"  # benchmark | scheme_c_free
GROUP_OBJECTIVE = "kernel"  # kernel | linear_surrogate
GROUP_RESIDUAL_MODE = "ols"  # ols | gls
GROUP_LAMBDA_REG = 0
GROUP_MAX_ITER = 200
GROUP_LR = 5e-1
GROUP_RANK_R = None
GROUP_ENFORCE_SYMMETRIC_DELTA = False
GROUP_DELTA_NORM_CAP = None
GROUP_FREE_ENFORCE_SYMMETRIC = False
GROUP_FREE_G_NORM_CAP = None
GROUP_KW_USE_PCA = False
GROUP_KW_PCA_COMPONENTS = LOOKBACK * 2
GROUP_PRINT_OBJECTIVE_EVERY = 0  # 0 disables printing

# Importance
IMPORTANCE_MODES = ("shap", "derivative")  # run both in one simulation pass; order preserved in output
IMPORTANCE_MODE = IMPORTANCE_MODES[0]  # legacy compatibility alias
SHAP_N_PERMUTATIONS = 20
SHAP_DERIVATIVE_MODE = "local_multivar_linear"  # local_multivar_linear | local_quadratic | linear_slope
MIRROR_MAGNITUDE_MODES = ("sum_l2", "multiple_l2")  # evaluate both in one run, post-hoc from same importance
MIRROR_MAGNITUDE_MODE = MIRROR_MAGNITUDE_MODES[0]  # primary mode used for the single S_CatNet run
MIRROR_USE_BALANCE = False
MIRROR_USE_SMOOTHNESS = False
MIRROR_SMOOTH_LAMBDA = 1.0

# Derivative-mode params (used only when 'derivative' is in IMPORTANCE_MODES)
DERIVATIVE_METHOD = "temporal_secant"  # temporal_secant | lag_perturb
DERIVATIVE_STEP_MIN = 5e-4
DERIVATIVE_STEP_MAX = 2e-3
DERIVATIVE_N_STEPS = 10
DERIVATIVE_N_AVG = 3
DERIVATIVE_SMOOTH_NOISE_STD = 0.0
DERIVATIVE_PARALLEL_GROUP_SIZE = 20

# Diagnostics (all print controls default to False here)
STANDARDIZE_INPUT = True
PRINT_KW_DIAGNOSTICS = False
KW_DIAGNOSTICS_MAX_GROUPS = 5
RELEVANT_PREFIX = "relevant_"
PRINT_MIRROR_DETAILS = False
PLOT_MIRROR_DISTRIBUTION = False

# Original-model importance diagnostics
DIAG_RUN_ORIGINAL_IMPORTANCE = True
DIAG_ORIG_TRAIN_EPOCHS = 1000
CATNET_TRAIN_EPOCHS = 1000
DIAG_ORIG_SHOW_PROGRESS = False
ORIGINAL_TOPK = 20

# =========================
# Hyperparameter tuning controls
# =========================
SHOW_PROGRESS = False
AUTO_TUNE_HPARAMS = True
TUNE_PRINT_RESULTS = False
TUNE_N_TRIALS = 50
TUNE_N_FOLDS = 3
TUNE_VAL_RATIO = 0.2
TUNE_MAX_EPOCHS = 1000
TUNE_PATIENCE = 12
TUNE_WEIGHT_DECAY_TARGET = 2e-2
TUNE_LAMBDA_WD = 5e-4
TUNE_LAMBDA_DROPOUT = 1e-3

assert N_RELEVANT + N_IRRELEVANT > 0
assert N_SAMPLES > LOOKBACK

# High-dimensional controls
GROUP_HIGH_DIM_MODE = "auto"  # auto | ridge | standard
GROUP_RIDGE_LAMBDA_W = N_SAMPLES * 0.1
GROUP_RIDGE_LAMBDA_Z = N_SAMPLES * 0.1
GROUP_RIDGE_Q_EPS = None
GROUP_RIDGE_G_DELTA = None

# Deprecated / currently unused compatibility parameters:
# TWO_STAGE_SPLIT_RATIO = 0.5
# TWO_STAGE_GAP = LOOKBACK
# SCREEN_LAMBDA_GRID = np.geomspace(1e-3, 1e3, 31)
# SCREEN_TRAIN_EPOCHS = 50
# SCREEN_GATE_THRESHOLD = 1e-4
# SCREEN_LOW_DIM_FACTOR = 1.0
# SCREEN_HIERARCHY_M = 10.0
# SCREEN_WARMUP_EPOCHS = 80

# Output controls
RESULTS_DIR = Path(f"{MODEL_TYPE}_simulation_runs")
RESULT_CSV_PATH = None


def _method_label(importance_mode: str, mirror_mode: str) -> str:
    derivative_label = SHAP_DERIVATIVE_MODE
    if str(importance_mode).lower() == "derivative" and str(DERIVATIVE_METHOD).lower() != "temporal_secant":
        derivative_label = DERIVATIVE_METHOD
    return f"{CATNET_VARIANT}_{GROUP_METHOD}_{importance_mode}_{derivative_label}_{mirror_mode}"


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
    out_path = RESULTS_DIR / f"{MODEL_TYPE}_simulation_results_{run_id}.csv"
    return _ensure_non_conflicting_path(out_path)


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


def _make_hidden_dim(input_dim: int, model_type: str) -> int:
    raw = max(8, int(HIDDEN_DIM_MULTIPLIER * np.log(max(2, int(input_dim)))))
    if str(model_type).lower() != "transformer":
        return raw
    nhead = max(1, int(TRANSFORMER_NHEAD))
    return int(math.ceil(raw / nhead) * nhead)


def _model_kwargs(model_type: str) -> Dict[str, Any]:
    if str(model_type).lower() != "transformer":
        return {}
    kwargs: Dict[str, Any] = {"nhead": int(TRANSFORMER_NHEAD)}
    if TRANSFORMER_DIM_FEEDFORWARD is not None:
        kwargs["dim_feedforward"] = int(TRANSFORMER_DIM_FEEDFORWARD)
    return kwargs


def _build_sequence_tensors(
    data_df: pd.DataFrame,
    y: Iterable[float],
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_rows = []
    for i in range(int(lookback), len(data_df)):
        X_rows.append(data_df.iloc[i - int(lookback):i].values)
    X_np = np.asarray(X_rows, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1)[int(lookback):]
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"Sequence length mismatch: X_seq={X_np.shape[0]}, y_seq={y_np.shape[0]}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


def _train_model_silent(
    model: nn.Module,
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    show_progress: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X_seq.to(device=device, dtype=torch.float32)
    y = y_seq.to(device=device, dtype=torch.float32).reshape(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.MSELoss()

    epochs = max(1, int(num_epochs))
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X).reshape(-1)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if bool(show_progress) and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"[diag-train] epoch={epoch}, loss={float(loss.item()):.6g}", flush=True)


def _tune_hparams(
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    model_type: str,
    model_kwargs: Dict[str, Any],
    hidden_dim: int,
    num_layers: int,
    random_state: int,
) -> Dict[str, Any] | None:
    if not AUTO_TUNE_HPARAMS:
        return None
    try:
        return find_optimal_params(
            X_train_torch=X_seq,
            y_train_torch=y_seq,
            model_type=model_type,
            model_kwargs=model_kwargs,
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


def _selection_metrics(selected_features: Dict[str, Any], relevant_set: set, irrelevant_set: set) -> Dict[str, float]:
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


def _oracle_from_mirror_stats(mirror_stats: Dict[str, float], relevant_set: set, q: float) -> Dict[str, Any]:
    candidate_cutoffs = sorted({float(v) for v in mirror_stats.values() if float(v) > 0.0}) + [float("inf")]
    best = {
        "cutoff": float("inf"),
        "selected_names": set(),
        "fdr": 0.0,
        "power": 0.0,
    }
    denom_rel = max(1, len(relevant_set))
    for cutoff in candidate_cutoffs:
        selected_names = {k for k, v in mirror_stats.items() if float(v) >= float(cutoff)}
        n_total = len(selected_names)
        tp = len(selected_names & relevant_set)
        fp = n_total - tp
        fdr = fp / n_total if n_total > 0 else 0.0
        power = tp / denom_rel
        if fdr > float(q):
            continue
        if (power > best["power"]) or (np.isclose(power, best["power"]) and fdr < best["fdr"]):
            best = {
                "cutoff": float(cutoff),
                "selected_names": selected_names,
                "fdr": float(fdr),
                "power": float(power),
            }
    return best


def _topk_relevant_hits(importance_map: Dict[str, float], relevant_set: set, k: int) -> Dict[str, float]:
    ranking = sorted(importance_map.items(), key=lambda kv: float(kv[1]), reverse=True)
    top = ranking[: max(0, int(k))]
    denom = max(1, len(top))
    hits = sum(1 for feat, _ in top if feat in relevant_set)
    return {
        "topk": float(len(top)),
        "topk_relevant_hits": float(hits),
        "topk_precision": float(hits / denom),
    }


def _resolve_mirror_modes() -> Tuple[str, ...]:
    if isinstance(MIRROR_MAGNITUDE_MODES, str):
        modes = (MIRROR_MAGNITUDE_MODES,)
    else:
        modes = tuple(str(m).strip() for m in MIRROR_MAGNITUDE_MODES)
    if len(modes) == 0:
        raise ValueError("MIRROR_MAGNITUDE_MODES must contain at least one mode.")
    valid = {"max_l1", "max_l2", "sum_l1", "sum_l2", "multiple_l1", "multiple_l2"}
    for m in modes:
        if m not in valid:
            raise ValueError(f"Unsupported mirror mode in MIRROR_MAGNITUDE_MODES: {m}")
    # preserve order while removing duplicates
    dedup = tuple(dict.fromkeys(modes))
    return dedup


def _resolve_importance_modes() -> Tuple[str, ...]:
    if isinstance(IMPORTANCE_MODES, str):
        modes = (IMPORTANCE_MODES,)
    else:
        modes = tuple(str(m).strip() for m in IMPORTANCE_MODES)
    if len(modes) == 0:
        raise ValueError("IMPORTANCE_MODES must contain at least one mode.")
    valid = {"shap", "derivative"}
    for mode in modes:
        if mode not in valid:
            raise ValueError(f"Unsupported importance mode in IMPORTANCE_MODES: {mode}")
    return tuple(dict.fromkeys(modes))


def _recompute_mirror_stats_from_details(
    details_df: pd.DataFrame,
    *,
    mirror_mode: str,
    use_balance: bool,
    use_smoothness: bool,
    smooth_lambda: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    eps = 1e-12
    for _, row in details_df.iterrows():
        feature = str(row["feature"])
        plus_vec = np.asarray(row["plus_lag_vector"], dtype=np.float64).reshape(-1)
        minus_vec = np.asarray(row["minus_lag_vector"], dtype=np.float64).reshape(-1)
        signed_product = float(np.dot(plus_vec, minus_vec))
        sign_function = 1.0 if signed_product >= 0.0 else -1.0
        mirror_abs_core, _, _, _, _ = _mirror_magnitude_from_vectors(plus_vec, minus_vec, mode=mirror_mode)
        bg_factor, qg_factor, _, _ = _mirror_balance_and_smoothness_factors(
            plus_vec,
            minus_vec,
            use_l2=str(mirror_mode).endswith("l2"),
            smooth_lambda=float(smooth_lambda),
            eps=eps,
        )
        mirror_abs = float(mirror_abs_core)
        if bool(use_balance):
            mirror_abs *= float(bg_factor)
        if bool(use_smoothness):
            mirror_abs *= float(qg_factor)
        out[feature] = float(sign_function * mirror_abs)
    return out


def _pick_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


@contextmanager
def _maybe_suppress_output(enable_output: bool):
    if bool(enable_output):
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def _write_header(path: Path, settings: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Simulation Settings\n")
        for k, v in settings.items():
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


def _prefixed_tune_fields(prefix: str, tuned: Dict[str, Any] | None) -> Dict[str, Any]:
    if tuned is None:
        return {
            f"{prefix}_tuned_lr": np.nan,
            f"{prefix}_tuned_weight_decay": np.nan,
            f"{prefix}_tuned_dropout": np.nan,
            f"{prefix}_tuned_cv_mse": np.nan,
            f"{prefix}_tuned_objective": np.nan,
            f"{prefix}_tuned_found": False,
        }
    return {
        f"{prefix}_tuned_lr": float(tuned.get("lr", np.nan)),
        f"{prefix}_tuned_weight_decay": float(tuned.get("weight_decay", np.nan)),
        f"{prefix}_tuned_dropout": float(tuned.get("dropout", np.nan)),
        f"{prefix}_tuned_cv_mse": float(tuned.get("cv_mse", np.nan)),
        f"{prefix}_tuned_objective": float(tuned.get("objective", np.nan)),
        f"{prefix}_tuned_found": True,
    }


def main() -> None:
    if str(CATNET_VARIANT).lower() != "s_catnet":
        raise ValueError("This script currently supports only 's_catnet'.")
    importance_modes = _resolve_importance_modes()
    primary_importance_mode = str(importance_modes[0])
    mirror_modes = _resolve_mirror_modes()
    primary_mirror_mode = str(mirror_modes[0])

    run_id = _make_run_id()
    result_csv_path = _create_run_output_path(run_id)
    settings = {
        "RUN_ID": run_id,
        "RESULT_CSV_PATH": str(result_csv_path.resolve()),
        "SEED": SEED,
        "NUM_SIMULATIONS": NUM_SIMULATIONS,
        "N_SAMPLES": N_SAMPLES,
        "N_RELEVANT": N_RELEVANT,
        "N_IRRELEVANT": N_IRRELEVANT,
        "LOOKBACK": LOOKBACK,
        "DATA_GENERATION_MODE": DATA_GENERATION_MODE,
        "FACTOR_MODEL": FACTOR_MODEL,
        "LATENT_DIM": LATENT_DIM,
        "LATENT_AR_PHI": LATENT_AR_PHI,
        "Y_MODE": Y_MODE,
        "Y_LINK_FUNCTION": Y_LINK_FUNCTION,
        "SIGNAL_AMPLITUDE": SIGNAL_AMPLITUDE,
        "NLP_SIGNAL_AMPLITUDE": NLP_SIGNAL_AMPLITUDE,
        "LABEL_NOISE": LABEL_NOISE,
        "LAG_DECAY": LAG_DECAY,
        "LAG_WEIGHT_NORM": LAG_WEIGHT_NORM,
        "LAG_WEIGHT_SCHEME": LAG_WEIGHT_SCHEME,
        "LAG_CHUNK_SIZE": LAG_CHUNK_SIZE,
        "LAG_CHUNK_WEIGHT_LOW": LAG_CHUNK_WEIGHT_LOW,
        "LAG_CHUNK_WEIGHT_HIGH": LAG_CHUNK_WEIGHT_HIGH,
        "STANDARDIZE_SIGNAL": STANDARDIZE_SIGNAL,
        "LINK_SCALE": LINK_SCALE,
        "MODEL_TYPE": MODEL_TYPE,
        "HIDDEN_DIM_MULTIPLIER": HIDDEN_DIM_MULTIPLIER,
        "NUM_LAYERS": NUM_LAYERS,
        "DROPOUT": DROPOUT,
        "OPTIM_LR": OPTIM_LR,
        "OPTIM_WEIGHT_DECAY": OPTIM_WEIGHT_DECAY,
        "TRANSFORMER_NHEAD": TRANSFORMER_NHEAD,
        "TRANSFORMER_DIM_FEEDFORWARD": TRANSFORMER_DIM_FEEDFORWARD,
        "CATNET_VARIANT": CATNET_VARIANT,
        "FDR_Q": FDR_Q,
        "GROUP_METHOD": GROUP_METHOD,
        "GROUP_OBJECTIVE": GROUP_OBJECTIVE,
        "GROUP_RESIDUAL_MODE": GROUP_RESIDUAL_MODE,
        "GROUP_LAMBDA_REG": GROUP_LAMBDA_REG,
        "GROUP_MAX_ITER": GROUP_MAX_ITER,
        "GROUP_LR": GROUP_LR,
        "GROUP_RANK_R": GROUP_RANK_R,
        "GROUP_ENFORCE_SYMMETRIC_DELTA": GROUP_ENFORCE_SYMMETRIC_DELTA,
        "GROUP_DELTA_NORM_CAP": GROUP_DELTA_NORM_CAP,
        "GROUP_FREE_ENFORCE_SYMMETRIC": GROUP_FREE_ENFORCE_SYMMETRIC,
        "GROUP_FREE_G_NORM_CAP": GROUP_FREE_G_NORM_CAP,
        "GROUP_KW_USE_PCA": GROUP_KW_USE_PCA,
        "GROUP_KW_PCA_COMPONENTS": GROUP_KW_PCA_COMPONENTS,
        "GROUP_PRINT_OBJECTIVE_EVERY": GROUP_PRINT_OBJECTIVE_EVERY,
        "IMPORTANCE_MODES": list(importance_modes),
        "PRIMARY_IMPORTANCE_MODE_FOR_DIAG": primary_importance_mode,
        "IMPORTANCE_MODE": IMPORTANCE_MODE,
        "SHAP_N_PERMUTATIONS": SHAP_N_PERMUTATIONS,
        "SHAP_DERIVATIVE_MODE": SHAP_DERIVATIVE_MODE,
        "MIRROR_MAGNITUDE_MODE": MIRROR_MAGNITUDE_MODE,
        "MIRROR_MAGNITUDE_MODES": list(mirror_modes),
        "PRIMARY_MIRROR_MODE_FOR_SINGLE_RUN": primary_mirror_mode,
        "MIRROR_USE_BALANCE": MIRROR_USE_BALANCE,
        "MIRROR_USE_SMOOTHNESS": MIRROR_USE_SMOOTHNESS,
        "MIRROR_SMOOTH_LAMBDA": MIRROR_SMOOTH_LAMBDA,
        "DERIVATIVE_METHOD": DERIVATIVE_METHOD,
        "DERIVATIVE_STEP_MIN": DERIVATIVE_STEP_MIN,
        "DERIVATIVE_STEP_MAX": DERIVATIVE_STEP_MAX,
        "DERIVATIVE_N_STEPS": DERIVATIVE_N_STEPS,
        "DERIVATIVE_N_AVG": DERIVATIVE_N_AVG,
        "DERIVATIVE_SMOOTH_NOISE_STD": DERIVATIVE_SMOOTH_NOISE_STD,
        "DERIVATIVE_PARALLEL_GROUP_SIZE": DERIVATIVE_PARALLEL_GROUP_SIZE,
        "STANDARDIZE_INPUT": STANDARDIZE_INPUT,
        "PRINT_KW_DIAGNOSTICS": PRINT_KW_DIAGNOSTICS,
        "KW_DIAGNOSTICS_MAX_GROUPS": KW_DIAGNOSTICS_MAX_GROUPS,
        "RELEVANT_PREFIX": RELEVANT_PREFIX,
        "PRINT_MIRROR_DETAILS": PRINT_MIRROR_DETAILS,
        "PLOT_MIRROR_DISTRIBUTION": PLOT_MIRROR_DISTRIBUTION,
        "DIAG_RUN_ORIGINAL_IMPORTANCE": DIAG_RUN_ORIGINAL_IMPORTANCE,
        "DIAG_ORIG_TRAIN_EPOCHS": DIAG_ORIG_TRAIN_EPOCHS,
        "CATNET_TRAIN_EPOCHS": CATNET_TRAIN_EPOCHS,
        "DIAG_ORIG_SHOW_PROGRESS": DIAG_ORIG_SHOW_PROGRESS,
        "ORIGINAL_TOPK": ORIGINAL_TOPK,
        "SHOW_PROGRESS": SHOW_PROGRESS,
        "AUTO_TUNE_HPARAMS": AUTO_TUNE_HPARAMS,
        "TUNE_PRINT_RESULTS": TUNE_PRINT_RESULTS,
        "TUNE_N_TRIALS": TUNE_N_TRIALS,
        "TUNE_N_FOLDS": TUNE_N_FOLDS,
        "TUNE_VAL_RATIO": TUNE_VAL_RATIO,
        "TUNE_MAX_EPOCHS": TUNE_MAX_EPOCHS,
        "TUNE_PATIENCE": TUNE_PATIENCE,
        "TUNE_WEIGHT_DECAY_TARGET": TUNE_WEIGHT_DECAY_TARGET,
        "TUNE_LAMBDA_WD": TUNE_LAMBDA_WD,
        "TUNE_LAMBDA_DROPOUT": TUNE_LAMBDA_DROPOUT,
        "GROUP_HIGH_DIM_MODE": GROUP_HIGH_DIM_MODE,
        "GROUP_RIDGE_LAMBDA_W": GROUP_RIDGE_LAMBDA_W,
        "GROUP_RIDGE_LAMBDA_Z": GROUP_RIDGE_LAMBDA_Z,
        "GROUP_RIDGE_Q_EPS": GROUP_RIDGE_Q_EPS,
        "GROUP_RIDGE_G_DELTA": GROUP_RIDGE_G_DELTA,
        # Deprecated / currently unused compatibility parameters were removed.
    }
    _write_header(result_csv_path, settings)
    wrote_results_header = False

    for sim_idx in range(int(NUM_SIMULATIONS)):
        run_seed = int(SEED) + 5 * int(sim_idx)
        _set_all_seeds(run_seed)
        effective_signal_amplitude = (
            float(NLP_SIGNAL_AMPLITUDE)
            if str(DATA_GENERATION_MODE).lower() == "nlp"
            else float(SIGNAL_AMPLITUDE)
        )

        data, y, relevant_columns, irrelevant_columns = simulate_deeplink_t_dataset(
            n_samples=int(N_SAMPLES),
            n_relevant=int(N_RELEVANT),
            n_irrelevant=int(N_IRRELEVANT),
            latent_dim=int(LATENT_DIM),
            factor_model=FACTOR_MODEL,
            data_generation_mode=DATA_GENERATION_MODE,
            latent_ar_phi=float(LATENT_AR_PHI),
            y_mode=Y_MODE,
            lookback=int(LOOKBACK),
            lag_decay=float(LAG_DECAY),
            lag_weight_norm=LAG_WEIGHT_NORM,
            lag_weight_scheme=LAG_WEIGHT_SCHEME,
            lag_chunk_size=int(LAG_CHUNK_SIZE),
            lag_chunk_weight_low=float(LAG_CHUNK_WEIGHT_LOW),
            lag_chunk_weight_high=float(LAG_CHUNK_WEIGHT_HIGH),
            standardize_signal=STANDARDIZE_SIGNAL,
            link_scale=float(LINK_SCALE),
            signal_amplitude=float(effective_signal_amplitude),
            link_function=Y_LINK_FUNCTION,
            noise_level=float(LABEL_NOISE),
            random_state=run_seed,
        )
        relevant_set = set(relevant_columns)
        irrelevant_set = set(irrelevant_columns)
        input_dim = int(data.shape[1])
        hidden_dim = _make_hidden_dim(input_dim, MODEL_TYPE)
        model_kwargs = _model_kwargs(MODEL_TYPE)
        X_seq, y_seq = _build_sequence_tensors(data, y, LOOKBACK)

        diag_fit = {}
        orig_topk_default = {"topk": np.nan, "topk_relevant_hits": np.nan, "topk_precision": np.nan}
        orig_topk_by_mode: Dict[str, Dict[str, float]] = {
            str(mode): dict(orig_topk_default) for mode in importance_modes
        }
        diag_tuned = None
        diag_lr, diag_wd, diag_dropout = float(OPTIM_LR), float(OPTIM_WEIGHT_DECAY), float(DROPOUT)

        if DIAG_RUN_ORIGINAL_IMPORTANCE:
            diag_tuned = _tune_hparams(
                X_seq,
                y_seq,
                model_type=MODEL_TYPE,
                model_kwargs=model_kwargs,
                hidden_dim=hidden_dim,
                num_layers=int(NUM_LAYERS),
                random_state=run_seed,
            )
            diag_lr, diag_wd, diag_dropout = _resolve_optim_params(diag_tuned)
            diag_model = build_sequence_model(
                model_type=MODEL_TYPE,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=int(NUM_LAYERS),
                output_dim=1,
                dropout=float(diag_dropout),
                **model_kwargs,
            )
            _train_model_silent(
                model=diag_model,
                X_seq=X_seq,
                y_seq=y_seq,
                num_epochs=int(DIAG_ORIG_TRAIN_EPOCHS),
                lr=float(diag_lr),
                weight_decay=float(diag_wd),
                show_progress=bool(DIAG_ORIG_SHOW_PROGRESS),
            )
            diag_fit = compute_sequence_fit_diagnostics(diag_model, X_seq, y_seq)

            with _maybe_suppress_output(enable_output=False):
                for importance_mode in importance_modes:
                    orig_importance = compute_group_scalar_importance_from_model(
                        model=diag_model,
                        X_seq=X_seq,
                        feature_names=list(data.columns),
                        importance_mode=str(importance_mode),
                        shap_n_permutations=int(SHAP_N_PERMUTATIONS),
                        shap_derivative_mode=SHAP_DERIVATIVE_MODE,
                        derivative_method=DERIVATIVE_METHOD,
                        derivative_step_min=float(DERIVATIVE_STEP_MIN),
                        derivative_step_max=float(DERIVATIVE_STEP_MAX),
                        derivative_n_steps=int(DERIVATIVE_N_STEPS),
                        derivative_n_avg=int(DERIVATIVE_N_AVG),
                        derivative_smooth_noise_std=float(DERIVATIVE_SMOOTH_NOISE_STD),
                        derivative_parallel_group_size=int(DERIVATIVE_PARALLEL_GROUP_SIZE),
                        mirror_magnitude_mode=primary_mirror_mode,
                        mirror_use_balance=bool(MIRROR_USE_BALANCE),
                        mirror_use_smoothness=bool(MIRROR_USE_SMOOTHNESS),
                        mirror_smooth_lambda=float(MIRROR_SMOOTH_LAMBDA),
                    )
                    orig_topk_by_mode[str(importance_mode)] = _topk_relevant_hits(
                        orig_importance, relevant_set, int(ORIGINAL_TOPK)
                    )

        # Align the main CatNet run with the notebook logic:
        # build the base model/optimizer from the fixed top-level config and
        # let S_CatNet_with_details handle any internal hyperparameter tuning.
        main_tuned = None
        main_lr, main_wd, main_dropout = np.nan, np.nan, np.nan

        main_model = build_sequence_model(
            model_type=MODEL_TYPE,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=int(NUM_LAYERS),
            output_dim=1,
            dropout=float(DROPOUT),
            **model_kwargs,
        )
        main_optimizer = torch.optim.Adam(
            main_model.parameters(),
            lr=float(OPTIM_LR),
            weight_decay=float(OPTIM_WEIGHT_DECAY),
        )

        shared_algo_kwargs = {
            "model": main_model,
            "optimizer": main_optimizer,
            "data": data,
            "y": y,
            "q": float(FDR_Q),
            "lookback": int(LOOKBACK),
            "show_progress": bool(SHOW_PROGRESS),
            "shap_n_permutations": int(SHAP_N_PERMUTATIONS),
            "model_type": MODEL_TYPE,
            "model_kwargs": model_kwargs,
            "relevant_prefix": RELEVANT_PREFIX,
            "standardize_input": bool(STANDARDIZE_INPUT),
            "importance_mode": tuple(str(m) for m in importance_modes),
            "derivative_method": str(DERIVATIVE_METHOD),
            "derivative_step_min": float(DERIVATIVE_STEP_MIN),
            "derivative_step_max": float(DERIVATIVE_STEP_MAX),
            "derivative_n_steps": int(DERIVATIVE_N_STEPS),
            "derivative_n_avg": int(DERIVATIVE_N_AVG),
            "derivative_smooth_noise_std": float(DERIVATIVE_SMOOTH_NOISE_STD),
            "derivative_parallel_group_size": int(DERIVATIVE_PARALLEL_GROUP_SIZE),
            "group_method": GROUP_METHOD,
            "group_objective": GROUP_OBJECTIVE,
            "group_residual_mode": GROUP_RESIDUAL_MODE,
            "group_lambda_reg": float(GROUP_LAMBDA_REG),
            "group_max_iter": int(GROUP_MAX_ITER),
            "group_lr": float(GROUP_LR),
            "group_rank_r": GROUP_RANK_R,
            "group_enforce_symmetric_delta": bool(GROUP_ENFORCE_SYMMETRIC_DELTA),
            "group_delta_norm_cap": GROUP_DELTA_NORM_CAP,
            "group_free_enforce_symmetric": bool(GROUP_FREE_ENFORCE_SYMMETRIC),
            "group_free_g_norm_cap": GROUP_FREE_G_NORM_CAP,
            "group_kw_use_pca": bool(GROUP_KW_USE_PCA),
            "group_kw_pca_components": GROUP_KW_PCA_COMPONENTS,
            "group_high_dim_mode": GROUP_HIGH_DIM_MODE,
            "group_ridge_lambda_w": GROUP_RIDGE_LAMBDA_W,
            "group_ridge_lambda_z": GROUP_RIDGE_LAMBDA_Z,
            "group_ridge_q_eps": GROUP_RIDGE_Q_EPS,
            "group_ridge_g_delta": GROUP_RIDGE_G_DELTA,
            "group_print_objective_every": int(GROUP_PRINT_OBJECTIVE_EVERY),
            "print_kw_diagnostics": bool(PRINT_KW_DIAGNOSTICS),
            "kw_diagnostics_max_groups": int(KW_DIAGNOSTICS_MAX_GROUPS),
            "shap_derivative_mode": SHAP_DERIVATIVE_MODE,
            "train_epochs": int(CATNET_TRAIN_EPOCHS),
            "auto_tune_hparams": True,
            "tune_n_trials": int(TUNE_N_TRIALS),
            "tune_n_folds": int(TUNE_N_FOLDS),
            "tune_val_ratio": float(TUNE_VAL_RATIO),
            "tune_max_epochs": int(TUNE_MAX_EPOCHS),
            "tune_patience": int(TUNE_PATIENCE),
            "tune_weight_decay_target": float(TUNE_WEIGHT_DECAY_TARGET),
            "tune_lambda_wd": float(TUNE_LAMBDA_WD),
            "tune_lambda_dropout": float(TUNE_LAMBDA_DROPOUT),
            "mirror_magnitude_mode": primary_mirror_mode,
            "mirror_use_balance": bool(MIRROR_USE_BALANCE),
            "mirror_use_smoothness": bool(MIRROR_USE_SMOOTHNESS),
            "mirror_smooth_lambda": float(MIRROR_SMOOTH_LAMBDA),
            "print_mirror_details": bool(PRINT_MIRROR_DETAILS),
            "plot_mirror_distribution": bool(PLOT_MIRROR_DISTRIBUTION),
            # Removed deprecated compatibility kwargs:
            # two_stage_* and screen_* were filtered by _pick_kwargs and never used.
        }

        call_kwargs = _pick_kwargs(S_CatNet_with_details, shared_algo_kwargs)
        with _maybe_suppress_output(enable_output=bool(SHOW_PROGRESS or PRINT_MIRROR_DETAILS)):
            mode_results = S_CatNet_with_details(**call_kwargs)

        for importance_mode in importance_modes:
            mode_key = str(importance_mode)
            if mode_key not in mode_results:
                raise KeyError(f"Missing importance mode result from S_CatNet_with_details: {mode_key}")
            _selected_features_primary, _mirror_stats_primary, _cutoff_primary, details_df = mode_results[mode_key]

            for mirror_mode in mirror_modes:
                if str(mirror_mode) == primary_mirror_mode:
                    mirror_stats = dict(_mirror_stats_primary)
                    cutoff = float(_cutoff_primary)
                    selected_features = dict(_selected_features_primary)
                else:
                    mirror_stats = _recompute_mirror_stats_from_details(
                        details_df,
                        mirror_mode=str(mirror_mode),
                        use_balance=bool(MIRROR_USE_BALANCE),
                        use_smoothness=bool(MIRROR_USE_SMOOTHNESS),
                        smooth_lambda=float(MIRROR_SMOOTH_LAMBDA),
                    )
                    cutoff = _select_cutoff_from_mirror_stats(mirror_stats, float(FDR_Q))
                    selected_features = _select_features_by_cutoff(mirror_stats, cutoff)
                ours = _selection_metrics(selected_features, relevant_set, irrelevant_set)
                oracle = _oracle_from_mirror_stats(mirror_stats, relevant_set, float(FDR_Q))
                oracle_names = oracle["selected_names"]
                oracle_rel = len(oracle_names & relevant_set)
                oracle_irr = len(oracle_names & irrelevant_set)
                orig_topk = orig_topk_by_mode.get(str(importance_mode), orig_topk_default)
                result_key = f"{run_id}_sim{sim_idx + 1}_{importance_mode}_{mirror_mode}"

                row = {
                    "run_id": run_id,
                    "result_key": result_key,
                    "result_file": str(result_csv_path.resolve()),
                    "simulation_id": sim_idx + 1,
                    "seed": run_seed,
                    "method": _method_label(str(importance_mode), str(mirror_mode)),
                    "model_type": MODEL_TYPE,
                    "catnet_variant": CATNET_VARIANT,
                    "importance_mode": str(importance_mode),
                    "derivative_method": str(DERIVATIVE_METHOD),
                    "mirror_magnitude_mode": str(mirror_mode),
                    "primary_importance_mode_for_diag": primary_importance_mode,
                    "primary_mirror_mode_for_single_run": primary_mirror_mode,
                    "posthoc_recompute_from_single_importance_run": True,
                    "cutoff": float(cutoff),
                    "strict_cutoff": float(cutoff),
                    "fdr": float(ours["fdr"]),
                    "power": float(ours["power"]),
                    "strict_fdr": float(ours["fdr"]),
                    "strict_power": float(ours["power"]),
                    "n_samples": int(N_SAMPLES),
                    "effective_train_samples": max(int(N_SAMPLES) - int(LOOKBACK), 0),
                    "n_relevant_features": int(len(relevant_set)),
                    "n_irrelevant_features": int(len(irrelevant_set)),
                    "selected_total_features": int(ours["selected_total_features"]),
                    "selected_relevant_features": int(ours["selected_relevant_features"]),
                    "selected_irrelevant_features": int(ours["selected_irrelevant_features"]),
                    "strict_selected_total_features": int(ours["selected_total_features"]),
                    "strict_selected_relevant_features": int(ours["selected_relevant_features"]),
                    "strict_selected_irrelevant_features": int(ours["selected_irrelevant_features"]),
                    "oracle_cutoff": float(oracle["cutoff"]),
                    "oracle_fdr": float(oracle["fdr"]),
                    "oracle_power": float(oracle["power"]),
                    "oracle_selected_relevant_features": int(oracle_rel),
                    "oracle_selected_irrelevant_features": int(oracle_irr),
                    "main_used_lr": float(main_lr),
                    "main_used_weight_decay": float(main_wd),
                    "main_used_dropout": float(main_dropout),
                    "diag_used_lr": float(diag_lr),
                    "diag_used_weight_decay": float(diag_wd),
                    "diag_used_dropout": float(diag_dropout),
                    "diag_top20_k": int(orig_topk["topk"]) if not np.isnan(orig_topk["topk"]) else np.nan,
                    "diag_top20_relevant_hits": int(orig_topk["topk_relevant_hits"])
                    if not np.isnan(orig_topk["topk_relevant_hits"])
                    else np.nan,
                    "diag_top20_precision": float(orig_topk["topk_precision"]),
                }

                row.update(
                    {
                        "diag_fit_mse": float(diag_fit.get("mse", np.nan)),
                        "diag_fit_mae": float(diag_fit.get("mae", np.nan)),
                        "diag_fit_r2": float(diag_fit.get("r2", np.nan)),
                        "diag_fit_corr_pred_target": float(diag_fit.get("corr_pred_target", np.nan)),
                        "diag_fit_std_pred_over_target": float(diag_fit.get("std_pred_over_target", np.nan)),
                        "diag_fit_mse_over_baseline": float(diag_fit.get("mse_over_baseline", np.nan)),
                        "diag_fit_near_constant_prediction": bool(diag_fit.get("near_constant_prediction", False))
                        if len(diag_fit) > 0
                        else np.nan,
                        "diag_fit_likely_underfit": bool(diag_fit.get("likely_underfit", False))
                        if len(diag_fit) > 0
                        else np.nan,
                    }
                )
                row.update(_prefixed_tune_fields("diag", diag_tuned))
                row.update(_prefixed_tune_fields("main", main_tuned))

                wrote_results_header = _append_row(result_csv_path, row, wrote_results_header)

    if bool(SHOW_PROGRESS):
        print(f"Saved results to: {result_csv_path.resolve()}")


if __name__ == "__main__":
    main()
