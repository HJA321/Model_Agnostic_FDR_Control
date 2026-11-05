#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from uuid import uuid4

import io
import numpy as np
import pandas as pd
import torch
from group_lasso import GroupLasso
from scipy import stats
from sklearn.preprocessing import StandardScaler

import linear_simulation as lin


# =========================
# Main parameters
# =========================
SEED = 42
NUM_SIMULATIONS = 25

DATA_MODE = "iid_group"  # iid_group | lag_group
DATA_LENGTH = 2500
NUM_FEATURES = 150
NUM_NOISES = 3350
GROUP_SIZE = 5

WITHIN_GROUP_CORR_VALUES = (0.1, 0.5, 0.9)
BETWEEN_GROUP_CORR_FACTORS = (0.2, 0.8)
AR_PHI_VALUES = (0.1, 0.5, 0.9)

BETA_STD_SCALE = 20.0
LABEL_NOISE_STD = 1.0

FDR_Q = 0.1
LOW_DIM_FACTOR = 2.0
SPLIT_RATIO = 1.0 / 3.0
SPLIT_GAP = 10
TIME_ORDERED_SPLIT = True
SCREEN_LAMBDA_GRID = np.geomspace(0.01, 0.5, 12)

COEF_METHOD = "full_lstsq"
CJ_METHOD = "grouped_matrix"
NOISE_STD = 1.0

# Sparse-oracle-ish Group Lasso baseline. We assume we know the target is sparse
# and stop once selected groups are no more than this multiplier times |S|.
GROUP_LASSO_LAMBDA_GRID = np.geomspace(0.001, 1.0, 30)
GROUP_LASSO_MAX_SELECTED_MULTIPLIER = 2.0
GROUP_LASSO_MIN_SELECTED = 1

# Known AR(1) label-noise experiment.
ERROR_AR_RHO = 0.5
ERROR_SEED_OFFSET = 100_000

SHOW_PROGRESS = False
RESULTS_DIR = Path("linear_extra_runs")
RESULT_CSV_PATH = None
METHOD_FILTER: tuple[str, ...] | None = None
ALLOWED_METHODS = {"group_lasso", "group_bh", "group_bh_gls_ar1", "group_gm_ols", "group_gm_gls_ar1"}

RESULT_COLUMNS = [
    "run_id",
    "result_key",
    "result_file",
    "experiment",
    "method",
    "status",
    "mode",
    "simulation_id",
    "ar_simulation_id",
    "scenario_simulation_id",
    "seed",
    "data_mode",
    "ar_phi",
    "within_group_corr",
    "between_group_corr_factor",
    "between_group_corr",
    "fdr_q",
    "cutoff",
    "fdr",
    "power",
    "num_selected_groups",
    "num_selected_null_groups",
    "num_selected_relevant_groups",
    "num_total_relevant_groups",
    "num_relevant_groups",
    "num_noise_groups",
    "num_total_groups",
    "n_samples",
    "n_features_total",
    "n_relevant_columns",
    "n_noise_columns",
    "group_size",
    "label_noise_std",
    "beta_nonzero_count",
    "stage1_selected_groups",
    "stage1_dropped_groups",
    "stage1_dropped_columns",
    "stage2_n2",
    "stage2_d2",
    "stage2_low_dim_ok",
    "screening_log_len",
    "mirror_stats_positive",
    "bh_cutoff_pvalue",
    "bh_min_pvalue",
    "bh_num_tests",
    "bh_num_rejected",
    "bh_uses_gls",
    "lambda",
    "selection_policy",
    "path_len",
    "sparse_target_groups",
    "sparse_max_selected_groups",
    "error_ar_rho",
]


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
    return _ensure_non_conflicting_path(RESULTS_DIR / f"linear_extra_results_{run_id}.csv")


def _write_header(path: Path, settings: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Linear Extra Simulation Settings\n")
        for k, v in settings.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            f.write(f"# {k}: {v}\n")
        f.write("# ---\n")
        f.flush()
        os.fsync(f.fileno())


def _append_row(path: Path, row: Dict[str, Any], wrote_header: bool) -> bool:
    row = {col: row.get(col, np.nan) for col in RESULT_COLUMNS}
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


@contextmanager
def _maybe_suppress_output(enable_output: bool):
    if bool(enable_output):
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def _normalize_methods(methods: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if methods is None:
        return None
    if isinstance(methods, str):
        requested = tuple(m.strip() for m in methods.split(",") if m.strip())
    else:
        requested = tuple(str(m).strip() for m in methods if str(m).strip())
    unknown = sorted(set(requested) - ALLOWED_METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Allowed: {sorted(ALLOWED_METHODS)}")
    return requested


def _sync_linear_globals() -> None:
    lin.SEED = int(SEED)
    lin.NUM_SIMULATIONS = int(NUM_SIMULATIONS)
    lin.DATA_MODE = str(DATA_MODE)
    lin.DATA_LENGTH = int(DATA_LENGTH)
    lin.NUM_FEATURES = int(NUM_FEATURES)
    lin.NUM_NOISES = int(NUM_NOISES)
    lin.GROUP_SIZE = int(GROUP_SIZE)
    lin.WITHIN_GROUP_CORR_VALUES = tuple(float(v) for v in WITHIN_GROUP_CORR_VALUES)
    lin.BETWEEN_GROUP_CORR_FACTORS = tuple(float(v) for v in BETWEEN_GROUP_CORR_FACTORS)
    lin.AR_PHI_VALUES = tuple(float(v) for v in AR_PHI_VALUES)
    lin.BETA_STD_SCALE = float(BETA_STD_SCALE)
    lin.LABEL_NOISE_STD = float(LABEL_NOISE_STD)
    lin.FDR_Q = float(FDR_Q)
    lin.LOW_DIM_FACTOR = float(LOW_DIM_FACTOR)
    lin.SPLIT_RATIO = float(SPLIT_RATIO)
    lin.SPLIT_GAP = int(SPLIT_GAP)
    lin.TIME_ORDERED_SPLIT = bool(TIME_ORDERED_SPLIT)
    lin.LAMBDA_GRID = np.asarray(SCREEN_LAMBDA_GRID, dtype=float)
    lin.NOISE_STD = float(NOISE_STD)
    lin.COEF_METHOD = str(COEF_METHOD)
    lin.CJ_METHOD = str(CJ_METHOD)


def _build_scenarios() -> list[dict[str, Any]]:
    if str(DATA_MODE).lower() == "iid_group":
        return [
            {
                "ar_phi": None,
                "within_group_corr": float(rho_w),
                "between_group_corr_factor": float(rho_b_factor),
                "between_group_corr": float(rho_w) * float(rho_b_factor),
            }
            for rho_w in WITHIN_GROUP_CORR_VALUES
            for rho_b_factor in BETWEEN_GROUP_CORR_FACTORS
        ]
    return [
        {
            "ar_phi": float(ar_phi),
            "within_group_corr": None,
            "between_group_corr_factor": float(rho_b_factor),
            "between_group_corr": float(rho_b_factor),
        }
        for ar_phi in AR_PHI_VALUES
        for rho_b_factor in BETWEEN_GROUP_CORR_FACTORS
    ]


def _make_ar1_noise(n: int, rho: float, sigma: float, seed: int) -> np.ndarray:
    rho = float(rho)
    sigma = float(sigma)
    rng = np.random.default_rng(int(seed))
    eps = np.zeros(int(n), dtype=np.float64)
    eps[0] = rng.normal(0.0, sigma)
    innovation_sd = sigma * np.sqrt(max(0.0, 1.0 - rho ** 2))
    for t in range(1, int(n)):
        eps[t] = rho * eps[t - 1] + rng.normal(0.0, innovation_sd)
    return eps


def _whiten_ar1(x_df: pd.DataFrame, y: np.ndarray, rho: float) -> Tuple[pd.DataFrame, np.ndarray]:
    x = x_df.to_numpy(dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y_arr.shape[0]:
        raise ValueError("X/y length mismatch before AR whitening.")
    rho = float(rho)
    scale0 = np.sqrt(max(0.0, 1.0 - rho ** 2))
    x_w = np.empty_like(x)
    y_w = np.empty_like(y_arr)
    x_w[0, :] = scale0 * x[0, :]
    y_w[0] = scale0 * y_arr[0]
    x_w[1:, :] = x[1:, :] - rho * x[:-1, :]
    y_w[1:] = y_arr[1:] - rho * y_arr[:-1]
    return pd.DataFrame(x_w, columns=x_df.columns), y_w


def _selection_metrics(selected_groups: Iterable[str], relevant_groups: set[str], noise_groups: set[str]) -> Dict[str, float]:
    selected = set(str(g) for g in selected_groups)
    tp = len(selected & relevant_groups)
    fp = len(selected & noise_groups)
    total = len(selected)
    return {
        "fdr": float(fp / total) if total > 0 else 0.0,
        "power": float(tp / max(1, len(relevant_groups))),
        "num_selected_groups": int(total),
        "num_selected_null_groups": int(fp),
        "num_selected_relevant_groups": int(tp),
    }


def _base_row(
    *,
    run_id: str,
    result_csv_path: Path,
    result_key: str,
    experiment: str,
    method: str,
    scenario: Dict[str, Any],
    simulation_id: int,
    scenario_sim_id: int,
    seed: int,
    mode: str,
    status: str,
    relevant_groups: set[str],
    noise_groups: set[str],
    selected_groups: Iterable[str],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = _selection_metrics(selected_groups, relevant_groups, noise_groups)
    row = {
        "run_id": run_id,
        "result_key": result_key,
        "result_file": str(result_csv_path.resolve()),
        "experiment": str(experiment),
        "method": str(method),
        "status": str(status),
        "mode": str(mode),
        "simulation_id": int(simulation_id),
        "ar_simulation_id": int(scenario_sim_id),
        "scenario_simulation_id": int(scenario_sim_id),
        "seed": int(seed),
        "data_mode": str(DATA_MODE),
        "ar_phi": float(scenario["ar_phi"]) if scenario.get("ar_phi") is not None else np.nan,
        "within_group_corr": (
            float(scenario["within_group_corr"]) if scenario.get("within_group_corr") is not None else np.nan
        ),
        "between_group_corr_factor": (
            float(scenario["between_group_corr_factor"])
            if scenario.get("between_group_corr_factor") is not None
            else np.nan
        ),
        "between_group_corr": (
            float(scenario["between_group_corr"]) if scenario.get("between_group_corr") is not None else np.nan
        ),
        "fdr_q": float(FDR_Q),
        "fdr": float(metrics["fdr"]),
        "power": float(metrics["power"]),
        "num_selected_groups": int(metrics["num_selected_groups"]),
        "num_selected_null_groups": int(metrics["num_selected_null_groups"]),
        "num_selected_relevant_groups": int(metrics["num_selected_relevant_groups"]),
        "num_total_relevant_groups": int(len(relevant_groups)),
        "num_relevant_groups": int(len(relevant_groups)),
        "num_noise_groups": int(len(noise_groups)),
        "num_total_groups": int(len(relevant_groups) + len(noise_groups)),
        "n_samples": int(DATA_LENGTH),
        "n_features_total": int(NUM_FEATURES + NUM_NOISES),
        "n_relevant_columns": int(NUM_FEATURES),
        "n_noise_columns": int(NUM_NOISES),
        "group_size": int(GROUP_SIZE),
        "label_noise_std": float(LABEL_NOISE_STD),
        "error_ar_rho": float(ERROR_AR_RHO) if str(experiment) == "ar_error" else np.nan,
    }
    if extra:
        row.update(extra)
    return row


def _stage_for_second_part(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    *,
    random_state: int,
    verbose: bool,
) -> Dict[str, Any]:
    n_total, d_total = x_df.shape
    if lin.check_low_dim(n_total, d_total, factor=float(LOW_DIM_FACTOR)):
        return {
            "mode": "low_dim_direct",
            "x_stage": x_df,
            "y_stage": np.asarray(y, dtype=np.float64),
            "group_to_columns_stage": group_to_columns,
            "selected_groups_stage1": list(group_to_columns.keys()),
            "stage1_dropped_groups": 0,
            "stage2_n2": int(n_total),
            "stage2_d2": int(d_total),
            "stage2_low_dim_ok": True,
            "screening_log_len": 0,
        }

    x1, y1, x2, y2, _, _ = lin.split_two_stage(
        x_df,
        y,
        split_ratio=float(SPLIT_RATIO),
        gap=int(SPLIT_GAP),
        time_ordered=bool(TIME_ORDERED_SPLIT),
        random_state=int(random_state),
    )
    best, screening_log = lin.run_group_lasso_screening(
        x1=x1,
        y1=y1,
        group_to_columns=group_to_columns,
        n2=len(y2),
        low_dim_factor=float(LOW_DIM_FACTOR),
        lambda_grid=np.asarray(SCREEN_LAMBDA_GRID, dtype=float),
        random_state=int(random_state),
        verbose=bool(verbose),
    )
    selected_groups = list(best["selected_groups"])
    x2_sel, group_to_columns_sel, selected_cols = lin.subset_group_mapping(x2, group_to_columns, selected_groups)
    stage2_low_dim_ok = lin.check_low_dim(len(y2), x2_sel.shape[1], factor=float(LOW_DIM_FACTOR))
    if len(selected_groups) > 0 and not stage2_low_dim_ok:
        raise RuntimeError(
            f"Screening did not make stage2 low-dimensional: d2={x2_sel.shape[1]}, n2={len(y2)}."
        )
    return {
        "mode": "high_dim_screen_then_method",
        "x_stage": x2_sel,
        "y_stage": np.asarray(y2, dtype=np.float64),
        "group_to_columns_stage": group_to_columns_sel,
        "selected_groups_stage1": selected_groups,
        "stage1_dropped_groups": int(len(group_to_columns) - len(selected_groups)),
        "stage2_n2": int(len(y2)),
        "stage2_d2": int(len(selected_cols)),
        "stage2_low_dim_ok": bool(stage2_low_dim_ok),
        "screening_log_len": int(len(screening_log)),
    }


def _run_group_gm_on_stage(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    group_is_relevant: Dict[str, bool],
    *,
    random_state: int,
    ar_gls_rho: float | None,
    verbose: bool,
) -> Dict[str, Any]:
    stage = _stage_for_second_part(
        x_df,
        y,
        group_to_columns,
        random_state=int(random_state),
        verbose=bool(verbose),
    )
    x_stage = stage["x_stage"]
    y_stage = stage["y_stage"]
    group_to_columns_stage = stage["group_to_columns_stage"]
    if ar_gls_rho is not None and len(y_stage) > 0:
        x_stage, y_stage = _whiten_ar1(x_stage, y_stage, rho=float(ar_gls_rho))

    if len(group_to_columns_stage) == 0:
        mirror_stats = {g: 0.0 for g in group_to_columns.keys()}
    else:
        group_is_relevant_stage = {g: bool(group_is_relevant[g]) for g in group_to_columns_stage.keys()}
        mirror_stats_stage, _, _ = lin.compute_group_mirror_stats_ols(
            x_selected_df=x_stage,
            y=y_stage,
            group_to_columns=group_to_columns_stage,
            group_is_relevant=group_is_relevant_stage,
            noise_std=float(NOISE_STD),
            random_state=int(random_state),
            print_cj=False,
            return_cj=False,
            coef_method=str(COEF_METHOD),
            cj_method=str(CJ_METHOD),
        )
        mirror_stats = {g: 0.0 for g in group_to_columns.keys()}
        mirror_stats.update(mirror_stats_stage)

    _, cutoff = lin.compute_cutoff(mirror_stats, q=float(FDR_Q))
    selected_groups = [g for g, stat in mirror_stats.items() if float(stat) >= float(cutoff)]
    stage.update(
        {
            "selected_groups": selected_groups,
            "cutoff": float(cutoff),
            "mirror_stats_positive": int(sum(1 for v in mirror_stats.values() if float(v) > 0.0)),
        }
    )
    return stage


def _run_group_lasso_sparse(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    *,
    num_relevant_groups: int,
    random_state: int,
) -> Dict[str, Any]:
    groups_vec, ordered_groups = lin.build_group_vector(x_df, group_to_columns)
    x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x_df.values)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    max_selected = max(
        int(GROUP_LASSO_MIN_SELECTED),
        int(np.ceil(float(GROUP_LASSO_MAX_SELECTED_MULTIPLIER) * int(num_relevant_groups))),
    )
    log_rows = []
    chosen = None
    fallback = None
    for lam in np.asarray(GROUP_LASSO_LAMBDA_GRID, dtype=float):
        gl = GroupLasso(
            groups=groups_vec,
            group_reg=float(lam),
            l1_reg=0.0,
            n_iter=2000,
            tol=1e-5,
            scale_reg="group_size",
            fit_intercept=True,
            random_state=int(random_state),
            supress_warning=True,
        )
        gl.fit(x_scaled, y_arr)
        selected_groups = lin.selected_groups_from_model(gl, ordered_groups)
        count = len(selected_groups)
        rec = {
            "lambda": float(lam),
            "num_selected_groups": int(count),
            "selected_groups": selected_groups,
        }
        log_rows.append(rec)
        if count > 0:
            dist = abs(count - int(num_relevant_groups))
            if fallback is None or dist < fallback["distance_to_target"]:
                fallback = {**rec, "distance_to_target": int(dist)}
        if int(GROUP_LASSO_MIN_SELECTED) <= count <= max_selected:
            chosen = {**rec, "selection_policy": "first_lambda_with_sparse_upper_bound"}
            break
    if chosen is None:
        if fallback is None:
            chosen = {
                "lambda": float(np.asarray(GROUP_LASSO_LAMBDA_GRID, dtype=float)[-1]),
                "num_selected_groups": 0,
                "selected_groups": [],
                "selection_policy": "fallback_empty",
            }
        else:
            chosen = {**fallback, "selection_policy": "fallback_closest_to_true_support_size"}
    return {
        "selected_groups": chosen["selected_groups"],
        "lambda": float(chosen["lambda"]),
        "selection_policy": str(chosen["selection_policy"]),
        "path_len": int(len(log_rows)),
        "sparse_target_groups": int(num_relevant_groups),
        "sparse_max_selected_groups": int(max_selected),
    }


def _fit_rss_and_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    coef, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if residuals.size > 0:
        rss = float(residuals[0])
    else:
        resid = y - design @ coef
        rss = float(np.dot(resid, resid))
    return max(rss, 0.0), int(rank)


def _partial_f_group_pvalues(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x_df.to_numpy(dtype=np.float64)
    n_obs = int(x.shape[0])
    if len(group_to_columns) == 0:
        return {}

    full_rss, full_rank = _fit_rss_and_rank(x, y_arr)
    df_den = int(n_obs - full_rank)
    if df_den <= 0:
        raise RuntimeError(f"Group BH needs positive residual df, got n={n_obs}, rank={full_rank}.")

    col_to_idx = {c: i for i, c in enumerate(x_df.columns)}
    pvalues: Dict[str, float] = {}
    eps = np.finfo(float).eps
    for group_name, group_cols in group_to_columns.items():
        group_idx = np.array([col_to_idx[c] for c in group_cols if c in col_to_idx], dtype=int)
        if group_idx.size == 0:
            pvalues[group_name] = 1.0
            continue
        keep = np.ones(x.shape[1], dtype=bool)
        keep[group_idx] = False
        reduced_rss, reduced_rank = _fit_rss_and_rank(x[:, keep], y_arr)
        df_num = int(full_rank - reduced_rank)
        if df_num <= 0:
            pvalues[group_name] = 1.0
            continue
        numerator = max(0.0, reduced_rss - full_rss) / float(df_num)
        denominator = full_rss / float(df_den)
        if denominator <= eps:
            pvalues[group_name] = 0.0 if numerator > eps else 1.0
            continue
        f_stat = numerator / denominator
        pvalues[group_name] = float(stats.f.sf(f_stat, df_num, df_den))
    return pvalues


def _bh_select_groups(pvalues: Dict[str, float], q: float) -> tuple[list[str], float]:
    clean = {g: float(p) for g, p in pvalues.items() if np.isfinite(float(p))}
    if len(clean) == 0:
        return [], 0.0
    ordered = sorted(clean.items(), key=lambda item: item[1])
    m = len(ordered)
    cutoff = 0.0
    reject_count = 0
    for k, (_, pvalue) in enumerate(ordered, start=1):
        if float(pvalue) <= float(q) * float(k) / float(m):
            cutoff = float(pvalue)
            reject_count = int(k)
    if reject_count == 0:
        return [], 0.0
    return [g for g, pvalue in clean.items() if float(pvalue) <= cutoff], cutoff


def _run_group_bh_on_stage(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    *,
    random_state: int,
    ar_gls_rho: float | None,
    verbose: bool,
) -> Dict[str, Any]:
    stage = _stage_for_second_part(
        x_df,
        y,
        group_to_columns,
        random_state=int(random_state),
        verbose=bool(verbose),
    )
    x_stage = stage["x_stage"]
    y_stage = stage["y_stage"]
    group_to_columns_stage = stage["group_to_columns_stage"]
    if ar_gls_rho is not None and len(y_stage) > 0:
        x_stage, y_stage = _whiten_ar1(x_stage, y_stage, rho=float(ar_gls_rho))

    if len(group_to_columns_stage) == 0:
        pvalues = {}
        selected_groups: list[str] = []
        cutoff = 0.0
    else:
        pvalues = _partial_f_group_pvalues(x_stage, y_stage, group_to_columns_stage)
        selected_groups, cutoff = _bh_select_groups(pvalues, q=float(FDR_Q))

    stage.update(
        {
            "selected_groups": selected_groups,
            "cutoff": float(cutoff),
            "bh_cutoff_pvalue": float(cutoff),
            "bh_min_pvalue": float(min(pvalues.values())) if pvalues else np.nan,
            "bh_num_tests": int(len(pvalues)),
            "bh_num_rejected": int(len(selected_groups)),
            "bh_uses_gls": bool(ar_gls_rho is not None),
        }
    )
    return stage


def _row_extra_from_stage(stage: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "cutoff",
        "stage1_dropped_groups",
        "stage2_n2",
        "stage2_d2",
        "stage2_low_dim_ok",
        "screening_log_len",
        "mirror_stats_positive",
        "bh_cutoff_pvalue",
        "bh_min_pvalue",
        "bh_num_tests",
        "bh_num_rejected",
        "bh_uses_gls",
        "lambda",
        "selection_policy",
        "path_len",
        "sparse_target_groups",
        "sparse_max_selected_groups",
    ]
    extra = {k: stage[k] for k in keys if k in stage}
    if "selected_groups_stage1" in stage:
        extra["stage1_selected_groups"] = int(len(stage["selected_groups_stage1"]))
    if "stage1_dropped_groups" in stage:
        extra["stage1_dropped_columns"] = int(stage["stage1_dropped_groups"]) * int(GROUP_SIZE)
    return extra


def _settings(run_id: str, result_csv_path: Path, experiment: str) -> Dict[str, Any]:
    return {
        "RUN_ID": run_id,
        "RESULT_CSV_PATH": str(result_csv_path.resolve()),
        "EXPERIMENT": experiment,
        "SEED": int(SEED),
        "NUM_SIMULATIONS": int(NUM_SIMULATIONS),
        "DATA_MODE": str(DATA_MODE),
        "DATA_LENGTH": int(DATA_LENGTH),
        "NUM_FEATURES": int(NUM_FEATURES),
        "NUM_NOISES": int(NUM_NOISES),
        "GROUP_SIZE": int(GROUP_SIZE),
        "WITHIN_GROUP_CORR_VALUES": list(WITHIN_GROUP_CORR_VALUES),
        "BETWEEN_GROUP_CORR_FACTORS": list(BETWEEN_GROUP_CORR_FACTORS),
        "AR_PHI_VALUES": list(AR_PHI_VALUES),
        "BETA_STD_SCALE": float(BETA_STD_SCALE),
        "LABEL_NOISE_STD": float(LABEL_NOISE_STD),
        "FDR_Q": float(FDR_Q),
        "LOW_DIM_FACTOR": float(LOW_DIM_FACTOR),
        "SPLIT_RATIO": float(SPLIT_RATIO),
        "SPLIT_GAP": int(SPLIT_GAP),
        "TIME_ORDERED_SPLIT": bool(TIME_ORDERED_SPLIT),
        "SCREEN_LAMBDA_GRID": np.asarray(SCREEN_LAMBDA_GRID, dtype=float).tolist(),
        "GROUP_LASSO_LAMBDA_GRID": np.asarray(GROUP_LASSO_LAMBDA_GRID, dtype=float).tolist(),
        "GROUP_LASSO_MAX_SELECTED_MULTIPLIER": float(GROUP_LASSO_MAX_SELECTED_MULTIPLIER),
        "ERROR_AR_RHO": float(ERROR_AR_RHO),
        "METHOD_FILTER": list(METHOD_FILTER) if METHOD_FILTER is not None else None,
        "COEF_METHOD": str(COEF_METHOD),
        "CJ_METHOD": str(CJ_METHOD),
        "NOISE_STD": float(NOISE_STD),
    }


def _run_one_scenario(
    *,
    run_id: str,
    result_csv_path: Path,
    experiment: str,
    scenario: Dict[str, Any],
    scenario_idx: int,
    sim_idx: int,
    global_sim_idx: int,
    group_to_columns: Dict[str, list],
    group_is_relevant: Dict[str, bool],
) -> list[Dict[str, Any]]:
    run_seed = int(SEED) + int(scenario_idx) * int(NUM_SIMULATIONS) + int(sim_idx)
    _set_all_seeds(run_seed)
    x_df, y_iid, beta = lin._generate_data_one_run(
        run_seed,
        ar_phi=float(scenario["ar_phi"]) if scenario.get("ar_phi") is not None else 0.0,
        within_group_corr=scenario.get("within_group_corr"),
        between_group_corr=float(scenario.get("between_group_corr") or 0.0),
    )

    y = np.asarray(y_iid, dtype=np.float64)
    if experiment == "ar_error":
        signal = x_df.to_numpy(dtype=np.float64) @ np.asarray(beta, dtype=np.float64).reshape(-1)
        eps = _make_ar1_noise(
            len(signal),
            rho=float(ERROR_AR_RHO),
            sigma=float(LABEL_NOISE_STD),
            seed=run_seed + int(ERROR_SEED_OFFSET),
        )
        y = signal + eps

    relevant_groups = {g for g, v in group_is_relevant.items() if bool(v)}
    noise_groups = {g for g, v in group_is_relevant.items() if not bool(v)}
    num_relevant_groups = len(relevant_groups)

    scenario_tag = (
        f"rhoW{float(scenario['within_group_corr']):.1f}_rhoBfac{float(scenario['between_group_corr_factor']):.1f}"
        if str(DATA_MODE).lower() == "iid_group"
        else f"ar{float(scenario['ar_phi']):.1f}_rhoBfac{float(scenario['between_group_corr_factor']):.1f}"
    )
    rows: list[Dict[str, Any]] = []

    methods = ["group_lasso", "group_bh"] if experiment == "iid_baselines" else [
        "group_gm_ols",
        "group_gm_gls_ar1",
        "group_lasso",
        "group_bh_gls_ar1",
    ]
    if METHOD_FILTER is not None:
        methods = [m for m in methods if m in METHOD_FILTER]

    for method in methods:
        selected_groups: Iterable[str] = []
        status = "ok"
        mode = "full_data"
        extra: Dict[str, Any] = {}
        try:
            with _maybe_suppress_output(enable_output=bool(SHOW_PROGRESS)):
                if method == "group_gm_ols":
                    stage = _run_group_gm_on_stage(
                        x_df,
                        y,
                        group_to_columns,
                        group_is_relevant,
                        random_state=run_seed,
                        ar_gls_rho=None,
                        verbose=bool(SHOW_PROGRESS),
                    )
                    selected_groups = stage["selected_groups"]
                    mode = stage["mode"]
                    extra = _row_extra_from_stage(stage)
                elif method == "group_gm_gls_ar1":
                    stage = _run_group_gm_on_stage(
                        x_df,
                        y,
                        group_to_columns,
                        group_is_relevant,
                        random_state=run_seed,
                        ar_gls_rho=float(ERROR_AR_RHO),
                        verbose=bool(SHOW_PROGRESS),
                    )
                    selected_groups = stage["selected_groups"]
                    mode = stage["mode"]
                    extra = _row_extra_from_stage(stage)
                    extra["error_ar_rho"] = float(ERROR_AR_RHO)
                elif method == "group_lasso":
                    gl_result = _run_group_lasso_sparse(
                        x_df,
                        y,
                        group_to_columns,
                        num_relevant_groups=int(num_relevant_groups),
                        random_state=run_seed,
                    )
                    selected_groups = gl_result["selected_groups"]
                    extra = _row_extra_from_stage(gl_result)
                elif method == "group_bh":
                    bh_result = _run_group_bh_on_stage(
                        x_df,
                        y,
                        group_to_columns,
                        random_state=run_seed,
                        ar_gls_rho=None,
                        verbose=bool(SHOW_PROGRESS),
                    )
                    selected_groups = bh_result["selected_groups"]
                    mode = bh_result["mode"]
                    extra = _row_extra_from_stage(bh_result)
                elif method == "group_bh_gls_ar1":
                    bh_result = _run_group_bh_on_stage(
                        x_df,
                        y,
                        group_to_columns,
                        random_state=run_seed,
                        ar_gls_rho=float(ERROR_AR_RHO),
                        verbose=bool(SHOW_PROGRESS),
                    )
                    selected_groups = bh_result["selected_groups"]
                    mode = bh_result["mode"]
                    extra = _row_extra_from_stage(bh_result)
                    extra["error_ar_rho"] = float(ERROR_AR_RHO)
                else:
                    raise ValueError(f"Unknown method: {method}")
        except Exception as exc:
            status = f"failed: {type(exc).__name__}: {exc}"
            selected_groups = []
            extra = {}

        result_key = f"{run_id}_{experiment}_{scenario_tag}_sim{sim_idx + 1}_{method}"
        rows.append(
            _base_row(
                run_id=run_id,
                result_csv_path=result_csv_path,
                result_key=result_key,
                experiment=experiment,
                method=method,
                scenario=scenario,
                simulation_id=int(global_sim_idx),
                scenario_sim_id=int(sim_idx + 1),
                seed=run_seed,
                mode=mode,
                status=status,
                relevant_groups=relevant_groups,
                noise_groups=noise_groups,
                selected_groups=selected_groups,
                extra={
                    **extra,
                    "beta_nonzero_count": int(np.sum(np.abs(beta) > 0.0)),
                },
            )
        )
    return rows


def run_experiment_job(
    *,
    experiment: str,
    data_mode: str,
    num_simulations: int = NUM_SIMULATIONS,
    num_features: int = NUM_FEATURES,
    num_noises: int = NUM_NOISES,
    group_size: int = GROUP_SIZE,
    seed: int = SEED,
    error_ar_rho: float = ERROR_AR_RHO,
    methods: str | Iterable[str] | None = None,
    result_csv: str | None = None,
    results_dir: str | Path | None = None,
    show_progress: bool = False,
) -> list[Path]:
    global SEED, NUM_SIMULATIONS, DATA_MODE, NUM_FEATURES, NUM_NOISES, GROUP_SIZE, ERROR_AR_RHO
    global RESULT_CSV_PATH, RESULTS_DIR, SHOW_PROGRESS, METHOD_FILTER

    if experiment not in {"iid_baselines", "ar_error", "both"}:
        raise ValueError("experiment must be one of: iid_baselines, ar_error, both.")
    if data_mode not in {"iid_group", "lag_group"}:
        raise ValueError("data_mode must be one of: iid_group, lag_group.")
    SEED = int(seed)
    NUM_SIMULATIONS = int(num_simulations)
    DATA_MODE = str(data_mode)
    NUM_FEATURES = int(num_features)
    NUM_NOISES = int(num_noises)
    GROUP_SIZE = int(group_size)
    ERROR_AR_RHO = float(error_ar_rho)
    RESULT_CSV_PATH = result_csv
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    SHOW_PROGRESS = bool(show_progress)
    METHOD_FILTER = _normalize_methods(methods)

    _sync_linear_globals()
    group_to_columns, group_is_relevant, _ = lin._build_problem_structure()

    output_paths: list[Path] = []
    experiments = ["iid_baselines", "ar_error"] if experiment == "both" else [str(experiment)]
    for experiment in experiments:
        run_id = _make_run_id()
        result_csv_path = _create_run_output_path(run_id)
        output_paths.append(result_csv_path)
        _write_header(result_csv_path, _settings(run_id, result_csv_path, experiment))
        wrote_results_header = False

        global_sim_idx = 0
        for scenario_idx, scenario in enumerate(_build_scenarios()):
            for sim_idx in range(int(NUM_SIMULATIONS)):
                global_sim_idx += 1
                rows = _run_one_scenario(
                    run_id=run_id,
                    result_csv_path=result_csv_path,
                    experiment=experiment,
                    scenario=scenario,
                    scenario_idx=scenario_idx,
                    sim_idx=sim_idx,
                    global_sim_idx=global_sim_idx,
                    group_to_columns=group_to_columns,
                    group_is_relevant=group_is_relevant,
                )
                for row in rows:
                    wrote_results_header = _append_row(result_csv_path, row, wrote_results_header)
                if SHOW_PROGRESS:
                    print(
                        f"[{experiment}/{DATA_MODE}] scenario={scenario_idx + 1}, "
                        f"sim={sim_idx + 1}/{int(NUM_SIMULATIONS)} saved",
                        flush=True,
                    )
        if SHOW_PROGRESS:
            print(f"Saved results to: {result_csv_path.resolve()}", flush=True)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra linear simulations: Group Lasso/Knockoff and AR-error GLS GM.")
    parser.add_argument("--experiment", choices=["iid_baselines", "ar_error", "both"], default="both")
    parser.add_argument("--data-mode", choices=["iid_group", "lag_group"], default=DATA_MODE)
    parser.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    parser.add_argument("--num-features", type=int, default=NUM_FEATURES)
    parser.add_argument("--num-noises", type=int, default=NUM_NOISES)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--error-ar-rho", type=float, default=ERROR_AR_RHO)
    parser.add_argument("--result-csv", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Optional comma-separated subset: group_lasso,group_bh,group_bh_gls_ar1,group_gm_ols,group_gm_gls_ar1.",
    )
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    run_experiment_job(
        experiment=str(args.experiment),
        data_mode=str(args.data_mode),
        num_simulations=int(args.num_simulations),
        num_features=int(args.num_features),
        num_noises=int(args.num_noises),
        group_size=int(args.group_size),
        seed=int(args.seed),
        error_ar_rho=float(args.error_ar_rho),
        methods=args.methods,
        result_csv=args.result_csv,
        results_dir=args.results_dir,
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
