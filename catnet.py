import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from statsmodels.nonparametric.smoothers_lowess import lowess
import random

from src.model_training.train import train_model
from src.model_training.hyperparameter_tuning import find_optimal_params
from src.model_training.model_factory import (
    build_sequence_model,
    get_model_arch_params,
    infer_model_type,
)
from src.FDR.shap_values import calc_shap_values
from src.FDR.kernel_dependence import *


def _standardize_feature_df(data: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Z-score standardization by column.
    Uses population std (ddof=0) and guards zero-variance columns.
    """
    data_f = data.astype(np.float64)
    col_mean = data_f.mean(axis=0)
    col_std = data_f.std(axis=0, ddof=0)
    col_std = col_std.mask(col_std <= eps, 1.0)
    data_std = (data_f - col_mean) / col_std
    return data_std.astype(np.float32)


# def _fit_standardize_params(data: pd.DataFrame, eps: float = 1e-12):
#     data_f = data.astype(np.float64)
#     col_mean = data_f.mean(axis=0)
#     col_std = data_f.std(axis=0, ddof=0)
#     col_std = col_std.mask(col_std <= eps, 1.0)
#     return col_mean, col_std


# def _apply_standardize_params(data: pd.DataFrame, col_mean: pd.Series, col_std: pd.Series) -> pd.DataFrame:
#     data_f = data.astype(np.float64)
#     data_std = (data_f - col_mean) / col_std
#     return data_std.astype(np.float32)


def _local_quadratic_smoothing(
    y,
    x,
    frac: float = 0.3,
    ridge: float = 1e-6,
    lag_batch_size: int = 16,
):
    """
    Local quadratic smoothing (tricube kernel) with torch backend.

    Returns:
        x_out, yhat where x_out is strictly increasing (safe for np.gradient).
        If x is 1D, returns 1D arrays.
        If x is 2D with shape (n, d), returns 2D arrays with shape (n, d).
    """
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_mat = x_arr.reshape(-1, 1)
        squeeze_out = True
    elif x_arr.ndim == 2:
        x_mat = x_arr
        squeeze_out = False
    else:
        raise ValueError("x must be 1D or 2D.")

    if x_mat.shape[0] != y_np.shape[0]:
        raise ValueError("x and y must have the same length on axis 0.")

    n, d = x_mat.shape
    if n == 0:
        empty = np.asarray([], dtype=np.float64)
        return (empty, empty) if squeeze_out else (
            np.zeros((0, d), dtype=np.float64),
            np.zeros((0, d), dtype=np.float64),
        )

    if n < 5:
        order = np.argsort(x_mat, axis=0)
        x_sorted = np.take_along_axis(x_mat, order, axis=0)
        y_col = y_np[:, None]
        y_sorted = np.take_along_axis(y_col, order, axis=0)
        x_out = np.maximum.accumulate(x_sorted + np.arange(n, dtype=np.float64)[:, None] * 1e-12, axis=0)
        if squeeze_out:
            return x_out[:, 0], y_sorted[:, 0]
        return x_out, y_sorted

    frac = float(np.clip(frac, 1e-3, 1.0))
    k = max(5, int(np.ceil(frac * n)))
    lag_batch_size = max(1, int(lag_batch_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_out = np.zeros((n, d), dtype=np.float64)
    yhat_out = np.zeros((n, d), dtype=np.float64)

    for c0 in range(0, d, lag_batch_size):
        c1 = min(d, c0 + lag_batch_size)
        x_chunk = x_mat[:, c0:c1]
        order = np.argsort(x_chunk, axis=0)
        xs = np.take_along_axis(x_chunk, order, axis=0)
        ys = np.take_along_axis(y_np[:, None], order, axis=0)
        x_chunk_out = np.maximum.accumulate(xs + np.arange(n, dtype=np.float64)[:, None] * 1e-12, axis=0)

        xt = torch.as_tensor(xs, dtype=torch.float64, device=device).transpose(0, 1)  # (b, n)
        yt = torch.as_tensor(ys, dtype=torch.float64, device=device).transpose(0, 1)  # (b, n)

        # Batched local-quadratic smoothing over lag dimension.
        d_ = xt.unsqueeze(2) - xt.unsqueeze(1)  # (b, n, n)
        abs_d = torch.abs(d_)
        h = torch.topk(abs_d, k=k, dim=2, largest=False).values[:, :, -1]
        h = torch.clamp(h, min=1e-12)
        u = abs_d / h.unsqueeze(2)
        w = torch.where(u < 1.0, (1.0 - u ** 3) ** 3, torch.zeros_like(u))

        d2 = d_ * d_
        d3 = d2 * d_
        d4 = d2 * d2
        yj = yt.unsqueeze(1).expand(-1, n, -1)

        S0 = torch.sum(w, dim=2)
        S1 = torch.sum(w * d_, dim=2)
        S2 = torch.sum(w * d2, dim=2)
        S3 = torch.sum(w * d3, dim=2)
        S4 = torch.sum(w * d4, dim=2)

        T0 = torch.sum(w * yj, dim=2)
        T1 = torch.sum(w * d_ * yj, dim=2)
        T2 = torch.sum(w * d2 * yj, dim=2)

        A = torch.stack(
            [
                torch.stack([S0, S1, S2], dim=2),
                torch.stack([S1, S2, S3], dim=2),
                torch.stack([S2, S3, S4], dim=2),
            ],
            dim=2,
        )  # (b, n, 3, 3)
        b_vec = torch.stack([T0, T1, T2], dim=2).unsqueeze(-1)  # (b, n, 3, 1)
        eye = torch.eye(3, dtype=torch.float64, device=device).view(1, 1, 3, 3)
        A_reg = A + float(ridge) * eye

        try:
            beta = torch.linalg.solve(A_reg, b_vec).squeeze(-1)
            yhat = beta[:, :, 0].detach().cpu().numpy().transpose(1, 0)  # (n, b)
            if not np.all(np.isfinite(yhat)):
                raise RuntimeError("non-finite yhat in batched smoothing")
            y_chunk_out = yhat
        except RuntimeError:
            # Robust fallback: per-lag LOWESS when local systems are singular/ill-conditioned.
            y_chunk_out = np.zeros((n, c1 - c0), dtype=np.float64)
            for ll in range(c1 - c0):
                lw = lowess(ys[:, ll], xs[:, ll], frac=frac)
                x_lw, y_lw = zip(*lw)
                x_lw = np.asarray(x_lw, dtype=np.float64)
                y_lw = np.asarray(y_lw, dtype=np.float64)
                x_chunk_out[:, ll] = np.maximum.accumulate(x_lw + np.arange(x_lw.shape[0], dtype=np.float64) * 1e-12)
                y_chunk_out[:, ll] = y_lw

        x_out[:, c0:c1] = x_chunk_out
        yhat_out[:, c0:c1] = y_chunk_out

    if squeeze_out:
        return x_out[:, 0], yhat_out[:, 0]
    return x_out, yhat_out


# Disabled (not used in current experiments):
# def _global_linear_slope(
#     y,
#     x,
#     eps: float = 1e-12,
# ):
#     """
#     Fast global slope estimator d y / d x using OLS slope:
#         slope = cov(x, y) / var(x)
#     """
#     x_np = np.asarray(x, dtype=np.float64).reshape(-1)
#     y_np = np.asarray(y, dtype=np.float64).reshape(-1)
#     if x_np.shape[0] != y_np.shape[0]:
#         raise ValueError("x and y must have the same length.")
#     if x_np.shape[0] == 0:
#         return 0.0
#     x_center = x_np - x_np.mean()
#     y_center = y_np - y_np.mean()
#     denom = float(np.dot(x_center, x_center))
#     if denom <= eps:
#         return 0.0
#     numer = float(np.dot(x_center, y_center))
#     return numer / denom


def _local_multivar_linear_lag_vector(
    y,
    x,
    frac: float = 0.3,
    ridge: float = 1e-4,
    query_batch_size: int = 64,
):
    """
    Multivariate local-linear gradient estimator for lag vectors.

    Supports:
      - single channel: x (n, d), y (n,) -> returns (d,)
      - batched channels: x (g, n, d), y (g, n) -> returns (g, d)
    """
    if isinstance(x, torch.Tensor):
        device = x.device
        x_t = x.detach().to(dtype=torch.float32)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    if isinstance(y, torch.Tensor):
        y_t = y.detach().to(device=device, dtype=torch.float32)
    else:
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)

    squeeze_out = False
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(0)
        y_t = y_t.reshape(1, -1)
        squeeze_out = True
    elif x_t.ndim == 3:
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(0)
    else:
        raise ValueError("x must have shape (n, d) or (g, n, d).")

    if y_t.ndim != 2:
        raise ValueError("y must have shape (n,) or (g, n).")
    if x_t.shape[0] != y_t.shape[0] or x_t.shape[1] != y_t.shape[1]:
        raise ValueError("x and y shapes are incompatible.")

    g, n, d = x_t.shape
    if n == 0:
        out_empty = torch.zeros((g, d), dtype=torch.float32, device=device)
        return out_empty[0].detach().cpu().numpy().astype(np.float64) if squeeze_out else out_empty.detach().cpu().numpy().astype(np.float64)

    frac = float(np.clip(frac, 1e-3, 1.0))
    k = max(int(np.ceil(frac * n)), d + 2, 8)
    k = min(k, n)
    query_batch_size = max(1, int(query_batch_size))

    grads = torch.zeros((g, n, d), dtype=torch.float32, device=device)
    eye = torch.eye(d + 1, dtype=torch.float32, device=device).view(1, 1, d + 1, d + 1)

    for i0 in range(0, n, query_batch_size):
        i1 = min(n, i0 + query_batch_size)
        xq = x_t[:, i0:i1, :]  # (g, b, d)

        dist = torch.cdist(xq, x_t, p=2)  # (g, b, n)
        h = torch.topk(dist, k=k, dim=2, largest=False).values[:, :, -1]
        h = torch.clamp(h, min=1e-6)
        u = dist / h.unsqueeze(2)
        w = torch.where(u < 1.0, (1.0 - u ** 3) ** 3, torch.zeros_like(u))  # (g, b, n)

        delta = x_t.unsqueeze(1) - xq.unsqueeze(2)  # (g, b, n, d)
        ones = torch.ones((g, i1 - i0, n, 1), dtype=torch.float32, device=device)
        z = torch.cat([ones, delta], dim=3)  # (g, b, n, d+1)

        zw = z * w.unsqueeze(3)
        a = z.transpose(2, 3) @ zw  # (g, b, d+1, d+1)
        yw = (w * y_t.unsqueeze(1)).unsqueeze(3)  # (g, b, n, 1)
        b_vec = z.transpose(2, 3) @ yw  # (g, b, d+1, 1)
        a_reg = a + float(ridge) * eye

        try:
            beta = torch.linalg.solve(a_reg, b_vec).squeeze(3)  # (g, b, d+1)
        except RuntimeError:
            beta = (torch.linalg.pinv(a_reg) @ b_vec).squeeze(3)

        grads[:, i0:i1, :] = beta[:, :, 1:]

    lag_vec = grads.sum(dim=1).detach().cpu().numpy().astype(np.float64)  # (g, d)
    return lag_vec[0] if squeeze_out else lag_vec


def compute_group_scalar_importance_from_model(
    model: nn.Module,
    X_seq,
    feature_names,
    importance_mode: str = "shap",
    shap_n_permutations: int = 20,
    shap_derivative_mode: str = "local_quadratic",  # local_quadratic | linear_slope | local_multivar_linear
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
    derivative_method: str = "temporal_secant",
    shap_background: torch.Tensor = None,
    shap_bg_size: int = 64,
    shap_eval_batch_size: int = None,
    shap_perm_batch_size: int = None,
    shap_sample_batch_size: int = None,
    shap_path_chunk_size: int = 16,
    shap_random_state: int = 0,
    mirror_magnitude_mode: str = "max_l1",
    mirror_use_balance: bool = False,
    mirror_use_smoothness: bool = False,
    mirror_smooth_lambda: float = 0.1,
):
    """
    Diagnostic helper: compute group-level scalar importance for each feature
    from a trained sequence model.

    This reuses the same group-level importance logic as S_CatNet:
      - SHAP path: group-player SHAP + lag-level derivative vector, then scalar aggregation
      - Derivative path: synchronized perturbation over all lags in one group

    Args:
        model: trained sequence model (output must be scalar per sample).
        X_seq: input sequence tensor/array with shape (n_seq, lookback, p).
        feature_names: list-like of length p.
    Returns:
        dict[str, float]: scalar importance per feature.
    """
    device = next(model.parameters()).device
    importance_mode = str(importance_mode).lower()
    if importance_mode not in {"shap", "derivative"}:
        raise ValueError("importance_mode must be 'shap' or 'derivative'.")
    shap_derivative_mode = str(shap_derivative_mode).lower()
    if shap_derivative_mode not in {"local_quadratic", "linear_slope", "local_multivar_linear"}:
        raise ValueError(
            "shap_derivative_mode must be 'local_quadratic', 'linear_slope', or 'local_multivar_linear'."
        )
    derivative_method = str(derivative_method).lower().strip()
    if derivative_method not in {"temporal_secant", "lag_perturb"}:
        raise ValueError("derivative_method must be 'temporal_secant' or 'lag_perturb'.")
    mirror_mode = str(mirror_magnitude_mode).lower().strip()
    if mirror_mode not in {"max_l1", "max_l2", "sum_l1", "sum_l2", "multiple_l1", "multiple_l2"}:
        raise ValueError(
            "mirror_magnitude_mode must be one of: "
            "max_l1, max_l2, sum_l1, sum_l2, multiple_l1, multiple_l2."
        )
    use_l2 = mirror_mode.endswith("l2")
    mirror_use_balance = bool(mirror_use_balance)
    mirror_use_smoothness = bool(mirror_use_smoothness)
    mirror_smooth_lambda = float(max(0.0, mirror_smooth_lambda))

    if isinstance(X_seq, torch.Tensor):
        X_t = X_seq.detach().to(device=device, dtype=torch.float32)
    else:
        X_t = torch.as_tensor(np.asarray(X_seq), dtype=torch.float32, device=device)
    if X_t.ndim != 3:
        raise ValueError(f"X_seq must have shape (n_seq, lookback, p), got {tuple(X_t.shape)}")

    n_seq, lookback_len, p = X_t.shape
    feature_names = [str(f) for f in feature_names]
    if len(feature_names) != p:
        raise ValueError(f"feature_names length ({len(feature_names)}) must equal p ({p}).")

    out = {}
    X_cpu = None

    if importance_mode == "shap":
        shap_values = calc_shap_values(
            model=model,
            X=X_t,
            feature_names=feature_names,
            background=shap_background,
            bg_size=shap_bg_size,
            n_permutations=shap_n_permutations,
            eval_batch_size=shap_eval_batch_size,
            perm_batch_size=shap_perm_batch_size,
            sample_batch_size=shap_sample_batch_size,
            path_chunk_size=shap_path_chunk_size,
            random_state=shap_random_state,
            player_mode="feature_group",
            return_torch=(shap_derivative_mode != "local_quadratic"),
        )
        for j, feat in enumerate(feature_names):
            shap_raw = shap_values[feat]
            if shap_derivative_mode == "local_quadratic":
                shap_vec = np.asarray(shap_raw, dtype=np.float64).reshape(-1)
                if X_cpu is None:
                    X_cpu = X_t.detach().cpu().numpy()
                x_lag = X_cpu[:, :, j]
                x_s, y_s = _local_quadratic_smoothing(shap_vec, x_lag, frac=0.3)
                lag_vec = np.zeros(lookback_len, dtype=np.float64)
                for lag_idx in range(lookback_len):
                    grad = np.gradient(y_s[:, lag_idx], x_s[:, lag_idx])
                    lag_vec[lag_idx] = float(np.mean(grad))
            elif shap_derivative_mode == "linear_slope":
                x_t = X_t[:, :, j]
                if isinstance(shap_raw, torch.Tensor):
                    y_t = shap_raw.to(device=device, dtype=torch.float32).unsqueeze(1)
                else:
                    y_t = torch.as_tensor(np.asarray(shap_raw), dtype=torch.float32, device=device).reshape(-1, 1)
                x_center = x_t - x_t.mean(dim=0, keepdim=True)
                y_center = y_t - y_t.mean(dim=0, keepdim=True)
                xtx = x_center.transpose(0, 1) @ x_center
                xty = x_center.transpose(0, 1) @ y_center
                ridge = 1e-6 * torch.eye(lookback_len, dtype=torch.float32, device=device)
                beta = torch.linalg.solve(xtx + ridge, xty).squeeze(1)
                lag_vec = beta.detach().cpu().numpy().astype(np.float64)
            else:
                x_t = X_t[:, :, j]
                lag_vec = _local_multivar_linear_lag_vector(
                    y=shap_raw,
                    x=x_t,
                    frac=0.3,
                    ridge=1e-4,
                    query_batch_size=64,
                )
            imp_core = float(np.linalg.norm(lag_vec)) if use_l2 else float(np.sum(np.abs(lag_vec)))
            # For original-model scalar diagnostics, optional Bg/Qg are computed
            # from the same lag vector on both sides, so they remain in [0,1]
            # and keep compatibility with mirror-stat style scaling controls.
            bg_factor, qg_factor, _, _ = _mirror_balance_and_smoothness_factors(
                lag_vec,
                lag_vec,
                use_l2=use_l2,
                smooth_lambda=mirror_smooth_lambda,
            )
            if mirror_use_balance:
                imp_core *= float(bg_factor)
            if mirror_use_smoothness:
                imp_core *= float(qg_factor)
            imp = float(imp_core)
            out[feat] = imp
        return out

    # importance_mode == "derivative"
    for j, feat in enumerate(feature_names):
        if derivative_method == "lag_perturb":
            lag_vec = _lag_perturb_fitted_derivative_importance_group_scalar(
                model=model,
                X=X_t,
                col_idx=j,
                derivative_eps=derivative_eps,
                derivative_step_min=derivative_step_min,
                derivative_step_max=derivative_step_max,
                derivative_n_avg=derivative_n_avg,
                derivative_smooth_noise_std=derivative_smooth_noise_std,
                derivative_parallel_group_size=derivative_parallel_group_size,
            )
        else:
            lag_vec = _stable_finite_difference_importance_group_scalar(
                model=model,
                X=X_t,
                col_idx=j,
                derivative_eps=derivative_eps,
                derivative_step_min=derivative_step_min,
                derivative_step_max=derivative_step_max,
                derivative_n_steps=derivative_n_steps,
                derivative_n_avg=derivative_n_avg,
                derivative_smooth_noise_std=derivative_smooth_noise_std,
                derivative_parallel_group_size=derivative_parallel_group_size,
            )
        imp_core = float(np.linalg.norm(lag_vec)) if use_l2 else float(np.sum(np.abs(lag_vec)))
        bg_factor, qg_factor, _, _ = _mirror_balance_and_smoothness_factors(
            lag_vec,
            lag_vec,
            use_l2=use_l2,
            smooth_lambda=mirror_smooth_lambda,
        )
        if mirror_use_balance:
            imp_core *= float(bg_factor)
        if mirror_use_smoothness:
            imp_core *= float(qg_factor)
        out[feat] = float(imp_core)
    return out


def compute_sequence_fit_diagnostics(
    model: nn.Module,
    X_seq,
    y_seq,
    eval_batch_size: int = 4096,
):
    """
    Compute basic regression-fit diagnostics on one sequence dataset.
    Useful to detect underfitting / near-constant predictions.

    Args:
        model: trained sequence model.
        X_seq: array/tensor with shape (n_seq, lookback, p).
        y_seq: array/tensor with shape (n_seq,) or (n_seq,1).
    Returns:
        dict with mse/mae/r2/corr/std ratio and baseline comparison.
    """
    if isinstance(X_seq, torch.Tensor):
        X_t = X_seq.detach()
    else:
        X_t = torch.as_tensor(np.asarray(X_seq), dtype=torch.float32)

    if isinstance(y_seq, torch.Tensor):
        y_np = y_seq.detach().cpu().numpy().reshape(-1).astype(np.float64)
    else:
        y_np = np.asarray(y_seq, dtype=np.float64).reshape(-1)

    if X_t.ndim != 3:
        raise ValueError(f"X_seq must have shape (n_seq, lookback, p), got {tuple(X_t.shape)}")
    if X_t.shape[0] != y_np.shape[0]:
        raise ValueError(f"X_seq and y_seq must have same n_seq, got {X_t.shape[0]} vs {y_np.shape[0]}")

    model_device = next(model.parameters()).device
    X_t = X_t.to(device=model_device, dtype=torch.float32)
    y_pred = _predict_scalar_in_batches(model, X_t, batch_size=int(eval_batch_size)).detach().cpu().numpy().astype(np.float64)

    eps = 1e-12
    resid = y_pred - y_np
    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))
    y_mean = float(np.mean(y_np))
    baseline_mse = float(np.mean((y_np - y_mean) ** 2))
    r2 = float(1.0 - mse / (baseline_mse + eps))
    corr = float(np.corrcoef(y_pred, y_np)[0, 1]) if y_np.size > 1 else 0.0
    std_pred = float(np.std(y_pred))
    std_y = float(np.std(y_np))
    std_ratio = float(std_pred / (std_y + eps))
    mse_to_baseline = float(mse / (baseline_mse + eps))

    near_constant = bool(std_ratio < 0.05)
    likely_underfit = bool((r2 < 0.05) and (mse_to_baseline > 0.95))

    return {
        "n_seq": int(y_np.shape[0]),
        "mse": mse,
        "mae": mae,
        "baseline_mse_mean_predictor": baseline_mse,
        "mse_over_baseline": mse_to_baseline,
        "r2": r2,
        "corr_pred_target": corr,
        "std_pred": std_pred,
        "std_target": std_y,
        "std_pred_over_target": std_ratio,
        "pred_mean": float(np.mean(y_pred)),
        "target_mean": y_mean,
        "near_constant_prediction": near_constant,
        "likely_underfit": likely_underfit,
    }


def _summarize_importance_vector(vec, eps: float = 1e-12) -> dict:
    """
    Summarize a lag-importance vector into scalar diagnostics.
    Useful for checking null-feature jitter around zero without printing full vectors.
    """
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "abs_mean": np.nan,
            "l1": np.nan,
            "l2": np.nan,
            "pos_frac": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "sign_balance": np.nan,
        }
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "l1": float(np.sum(np.abs(arr))),
        "l2": float(np.linalg.norm(arr)),
        "pos_frac": float(np.mean(arr > 0)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "sign_balance": float((n_pos - n_neg) / max(arr.size, 1)),
    }


def _mirror_magnitude_from_vectors(lag_vec_plus, lag_vec_minus, mode: str = "max_l1"):
    """
    Magnitude part of mirror statistic.
    mode:
      - max_l1: max(||v+||_1, ||v-||_1)
      - max_l2: max(||v+||_2, ||v-||_2)
      - sum_l1: ||v+||_1 + ||v-||_1
      - sum_l2: ||v+||_2 + ||v-||_2
      - multiple_l1: ||v+||_1 * ||v-||_1
      - multiple_l2: ||v+||_2 * ||v-||_2
    """
    v_plus = np.asarray(lag_vec_plus, dtype=np.float64).reshape(-1)
    v_minus = np.asarray(lag_vec_minus, dtype=np.float64).reshape(-1)
    l1_plus = float(np.abs(v_plus).sum())
    l1_minus = float(np.abs(v_minus).sum())
    l2_plus = float(np.linalg.norm(v_plus))
    l2_minus = float(np.linalg.norm(v_minus))

    m = str(mode).lower().strip()
    if m == "max_l1":
        mag = float(max(l1_plus, l1_minus))
    elif m == "max_l2":
        mag = float(max(l2_plus, l2_minus))
    elif m == "sum_l1":
        mag = float(l1_plus + l1_minus)
    elif m == "sum_l2":
        mag = float(l2_plus + l2_minus)
    elif m == "multiple_l1":
        mag = float(l1_plus * l1_minus)
    elif m == "multiple_l2":
        mag = float(l2_plus * l2_minus)
    else:
        raise ValueError(
            "mirror_magnitude_mode must be one of: "
            "max_l1, max_l2, sum_l1, sum_l2, multiple_l1, multiple_l2."
        )
    return mag, l1_plus, l1_minus, l2_plus, l2_minus


def _mirror_balance_and_smoothness_factors(
    lag_vec_plus,
    lag_vec_minus,
    use_l2: bool = True,
    smooth_lambda: float = 0.1,
    eps: float = 1e-12,
):
    """
    Compute optional multiplicative factors for mirror-stat magnitude:
      B_g = 2 * sqrt(a*b) / (a + b), where a,b are plus/minus norms (L1 or L2)
      Q_g = 0.5 * (q(v_plus) + q(v_minus)),
            q(v) = ||v||_2^2 / (||v||_2^2 + lambda * sum_t (v_{t+1}-v_t)^2)
    Both factors are in [0, 1] up to numerical tolerance.
    """
    vp = np.asarray(lag_vec_plus, dtype=np.float64).reshape(-1)
    vm = np.asarray(lag_vec_minus, dtype=np.float64).reshape(-1)
    if bool(use_l2):
        a = float(np.linalg.norm(vp))
        b = float(np.linalg.norm(vm))
    else:
        a = float(np.abs(vp).sum())
        b = float(np.abs(vm).sum())
    b_factor = float((2.0 * np.sqrt(max(a, 0.0) * max(b, 0.0))) / (a + b + eps))
    b_factor = float(np.clip(b_factor, 0.0, 1.0))

    lam = float(max(0.0, smooth_lambda))

    def _q_factor(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        signal = float(np.dot(v, v))
        if v.size <= 1:
            return 1.0
        rough = float(np.dot(np.diff(v), np.diff(v)))
        denom = signal + lam * rough + eps
        qv = float(signal / denom)
        return float(np.clip(qv, 0.0, 1.0))

    q_plus = _q_factor(vp)
    q_minus = _q_factor(vm)
    q_factor = float(0.5 * (q_plus + q_minus))
    return b_factor, q_factor, q_plus, q_minus


def _select_cutoff_from_mirror_stats(mirror_stats: dict, q: float) -> float:
    """
    Select mirror cutoff for target FDR level q.

    Priority:
    1) Return the first positive threshold t with estimated ratio(t) <= q.
    2) If none exists, return the positive t with ratio(t) closest to q.
       Tie-break uses larger t for a more conservative cutoff.
    3) If no positive stats exist, return 0.0.
    """
    stats_arr = np.asarray([float(v) for v in mirror_stats.values()], dtype=np.float64)
    pos_thresholds = np.sort(stats_arr[stats_arr > 0.0])
    if pos_thresholds.size == 0:
        return 0.0

    best_t = float(pos_thresholds[0])
    best_gap = float("inf")

    for t in pos_thresholds:
        num_j_less = int(np.sum(stats_arr <= -t))
        num_j_more = int(np.sum(stats_arr >= t))
        if num_j_more <= 0:
            continue
        ratio = (num_j_less + 1) / num_j_more
        if ratio <= q:
            return float(t)

        gap = abs(ratio - q)
        if (gap < best_gap) or (np.isclose(gap, best_gap) and float(t) > best_t):
            best_gap = float(gap)
            best_t = float(t)

    return float(best_t)


def _select_features_by_cutoff(mirror_stats: dict, cutoff: float) -> dict:
    # Important edge case: when cutoff <= 0 (e.g., no positive mirror stats),
    # selecting with >= would include zero-padded screened-out features.
    if float(cutoff) <= 0.0:
        return {k: v for k, v in mirror_stats.items() if float(v) > 0.0}
    return {k: v for k, v in mirror_stats.items() if float(v) >= float(cutoff)}



"""
Inputs: 
    model: the pre-trained LSTM model
    optimizer: the optimizer used to train the model
    data: dataframe storing the features (normalized)
    y: the response vector
    q: the False Discovery Rate level
Outputs: the set of the index of the significant features
"""
# def CatNet(
#     model: nn.Module,
#     optimizer:optim.Optimizer,
#     data:pd.DataFrame,
#     y,
#     q,
#     lookback = 60,
#     show_progress: bool = False,
#     shap_n_permutations: int = 20,
#     model_type: str = None,
#     model_kwargs: dict = None,
#     standardize_input: bool = True,
#     train_epochs: int = 500,
#     mirror_magnitude_mode: str = "max_l1",
#     mirror_use_balance: bool = False,
#     mirror_use_smoothness: bool = False,
#     mirror_smooth_lambda: float = 0.1,
#     random_state: int = 42,
# ):
#     selected_features, mirror_stats, cutoff = _run_s_catnet_variant_per_group_mirror(
#         model=model,
#         optimizer=optimizer,
#         data=data,
#         y=y,
#         q=q,
#         lookback=lookback,
#         show_progress=show_progress,
#         shap_n_permutations=shap_n_permutations,
#         model_type=model_type,
#         model_kwargs=model_kwargs,
#         importance_mode="shap",
#         standardize_input=standardize_input,
#         train_epochs=train_epochs,
#         auto_tune_hparams=False,
#         mirror_magnitude_mode=mirror_magnitude_mode,
#         mirror_use_balance=mirror_use_balance,
#         mirror_use_smoothness=mirror_use_smoothness,
#         mirror_smooth_lambda=mirror_smooth_lambda,
#         random_state=random_state,
#     )
#     return selected_features, mirror_stats, cutoff

def S_CatNet(
    model: nn.Module,
    optimizer:optim.Optimizer,
    data:pd.DataFrame,
    y,
    q,
    lookback = 60,
    show_progress: bool = False,
    shap_n_permutations: int = 20,
    model_type: str = None,
    model_kwargs: dict = None,
    standardize_input: bool = True,
    importance_mode: str = "shap",
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
    derivative_method: str = "temporal_secant",
    group_method: str = "benchmark",
    group_objective: str = "kernel",
    group_residual_mode: str = "ols",
    group_lambda_reg: float = 1e-2,
    group_max_iter: int = 60,
    group_lr: float = 1e-2,
    group_rank_r: int = None,
    group_enforce_symmetric_delta: bool = False,
    group_delta_norm_cap: float = None,
    group_free_enforce_symmetric: bool = False,
    group_free_g_norm_cap: float = None,
    group_kw_use_pca: bool = False,
    group_kw_pca_components: int = None,
    group_high_dim_mode: str = "auto",  # auto | ridge | standard
    group_ridge_lambda_w: float = None,
    group_ridge_lambda_z: float = None,
    group_ridge_q_eps: float = None,
    group_ridge_g_delta: float = None,
    group_print_objective_every: int = 0,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_max_groups: int = 5,
    shap_derivative_mode: str = "local_quadratic",  # local_quadratic | linear_slope | local_multivar_linear
    train_epochs: int = 500,
    auto_tune_hparams: bool = True,
    tune_n_trials: int = 24,
    tune_n_folds: int = 3,
    tune_val_ratio: float = 0.2,
    tune_max_epochs: int = 200,
    tune_patience: int = 12,
    tune_weight_decay_target: float = 2e-2,
    tune_lambda_wd: float = 5e-4,
    tune_lambda_dropout: float = 1e-3,
    mirror_magnitude_mode: str = "sum_l1",
    mirror_use_balance: bool = False,
    mirror_use_smoothness: bool = False,
    mirror_smooth_lambda: float = 0.1,
    return_importance_details: bool = False,
    random_state: int = 42,
):
    if show_progress:
        print(
            f"S_CatNet route: single-stage variant with high_dim_mode={str(group_high_dim_mode).lower()} "
            "(auto/ridge/standard).",
            flush=True,
        )

    run_result = _run_s_catnet_variant(
        model=model,
        optimizer=optimizer,
        data=data,
        y=y,
        q=q,
        lookback=lookback,
        show_progress=show_progress,
        shap_n_permutations=shap_n_permutations,
        model_type=model_type,
        model_kwargs=model_kwargs,
        importance_mode=importance_mode,
        derivative_method=derivative_method,
        derivative_eps=derivative_eps,
        derivative_step_min=derivative_step_min,
        derivative_step_max=derivative_step_max,
        derivative_n_steps=derivative_n_steps,
        derivative_n_avg=derivative_n_avg,
        derivative_smooth_noise_std=derivative_smooth_noise_std,
        derivative_parallel_group_size=derivative_parallel_group_size,
        standardize_input=standardize_input,
        group_method=group_method,
        group_objective=group_objective,
        group_residual_mode=group_residual_mode,
        group_lambda_reg=group_lambda_reg,
        group_max_iter=group_max_iter,
        group_lr=group_lr,
        group_rank_r=group_rank_r,
        group_enforce_symmetric_delta=group_enforce_symmetric_delta,
        group_delta_norm_cap=group_delta_norm_cap,
        group_free_enforce_symmetric=group_free_enforce_symmetric,
        group_free_g_norm_cap=group_free_g_norm_cap,
        group_kw_use_pca=group_kw_use_pca,
        group_kw_pca_components=group_kw_pca_components,
        group_high_dim_mode=group_high_dim_mode,
        group_ridge_lambda_w=group_ridge_lambda_w,
        group_ridge_lambda_z=group_ridge_lambda_z,
        group_ridge_q_eps=group_ridge_q_eps,
        group_ridge_g_delta=group_ridge_g_delta,
        group_print_objective_every=group_print_objective_every,
        print_kw_diagnostics=print_kw_diagnostics,
        kw_diagnostics_max_groups=kw_diagnostics_max_groups,
        shap_derivative_mode=shap_derivative_mode,
        train_epochs=train_epochs,
        auto_tune_hparams=auto_tune_hparams,
        tune_n_trials=tune_n_trials,
        tune_n_folds=tune_n_folds,
        tune_val_ratio=tune_val_ratio,
        tune_max_epochs=tune_max_epochs,
        tune_patience=tune_patience,
        tune_weight_decay_target=tune_weight_decay_target,
        tune_lambda_wd=tune_lambda_wd,
        tune_lambda_dropout=tune_lambda_dropout,
        mirror_magnitude_mode=mirror_magnitude_mode,
        mirror_use_balance=mirror_use_balance,
        mirror_use_smoothness=mirror_use_smoothness,
        mirror_smooth_lambda=mirror_smooth_lambda,
        return_importance_details=bool(return_importance_details),
        random_state=random_state,
    )
    if not isinstance(importance_mode, str):
        return run_result
    if bool(return_importance_details):
        selected_features, mirror_stats, cutoff, importance_details = run_result
        return selected_features, mirror_stats, cutoff, importance_details
    selected_features, mirror_stats, cutoff = run_result
    return selected_features, mirror_stats, cutoff


# def _oracle_select_from_mirror_stats(
#     mirror_stats: dict,
#     relevant_features: set,
#     q: float,
# ):
#     """
#     Oracle cutoff selection:
#     - uses the same mirror statistics as S_CatNet
#     - knows which features are truly relevant
#     - selects cutoff maximizing power under FDR <= q
#     """
#     if len(relevant_features) == 0:
#         raise ValueError("relevant_features is empty; cannot compute oracle power.")
# 
#     # Keep the same positive-threshold convention as S_CatNet and add +inf (select none)
#     positive_stats = sorted({float(v) for v in mirror_stats.values() if float(v) > 0.0})
#     candidate_cutoffs = positive_stats + [float("inf")]
# 
#     best = {
#         "cutoff": float("inf"),
#         "selected_features": {},
#         "fdr": 0.0,
#         "power": 0.0,
#         "num_selected": 0,
#     }
# 
#     for cutoff in candidate_cutoffs:
#         selected = {k: v for k, v in mirror_stats.items() if float(v) >= cutoff}
#         selected_names = set(selected.keys())
# 
#         num_selected = len(selected_names)
#         tp = len(selected_names & relevant_features)
#         fp = num_selected - tp
# 
#         fdr = (fp / num_selected) if num_selected > 0 else 0.0
#         power = tp / len(relevant_features)
# 
#         # Feasible candidates only
#         if fdr > q:
#             continue
# 
#         # Optimize for power first, then lower FDR, then fewer selected (more conservative tie-break)
#         if (
#             power > best["power"]
#             or (
#                 np.isclose(power, best["power"])
#                 and (fdr < best["fdr"] or (np.isclose(fdr, best["fdr"]) and num_selected < best["num_selected"]))
#             )
#         ):
#             best = {
#                 "cutoff": float(cutoff),
#                 "selected_features": selected,
#                 "fdr": float(fdr),
#                 "power": float(power),
#                 "num_selected": int(num_selected),
#             }
# 
#     return best


# def S_CatNet_with_oracle(
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     data: pd.DataFrame,
#     y,
#     q,
#     lookback=60,
#     show_progress: bool = False,
#     shap_n_permutations: int = 20,
#     model_type: str = None,
#     model_kwargs: dict = None,
#     relevant_prefix: str = "relevant_",
#     standardize_input: bool = True,
#     importance_mode: str = "shap",
#     derivative_eps: float = 1e-3,
#     derivative_step_min: float = None,
#     derivative_step_max: float = None,
#     derivative_n_steps: int = 1,
#     derivative_n_avg: int = 1,
#     derivative_smooth_noise_std: float = 0.0,
#     derivative_parallel_group_size: int = 8,
#     group_method: str = "benchmark",
#     group_objective: str = "kernel",
#     group_residual_mode: str = "ols",
#     group_lambda_reg: float = 1e-2,
#     group_max_iter: int = 60,
#     group_lr: float = 1e-2,
#     group_rank_r: int = None,
#     group_enforce_symmetric_delta: bool = False,
#     group_delta_norm_cap: float = None,
#     group_free_enforce_symmetric: bool = False,
#     group_free_g_norm_cap: float = None,
#     group_kw_use_pca: bool = False,
#     group_kw_pca_components: int = None,
#     group_high_dim_mode: str = "auto",
#     group_ridge_lambda_w: float = None,
#     group_ridge_lambda_z: float = None,
#     group_ridge_q_eps: float = None,
#     group_ridge_g_delta: float = None,
#     group_print_objective_every: int = 0,
#     print_kw_diagnostics: bool = False,
#     kw_diagnostics_max_groups: int = 5,
#     shap_derivative_mode: str = "local_quadratic",
#     train_epochs: int = 500,
#     auto_tune_hparams: bool = True,
#     tune_n_trials: int = 24,
#     tune_n_folds: int = 3,
#     tune_val_ratio: float = 0.2,
#     tune_max_epochs: int = 200,
#     tune_patience: int = 12,
#     tune_weight_decay_target: float = 2e-2,
#     tune_lambda_wd: float = 5e-4,
#     tune_lambda_dropout: float = 1e-3,
#     mirror_magnitude_mode: str = "sum_l1",
#     mirror_use_balance: bool = False,
#     mirror_use_smoothness: bool = False,
#     mirror_smooth_lambda: float = 0.1,
#     random_state: int = 42,
# ):
#     """
#     Run S_CatNet once, then compute an oracle cutoff on the same mirror statistics.
#     """
#     selected_features, mirror_stats, cutoff = S_CatNet(
#         model=model,
#         optimizer=optimizer,
#         data=data,
#         y=y,
#         q=q,
#         lookback=lookback,
#         show_progress=show_progress,
#         shap_n_permutations=shap_n_permutations,
#         model_type=model_type,
#         model_kwargs=model_kwargs,
#         standardize_input=standardize_input,
#         importance_mode=importance_mode,
#         derivative_eps=derivative_eps,
#         derivative_step_min=derivative_step_min,
#         derivative_step_max=derivative_step_max,
#         derivative_n_steps=derivative_n_steps,
#         derivative_n_avg=derivative_n_avg,
#         derivative_smooth_noise_std=derivative_smooth_noise_std,
#         derivative_parallel_group_size=derivative_parallel_group_size,
#         group_method=group_method,
#         group_objective=group_objective,
#         group_residual_mode=group_residual_mode,
#         group_lambda_reg=group_lambda_reg,
#         group_max_iter=group_max_iter,
#         group_lr=group_lr,
#         group_rank_r=group_rank_r,
#         group_enforce_symmetric_delta=group_enforce_symmetric_delta,
#         group_delta_norm_cap=group_delta_norm_cap,
#         group_free_enforce_symmetric=group_free_enforce_symmetric,
#         group_free_g_norm_cap=group_free_g_norm_cap,
#         group_kw_use_pca=group_kw_use_pca,
#         group_kw_pca_components=group_kw_pca_components,
#         group_high_dim_mode=group_high_dim_mode,
#         group_ridge_lambda_w=group_ridge_lambda_w,
#         group_ridge_lambda_z=group_ridge_lambda_z,
#         group_ridge_q_eps=group_ridge_q_eps,
#         group_ridge_g_delta=group_ridge_g_delta,
#         group_print_objective_every=group_print_objective_every,
#         print_kw_diagnostics=print_kw_diagnostics,
#         kw_diagnostics_max_groups=kw_diagnostics_max_groups,
#         shap_derivative_mode=shap_derivative_mode,
#         train_epochs=train_epochs,
#         auto_tune_hparams=auto_tune_hparams,
#         tune_n_trials=tune_n_trials,
#         tune_n_folds=tune_n_folds,
#         tune_val_ratio=tune_val_ratio,
#         tune_max_epochs=tune_max_epochs,
#         tune_patience=tune_patience,
#         tune_weight_decay_target=tune_weight_decay_target,
#         tune_lambda_wd=tune_lambda_wd,
#         tune_lambda_dropout=tune_lambda_dropout,
#         mirror_magnitude_mode=mirror_magnitude_mode,
#         mirror_use_balance=mirror_use_balance,
#         mirror_use_smoothness=mirror_use_smoothness,
#         mirror_smooth_lambda=mirror_smooth_lambda,
#         random_state=random_state,
#     )
# 
#     relevant_features = {str(c) for c in data.columns if str(c).startswith(relevant_prefix)}
#     oracle = _oracle_select_from_mirror_stats(
#         mirror_stats=mirror_stats,
#         relevant_features=relevant_features,
#         q=q,
#     )
# 
#     return {
#         "selected_features": selected_features,
#         "mirror_stats": mirror_stats,
#         "cutoff": cutoff,
#         "oracle_selected_features": oracle["selected_features"],
#         "oracle_cutoff": oracle["cutoff"],
#         "oracle_fdr": oracle["fdr"],
#         "oracle_power": oracle["power"],
#         "oracle_num_selected": oracle["num_selected"],
#     }


def _predict_scalar_in_batches(model: nn.Module, X: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    preds = []
    with torch.inference_mode():
        for k in range(0, X.shape[0], batch_size):
            out = model(X[k:k + batch_size]).reshape(-1)
            preds.append(out)
    return torch.cat(preds, dim=0)


# def _build_group_specific_mirror_input(
#     X_base: torch.Tensor,
#     U_g: torch.Tensor,
#     V_g: torch.Tensor,
#     group_idx: int,
# ) -> torch.Tensor:
#     """
#     Build per-group mirror input:
#       [X_0, ..., X_{j-1}, X_j_plus, X_j_minus, X_{j+1}, ..., X_{p-1}]
#     Shape: (n_seq, lookback, p+1)
#     """
#     n_seq, lookback_len, p = X_base.shape
#     X_new = torch.zeros((n_seq, lookback_len, p + 1), dtype=X_base.dtype, device=X_base.device)
#     if group_idx > 0:
#         X_new[:, :, :group_idx] = X_base[:, :, :group_idx]
#     X_new[:, :, group_idx] = U_g
#     X_new[:, :, group_idx + 1] = V_g
#     if group_idx + 1 < p:
#         X_new[:, :, group_idx + 2:] = X_base[:, :, group_idx + 1:]
#     return X_new


# def _warmstart_group_mirror_model(
#     base_model: nn.Module,
#     mirror_model: nn.Module,
#     model_type: str,
#     group_idx: int,
#     p: int,
# ):
#     """
#     Warm start mirror model from base model.
#     For input-layer weights:
#       - copy non-group features directly
#       - split X_j weight equally to X_j_plus and X_j_minus
#     """
#     mt = str(model_type).lower()
# 
#     # Copy all shape-compatible parameters first.
#     with torch.no_grad():
#         base_sd = base_model.state_dict()
#         new_sd = mirror_model.state_dict()
#         for k in new_sd.keys():
#             if k in base_sd and tuple(new_sd[k].shape) == tuple(base_sd[k].shape):
#                 new_sd[k].copy_(base_sd[k])
# 
#         if mt == "lstm":
#             w_base = base_model.lstm.weight_ih_l0.data
#             w_new = mirror_model.lstm.weight_ih_l0.data
#         elif mt == "gru":
#             w_base = base_model.gru.weight_ih_l0.data
#             w_new = mirror_model.gru.weight_ih_l0.data
#         elif mt == "rnn":
#             w_base = base_model.rnn.weight_ih_l0.data
#             w_new = mirror_model.rnn.weight_ih_l0.data
#         elif mt == "transformer":
#             w_base = base_model.input_proj.weight.data
#             w_new = mirror_model.input_proj.weight.data
#         else:
#             raise ValueError("Unsupported model_type for warm start.")
# 
#         if w_base.shape[1] != p or w_new.shape[1] != p + 1:
#             raise ValueError("Input-layer weight shape mismatch for warm-start mapping.")
# 
#         w_new.zero_()
#         # left block unchanged
#         if group_idx > 0:
#             w_new[:, :group_idx] = w_base[:, :group_idx]
#         # split group j into plus/minus
#         w_new[:, group_idx] = 0.5 * w_base[:, group_idx]
#         w_new[:, group_idx + 1] = 0.5 * w_base[:, group_idx]
#         # right block shifted by +1
#         if group_idx + 1 < p:
#             w_new[:, group_idx + 2:] = w_base[:, group_idx + 1:]


# def _prepare_group_finetune_trainables(
#     mirror_model: nn.Module,
#     model_type: str,
#     group_idx: int,
# ):
#     """
#     Freeze backbone; only tune input projection and final head.
#     Input-layer gradients are masked to update only mirror columns [j, j+1].
#     """
#     mt = str(model_type).lower()
#     for param in mirror_model.parameters():
#         param.requires_grad = False
# 
#     if mt == "lstm":
#         input_weight = mirror_model.lstm.weight_ih_l0
#         input_bias = mirror_model.lstm.bias_ih_l0
#     elif mt == "gru":
#         input_weight = mirror_model.gru.weight_ih_l0
#         input_bias = mirror_model.gru.bias_ih_l0
#     elif mt == "rnn":
#         input_weight = mirror_model.rnn.weight_ih_l0
#         input_bias = mirror_model.rnn.bias_ih_l0
#     elif mt == "transformer":
#         input_weight = mirror_model.input_proj.weight
#         input_bias = mirror_model.input_proj.bias
#     else:
#         raise ValueError("Unsupported model_type for finetune parameter selection.")
# 
#     input_weight.requires_grad = True
#     if input_bias is not None:
#         input_bias.requires_grad = True
#     mirror_model.fc.weight.requires_grad = True
#     mirror_model.fc.bias.requires_grad = True
# 
#     # gradient mask on input columns: only plus/minus columns are trainable
#     with torch.no_grad():
#         mask = torch.zeros_like(input_weight)
#         mask[:, group_idx] = 1.0
#         mask[:, group_idx + 1] = 1.0
# 
#     def _mask_grad(grad, m=mask):
#         return grad * m
# 
#     input_weight.register_hook(_mask_grad)


# def _pair_shap_exact_two_players(
#     model: nn.Module,
#     X_group: torch.Tensor,
#     plus_col_idx: int,
#     minus_col_idx: int,
#     bg_size: int = 64,
#     n_background_draws: int = 1,
#     random_state: int = 0,
#     eval_batch_size: int = 4096,
#     return_drawwise: bool = False,
# ):
#     """
#     Exact 2-player SHAP for (plus, minus), with all other channels fixed at sample values.
#     """
#     device = X_group.device
#     n = X_group.shape[0]
#     rng = np.random.default_rng(int(random_state))
# 
#     bg_size = int(max(1, min(bg_size, n)))
#     n_background_draws = int(max(1, n_background_draws))
# 
#     x11 = X_group
#     phi_plus_sum = torch.zeros(n, dtype=X_group.dtype, device=device)
#     phi_minus_sum = torch.zeros(n, dtype=X_group.dtype, device=device)
#     if return_drawwise:
#         phi_plus_draws = []
#         phi_minus_draws = []
# 
#     for _ in range(n_background_draws):
#         bg_idx = torch.as_tensor(rng.choice(n, size=bg_size, replace=False), device=device, dtype=torch.long)
#         background = X_group.index_select(0, bg_idx)
#         sel = torch.as_tensor(rng.integers(0, bg_size, size=n), device=device, dtype=torch.long)
#         baseline = background.index_select(0, sel)
# 
#         x00 = X_group.clone()
#         x00[:, :, plus_col_idx] = baseline[:, :, plus_col_idx]
#         x00[:, :, minus_col_idx] = baseline[:, :, minus_col_idx]
# 
#         x10 = x00.clone()
#         x10[:, :, plus_col_idx] = x11[:, :, plus_col_idx]
# 
#         x01 = x00.clone()
#         x01[:, :, minus_col_idx] = x11[:, :, minus_col_idx]
# 
#         f00 = _predict_scalar_in_batches(model, x00, batch_size=eval_batch_size)
#         f10 = _predict_scalar_in_batches(model, x10, batch_size=eval_batch_size)
#         f01 = _predict_scalar_in_batches(model, x01, batch_size=eval_batch_size)
#         f11 = _predict_scalar_in_batches(model, x11, batch_size=eval_batch_size)
# 
#         phi_plus_d = 0.5 * ((f10 - f00) + (f11 - f01))
#         phi_minus_d = 0.5 * ((f01 - f00) + (f11 - f10))
#         phi_plus_sum += phi_plus_d
#         phi_minus_sum += phi_minus_d
#         if return_drawwise:
#             phi_plus_draws.append(phi_plus_d)
#             phi_minus_draws.append(phi_minus_d)
# 
#     phi_plus = phi_plus_sum / float(n_background_draws)
#     phi_minus = phi_minus_sum / float(n_background_draws)
#     if return_drawwise:
#         return phi_plus, phi_minus, torch.stack(phi_plus_draws, dim=0), torch.stack(phi_minus_draws, dim=0)
#     return phi_plus, phi_minus


def _stable_finite_difference_importance_group_scalar(
    model: nn.Module,
    X: torch.Tensor,
    col_idx: int,
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
):
    """
    Real-time-series secant derivative vector for one channel.

    For each temporal shift k (k=1..K), we:
      1) use X_ref = X[k:] and X_prev = X[:-k] (aligned by sample index),
      2) keep all other features fixed,
      3) for each lag component l, replace only X_ref[:, l, col_idx] by X_prev[:, l, col_idx],
      4) compute secant slope per sample:
            (f(X_ref) - f(X_alt_{k,l})) / (X_ref[:, l, col_idx] - X_prev[:, l, col_idx]),
      5) average over samples and over k.

    Returns:
        np.ndarray with shape (lookback,), i.e., lag-wise derivative-like vector.

    Notes:
      - derivative_step_min / derivative_step_max are kept only for API compatibility.
      - Effective temporal horizon is K = max(derivative_n_steps, derivative_n_avg).
    """
    n, T, p = X.shape
    del p, derivative_step_min, derivative_step_max  # kept only for backward-compatible signature
    max_k = max(1, int(derivative_n_steps), int(derivative_n_avg))
    max_k = min(max_k, max(0, n - 1))
    lag_batch_size = max(1, int(derivative_parallel_group_size))
    device = X.device
    dx_eps = float(max(abs(float(derivative_eps)), 1e-8))

    if max_k <= 0:
        return np.zeros(T, dtype=np.float64)

    lag_sum = torch.zeros(T, dtype=torch.float64, device=device)
    lag_count = torch.zeros(T, dtype=torch.float64, device=device)

    for k in range(1, max_k + 1):
        X_ref = X[k:]      # (n-k, T, p)
        X_prev = X[:-k]    # (n-k, T, p)
        n_k = int(X_ref.shape[0])
        if n_k <= 0:
            continue

        if derivative_smooth_noise_std > 0:
            # Add the same noise to aligned pairs so the true temporal difference is preserved.
            noise = float(derivative_smooth_noise_std) * torch.randn_like(X_ref)
            X_ref_eff = X_ref + noise
            X_prev_eff = X_prev + noise
        else:
            X_ref_eff = X_ref
            X_prev_eff = X_prev

        y_ref = _predict_scalar_in_batches(model, X_ref_eff).reshape(-1)  # (n_k,)

        for l0 in range(0, T, lag_batch_size):
            l1 = min(T, l0 + lag_batch_size)
            lag_ids = list(range(l0, l1))
            g = len(lag_ids)
            if g <= 0:
                continue

            X_alt = X_ref_eff.unsqueeze(0).repeat(g, 1, 1, 1)  # (g, n_k, T, p)
            for gi, lag_idx in enumerate(lag_ids):
                X_alt[gi, :, lag_idx, col_idx] = X_prev_eff[:, lag_idx, col_idx]

            y_alt = _predict_scalar_in_batches(model, X_alt.reshape(g * n_k, T, X.shape[2])).reshape(g, n_k)

            for gi, lag_idx in enumerate(lag_ids):
                dx = X_ref_eff[:, lag_idx, col_idx] - X_prev_eff[:, lag_idx, col_idx]
                valid = torch.isfinite(dx) & (torch.abs(dx) > dx_eps)
                if not torch.any(valid):
                    continue
                secant = (y_ref[valid] - y_alt[gi, valid]) / dx[valid]
                secant = secant[torch.isfinite(secant)]
                if secant.numel() == 0:
                    continue
                lag_sum[lag_idx] += secant.mean().to(torch.float64)
                lag_count[lag_idx] += 1.0

    lag_vec = torch.zeros(T, dtype=torch.float64, device=device)
    mask = lag_count > 0
    lag_vec[mask] = lag_sum[mask] / lag_count[mask]
    return lag_vec.detach().cpu().numpy().astype(np.float64)


def _lag_perturb_fitted_derivative_importance_group_scalar(
    model: nn.Module,
    X: torch.Tensor,
    col_idx: int,
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_avg: int = 2,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
):
    """
    Fitted-value lag perturbation derivative for one channel.

    For each lag component l, perturb X[:, l, col_idx] by a small +/- step,
    re-evaluate fitted y = f(X), and average the central finite-difference
    slope across samples and perturbation steps. This is intentionally a
    fitted-derivative baseline and does not use the temporal secant alignment.
    """
    n, T, p = X.shape
    del p
    if n <= 0:
        return np.zeros(T, dtype=np.float64)

    device = X.device
    dx_eps = float(max(abs(float(derivative_eps)), 1e-8))
    step_min = dx_eps if derivative_step_min is None else abs(float(derivative_step_min))
    step_max = step_min if derivative_step_max is None else abs(float(derivative_step_max))
    step_min = max(step_min, dx_eps)
    step_max = max(step_max, step_min)
    n_avg = max(1, int(derivative_n_avg))
    if n_avg == 1:
        steps = torch.tensor([step_max], dtype=torch.float32, device=device)
    else:
        steps = torch.linspace(step_min, step_max, n_avg, dtype=torch.float32, device=device)

    lag_batch_size = max(1, int(derivative_parallel_group_size))
    lag_sum = torch.zeros(T, dtype=torch.float64, device=device)
    lag_count = torch.zeros(T, dtype=torch.float64, device=device)

    for step in steps:
        h = float(step.detach().item())
        if h <= 0:
            continue
        if derivative_smooth_noise_std > 0:
            X_eff = X + float(derivative_smooth_noise_std) * torch.randn_like(X)
        else:
            X_eff = X

        for l0 in range(0, T, lag_batch_size):
            l1 = min(T, l0 + lag_batch_size)
            lag_ids = list(range(l0, l1))
            g = len(lag_ids)
            if g <= 0:
                continue

            X_alt = X_eff.unsqueeze(0).repeat(2 * g, 1, 1, 1)
            for gi, lag_idx in enumerate(lag_ids):
                X_alt[2 * gi, :, lag_idx, col_idx] += h
                X_alt[2 * gi + 1, :, lag_idx, col_idx] -= h

            y_alt = _predict_scalar_in_batches(
                model,
                X_alt.reshape(2 * g * n, T, X.shape[2]),
            ).reshape(g, 2, n)
            slopes = (y_alt[:, 0, :] - y_alt[:, 1, :]) / (2.0 * h)
            slopes = torch.where(torch.isfinite(slopes), slopes, torch.zeros_like(slopes))

            for gi, lag_idx in enumerate(lag_ids):
                valid = torch.isfinite(slopes[gi])
                if not torch.any(valid):
                    continue
                lag_sum[lag_idx] += slopes[gi, valid].mean().to(torch.float64)
                lag_count[lag_idx] += 1.0

    lag_vec = torch.zeros(T, dtype=torch.float64, device=device)
    mask = lag_count > 0
    lag_vec[mask] = lag_sum[mask] / lag_count[mask]
    return lag_vec.detach().cpu().numpy().astype(np.float64)


def _compute_variant_outputs_for_mode(
    *,
    new_model: nn.Module,
    mirrored_data_train: torch.Tensor,
    mirrored_seq_t: torch.Tensor,
    features,
    q: float,
    importance_mode: str,
    shap_n_permutations: int,
    shap_derivative_mode: str,
    derivative_method: str,
    derivative_eps: float,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
    mirror_magnitude_mode: str = "max_l1",
    mirror_use_balance: bool = False,
    mirror_use_smoothness: bool = False,
    mirror_smooth_lambda: float = 0.1,
    return_importance_details: bool = False,
    show_progress: bool = False,
    kw_diagnostics_max_groups: int = 5,
):
    importance_mode = str(importance_mode).lower().strip()
    if importance_mode not in {"shap", "derivative"}:
        raise ValueError("importance_mode must be 'shap' or 'derivative'.")
    derivative_method = str(derivative_method).lower().strip()
    if derivative_method not in {"temporal_secant", "lag_perturb"}:
        raise ValueError("derivative_method must be 'temporal_secant' or 'lag_perturb'.")

    p = int(mirrored_seq_t.shape[2] // 2)
    lookback_len = int(mirrored_seq_t.shape[1])
    mirror_stats = {}
    importance_details = {}
    eps = 1e-12
    device = mirrored_seq_t.device

    if importance_mode == "shap":
        if show_progress:
            print(
                f"Variant(group): computing SHAP values (feature_group players), derivative_mode={shap_derivative_mode}",
                flush=True,
            )
        new_features = []
        for feature_name in [str(f) for f in features]:
            new_features.extend([f"{feature_name}_plus", f"{feature_name}_minus"])
        shap_values = calc_shap_values(
            new_model,
            mirrored_data_train,
            new_features,
            n_permutations=shap_n_permutations,
            player_mode="feature_group",
            return_torch=(shap_derivative_mode != "local_quadratic"),
            print_diagnostics=bool(show_progress),
        )
        mirrored_seq_cpu = mirrored_seq_t.detach().cpu().numpy() if shap_derivative_mode == "local_quadratic" else None
    else:
        if show_progress:
            print("Variant: computing fitted derivatives", flush=True)
        shap_values = None
        mirrored_seq_cpu = None

    if show_progress:
        print(f"Variant(group): computing mirror statistics (importance_mode={importance_mode})", flush=True)

    eye_lag = torch.eye(lookback_len, dtype=torch.float32, device=device)
    for j in range(p):
        feature = features[j]
        plus_col_idx = 2 * j
        minus_col_idx = 2 * j + 1

        if importance_mode == "shap":
            # Group-level SHAP returns one sample-wise scalar contribution per group.
            # We derive lag-level importance as a length-L vector.
            Xplus_shap = shap_values[f"{feature}_plus"]
            Xminus_shap = shap_values[f"{feature}_minus"]
            if shap_derivative_mode == "local_quadratic":
                Xplus_shap = np.asarray(Xplus_shap, dtype=np.float64).reshape(-1)
                Xminus_shap = np.asarray(Xminus_shap, dtype=np.float64).reshape(-1)
                x_plus_lag = mirrored_seq_cpu[:, :, plus_col_idx]
                x_minus_lag = mirrored_seq_cpu[:, :, minus_col_idx]
                x_plus_s, y_plus_s = _local_quadratic_smoothing(Xplus_shap, x_plus_lag, frac=0.3)
                x_minus_s, y_minus_s = _local_quadratic_smoothing(Xminus_shap, x_minus_lag, frac=0.3)
                lag_vec_plus = np.zeros(lookback_len, dtype=np.float64)
                lag_vec_minus = np.zeros(lookback_len, dtype=np.float64)
                for lag_idx in range(lookback_len):
                    grad_plus = np.gradient(y_plus_s[:, lag_idx], x_plus_s[:, lag_idx])
                    grad_minus = np.gradient(y_minus_s[:, lag_idx], x_minus_s[:, lag_idx])
                    lag_vec_plus[lag_idx] = float(np.mean(grad_plus))
                    lag_vec_minus[lag_idx] = float(np.mean(grad_minus))
            elif shap_derivative_mode == "linear_slope":
                x_plus_t = mirrored_seq_t[:, :, plus_col_idx]
                x_minus_t = mirrored_seq_t[:, :, minus_col_idx]
                x_pair = torch.stack([x_plus_t, x_minus_t], dim=0)  # (2, n, L)
                if isinstance(Xplus_shap, torch.Tensor):
                    y_plus_t = Xplus_shap.to(device=device, dtype=torch.float32).reshape(-1)
                else:
                    y_plus_t = torch.as_tensor(np.asarray(Xplus_shap), dtype=torch.float32, device=device).reshape(-1)
                if isinstance(Xminus_shap, torch.Tensor):
                    y_minus_t = Xminus_shap.to(device=device, dtype=torch.float32).reshape(-1)
                else:
                    y_minus_t = torch.as_tensor(np.asarray(Xminus_shap), dtype=torch.float32, device=device).reshape(-1)
                y_pair = torch.stack([y_plus_t, y_minus_t], dim=0)  # (2, n)

                x_center = x_pair - x_pair.mean(dim=1, keepdim=True)
                y_center = y_pair.unsqueeze(2) - y_pair.mean(dim=1, keepdim=True).unsqueeze(2)
                xtx = x_center.transpose(1, 2) @ x_center  # (2, L, L)
                xty = x_center.transpose(1, 2) @ y_center  # (2, L, 1)
                ridge_scale_pair = torch.clamp(
                    torch.diagonal(xtx, dim1=1, dim2=2).mean(dim=1),
                    min=1e-12,
                )
                ridge_pair = (1e-6 * ridge_scale_pair).view(2, 1, 1) * eye_lag.unsqueeze(0)
                beta_pair = torch.linalg.solve(xtx + ridge_pair, xty).squeeze(2)  # (2, L)
                if show_progress and j < int(max(0, kw_diagnostics_max_groups)):
                    print(
                        f"[group={feature}] shap-derivative ridge(linear_slope): "
                        f"plus={float((1e-6 * ridge_scale_pair[0]).detach().item()):.6g}, "
                        f"minus={float((1e-6 * ridge_scale_pair[1]).detach().item()):.6g}",
                        flush=True,
                    )

                lag_vec_plus = beta_pair[0].detach().cpu().numpy().astype(np.float64)
                lag_vec_minus = beta_pair[1].detach().cpu().numpy().astype(np.float64)
            else:
                x_plus_t = mirrored_seq_t[:, :, plus_col_idx]
                x_minus_t = mirrored_seq_t[:, :, minus_col_idx]
                x_pair = torch.stack([x_plus_t, x_minus_t], dim=0)  # (2, n, L)
                if isinstance(Xplus_shap, torch.Tensor):
                    y_plus_t = Xplus_shap.to(device=device, dtype=torch.float32).reshape(-1)
                else:
                    y_plus_t = torch.as_tensor(np.asarray(Xplus_shap), dtype=torch.float32, device=device).reshape(-1)
                if isinstance(Xminus_shap, torch.Tensor):
                    y_minus_t = Xminus_shap.to(device=device, dtype=torch.float32).reshape(-1)
                else:
                    y_minus_t = torch.as_tensor(np.asarray(Xminus_shap), dtype=torch.float32, device=device).reshape(-1)
                y_pair = torch.stack([y_plus_t, y_minus_t], dim=0)  # (2, n)
                x_pair_center = x_pair - x_pair.mean(dim=1, keepdim=True)
                ridge_local = float(1e-4 * torch.clamp((x_pair_center * x_pair_center).mean(), min=1e-12).detach().item())
                lag_pair = _local_multivar_linear_lag_vector(
                    y=y_pair,
                    x=x_pair,
                    frac=0.3,
                    ridge=ridge_local,
                    query_batch_size=64,
                )
                if show_progress and j < int(max(0, kw_diagnostics_max_groups)):
                    print(
                        f"[group={feature}] shap-derivative ridge(local_multivar_linear): {ridge_local:.6g}",
                        flush=True,
                    )
                lag_vec_plus = lag_pair[0]
                lag_vec_minus = lag_pair[1]
        else:
            derivative_fn = (
                _lag_perturb_fitted_derivative_importance_group_scalar
                if derivative_method == "lag_perturb"
                else _stable_finite_difference_importance_group_scalar
            )
            shared_derivative_kwargs = dict(
                model=new_model,
                X=mirrored_data_train,
                derivative_eps=derivative_eps,
                derivative_step_min=derivative_step_min,
                derivative_step_max=derivative_step_max,
                derivative_n_avg=derivative_n_avg,
                derivative_smooth_noise_std=derivative_smooth_noise_std,
                derivative_parallel_group_size=derivative_parallel_group_size,
            )
            if derivative_method != "lag_perturb":
                shared_derivative_kwargs["derivative_n_steps"] = derivative_n_steps
            lag_vec_plus = derivative_fn(col_idx=plus_col_idx, **shared_derivative_kwargs)
            lag_vec_minus = derivative_fn(col_idx=minus_col_idx, **shared_derivative_kwargs)
            lag_vec_plus = np.asarray(lag_vec_plus, dtype=np.float64).reshape(-1)
            lag_vec_minus = np.asarray(lag_vec_minus, dtype=np.float64).reshape(-1)

        signed_product = float(np.dot(lag_vec_plus, lag_vec_minus))
        sign_function = 1.0 if signed_product >= 0.0 else -1.0
        mirror_abs_core, l1_plus, l1_minus, _, _ = _mirror_magnitude_from_vectors(
            lag_vec_plus,
            lag_vec_minus,
            mode=mirror_magnitude_mode,
        )
        bg_factor, qg_factor, qg_plus, qg_minus = _mirror_balance_and_smoothness_factors(
            lag_vec_plus,
            lag_vec_minus,
            use_l2=mirror_magnitude_mode.endswith("l2"),
            smooth_lambda=mirror_smooth_lambda,
            eps=eps,
        )
        mirror_abs = float(mirror_abs_core)
        if mirror_use_balance:
            mirror_abs *= float(bg_factor)
        if mirror_use_smoothness:
            mirror_abs *= float(qg_factor)
        mirror_stat = float(sign_function * mirror_abs)
        feature_name = str(feature)
        mirror_stats[feature_name] = float(mirror_stat)
        if return_importance_details:
            plus_summary = _summarize_importance_vector(lag_vec_plus)
            minus_summary = _summarize_importance_vector(lag_vec_minus)
            importance_details[feature_name] = {
                "L_plus": float(l1_plus),
                "L_minus": float(l1_minus),
                "sign_pm": float(sign_function),
                "signed_product": signed_product,
                "mirror_abs_core": float(mirror_abs_core),
                "mirror_bg_factor": float(bg_factor),
                "mirror_qg_factor": float(qg_factor),
                "mirror_qg_plus": float(qg_plus),
                "mirror_qg_minus": float(qg_minus),
                "mirror_use_balance": bool(mirror_use_balance),
                "mirror_use_smoothness": bool(mirror_use_smoothness),
                "mirror_smooth_lambda": float(mirror_smooth_lambda),
                "mirror_abs_final": float(mirror_abs),
                "mirror_magnitude_mode": mirror_magnitude_mode,
                "plus_lag_vector": np.asarray(lag_vec_plus, dtype=np.float64).tolist(),
                "minus_lag_vector": np.asarray(lag_vec_minus, dtype=np.float64).tolist(),
                "plus_mean": plus_summary["mean"],
                "plus_std": plus_summary["std"],
                "plus_abs_mean": plus_summary["abs_mean"],
                "plus_l1": plus_summary["l1"],
                "plus_l2": plus_summary["l2"],
                "plus_pos_frac": plus_summary["pos_frac"],
                "plus_p10": plus_summary["p10"],
                "plus_p50": plus_summary["p50"],
                "plus_p90": plus_summary["p90"],
                "plus_sign_balance": plus_summary["sign_balance"],
                "minus_mean": minus_summary["mean"],
                "minus_std": minus_summary["std"],
                "minus_abs_mean": minus_summary["abs_mean"],
                "minus_l1": minus_summary["l1"],
                "minus_l2": minus_summary["l2"],
                "minus_pos_frac": minus_summary["pos_frac"],
                "minus_p10": minus_summary["p10"],
                "minus_p50": minus_summary["p50"],
                "minus_p90": minus_summary["p90"],
                "minus_sign_balance": minus_summary["sign_balance"],
            }

    cutoff = _select_cutoff_from_mirror_stats(mirror_stats, q)
    selected_features = _select_features_by_cutoff(mirror_stats, cutoff)
    return selected_features, mirror_stats, float(cutoff), importance_details


# def _compute_fdr_power_from_selected(selected_features: dict, relevant_features: set):
#     selected_names = set(selected_features.keys())
#     num_selected = len(selected_names)
#     tp = len(selected_names & relevant_features)
#     fp = num_selected - tp
#     fdr = (fp / num_selected) if num_selected > 0 else 0.0
#     power = (tp / len(relevant_features)) if len(relevant_features) > 0 else 0.0
#     return {
#         "fdr": float(fdr),
#         "power": float(power),
#         "num_selected": int(num_selected),
#         "num_tp": int(tp),
#         "num_fp": int(fp),
#     }


def _run_s_catnet_variant(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: pd.DataFrame,
    y,
    q: float,
    lookback: int = 60,
    show_progress: bool = False,
    shap_n_permutations: int = 20,
    model_type: str = None,
    model_kwargs: dict = None,
    cj_lookback: int = None,
    importance_mode: str = "shap",  # "shap" or "derivative"
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
    derivative_method: str = "temporal_secant",  # temporal_secant | lag_perturb
    standardize_input: bool = True,
    group_method: str = "benchmark",  # benchmark | scheme_c_free
    group_objective: str = "kernel",  # kernel | linear_surrogate
    group_residual_mode: str = "ols", # ols | gls
    group_lambda_reg: float = 1e-2,
    group_max_iter: int = 60,
    group_lr: float = 1e-2,
    group_rank_r: int = None,
    group_enforce_symmetric_delta: bool = False,
    group_delta_norm_cap: float = None,
    group_free_enforce_symmetric: bool = False,
    group_free_g_norm_cap: float = None,
    group_kw_use_pca: bool = False,
    group_kw_pca_components: int = None,
    group_high_dim_mode: str = "auto",
    group_ridge_lambda_w: float = None,
    group_ridge_lambda_z: float = None,
    group_ridge_q_eps: float = None,
    group_ridge_g_delta: float = None,
    group_print_objective_every: int = 0,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_max_groups: int = 5,
    shap_derivative_mode: str = "local_quadratic",  # local_quadratic | linear_slope | local_multivar_linear
    return_importance_details: bool = False,
    train_epochs: int = 500,
    auto_tune_hparams: bool = True,
    tune_n_trials: int = 24,
    tune_n_folds: int = 3,
    tune_val_ratio: float = 0.2,
    tune_max_epochs: int = 200,
    tune_patience: int = 12,
    tune_weight_decay_target: float = 2e-2,
    tune_lambda_wd: float = 5e-4,
    tune_lambda_dropout: float = 1e-3,
    mirror_magnitude_mode: str = "max_l1",
    mirror_use_balance: bool = False,
    mirror_use_smoothness: bool = False,
    mirror_smooth_lambda: float = 0.1,
    random_state: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(random_state)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_type = infer_model_type(model) if model_type is None else str(model_type).lower()
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if isinstance(importance_mode, str):
        importance_modes = (str(importance_mode).lower(),)
        multi_mode_request = False
    else:
        importance_modes = tuple(dict.fromkeys(str(m).lower() for m in importance_mode))
        multi_mode_request = True
    if len(importance_modes) == 0:
        raise ValueError("importance_mode must contain at least one mode.")
    for mode in importance_modes:
        if mode not in {"shap", "derivative"}:
            raise ValueError("importance_mode must be 'shap' or 'derivative'.")
    shap_derivative_mode = str(shap_derivative_mode).lower()
    if shap_derivative_mode not in {"local_quadratic", "linear_slope", "local_multivar_linear"}:
        raise ValueError(
            "shap_derivative_mode must be 'local_quadratic', 'linear_slope', or 'local_multivar_linear'."
        )
    derivative_method = str(derivative_method).lower().strip()
    if derivative_method not in {"temporal_secant", "lag_perturb"}:
        raise ValueError("derivative_method must be 'temporal_secant' or 'lag_perturb'.")
    mirror_magnitude_mode = str(mirror_magnitude_mode).lower().strip()
    if mirror_magnitude_mode not in {"max_l1", "max_l2", "sum_l1", "sum_l2", "multiple_l1", "multiple_l2"}:
        raise ValueError(
            "mirror_magnitude_mode must be one of: "
            "max_l1, max_l2, sum_l1, sum_l2, multiple_l1, multiple_l2."
        )
    mirror_use_balance = bool(mirror_use_balance)
    mirror_use_smoothness = bool(mirror_use_smoothness)
    mirror_smooth_lambda = float(max(0.0, mirror_smooth_lambda))
    # Keep original high-dimensional handling:
    # group_high_dim_mode can be auto / ridge / standard.
    group_high_dim_mode = str(group_high_dim_mode).lower()
    if group_high_dim_mode not in {"auto", "ridge", "standard"}:
        raise ValueError("group_high_dim_mode must be one of: auto, ridge, standard.")
    if standardize_input:
        data = _standardize_feature_df(data)
        if show_progress:
            print("Variant: standardized input features (z-score) before c_j/model steps", flush=True)

    features = data.columns
    n, p = data.shape

    if cj_lookback is not None and int(cj_lookback) != int(lookback) and show_progress:
        print(
            f"Variant(group): cj_lookback={cj_lookback} is deprecated in group mode; using lookback={lookback}.",
            flush=True,
        )

    def load_data_X(data_X: pd.DataFrame, lookback_days=lookback):
        data_X_new = []
        for i in range(lookback_days, len(data_X)):
            data_X_new.append(data_X.iloc[i - lookback_days:i].values)
        return np.array(data_X_new)

    # Build sequence tensor first: each group is one feature's lag block (n_seq x lookback)
    data_seq_np = load_data_X(data, lookback).astype(np.float32)  # (n_seq, lookback, p)
    data_seq_t = torch.from_numpy(data_seq_np).to(device=device, dtype=torch.float32)
    n_seq, lookback_len, _ = data_seq_t.shape
    noise_seq_t = torch.randn_like(data_seq_t)
    y_seq_np = np.asarray(y[lookback:], dtype=np.float32).reshape(-1)
    if y_seq_np.shape[0] != n_seq:
        raise ValueError(
            f"Length mismatch after lookback: y_seq={y_seq_np.shape[0]} vs sequence samples={n_seq}."
        )

    tuned_params = None
    if bool(auto_tune_hparams):
        if show_progress:
            print(
                f"Variant(group): tuning lr/weight_decay/dropout via time-series CV "
                f"(single-GPU sequential trials, n_trials={int(tune_n_trials)})",
                flush=True,
            )
            print(
                "Variant(group): tuning search params=['lr','weight_decay','dropout'] "
                f"with ranges: lr=[1e-4,5e-3], weight_decay=[1e-3,3e-1], dropout=[0.0,0.2], "
                f"max_epochs={int(tune_max_epochs)}, patience={int(tune_patience)}",
                flush=True,
            )
        base_arch_params = get_model_arch_params(model, model_type=model_type)
        tune_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        tune_model_kwargs = {k: v for k, v in tune_model_kwargs.items() if k not in {"hidden_dim", "num_layers", "dropout"}}
        try:
            tuned_params = find_optimal_params(
                X_train_torch=data_seq_t,
                y_train_torch=torch.from_numpy(y_seq_np).to(device=device, dtype=torch.float32),
                model_type=model_type,
                model_kwargs=tune_model_kwargs,
                fixed_hidden_dim=int(base_arch_params["hidden_dim"]),
                fixed_num_layers=int(base_arch_params["num_layers"]),
                n_trials=tune_n_trials,
                n_folds=tune_n_folds,
                val_ratio=tune_val_ratio,
                cv_gap=int(lookback),
                max_epochs=tune_max_epochs,
                patience=tune_patience,
                wd_target=tune_weight_decay_target,
                lambda_wd=tune_lambda_wd,
                lambda_dropout=tune_lambda_dropout,
                random_state=seed,
            )
            if show_progress:
                print(
                    "Variant(group): tuning result "
                    f"best_params={{'lr': {float(tuned_params['lr']):.6g}, "
                    f"'weight_decay': {float(tuned_params['weight_decay']):.6g}, "
                    f"'dropout': {float(tuned_params['dropout']):.6g}}}, "
                    f"cv_mse={float(tuned_params.get('cv_mse', float('nan'))):.6g}, "
                    f"objective={float(tuned_params.get('objective', float('nan'))):.6g}",
                    flush=True,
                )
                trial_results = tuned_params.get("trial_results", [])
                if len(trial_results) > 0:
                    print("Variant(group): tuning trial details (all searched params):", flush=True)
                    for tr in trial_results:
                        print(
                            f"  trial={int(tr.get('trial_index', -1)):02d} "
                            f"lr={float(tr['lr']):.6g}, "
                            f"weight_decay={float(tr['weight_decay']):.6g}, "
                            f"dropout={float(tr['dropout']):.6g}, "
                            f"cv_mse={float(tr.get('cv_mse', float('nan'))):.6g}, "
                            f"wd_penalty={float(tr.get('wd_penalty', float('nan'))):.6g}, "
                            f"dropout_penalty={float(tr.get('dropout_penalty', float('nan'))):.6g}, "
                            f"objective={float(tr.get('objective', float('nan'))):.6g}",
                            flush=True,
                        )
        except Exception as exc:
            tuned_params = None
            if show_progress:
                print(
                    f"Variant(group): tuning skipped due to error; fallback to provided optimizer/model params. "
                    f"error={exc}",
                    flush=True,
                )

    if show_progress:
        print(f"Variant(group): building mirrored lag-blocks for {p} features", flush=True)

    mirrored_seq_t = torch.zeros((n_seq, lookback_len, 2 * p), dtype=torch.float32, device=device)
    group_info = {}

    for j in range(p):
        feature_name = str(features[j])
        Xg = data_seq_t[:, :, j]                          # (n_seq, lookback)
        Zg = noise_seq_t[:, :, j]                         # (n_seq, lookback)

        if p > 1:
            mask = [k for k in range(p) if k != j]
            W = data_seq_t[:, :, mask].reshape(n_seq, -1)  # (n_seq, lookback*(p-1))
        else:
            W = torch.zeros((n_seq, 0), dtype=torch.float32, device=device)

        g_prefix = f"[group={feature_name}]"
        G_j, g_info = find_Gj(
            Xg=Xg,
            W=W,
            method=group_method,
            Zg=Zg,
            objective=group_objective,
            residual_mode=group_residual_mode,
            lambda_reg=group_lambda_reg,
            max_iter=group_max_iter,
            lr=group_lr,
            rank_r=group_rank_r,
            enforce_symmetric_delta=group_enforce_symmetric_delta,
            delta_norm_cap=group_delta_norm_cap,
            free_enforce_symmetric=group_free_enforce_symmetric,
            free_g_norm_cap=group_free_g_norm_cap,
            kw_use_pca=group_kw_use_pca,
            kw_pca_components=group_kw_pca_components,
            high_dim_mode=group_high_dim_mode,
            ridge_lambda_w=group_ridge_lambda_w,
            ridge_lambda_z=group_ridge_lambda_z,
            ridge_q_eps=group_ridge_q_eps,
            ridge_g_delta=group_ridge_g_delta,
            print_objective_every=group_print_objective_every,
            objective_print_prefix=g_prefix,
            random_state=seed + j,
            print_kw_diagnostics=bool(print_kw_diagnostics and (j < kw_diagnostics_max_groups)),
            kw_diagnostics_prefix=g_prefix,
            return_info=True,
            return_torch=True,
        )
        group_info[feature_name] = g_info
        if show_progress and j < int(max(0, kw_diagnostics_max_groups)):
            ridge_info = g_info.get("ridge_params", None)
            if bool(g_info.get("use_high_dim_ridge", False)) and isinstance(ridge_info, dict):
                print(
                    f"{g_prefix} ridge(auto-resolved): "
                    f"lambda_w={float(ridge_info.get('ridge_lambda_w', float('nan'))):.6g}, "
                    f"lambda_z={float(ridge_info.get('ridge_lambda_z', float('nan'))):.6g}, "
                    f"q_eps={float(ridge_info.get('ridge_q_eps', float('nan'))):.6g}, "
                    f"g_delta={float(ridge_info.get('ridge_g_delta', float('nan'))):.6g}",
                    flush=True,
                )

        U_g, V_g = build_group_mirror_pair(
            Xg=Xg,
            W=W,
            Zg=Zg,
            G=G_j,
            residual_mode=group_residual_mode,
            high_dim_mode=group_high_dim_mode,
            ridge_lambda_w=group_ridge_lambda_w,
            ridge_lambda_z=group_ridge_lambda_z,
            ridge_q_eps=group_ridge_q_eps,
            return_torch=True,
        )

        mirrored_seq_t[:, :, 2 * j] = U_g
        mirrored_seq_t[:, :, 2 * j + 1] = V_g

        if show_progress:
            progress_bar_len = 30
            progress_done = int(progress_bar_len * (j + 1) / p)
            progress_bar = "#" * progress_done + "-" * (progress_bar_len - progress_done)
            progress_end = "\n" if (j + 1) == p else "\r"
            print(f"Variant(group) Progress [{progress_bar}] {j + 1}/{p}", end=progress_end, flush=True)

    mirrored_data_train = mirrored_seq_t
    y_train = torch.from_numpy(y_seq_np).to(device=device, dtype=torch.float32)

    arch_params = get_model_arch_params(model, model_type=model_type)
    extra_arch_kwargs = {k: v for k, v in arch_params.items() if k not in {"hidden_dim", "num_layers", "dropout"}}
    extra_arch_kwargs.update(model_kwargs)
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    if tuned_params is not None:
        arch_params["dropout"] = float(tuned_params["dropout"])
        lr = float(tuned_params["lr"])
        weight_decay = float(tuned_params["weight_decay"])

    new_model = build_sequence_model(
        model_type=model_type,
        input_dim=2 * p,
        hidden_dim=arch_params["hidden_dim"],
        num_layers=arch_params["num_layers"],
        output_dim=1,
        dropout=arch_params["dropout"],
        **extra_arch_kwargs,
    )
    new_model.to(device)
    criterion = nn.MSELoss()
    new_optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=weight_decay)

    if show_progress:
        print(f"Variant: training mirrored {model_type} model", flush=True)
    train_model(new_model, mirrored_data_train, y_train, train_epochs, criterion, new_optimizer)
    new_model.eval()

    rng_state_torch_cpu = torch.get_rng_state()
    rng_state_torch_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng_state_numpy = np.random.get_state()
    rng_state_python = random.getstate()

    mode_outputs = {}
    for mode in importance_modes:
        # Keep per-mode RNG aligned so shap/derivative each start from identical
        # post-training randomness state as in separate runs.
        torch.set_rng_state(rng_state_torch_cpu)
        if rng_state_torch_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_torch_cuda)
        np.random.set_state(rng_state_numpy)
        random.setstate(rng_state_python)

        selected_features, mirror_stats, cutoff, importance_details = _compute_variant_outputs_for_mode(
            new_model=new_model,
            mirrored_data_train=mirrored_data_train,
            mirrored_seq_t=mirrored_seq_t,
            features=features,
            q=q,
            importance_mode=mode,
            shap_n_permutations=shap_n_permutations,
            shap_derivative_mode=shap_derivative_mode,
            derivative_method=derivative_method,
            derivative_eps=derivative_eps,
            derivative_step_min=derivative_step_min,
            derivative_step_max=derivative_step_max,
            derivative_n_steps=derivative_n_steps,
            derivative_n_avg=derivative_n_avg,
            derivative_smooth_noise_std=derivative_smooth_noise_std,
            derivative_parallel_group_size=derivative_parallel_group_size,
            mirror_magnitude_mode=mirror_magnitude_mode,
            mirror_use_balance=mirror_use_balance,
            mirror_use_smoothness=mirror_use_smoothness,
            mirror_smooth_lambda=mirror_smooth_lambda,
            return_importance_details=bool(return_importance_details),
            show_progress=show_progress,
            kw_diagnostics_max_groups=kw_diagnostics_max_groups,
        )
        if return_importance_details:
            mode_outputs[str(mode)] = (selected_features, mirror_stats, float(cutoff), importance_details)
        else:
            mode_outputs[str(mode)] = (selected_features, mirror_stats, float(cutoff))

    if multi_mode_request:
        return mode_outputs
    return mode_outputs[str(importance_modes[0])]


# def _run_s_catnet_variant_per_group_mirror(
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     data: pd.DataFrame,
#     y,
#     q: float,
#     lookback: int = 60,
#     show_progress: bool = False,
#     shap_n_permutations: int = 20,
#     model_type: str = None,
#     model_kwargs: dict = None,
#     cj_lookback: int = None,
#     importance_mode: str = "shap",
#     derivative_eps: float = 1e-3,
#     derivative_step_min: float = None,
#     derivative_step_max: float = None,
#     derivative_n_steps: int = 1,
#     derivative_n_avg: int = 1,
#     derivative_smooth_noise_std: float = 0.0,
#     derivative_parallel_group_size: int = 8,
#     standardize_input: bool = True,
#     group_method: str = "benchmark",
#     group_objective: str = "kernel",
#     group_residual_mode: str = "ols",
#     group_lambda_reg: float = 1e-2,
#     group_max_iter: int = 60,
#     group_lr: float = 1e-2,
#     group_rank_r: int = None,
#     group_enforce_symmetric_delta: bool = False,
#     group_delta_norm_cap: float = None,
#     group_free_enforce_symmetric: bool = False,
#     group_free_g_norm_cap: float = None,
#     group_kw_use_pca: bool = False,
#     group_kw_pca_components: int = None,
#     group_high_dim_mode: str = "auto",
#     group_ridge_lambda_w: float = None,
#     group_ridge_lambda_z: float = None,
#     group_ridge_q_eps: float = None,
#     group_ridge_g_delta: float = None,
#     group_print_objective_every: int = 0,
#     print_kw_diagnostics: bool = False,
#     kw_diagnostics_max_groups: int = 5,
#     shap_derivative_mode: str = "local_quadratic",
#     return_importance_details: bool = False,
#     train_epochs: int = 500,
#     auto_tune_hparams: bool = True,
#     tune_n_trials: int = 24,
#     tune_n_folds: int = 3,
#     tune_val_ratio: float = 0.2,
#     tune_max_epochs: int = 200,
#     tune_patience: int = 12,
#     tune_weight_decay_target: float = 2e-2,
#     tune_lambda_wd: float = 5e-4,
#     tune_lambda_dropout: float = 1e-3,
#     mirror_magnitude_mode: str = "max_l1",
#     mirror_use_balance: bool = False,
#     mirror_use_smoothness: bool = False,
#     mirror_smooth_lambda: float = 0.1,
#     random_state: int = 42,
# ):
#     """
#     Per-group mirror workflow (CatNet-style):
#       1) train base model on original p-channel input
#       2) for each group j:
#          - build mirror pair (X_j+, X_j-)
#          - build p+1 input and warm-start mirror model
#          - freeze backbone, finetune input(+/- columns) + final head
#          - compute pair-only SHAP for (j+, j-) and construct mirror stat
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
#     seed = int(random_state)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
# 
#     model_type = infer_model_type(model) if model_type is None else str(model_type).lower()
#     model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
#     importance_mode = str(importance_mode).lower()
#     if importance_mode not in {"shap", "derivative"}:
#         raise ValueError("importance_mode must be 'shap' or 'derivative'.")
#     shap_derivative_mode = str(shap_derivative_mode).lower()
#     if shap_derivative_mode not in {"local_quadratic", "linear_slope", "local_multivar_linear"}:
#         raise ValueError(
#             "shap_derivative_mode must be 'local_quadratic', 'linear_slope', or 'local_multivar_linear'."
#         )
#     mirror_magnitude_mode = str(mirror_magnitude_mode).lower().strip()
#     if mirror_magnitude_mode not in {"max_l1", "max_l2", "sum_l1", "sum_l2", "multiple_l1", "multiple_l2"}:
#         raise ValueError(
#             "mirror_magnitude_mode must be one of: "
#             "max_l1, max_l2, sum_l1, sum_l2, multiple_l1, multiple_l2."
#         )
#     mirror_use_balance = bool(mirror_use_balance)
#     mirror_use_smoothness = bool(mirror_use_smoothness)
#     mirror_smooth_lambda = float(max(0.0, mirror_smooth_lambda))
#     # Keep original high-dimensional handling:
#     # group_high_dim_mode can be auto / ridge / standard.
#     group_high_dim_mode = str(group_high_dim_mode).lower()
#     if group_high_dim_mode not in {"auto", "ridge", "standard"}:
#         raise ValueError("group_high_dim_mode must be one of: auto, ridge, standard.")
# 
#     if standardize_input:
#         data = _standardize_feature_df(data)
#         if show_progress:
#             print("Variant(per-group): standardized input features (z-score)", flush=True)
# 
#     features = data.columns
#     features_dict = {i: str(features[i]) for i in range(len(features))}
#     n, p = data.shape
#     mirror_stats = {}
#     importance_details = {}
# 
#     if cj_lookback is not None and int(cj_lookback) != int(lookback) and show_progress:
#         print(
#             f"Variant(per-group): cj_lookback={cj_lookback} is deprecated in group mode; using lookback={lookback}.",
#             flush=True,
#         )
# 
#     def load_data_X(data_X: pd.DataFrame, lookback_days=lookback):
#         data_X_new = []
#         for i in range(lookback_days, len(data_X)):
#             data_X_new.append(data_X.iloc[i - lookback_days:i].values)
#         return np.array(data_X_new)
# 
#     data_seq_np = load_data_X(data, lookback).astype(np.float32)  # (n_seq, lookback, p)
#     data_seq_t = torch.from_numpy(data_seq_np).to(device=device, dtype=torch.float32)
#     n_seq, lookback_len, _ = data_seq_t.shape
#     noise_seq_t = torch.randn_like(data_seq_t)
#     y_seq_np = np.asarray(y[lookback:], dtype=np.float32).reshape(-1)
#     if y_seq_np.shape[0] != n_seq:
#         raise ValueError(
#             f"Length mismatch after lookback: y_seq={y_seq_np.shape[0]} vs sequence samples={n_seq}."
#         )
#     y_train = torch.from_numpy(y_seq_np).to(device=device, dtype=torch.float32)
# 
#     tuned_params = None
#     if bool(auto_tune_hparams):
#         if show_progress:
#             print(
#                 f"Variant(per-group): tuning lr/weight_decay/dropout via time-series CV "
#                 f"(n_trials={int(tune_n_trials)})",
#                 flush=True,
#             )
#         base_arch_params = get_model_arch_params(model, model_type=model_type)
#         tune_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
#         tune_model_kwargs = {k: v for k, v in tune_model_kwargs.items() if k not in {"hidden_dim", "num_layers", "dropout"}}
#         try:
#             tuned_params = find_optimal_params(
#                 X_train_torch=data_seq_t,
#                 y_train_torch=y_train,
#                 model_type=model_type,
#                 model_kwargs=tune_model_kwargs,
#                 fixed_hidden_dim=int(base_arch_params["hidden_dim"]),
#                 fixed_num_layers=int(base_arch_params["num_layers"]),
#                 n_trials=tune_n_trials,
#                 n_folds=tune_n_folds,
#                 val_ratio=tune_val_ratio,
#                 cv_gap=int(lookback),
#                 max_epochs=tune_max_epochs,
#                 patience=tune_patience,
#                 wd_target=tune_weight_decay_target,
#                 lambda_wd=tune_lambda_wd,
#                 lambda_dropout=tune_lambda_dropout,
#                 random_state=seed,
#             )
#         except Exception as exc:
#             tuned_params = None
#             if show_progress:
#                 print(
#                     f"Variant(per-group): tuning skipped due to error; "
#                     f"fallback to provided optimizer/model params. error={exc}",
#                     flush=True,
#                 )
#         if show_progress and tuned_params is not None:
#             print(
#                 "Variant(per-group): tuning result "
#                 f"best_params={{'lr': {float(tuned_params['lr']):.6g}, "
#                 f"'weight_decay': {float(tuned_params['weight_decay']):.6g}, "
#                 f"'dropout': {float(tuned_params['dropout']):.6g}}}, "
#                 f"cv_mse={float(tuned_params.get('cv_mse', float('nan'))):.6g}, "
#                 f"objective={float(tuned_params.get('objective', float('nan'))):.6g}",
#                 flush=True,
#             )
# 
#     # Hyperparameters for per-group training on mirror-augmented inputs.
#     base_arch_params = get_model_arch_params(model, model_type=model_type)
#     extra_arch_kwargs = {k: v for k, v in base_arch_params.items() if k not in {"hidden_dim", "num_layers", "dropout"}}
#     extra_arch_kwargs.update(model_kwargs)
#     base_lr = optimizer.param_groups[0]['lr']
#     base_weight_decay = optimizer.param_groups[0]['weight_decay']
#     if tuned_params is not None:
#         base_arch_params["dropout"] = float(tuned_params["dropout"])
#         base_lr = float(tuned_params["lr"])
#         base_weight_decay = float(tuned_params["weight_decay"])
# 
#     mirror_finetune_epochs = int(max(1, int(train_epochs)))
#     if show_progress:
#         print(
#             f"Variant(per-group): per-group mirror training epochs={mirror_finetune_epochs} "
#             f"(random initialization + full-parameter optimization)",
#             flush=True,
#         )
# 
#     eps = 1e-12
#     eye_lag = torch.eye(lookback_len, dtype=torch.float32, device=device)
# 
#     for j in range(p):
#         feature_name = str(features[j])
#         g_prefix = f"[group={feature_name}]"
#         if show_progress:
#             print(f"{g_prefix} start: building mirror direction and per-group model", flush=True)
#         Xg = data_seq_t[:, :, j]
#         Zg = noise_seq_t[:, :, j]
#         if p > 1:
#             mask = [k for k in range(p) if k != j]
#             W = data_seq_t[:, :, mask].reshape(n_seq, -1)
#         else:
#             W = torch.zeros((n_seq, 0), dtype=torch.float32, device=device)
# 
#         G_j, g_info = find_Gj(
#             Xg=Xg,
#             W=W,
#             method=group_method,
#             Zg=Zg,
#             objective=group_objective,
#             residual_mode=group_residual_mode,
#             lambda_reg=group_lambda_reg,
#             max_iter=group_max_iter,
#             lr=group_lr,
#             rank_r=group_rank_r,
#             enforce_symmetric_delta=group_enforce_symmetric_delta,
#             delta_norm_cap=group_delta_norm_cap,
#             free_enforce_symmetric=group_free_enforce_symmetric,
#             free_g_norm_cap=group_free_g_norm_cap,
#             kw_use_pca=group_kw_use_pca,
#             kw_pca_components=group_kw_pca_components,
#             high_dim_mode=group_high_dim_mode,
#             ridge_lambda_w=group_ridge_lambda_w,
#             ridge_lambda_z=group_ridge_lambda_z,
#             ridge_q_eps=group_ridge_q_eps,
#             ridge_g_delta=group_ridge_g_delta,
#             print_objective_every=group_print_objective_every,
#             objective_print_prefix=g_prefix,
#             random_state=seed + j,
#             print_kw_diagnostics=bool(print_kw_diagnostics and (j < kw_diagnostics_max_groups)),
#             kw_diagnostics_prefix=g_prefix,
#             return_info=True,
#             return_torch=True,
#         )
# 
#         U_g, V_g = build_group_mirror_pair(
#             Xg=Xg,
#             W=W,
#             Zg=Zg,
#             G=G_j,
#             residual_mode=group_residual_mode,
#             high_dim_mode=group_high_dim_mode,
#             ridge_lambda_w=group_ridge_lambda_w,
#             ridge_lambda_z=group_ridge_lambda_z,
#             ridge_q_eps=group_ridge_q_eps,
#             return_torch=True,
#         )
# 
#         X_group = _build_group_specific_mirror_input(data_seq_t, U_g, V_g, group_idx=j)  # (n, T, p+1)
#         plus_col_idx = j
#         minus_col_idx = j + 1
# 
#         mirror_model = build_sequence_model(
#             model_type=model_type,
#             input_dim=p + 1,
#             hidden_dim=base_arch_params["hidden_dim"],
#             num_layers=base_arch_params["num_layers"],
#             output_dim=1,
#             dropout=base_arch_params["dropout"],
#             **extra_arch_kwargs,
#         ).to(device)
# 
#         # Full retraining per group: no warm-start, no freezing.
#         for pp in mirror_model.parameters():
#             pp.requires_grad = True
# 
#         trainable_params = [pp for pp in mirror_model.parameters() if pp.requires_grad]
#         mirror_optimizer = optim.Adam(trainable_params, lr=base_lr, weight_decay=base_weight_decay)
#         mirror_criterion = nn.MSELoss()
#         if show_progress and j < int(max(0, kw_diagnostics_max_groups)):
#             print(f"{g_prefix} training mirror model from scratch (input_dim=p+1, full optimization)", flush=True)
#         train_model(
#             mirror_model,
#             X_group,
#             y_train,
#             mirror_finetune_epochs,
#             mirror_criterion,
#             mirror_optimizer,
#             print_every=max(1, int(mirror_finetune_epochs // 10)),
#         )
#         mirror_model.eval()
#         if show_progress:
#             with torch.inference_mode():
#                 y_hat_group = _predict_scalar_in_batches(mirror_model, X_group, batch_size=4096)
#                 y_true_group = y_train.reshape(-1).to(device=y_hat_group.device, dtype=y_hat_group.dtype)
#                 group_mse = float(torch.mean((y_hat_group - y_true_group) ** 2).detach().item())
#                 group_mae = float(torch.mean(torch.abs(y_hat_group - y_true_group)).detach().item())
#             print(
#                 f"{g_prefix} mirror-model fit: mse={group_mse:.6g}, mae={group_mae:.6g}",
#                 flush=True,
#             )
# 
#         if importance_mode == "shap":
#             if show_progress:
#                 print(f"{g_prefix} importance: pair-only exact SHAP", flush=True)
#             n_shap_draws = max(1, int(shap_n_permutations))
#             Xplus_shap, Xminus_shap, Xplus_shap_draws, Xminus_shap_draws = _pair_shap_exact_two_players(
#                 model=mirror_model,
#                 X_group=X_group,
#                 plus_col_idx=plus_col_idx,
#                 minus_col_idx=minus_col_idx,
#                 bg_size=max(64, int(shap_n_permutations)),
#                 n_background_draws=n_shap_draws,
#                 random_state=seed + j,
#                 eval_batch_size=4096,
#                 return_drawwise=True,
#             )
#             if show_progress and j < int(max(0, kw_diagnostics_max_groups)):
#                 print(
#                     f"{g_prefix} SHAP averaging: draws={n_shap_draws}, "
#                     "derivative vectors averaged over draws",
#                     flush=True,
#                 )
# 
#             x_plus_t = X_group[:, :, plus_col_idx]
#             x_minus_t = X_group[:, :, minus_col_idx]
#             x_pair = torch.stack([x_plus_t, x_minus_t], dim=0)
#             lag_plus_draws = []
#             lag_minus_draws = []
# 
#             if shap_derivative_mode == "local_quadratic":
#                 x_plus_lag = x_plus_t.detach().cpu().numpy()
#                 x_minus_lag = x_minus_t.detach().cpu().numpy()
# 
#             for d in range(n_shap_draws):
#                 y_plus_d = Xplus_shap_draws[d].reshape(-1)
#                 y_minus_d = Xminus_shap_draws[d].reshape(-1)
#                 if shap_derivative_mode == "local_quadratic":
#                     y_plus_np = np.asarray(y_plus_d.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
#                     y_minus_np = np.asarray(y_minus_d.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
#                     x_plus_s, y_plus_s = _local_quadratic_smoothing(y_plus_np, x_plus_lag, frac=0.3)
#                     x_minus_s, y_minus_s = _local_quadratic_smoothing(y_minus_np, x_minus_lag, frac=0.3)
#                     lag_vec_plus_d = np.zeros(lookback_len, dtype=np.float64)
#                     lag_vec_minus_d = np.zeros(lookback_len, dtype=np.float64)
#                     for lag_idx in range(lookback_len):
#                         grad_plus = np.gradient(y_plus_s[:, lag_idx], x_plus_s[:, lag_idx])
#                         grad_minus = np.gradient(y_minus_s[:, lag_idx], x_minus_s[:, lag_idx])
#                         lag_vec_plus_d[lag_idx] = float(np.sum(grad_plus))
#                         lag_vec_minus_d[lag_idx] = float(np.sum(grad_minus))
#                 elif shap_derivative_mode == "linear_slope":
#                     y_pair = torch.stack([y_plus_d, y_minus_d], dim=0)
#                     x_center = x_pair - x_pair.mean(dim=1, keepdim=True)
#                     y_center = y_pair.unsqueeze(2) - y_pair.mean(dim=1, keepdim=True).unsqueeze(2)
#                     xtx = x_center.transpose(1, 2) @ x_center
#                     xty = x_center.transpose(1, 2) @ y_center
#                     ridge_scale_pair = torch.clamp(torch.diagonal(xtx, dim1=1, dim2=2).mean(dim=1), min=1e-12)
#                     ridge_pair = (1e-6 * ridge_scale_pair).view(2, 1, 1) * eye_lag.unsqueeze(0)
#                     beta_pair = torch.linalg.solve(xtx + ridge_pair, xty).squeeze(2)
#                     lag_vec_plus_d = beta_pair[0].detach().cpu().numpy().astype(np.float64)
#                     lag_vec_minus_d = beta_pair[1].detach().cpu().numpy().astype(np.float64)
#                 else:
#                     y_pair = torch.stack([y_plus_d, y_minus_d], dim=0)
#                     x_pair_center = x_pair - x_pair.mean(dim=1, keepdim=True)
#                     ridge_local = float(1e-4 * torch.clamp((x_pair_center * x_pair_center).mean(), min=1e-12).detach().item())
#                     lag_pair = _local_multivar_linear_lag_vector(
#                         y=y_pair,
#                         x=x_pair,
#                         frac=0.3,
#                         ridge=ridge_local,
#                         query_batch_size=64,
#                     )
#                     lag_vec_plus_d = lag_pair[0]
#                     lag_vec_minus_d = lag_pair[1]
# 
#                 lag_plus_draws.append(lag_vec_plus_d)
#                 lag_minus_draws.append(lag_vec_minus_d)
# 
#             lag_vec_plus = np.mean(np.stack(lag_plus_draws, axis=0), axis=0).astype(np.float64)
#             lag_vec_minus = np.mean(np.stack(lag_minus_draws, axis=0), axis=0).astype(np.float64)
#         else:
#             if show_progress:
#                 print(f"{g_prefix} importance: derivative mode={importance_mode}", flush=True)
#             imp_plus = _stable_finite_difference_importance_group_scalar(
#                 model=mirror_model,
#                 X=X_group,
#                 col_idx=plus_col_idx,
#                 derivative_eps=derivative_eps,
#                 derivative_step_min=derivative_step_min,
#                 derivative_step_max=derivative_step_max,
#                 derivative_n_steps=derivative_n_steps,
#                 derivative_n_avg=derivative_n_avg,
#                 derivative_smooth_noise_std=derivative_smooth_noise_std,
#                 derivative_parallel_group_size=derivative_parallel_group_size,
#             )
#             imp_minus = _stable_finite_difference_importance_group_scalar(
#                 model=mirror_model,
#                 X=X_group,
#                 col_idx=minus_col_idx,
#                 derivative_eps=derivative_eps,
#                 derivative_step_min=derivative_step_min,
#                 derivative_step_max=derivative_step_max,
#                 derivative_n_steps=derivative_n_steps,
#                 derivative_n_avg=derivative_n_avg,
#                 derivative_smooth_noise_std=derivative_smooth_noise_std,
#                 derivative_parallel_group_size=derivative_parallel_group_size,
#             )
#             lag_vec_plus = np.asarray([imp_plus], dtype=np.float64)
#             lag_vec_minus = np.asarray([imp_minus], dtype=np.float64)
# 
#         signed_product = float(np.dot(lag_vec_plus, lag_vec_minus))
#         sign_function = 1.0 if signed_product >= 0.0 else -1.0
#         mirror_abs_core, l1_plus, l1_minus, _, _ = _mirror_magnitude_from_vectors(
#             lag_vec_plus,
#             lag_vec_minus,
#             mode=mirror_magnitude_mode,
#         )
#         bg_factor, qg_factor, qg_plus, qg_minus = _mirror_balance_and_smoothness_factors(
#             lag_vec_plus,
#             lag_vec_minus,
#             use_l2=mirror_magnitude_mode.endswith("l2"),
#             smooth_lambda=mirror_smooth_lambda,
#             eps=eps,
#         )
#         mirror_abs = float(mirror_abs_core)
#         if mirror_use_balance:
#             mirror_abs *= float(bg_factor)
#         if mirror_use_smoothness:
#             mirror_abs *= float(qg_factor)
#         mirror_stat_value = float(sign_function * mirror_abs)
#         mirror_stats[feature_name] = mirror_stat_value
#         if show_progress:
#             print(
#                 f"{g_prefix} mirror-stat: sign={sign_function:.6g}, "
#                 f"L_plus={l1_plus:.6g}, L_minus={l1_minus:.6g}, "
#                 f"Bg={bg_factor:.6g}, Qg={qg_factor:.6g}, stat={mirror_stat_value:.6g}",
#                 flush=True,
#             )
#         if return_importance_details:
#             plus_summary = _summarize_importance_vector(lag_vec_plus)
#             minus_summary = _summarize_importance_vector(lag_vec_minus)
#             importance_details[feature_name] = {
#                 "L_plus": float(l1_plus),
#                 "L_minus": float(l1_minus),
#                 "sign_pm": float(sign_function),
#                 "signed_product": signed_product,
#                 "mirror_abs_core": float(mirror_abs_core),
#                 "mirror_bg_factor": float(bg_factor),
#                 "mirror_qg_factor": float(qg_factor),
#                 "mirror_qg_plus": float(qg_plus),
#                 "mirror_qg_minus": float(qg_minus),
#                 "mirror_use_balance": bool(mirror_use_balance),
#                 "mirror_use_smoothness": bool(mirror_use_smoothness),
#                 "mirror_smooth_lambda": float(mirror_smooth_lambda),
#                 "mirror_abs_final": float(mirror_abs),
#                 "mirror_magnitude_mode": mirror_magnitude_mode,
#                 "plus_lag_vector": np.asarray(lag_vec_plus, dtype=np.float64).tolist(),
#                 "minus_lag_vector": np.asarray(lag_vec_minus, dtype=np.float64).tolist(),
#                 "plus_mean": plus_summary["mean"],
#                 "plus_std": plus_summary["std"],
#                 "plus_abs_mean": plus_summary["abs_mean"],
#                 "plus_l1": plus_summary["l1"],
#                 "plus_l2": plus_summary["l2"],
#                 "plus_pos_frac": plus_summary["pos_frac"],
#                 "plus_p10": plus_summary["p10"],
#                 "plus_p50": plus_summary["p50"],
#                 "plus_p90": plus_summary["p90"],
#                 "plus_sign_balance": plus_summary["sign_balance"],
#                 "minus_mean": minus_summary["mean"],
#                 "minus_std": minus_summary["std"],
#                 "minus_abs_mean": minus_summary["abs_mean"],
#                 "minus_l1": minus_summary["l1"],
#                 "minus_l2": minus_summary["l2"],
#                 "minus_pos_frac": minus_summary["pos_frac"],
#                 "minus_p10": minus_summary["p10"],
#                 "minus_p50": minus_summary["p50"],
#                 "minus_p90": minus_summary["p90"],
#                 "minus_sign_balance": minus_summary["sign_balance"],
#             }
# 
#         if show_progress:
#             progress_bar_len = 30
#             progress_done = int(progress_bar_len * (j + 1) / p)
#             progress_bar = "#" * progress_done + "-" * (progress_bar_len - progress_done)
#             progress_end = "\n" if (j + 1) == p else "\r"
#             print(f"Variant(per-group) Progress [{progress_bar}] {j + 1}/{p}", end=progress_end, flush=True)
# 
#     cutoff = _select_cutoff_from_mirror_stats(mirror_stats, q)
#     selected_features = _select_features_by_cutoff(mirror_stats, cutoff)
#     if return_importance_details:
#         return selected_features, mirror_stats, float(cutoff), importance_details
#     return selected_features, mirror_stats, float(cutoff)


# def S_CatNet_ablation_with_oracle(
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     data: pd.DataFrame,
#     y,
#     q,
#     lookback=60,
#     show_progress: bool = False,
#     shap_n_permutations: int = 10,
#     model_type: str = None,
#     model_kwargs: dict = None,
#     relevant_prefix: str = "relevant_",
#     derivative_eps: float = 1e-3,
#     derivative_step_min: float = None,
#     derivative_step_max: float = None,
#     derivative_n_steps: int = 20,
#     derivative_n_avg: int = 20,
#     derivative_smooth_noise_std: float = 0.0,
#     derivative_parallel_group_size: int = 8,
#     standardize_input: bool = True,
#     group_objective: str = "kernel",
#     group_residual_mode: str = "ols",
#     group_lambda_reg: float = 1e-2,
#     group_max_iter: int = 60,
#     group_lr: float = 1e-2,
#     group_rank_r: int = None,
#     group_enforce_symmetric_delta: bool = False,
#     group_delta_norm_cap: float = None,
#     group_free_enforce_symmetric: bool = False,
#     group_free_g_norm_cap: float = None,
#     group_kw_use_pca: bool = False,
#     group_kw_pca_components: int = None,
#     group_high_dim_mode: str = "auto",
#     group_ridge_lambda_w: float = None,
#     group_ridge_lambda_z: float = None,
#     group_ridge_q_eps: float = None,
#     group_ridge_g_delta: float = None,
#     group_print_objective_every: int = 0,
#     print_kw_diagnostics: bool = False,
#     kw_diagnostics_max_groups: int = 5,
#     shap_derivative_mode: str = "local_quadratic",
#     train_epochs: int = 500,
#     auto_tune_hparams: bool = True,
#     tune_n_trials: int = 24,
#     tune_n_folds: int = 3,
#     tune_val_ratio: float = 0.2,
#     tune_max_epochs: int = 200,
#     tune_patience: int = 12,
#     tune_weight_decay_target: float = 2e-2,
#     tune_lambda_wd: float = 5e-4,
#     tune_lambda_dropout: float = 1e-3,
#     mirror_magnitude_mode: str = "max_l1",
#     mirror_use_balance: bool = False,
#     mirror_use_smoothness: bool = False,
#     mirror_smooth_lambda: float = 0.1,
#     random_state: int = 42,
# ):
#     """
#     Group-Neural-GM ablation suite:
#     1) benchmark + SHAP-lag importance
#     2) scheme_c_free + SHAP-lag importance
#     3) benchmark + derivative-lag importance
# 
#     Returns one dict containing all method results.
#     Oracle metrics are reported only for group_benchmark_shap.
#     """
#     relevant_features = {str(c) for c in data.columns if str(c).startswith(relevant_prefix)}
#     methods = [
#         ("group_benchmark_shap", "benchmark", "shap"),
#         ("group_scheme_c_free_shap", "scheme_c_free", "shap"),
#         ("group_benchmark_derivative", "benchmark", "derivative"),
#     ]
# 
#     all_results = {}
#     for method_name, group_method, importance_mode in methods:
#         if show_progress:
#             print(
#                 f"[Ablation] Running {method_name} (group_method={group_method}, importance={importance_mode})",
#                 flush=True,
#             )
#         selected_features, mirror_stats, cutoff = _run_s_catnet_variant(
#             model=model,
#             optimizer=optimizer,
#             data=data,
#             y=y,
#             q=q,
#             lookback=lookback,
#             show_progress=show_progress,
#             shap_n_permutations=shap_n_permutations,
#             model_type=model_type,
#             model_kwargs=model_kwargs,
#             importance_mode=importance_mode,
#             derivative_eps=derivative_eps,
#             derivative_step_min=derivative_step_min,
#             derivative_step_max=derivative_step_max,
#             derivative_n_steps=derivative_n_steps,
#             derivative_n_avg=derivative_n_avg,
#             derivative_smooth_noise_std=derivative_smooth_noise_std,
#             derivative_parallel_group_size=derivative_parallel_group_size,
#             standardize_input=standardize_input,
#             group_method=group_method,
#             group_objective=group_objective,
#             group_residual_mode=group_residual_mode,
#             group_lambda_reg=group_lambda_reg,
#             group_max_iter=group_max_iter,
#             group_lr=group_lr,
#             group_rank_r=group_rank_r,
#             group_enforce_symmetric_delta=group_enforce_symmetric_delta,
#             group_delta_norm_cap=group_delta_norm_cap,
#             group_free_enforce_symmetric=group_free_enforce_symmetric,
#             group_free_g_norm_cap=group_free_g_norm_cap,
#             group_kw_use_pca=group_kw_use_pca,
#             group_kw_pca_components=group_kw_pca_components,
#             group_high_dim_mode=group_high_dim_mode,
#             group_ridge_lambda_w=group_ridge_lambda_w,
#             group_ridge_lambda_z=group_ridge_lambda_z,
#             group_ridge_q_eps=group_ridge_q_eps,
#             group_ridge_g_delta=group_ridge_g_delta,
#             group_print_objective_every=group_print_objective_every,
#             print_kw_diagnostics=print_kw_diagnostics,
#             kw_diagnostics_max_groups=kw_diagnostics_max_groups,
#             shap_derivative_mode=shap_derivative_mode,
#             train_epochs=train_epochs,
#             auto_tune_hparams=auto_tune_hparams,
#             tune_n_trials=tune_n_trials,
#             tune_n_folds=tune_n_folds,
#             tune_val_ratio=tune_val_ratio,
#             tune_max_epochs=tune_max_epochs,
#             tune_patience=tune_patience,
#             tune_weight_decay_target=tune_weight_decay_target,
#             tune_lambda_wd=tune_lambda_wd,
#             tune_lambda_dropout=tune_lambda_dropout,
#             mirror_magnitude_mode=mirror_magnitude_mode,
#             mirror_use_balance=mirror_use_balance,
#             mirror_use_smoothness=mirror_use_smoothness,
#             mirror_smooth_lambda=mirror_smooth_lambda,
#             random_state=random_state,
#         )
# 
#         ours_metrics = _compute_fdr_power_from_selected(selected_features, relevant_features)
#         result_payload = {
#             "selected_features": selected_features,
#             "mirror_stats": mirror_stats,
#             "cutoff": cutoff,
#             "fdr": ours_metrics["fdr"],
#             "power": ours_metrics["power"],
#             "num_selected": ours_metrics["num_selected"],
#             "num_tp": ours_metrics["num_tp"],
#             "num_fp": ours_metrics["num_fp"],
#         }
#         if method_name == "group_benchmark_shap":
#             oracle = _oracle_select_from_mirror_stats(
#                 mirror_stats=mirror_stats,
#                 relevant_features=relevant_features,
#                 q=q,
#             )
#             result_payload.update(
#                 {
#                     "oracle_selected_features": oracle["selected_features"],
#                     "oracle_cutoff": oracle["cutoff"],
#                     "oracle_fdr": oracle["fdr"],
#                     "oracle_power": oracle["power"],
#                     "oracle_num_selected": oracle["num_selected"],
#                 }
#             )
# 
#         all_results[method_name] = result_payload
# 
#     return all_results


def _build_details_df_from_importance(
    mirror_stats: dict,
    importance_details: dict,
    relevant_prefix: str = "relevant_",
) -> pd.DataFrame:
    rows = []
    for feature, mirror_stat in mirror_stats.items():
        row = {"feature": feature, "mirror_stat": float(mirror_stat)}
        row.update(importance_details.get(feature, {}))
        rows.append(row)
    details_df = pd.DataFrame(rows)
    if "feature" in details_df.columns:
        details_df["is_relevant"] = details_df["feature"].astype(str).str.startswith(relevant_prefix)
    else:
        details_df["is_relevant"] = False
    show_cols = [
        "feature",
        "L_plus",
        "L_minus",
        "sign_pm",
        "plus_lag_vector",
        "minus_lag_vector",
        "plus_mean",
        "plus_std",
        "plus_abs_mean",
        "plus_l2",
        "plus_pos_frac",
        "plus_p10",
        "plus_p50",
        "plus_p90",
        "minus_mean",
        "minus_std",
        "minus_abs_mean",
        "minus_l2",
        "minus_pos_frac",
        "minus_p10",
        "minus_p50",
        "minus_p90",
        "mirror_bg_factor",
        "mirror_qg_factor",
        "mirror_abs_core",
        "mirror_abs_final",
        "mirror_stat",
        "is_relevant",
    ]
    show_cols = [c for c in show_cols if c in details_df.columns]
    if len(show_cols) > 0:
        details_df = details_df[show_cols]
    return details_df


def S_CatNet_with_details(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: pd.DataFrame,
    y,
    q,
    lookback=60,
    show_progress: bool = False,
    shap_n_permutations: int = 20,
    model_type: str = None,
    model_kwargs: dict = None,
    print_mirror_details: bool = True,
    relevant_prefix: str = "relevant_",
    plot_mirror_distribution: bool = False,
    standardize_input: bool = True,
    importance_mode: str = "shap",
    derivative_eps: float = 1e-3,
    derivative_step_min: float = None,
    derivative_step_max: float = None,
    derivative_n_steps: int = 1,
    derivative_n_avg: int = 1,
    derivative_smooth_noise_std: float = 0.0,
    derivative_parallel_group_size: int = 8,
    derivative_method: str = "temporal_secant",
    group_method: str = "benchmark",
    group_objective: str = "kernel",
    group_residual_mode: str = "ols",
    group_lambda_reg: float = 1e-2,
    group_max_iter: int = 60,
    group_lr: float = 1e-2,
    group_rank_r: int = None,
    group_enforce_symmetric_delta: bool = False,
    group_delta_norm_cap: float = None,
    group_free_enforce_symmetric: bool = False,
    group_free_g_norm_cap: float = None,
    group_kw_use_pca: bool = False,
    group_kw_pca_components: int = None,
    group_high_dim_mode: str = "auto",
    group_ridge_lambda_w: float = None,
    group_ridge_lambda_z: float = None,
    group_ridge_q_eps: float = None,
    group_ridge_g_delta: float = None,
    group_print_objective_every: int = 0,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_max_groups: int = 5,
    shap_derivative_mode: str = "local_quadratic",
    train_epochs: int = 500,
    auto_tune_hparams: bool = True,
    tune_n_trials: int = 24,
    tune_n_folds: int = 3,
    tune_val_ratio: float = 0.2,
    tune_max_epochs: int = 200,
    tune_patience: int = 12,
    tune_weight_decay_target: float = 2e-2,
    tune_lambda_wd: float = 5e-4,
    tune_lambda_dropout: float = 1e-3,
    mirror_magnitude_mode: str = "sum_l1",
    mirror_use_balance: bool = False,
    mirror_use_smoothness: bool = False,
    mirror_smooth_lambda: float = 0.1,
    random_state: int = 42,
):
    """
    Same core algorithm as S_CatNet (single-stage), with mirror-stat diagnostics table.
    """
    run_result = S_CatNet(
        model=model,
        optimizer=optimizer,
        data=data,
        y=y,
        q=q,
        lookback=lookback,
        show_progress=show_progress,
        shap_n_permutations=shap_n_permutations,
        model_type=model_type,
        model_kwargs=model_kwargs,
        importance_mode=importance_mode,
        derivative_method=derivative_method,
        derivative_eps=derivative_eps,
        derivative_step_min=derivative_step_min,
        derivative_step_max=derivative_step_max,
        derivative_n_steps=derivative_n_steps,
        derivative_n_avg=derivative_n_avg,
        derivative_smooth_noise_std=derivative_smooth_noise_std,
        derivative_parallel_group_size=derivative_parallel_group_size,
        standardize_input=standardize_input,
        group_method=group_method,
        group_objective=group_objective,
        group_residual_mode=group_residual_mode,
        group_lambda_reg=group_lambda_reg,
        group_max_iter=group_max_iter,
        group_lr=group_lr,
        group_rank_r=group_rank_r,
        group_enforce_symmetric_delta=group_enforce_symmetric_delta,
        group_delta_norm_cap=group_delta_norm_cap,
        group_free_enforce_symmetric=group_free_enforce_symmetric,
        group_free_g_norm_cap=group_free_g_norm_cap,
        group_kw_use_pca=group_kw_use_pca,
        group_kw_pca_components=group_kw_pca_components,
        group_high_dim_mode=group_high_dim_mode,
        group_ridge_lambda_w=group_ridge_lambda_w,
        group_ridge_lambda_z=group_ridge_lambda_z,
        group_ridge_q_eps=group_ridge_q_eps,
        group_ridge_g_delta=group_ridge_g_delta,
        group_print_objective_every=group_print_objective_every,
        print_kw_diagnostics=print_kw_diagnostics,
        kw_diagnostics_max_groups=kw_diagnostics_max_groups,
        shap_derivative_mode=shap_derivative_mode,
        train_epochs=train_epochs,
        auto_tune_hparams=auto_tune_hparams,
        tune_n_trials=tune_n_trials,
        tune_n_folds=tune_n_folds,
        tune_val_ratio=tune_val_ratio,
        tune_max_epochs=tune_max_epochs,
        tune_patience=tune_patience,
        tune_weight_decay_target=tune_weight_decay_target,
        tune_lambda_wd=tune_lambda_wd,
        tune_lambda_dropout=tune_lambda_dropout,
        mirror_magnitude_mode=mirror_magnitude_mode,
        mirror_use_balance=mirror_use_balance,
        mirror_use_smoothness=mirror_use_smoothness,
        mirror_smooth_lambda=mirror_smooth_lambda,
        return_importance_details=True,
        random_state=random_state,
    )
    if not isinstance(importance_mode, str):
        mode_results = {}
        for mode, payload in run_result.items():
            selected_features_mode, mirror_stats_mode, cutoff_mode, importance_details_mode = payload
            details_df_mode = _build_details_df_from_importance(
                mirror_stats=mirror_stats_mode,
                importance_details=importance_details_mode,
                relevant_prefix=relevant_prefix,
            )
            mode_results[str(mode)] = (
                selected_features_mode,
                mirror_stats_mode,
                cutoff_mode,
                details_df_mode,
            )
        return mode_results

    selected_features, mirror_stats, cutoff, importance_details = run_result
    details_df = _build_details_df_from_importance(
        mirror_stats=mirror_stats,
        importance_details=importance_details,
        relevant_prefix=relevant_prefix,
    )
    if print_mirror_details:
        print("\n=== Per-feature Mirror Details ===", flush=True)
        float_cols = details_df.select_dtypes(include=[np.number]).columns
        formatters = {c: (lambda x: f"{x:.6e}") for c in float_cols if c != "is_relevant"}
        print(details_df.to_string(index=False, formatters=formatters), flush=True)
        print("\n=== Per-feature Lag Derivative Vectors (for mirror stat) ===", flush=True)
        for _, r in details_df.iterrows():
            feat = str(r["feature"])
            plus_vec = importance_details.get(feat, {}).get("plus_lag_vector", None)
            minus_vec = importance_details.get(feat, {}).get("minus_lag_vector", None)
            if plus_vec is None or minus_vec is None:
                continue
            plus_str = ", ".join(f"{float(v):.6e}" for v in plus_vec)
            minus_str = ", ".join(f"{float(v):.6e}" for v in minus_vec)
            print(f"{feat}:", flush=True)
            print(f"  plus_lag_vector = [{plus_str}]", flush=True)
            print(f"  minus_lag_vector = [{minus_str}]", flush=True)

        print("\n=== Mirror Statistic Distribution Summary ===", flush=True)
        eps = 1e-12
        rel_stats = details_df.loc[details_df["is_relevant"], "mirror_stat"].to_numpy()
        irr_stats = details_df.loc[~details_df["is_relevant"], "mirror_stat"].to_numpy()

        if rel_stats.size > 0:
            print(
                f"Relevant count={rel_stats.size}, "
                f"mean={rel_stats.mean():.6f}, median={np.median(rel_stats):.6f}, "
                f"p10={np.percentile(rel_stats, 10):.6f}, p90={np.percentile(rel_stats, 90):.6f}",
                flush=True,
            )
        else:
            print("Relevant group is empty under current prefix rule.", flush=True)

        if irr_stats.size > 0:
            irr_mean = float(irr_stats.mean())
            irr_median = float(np.median(irr_stats))
            irr_p10 = float(np.percentile(irr_stats, 10))
            irr_p90 = float(np.percentile(irr_stats, 90))
            irr_std = float(np.std(irr_stats))
            n_pos = int(np.sum(irr_stats > 0))
            n_neg = int(np.sum(irr_stats < 0))
            sign_imbalance = abs(n_pos - n_neg) / max(irr_stats.size, 1)
            quantile_mirror_gap = abs(irr_p90 + irr_p10)
            quantile_mirror_norm = quantile_mirror_gap / (abs(irr_p90) + abs(irr_p10) + eps)
            abs_t = float(np.percentile(np.abs(irr_stats), 90))
            right_tail = float(np.mean(irr_stats >= abs_t))
            left_tail = float(np.mean(irr_stats <= -abs_t))
            tail_imbalance = abs(right_tail - left_tail)
            centered = irr_stats - irr_mean
            irr_skew = float(np.mean(centered ** 3) / (irr_std ** 3 + eps))

            print(
                f"Irrelevant count={irr_stats.size}, "
                f"mean={irr_mean:.6f}, median={irr_median:.6f}, "
                f"p10={irr_p10:.6f}, p90={irr_p90:.6f}",
                flush=True,
            )
            print(
                "Irrelevant symmetry diagnostics: "
                f"sign_imbalance={sign_imbalance:.6f}, "
                f"quantile_mirror_gap={quantile_mirror_gap:.6f}, "
                f"quantile_mirror_norm={quantile_mirror_norm:.6f}, "
                f"tail_imbalance={tail_imbalance:.6f}, "
                f"skewness={irr_skew:.6f}",
                flush=True,
            )

            if {"L_plus", "L_minus"}.issubset(set(details_df.columns)):
                irr_plus = details_df.loc[~details_df["is_relevant"], "L_plus"].to_numpy(dtype=np.float64)
                irr_minus = details_df.loc[~details_df["is_relevant"], "L_minus"].to_numpy(dtype=np.float64)
                msg = (
                    "Irrelevant vector-importance diagnostics: "
                    f"L_plus(avg)={np.nanmean(irr_plus):.6e}, "
                    f"L_minus(avg)={np.nanmean(irr_minus):.6e}"
                )
                if {"plus_l2", "minus_l2"}.issubset(set(details_df.columns)):
                    irr_plus_l2 = details_df.loc[~details_df["is_relevant"], "plus_l2"].to_numpy(dtype=np.float64)
                    irr_minus_l2 = details_df.loc[~details_df["is_relevant"], "minus_l2"].to_numpy(dtype=np.float64)
                    msg += (
                        f", plus_l2(avg)={np.nanmean(irr_plus_l2):.6e}, "
                        f"minus_l2(avg)={np.nanmean(irr_minus_l2):.6e}"
                    )
                print(
                    msg,
                    flush=True,
                )
        else:
            print("Irrelevant group is empty under current prefix rule.", flush=True)

        print(f"Cutoff={cutoff:.6f}, Selected={len(selected_features)}", flush=True)

        if plot_mirror_distribution:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            if rel_stats.size > 0:
                axes[0].hist(rel_stats, bins=30, alpha=0.55, label="Relevant", density=True)
            if irr_stats.size > 0:
                axes[0].hist(irr_stats, bins=30, alpha=0.55, label="Irrelevant", density=True)
            axes[0].axvline(0.0, color="gray", linestyle=":", linewidth=1)
            axes[0].axvline(cutoff, color="black", linestyle="--", linewidth=1, label=f"Cutoff={cutoff:.3f}")
            axes[0].set_title("Mirror Statistic Distribution")
            axes[0].set_xlabel("mirror_stat")
            axes[0].set_ylabel("Density")
            axes[0].legend()

            box_data = []
            box_labels = []
            if rel_stats.size > 0:
                box_data.append(rel_stats)
                box_labels.append("Relevant")
            if irr_stats.size > 0:
                box_data.append(irr_stats)
                box_labels.append("Irrelevant")
            if box_data:
                axes[1].boxplot(box_data, labels=box_labels, showfliers=False)
                axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
                axes[1].axhline(cutoff, color="black", linestyle="--", linewidth=1)
            axes[1].set_title("Mirror Statistic Boxplot")
            axes[1].set_ylabel("mirror_stat")

            if irr_stats.size > 0:
                axes[2].hist(irr_stats, bins=30, alpha=0.55, label="Irrelevant T", density=True)
                axes[2].hist(-irr_stats, bins=30, alpha=0.55, label="Mirrored -T", density=True)
                axes[2].axvline(0.0, color="gray", linestyle=":", linewidth=1)
            axes[2].set_title("Irrelevant Symmetry Check")
            axes[2].set_xlabel("value")
            axes[2].set_ylabel("Density")
            axes[2].legend()

            plt.tight_layout()
            plt.show()

    return selected_features, mirror_stats, cutoff, details_df
