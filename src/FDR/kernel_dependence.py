import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline


# =========================================================
# Basic kernels
# =========================================================

# Debug plotting for I_j(c) curves (no notebook change needed).
DEBUG_PLOT_IJC = True
DEBUG_PLOT_FEATURE_INDICES = {0, 1, 2, 3, 4}
DEBUG_PLOT_MAX_FIGURES = 10
DEBUG_PLOT_DIR = "debug_kernel_dependence_curves"
_PLOTTED_FEATURES = set()

# Bandwidth strategy (legacy scalar c_j path)
USE_MEDIAN_HEURISTIC_BW = True
MEDIAN_BW_MAX_POINTS = 320

# High-dimensional bandwidth tuning (important for W with large dimension)
HIGH_DIM_BW_THRESHOLD = 256
HIGH_DIM_BW_PROJ_MAX = 64
HIGH_DIM_BW_PROJ_MIN = 16
HIGH_DIM_BW_PROJ_LOG_FACTOR = 8.0
BANDWIDTH_STD_EPS = 1e-6
BANDWIDTH_MIN = 1e-6
BANDWIDTH_RANDOM_PROJ_SEED = 1234

# For time-series lag blocks, random projection may mix coordinates and weaken
# interpretable lag geometry. Keep it off by default; enable only if needed.
USE_RANDOM_PROJECTION_FOR_BW = False

# High-dimensional auto switch: use ridge-preconditioned mirror geometry
# when W has more columns than samples.
HIGH_DIM_AUTO_D_GT_N = True


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_plot_ij_curve(j: int, c_values: np.ndarray, I_values: np.ndarray, c_optimal: float):
    if not DEBUG_PLOT_IJC:
        return
    if j not in DEBUG_PLOT_FEATURE_INDICES:
        return
    if len(_PLOTTED_FEATURES) >= DEBUG_PLOT_MAX_FIGURES and j not in _PLOTTED_FEATURES:
        return
    if j in _PLOTTED_FEATURES:
        return

    os.makedirs(DEBUG_PLOT_DIR, exist_ok=True)
    path = os.path.join(DEBUG_PLOT_DIR, f"Ijc_feature_{j}.png")

    c_arr = np.asarray(c_values, dtype=float)
    i_arr = np.asarray(I_values, dtype=float)
    finite_mask = np.isfinite(c_arr) & np.isfinite(i_arr)
    if not np.any(finite_mask):
        return

    c_arr = c_arr[finite_mask]
    i_arr = i_arr[finite_mask]
    c_min_idx = int(np.argmin(i_arr))
    c_grid_min = float(c_arr[c_min_idx])
    i_grid_min = float(i_arr[c_min_idx])

    plt.figure(figsize=(7, 4.5))
    plt.plot(c_arr, i_arr, marker="o", linewidth=1.3, markersize=3, label="Grid I_j(c)")
    plt.axvline(c_optimal, color="red", linestyle="--", linewidth=1.2, label=f"c_opt={c_optimal:.6g}")
    plt.scatter([c_grid_min], [i_grid_min], color="black", s=35, zorder=3, label=f"grid_min={c_grid_min:.6g}")
    plt.xlabel("c")
    plt.ylabel("I_j(c)")
    plt.title(f"I_j(c) curve for feature index j={j}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    _PLOTTED_FEATURES.add(j)
    print(f"[I_j(c) debug] saved curve for feature j={j} to: {path}", flush=True)


def _subsample_rows_for_bw(X: torch.Tensor, max_points: int = MEDIAN_BW_MAX_POINTS) -> torch.Tensor:
    n = X.shape[0]
    if n <= max_points:
        return X
    idx = torch.linspace(0, n - 1, steps=max_points, device=X.device).long()
    return X.index_select(0, idx)


def _robust_positive_scalar(val: torch.Tensor, fallback: torch.Tensor) -> float:
    v = float(val.detach().item())
    if not np.isfinite(v) or v <= 0.0:
        fb = float(fallback.detach().item())
        if not np.isfinite(fb) or fb <= 0.0:
            return 1.0
        return fb
    return v


def _standardize_columns_for_bw(X: torch.Tensor, eps: float = BANDWIDTH_STD_EPS) -> torch.Tensor:
    """
    Column-wise standardization before bandwidth estimation.
    This strongly stabilizes median-distance in high-dimensional W.
    """
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.clamp(sd, min=eps)
    return (X - mu) / sd


def _choose_proj_dim(n: int, d: int) -> int:
    k_log = int(np.ceil(HIGH_DIM_BW_PROJ_LOG_FACTOR * np.log(max(n, 2))))
    k = max(HIGH_DIM_BW_PROJ_MIN, min(HIGH_DIM_BW_PROJ_MAX, k_log, d))
    return int(max(1, k))


def _random_project_for_bw(X: torch.Tensor, out_dim: int, seed: int = BANDWIDTH_RANDOM_PROJ_SEED) -> torch.Tensor:
    """
    JL-style random projection with N(0, 1/out_dim), approximately preserving distances.
    """
    d = X.shape[1]
    if out_dim >= d:
        return X
    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(seed))
    R = torch.randn((d, out_dim), dtype=X.dtype, device=X.device, generator=gen) / np.sqrt(float(out_dim))
    return X @ R


def _median_pairwise_distance(X: torch.Tensor) -> float:
    n = X.shape[0]
    if n <= 1:
        return 1.0
    d = torch.pdist(X, p=2)
    if d.numel() == 0:
        std = torch.std(X, unbiased=False)
        return _robust_positive_scalar(std, torch.tensor(1.0, device=X.device))
    med = torch.median(d)
    std = torch.std(X, unbiased=False)
    return max(BANDWIDTH_MIN, _robust_positive_scalar(med, std))


def _median_bandwidth_1d(x: torch.Tensor) -> float:
    x_use = _subsample_rows_for_bw(x.view(-1, 1), max_points=MEDIAN_BW_MAX_POINTS)
    return _median_pairwise_distance(x_use)


def _fit_kernel_transform(X: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Fit a deterministic transform used both for bandwidth estimation and
    actual kernel construction, so the geometry is consistent.
    """
    X = X if X.ndim == 2 else X.view(X.shape[0], -1)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.clamp(sd, min=BANDWIDTH_STD_EPS)

    proj = None
    d = X.shape[1]
    if USE_RANDOM_PROJECTION_FOR_BW and d >= HIGH_DIM_BW_THRESHOLD:
        k = _choose_proj_dim(n=X.shape[0], d=d)
        gen = torch.Generator(device=X.device)
        gen.manual_seed(int(BANDWIDTH_RANDOM_PROJ_SEED))
        proj = torch.randn((d, k), dtype=X.dtype, device=X.device, generator=gen) / np.sqrt(float(k))

    return {"mu": mu, "sd": sd, "proj": proj}


def _apply_kernel_transform(X: torch.Tensor, tfm: Dict[str, torch.Tensor]) -> torch.Tensor:
    Xs = (X - tfm["mu"]) / tfm["sd"]
    proj = tfm["proj"]
    if proj is not None:
        Xs = Xs @ proj
    return Xs


def _median_bandwidth_multivariate(X: torch.Tensor) -> float:
    """
    Robust bandwidth for multivariate kernel.

    For high-dimensional W (e.g., d in thousands), direct pdist in original space can be unstable.
    We use:
    1) row subsample
    2) column standardization
    3) optional random projection to moderate dimension
    4) median pairwise distance in the transformed space

    This works much better for settings like n~1000, d~5000.
    """
    X_use = _subsample_rows_for_bw(X, max_points=MEDIAN_BW_MAX_POINTS)
    if X_use.ndim != 2:
        raise ValueError("_median_bandwidth_multivariate expects a 2D tensor.")
    tfm = _fit_kernel_transform(X_use)
    X_bw = _apply_kernel_transform(X_use, tfm)
    return _median_pairwise_distance(X_bw)


# Computes the Gaussian (RBF) kernel matrix for a batch of 1D vectors.
# x_batch shape: (batch, n)
def _rbf_kernel_matrix_batch_1d(x_batch: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    sigma = float(max(BANDWIDTH_MIN, sigma))
    x2 = x_batch * x_batch
    dist = -2.0 * torch.bmm(x_batch.unsqueeze(2), x_batch.unsqueeze(1))
    dist = dist + x2.unsqueeze(2) + x2.unsqueeze(1)
    return torch.exp(-dist / (2.0 * sigma ** 2))


# Computes the Gaussian (RBF) kernel matrix for a batch of multivariate vectors.
# x_batch shape: (batch, n, d)
def _rbf_kernel_matrix_batch_multivariate(x_batch: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    sigma = float(max(BANDWIDTH_MIN, sigma))
    x_norm2 = (x_batch * x_batch).sum(dim=2)  # (batch, n)
    gram = torch.bmm(x_batch, x_batch.transpose(1, 2))  # (batch, n, n)
    dist = x_norm2.unsqueeze(2) + x_norm2.unsqueeze(1) - 2.0 * gram
    dist = torch.clamp(dist, min=0.0)
    return torch.exp(-dist / (2.0 * sigma ** 2))


def _rbf_kernel_matrix_batch_multivariate_per_sigma(
    x_batch: torch.Tensor,
    sigma_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian (RBF) kernel with per-batch bandwidth.
    x_batch: (batch, n, d)
    sigma_batch: (batch,)
    """
    sigma_batch = torch.clamp(sigma_batch.to(dtype=x_batch.dtype, device=x_batch.device), min=BANDWIDTH_MIN)
    x_norm2 = (x_batch * x_batch).sum(dim=2)  # (batch, n)
    gram = torch.bmm(x_batch, x_batch.transpose(1, 2))  # (batch, n, n)
    dist = x_norm2.unsqueeze(2) + x_norm2.unsqueeze(1) - 2.0 * gram
    dist = torch.clamp(dist, min=0.0)
    denom = (2.0 * sigma_batch * sigma_batch).view(-1, 1, 1)
    return torch.exp(-dist / denom)


# Centers a batch of kernel matrices using HAH form in broadcast style.
# K shape: (batch, n, n)
def _center_kernel_matrix_batch(K: torch.Tensor) -> torch.Tensor:
    row_mean = K.mean(dim=2, keepdim=True)
    col_mean = K.mean(dim=1, keepdim=True)
    total_mean = K.mean(dim=(1, 2), keepdim=True)
    return K - row_mean - col_mean + total_mean


# Deprecated / currently unused legacy scalar c_j path (commented out).
# # Backward-compatible single-vector API.
# def calculate_kernel_matrix(X_sample: np.ndarray, sigma: float = 1.0) -> torch.Tensor:
#     device = _device()
#     x = torch.as_tensor(X_sample, dtype=torch.float32, device=device).view(1, -1)
#     return _rbf_kernel_matrix_batch_1d(x, sigma=sigma).squeeze(0)
# 
# 
# # Backward-compatible single-matrix API.
# def center_matrix(A: torch.Tensor) -> torch.Tensor:
#     A_b = A.unsqueeze(0)
#     return _center_kernel_matrix_batch(A_b).squeeze(0)
# 
# 
# # =========================================================
# # Kernel dependence: legacy scalar c_j APIs
# # =========================================================
# 
# def _build_W_tensor(
#     X: np.ndarray,
#     j: int,
#     tau_plan,
#     device: torch.device,
# ):
#     """
#     Build lag-specific W = X_{-j} slices for all tau values once.
#     Returns list of tensors W_tau, each shape (n_tau, p-1).
#     """
#     X_np = np.asarray(X)
#     if X_np.ndim != 2:
#         raise ValueError("X must be a 2D array with shape (n, p).")
# 
#     W_np = np.delete(X_np, j, axis=1)
#     W_full = torch.as_tensor(W_np, dtype=torch.float32, device=device)
# 
#     W_list = []
#     for _, su, eu, _, _, _ in tau_plan:
#         W_list.append(W_full[su:eu, :])
#     return W_list
# 
# 
# def _precompute_kw_list(
#     X: np.ndarray,
#     xj: torch.Tensor,
#     j: int,
#     tau_plan,
#     device: torch.device,
#     sigma: float = 1.0,
# ):
#     """
#     Precompute K_W and sigma_UV per tau for scalar c_j path.
#     """
#     W_list = _build_W_tensor(X, j, tau_plan, device=device)
#     kw_list = []
#     sigma_uv_list = []
# 
#     with torch.no_grad():
#         for idx, W_tau in enumerate(W_list):
#             n_tau = W_tau.shape[0]
#             if n_tau <= 1:
#                 kw_list.append(None)
#                 sigma_uv_list.append(float(max(BANDWIDTH_MIN, sigma)))
#                 continue
# 
#             if USE_MEDIAN_HEURISTIC_BW:
#                 _, su, eu, _, _, _ = tau_plan[idx]
#                 xj_tau = xj[su:eu]
#                 sigma_uv_tau = _median_bandwidth_1d(xj_tau)
#                 sigma_w_tau = _median_bandwidth_multivariate(W_tau)
#             else:
#                 sigma_uv_tau = float(max(BANDWIDTH_MIN, sigma))
#                 sigma_w_tau = float(max(BANDWIDTH_MIN, sigma))
# 
#             Kw = _rbf_kernel_matrix_batch_multivariate(W_tau.unsqueeze(0), sigma=sigma_w_tau).squeeze(0)
#             kw_list.append(Kw)
#             sigma_uv_list.append(sigma_uv_tau)
# 
#     return kw_list, sigma_uv_list
# 
# 
# def _kernel_dependence_batch_for_c_values(
#     c_values: torch.Tensor,
#     X: np.ndarray,
#     xj: torch.Tensor,
#     zj: torch.Tensor,
#     j: int,
#     tau_plan,
#     sigma: float = 1.0,
#     precomputed_kw_list=None,
#     precomputed_sigma_uv_list=None,
# ) -> torch.Tensor:
#     """
#     Legacy scalar objective:
#         I_j(c) = sum_tau w_tau * [ (H K_U H) ◦ (H K_V H) ◦ K_W ]_{++} / n_tau^2
#     """
#     device = xj.device
#     c_values = c_values.view(-1, 1)
# 
#     xj_plus_all = xj.unsqueeze(0) + c_values * zj.unsqueeze(0)
#     xj_minus_all = xj.unsqueeze(0) - c_values * zj.unsqueeze(0)
# 
#     batch_size = c_values.shape[0]
#     I = torch.zeros(batch_size, dtype=xj.dtype, device=device)
# 
#     if precomputed_kw_list is None:
#         precomputed_kw_list, precomputed_sigma_uv_list = _precompute_kw_list(
#             X=X,
#             xj=xj,
#             j=j,
#             tau_plan=tau_plan,
#             device=device,
#             sigma=sigma,
#         )
# 
#     with torch.no_grad():
#         for idx, (_, su, eu, sv, ev, weight) in enumerate(tau_plan):
#             u = xj_plus_all[:, su:eu]
#             v = xj_minus_all[:, sv:ev]
#             n_tau = u.shape[1]
# 
#             if n_tau <= 1:
#                 continue
# 
#             sigma_uv_tau = (
#                 precomputed_sigma_uv_list[idx]
#                 if precomputed_sigma_uv_list is not None
#                 else float(max(BANDWIDTH_MIN, sigma))
#             )
# 
#             Ku = _rbf_kernel_matrix_batch_1d(u, sigma=sigma_uv_tau)
#             Kv = _rbf_kernel_matrix_batch_1d(v, sigma=sigma_uv_tau)
# 
#             Ku_c = _center_kernel_matrix_batch(Ku)
#             Kv_c = _center_kernel_matrix_batch(Kv)
# 
#             Kw = precomputed_kw_list[idx]
#             if Kw is None:
#                 continue
# 
#             I_tau = (Ku_c * Kv_c * Kw.unsqueeze(0)).sum(dim=(1, 2)) / (n_tau ** 2)
#             I += weight * I_tau
# 
#     return I
# 
# 
# # =========================================================
# # Public legacy scalar APIs
# # =========================================================
# 
# def kernel_dependence(
#     cj: float,
#     X: np.ndarray,
#     xj: np.ndarray,
#     zj: np.ndarray,
#     j: int,
#     sigma: float = 1.0,
# ) -> float:
#     device = _device()
#     xj_t = torch.as_tensor(xj, dtype=torch.float32, device=device)
#     zj_t = torch.as_tensor(zj, dtype=torch.float32, device=device)
#     c_t = torch.tensor([cj], dtype=torch.float32, device=device)
# 
#     n = xj_t.shape[0]
#     tau_plan = [(0, 0, n, 0, n, 1.0)]
#     kw_list, sigma_uv_list = _precompute_kw_list(
#         X=X, xj=xj_t, j=j, tau_plan=tau_plan, device=device, sigma=sigma
#     )
#     I = _kernel_dependence_batch_for_c_values(
#         c_t,
#         X,
#         xj_t,
#         zj_t,
#         j,
#         tau_plan,
#         sigma=sigma,
#         precomputed_kw_list=kw_list,
#         precomputed_sigma_uv_list=sigma_uv_list,
#     )
#     return float(I.item())
# 
# 
# def timeseries_kernel_dependence(
#     cj: float,
#     X: np.ndarray,
#     xj: np.ndarray,
#     zj: np.ndarray,
#     j: int,
#     lookback: int = 60,
#     sigma: float = 1.0,
# ) -> float:
#     device = _device()
#     xj_t = torch.as_tensor(xj, dtype=torch.float32, device=device)
#     zj_t = torch.as_tensor(zj, dtype=torch.float32, device=device)
#     c_t = torch.tensor([cj], dtype=torch.float32, device=device)
# 
#     n = xj_t.shape[0]
#     weights = np.exp(-1.0 * np.arange(lookback + 1))
#     weights = weights / np.sum(weights)
# 
#     tau_plan = [(0, 0, n, 0, n, float(weights[0]))]
#     for tau in range(1, lookback + 1):
#         tau_plan.append((tau, tau, n, 0, n - tau, float(weights[tau])))
# 
#     kw_list, sigma_uv_list = _precompute_kw_list(
#         X=X, xj=xj_t, j=j, tau_plan=tau_plan, device=device, sigma=sigma
#     )
#     I = _kernel_dependence_batch_for_c_values(
#         c_t,
#         X,
#         xj_t,
#         zj_t,
#         j,
#         tau_plan,
#         sigma=sigma,
#         precomputed_kw_list=kw_list,
#         precomputed_sigma_uv_list=sigma_uv_list,
#     )
#     return float(I.item())
# 
# 
# def find_cj(
#     X,
#     xj,
#     zj,
#     j,
#     lookback: int = 30,
#     sigma: float = 1.0,
#     c_min: float = 1e-6,
#     c_max: float = None,
#     n_c_grid: int = 80,
#     c_batch_size: int = 16,
# ):
#     """
#     Legacy scalar c_j search (kept for backward compatibility).
#     """
#     device = _device()
# 
#     xj_t = torch.as_tensor(np.asarray(xj), dtype=torch.float32, device=device)
#     zj_t = torch.as_tensor(np.asarray(zj), dtype=torch.float32, device=device)
#     n = xj_t.shape[0]
# 
#     weights = np.exp(-1.0 * np.arange(lookback + 1))
#     weights = weights / np.sum(weights)
# 
#     tau_plan = [(0, 0, n, 0, n, float(weights[0]))]
#     for tau in range(1, lookback + 1):
#         tau_plan.append((tau, tau, n, 0, n - tau, float(weights[tau])))
# 
#     if c_max is None:
#         xj_std = float(np.std(np.asarray(xj)))
#         zj_std = float(np.std(np.asarray(zj)))
#         scale_ratio = xj_std / (zj_std + 1e-12)
#         c_max = float(np.clip(2.0 * scale_ratio, 1.0, 5.0))
# 
#     c_min = float(max(1e-6, c_min))
#     c_max = float(max(c_min + 1e-6, c_max))
#     n_c_grid = int(max(10, n_c_grid))
#     c_values = np.linspace(c_min, c_max, n_c_grid)
# 
#     kw_list, sigma_uv_list = _precompute_kw_list(
#         X=X, xj=xj_t, j=j, tau_plan=tau_plan, device=device, sigma=sigma
#     )
# 
#     c_batch_size = int(max(1, c_batch_size))
#     I_kj_values = []
# 
#     for start in range(0, len(c_values), c_batch_size):
#         c_chunk_np = c_values[start:start + c_batch_size]
#         c_chunk_t = torch.as_tensor(c_chunk_np, dtype=torch.float32, device=device)
# 
#         I_chunk = _kernel_dependence_batch_for_c_values(
#             c_chunk_t,
#             X,
#             xj_t,
#             zj_t,
#             j,
#             tau_plan,
#             sigma=sigma,
#             precomputed_kw_list=kw_list,
#             precomputed_sigma_uv_list=sigma_uv_list,
#         )
#         I_kj_values.extend(I_chunk.detach().cpu().numpy().tolist())
# 
#     min_index = int(np.argmin(I_kj_values))
# 
#     window_size = 5
#     start_index = max(0, min_index - window_size)
#     end_index = min(len(c_values), min_index + window_size + 1)
# 
#     c_window = c_values[start_index:end_index]
#     I_kj_window = np.asarray(I_kj_values[start_index:end_index], dtype=float)
# 
#     if len(c_window) < 4:
#         return float(c_values[min_index])
# 
#     spline = UnivariateSpline(c_window, I_kj_window, k=3, s=0)
#     c_finer = np.linspace(c_window[0], c_window[-1], 100)
#     I_kj_finer = spline(c_finer)
#     min_index_finer = int(np.argmin(I_kj_finer))
#     c_optimal = float(c_finer[min_index_finer])
# 
#     _maybe_plot_ij_curve(j=j, c_values=c_values, I_values=np.asarray(I_kj_values, dtype=float), c_optimal=c_optimal)
#     return c_optimal
# 
# 
# # =========================================================
# # Group Neural Gaussian Mirror: matrix G APIs
# =========================================================

def _to_2d_torch(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim != 2:
        raise ValueError("Expected a 2D array/tensor.")
    return t


def _symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.T)


def _sqrtm_psd(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    A = _symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def _invsqrtm_psd(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    A = _symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T


def _psd_project(A: torch.Tensor) -> torch.Tensor:
    A = _symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=0.0)
    return evecs @ torch.diag(evals) @ evecs.T


def _build_group_ridge_geometry(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    ridge_lambda_w: float = 1.0,
    ridge_lambda_z: float = 1.0,
    ridge_q_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    High-dimensional ridge-preconditioned geometry:
      P_w = W (W^T W + lambda_w I)^(-1) W^T
      R   = (I - P_w) Xg
      B   = [W, R]
      A   = I - B (B^T B + lambda_z I)^(-1) B^T
      Q0  = A Zg
      Q   = Q0 (Q0^T Q0 + eps I)^(-1/2)
    """
    n = Xg.shape[0]
    if W.numel() == 0 or W.shape[1] == 0:
        R = Xg
        B = R
    else:
        dw = W.shape[1]
        lam_w = float(max(1e-12, ridge_lambda_w))
        Iw = torch.eye(dw, dtype=W.dtype, device=W.device)
        XtX_w = _symmetrize(W.T @ W) + lam_w * Iw
        K_w = torch.linalg.solve(XtX_w, W.T)  # (dw, n)
        Pw = W @ K_w  # (n, n)
        Mw = torch.eye(n, dtype=W.dtype, device=W.device) - Pw
        R = Mw @ Xg
        B = torch.cat([W, R], dim=1)

    db = B.shape[1]
    lam_z = float(max(1e-12, ridge_lambda_z))
    Ib = torch.eye(db, dtype=B.dtype, device=B.device)
    BtB = _symmetrize(B.T @ B) + lam_z * Ib
    K_b = torch.linalg.solve(BtB, B.T)  # (db, n)
    A = torch.eye(n, dtype=B.dtype, device=B.device) - B @ K_b

    Q0 = A @ Zg
    QtQ = _symmetrize(Q0.T @ Q0)
    invsqrt_q = _invsqrtm_psd(QtQ + float(max(1e-12, ridge_q_eps)) * torch.eye(QtQ.shape[0], dtype=QtQ.dtype, device=QtQ.device))
    Q = Q0 @ invsqrt_q

    return {
        "R": R,
        "Q": Q,
        "A": A,
    }


def _resolve_ridge_params(
    n: int,
    d_w: int,
    m: int,
    ridge_lambda_w: Optional[float],
    ridge_lambda_z: Optional[float],
    ridge_q_eps: Optional[float],
    ridge_g_delta: Optional[float],
) -> Tuple[float, float, float, float]:
    """
    Dimension-adaptive defaults for standardized data.
    """
    n_f = float(max(1, int(n)))
    d_f = float(max(0, int(d_w)))
    m_f = float(max(1, int(m)))
    k_w = d_f / n_f
    k_b = (d_f + m_f) / n_f

    lam_w = float(ridge_lambda_w) if ridge_lambda_w is not None else n_f * float(np.clip(k_w, 0.1, 1.0))
    lam_z = float(ridge_lambda_z) if ridge_lambda_z is not None else n_f * float(np.clip(k_b, 0.1, 1.0))
    q_eps = float(ridge_q_eps) if ridge_q_eps is not None else 1e-6
    g_delta = float(ridge_g_delta) if ridge_g_delta is not None else 1e-6
    return lam_w, lam_z, q_eps, g_delta


def _build_group_ridge_benchmark(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    ridge_lambda_w: float = 1.0,
    ridge_lambda_z: float = 1.0,
    ridge_q_eps: float = 1e-6,
    ridge_g_delta: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    High-dimensional ridge benchmark:
      Sigma = (Xg^T M_w Xg)/n, with M_w from ridge smoother on W
      G0 = (PSD(Sigma) + delta I)^(1/2)
    Also returns ridge-preconditioned Q for mirror construction.
    """
    geom = _build_group_ridge_geometry(
        Xg=Xg,
        W=W,
        Zg=Zg,
        ridge_lambda_w=ridge_lambda_w,
        ridge_lambda_z=ridge_lambda_z,
        ridge_q_eps=ridge_q_eps,
    )
    R = geom["R"]
    n = Xg.shape[0]
    Sigma = _symmetrize((R.T @ R) / float(n))
    Sigma_psd = _psd_project(Sigma)
    m = Xg.shape[1]
    G0 = _sqrtm_psd(Sigma_psd + float(max(1e-12, ridge_g_delta)) * torch.eye(m, dtype=Xg.dtype, device=Xg.device))
    info = {
        "RtR": (R.T @ R).detach().cpu().numpy(),
        "residual_mode": "ridge",
        "rank_R": int(torch.linalg.matrix_rank(R).item()),
        "rank_Q": int(torch.linalg.matrix_rank(geom["Q"]).item()),
        "ridge_lambda_w": float(ridge_lambda_w),
        "ridge_lambda_z": float(ridge_lambda_z),
        "ridge_q_eps": float(ridge_q_eps),
        "ridge_g_delta": float(ridge_g_delta),
    }
    return G0, {"base_info": info, "geometry": geom}


def _residualize_ols(target: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return target
    coef = torch.linalg.lstsq(basis, target).solution
    return target - basis @ coef


def _residualize_gls(target: torch.Tensor, basis: torch.Tensor, omega_inv: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return target
    BtOi = basis.T @ omega_inv
    A = BtOi @ basis + eps * torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    B = BtOi @ target
    coef = torch.linalg.solve(A, B)
    return target - basis @ coef


def _estimate_residual_covariance(
    Xg: torch.Tensor,
    W: torch.Tensor,
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (Sigma_hat, R) where R is residualized group block.
    """
    n = Xg.shape[0]
    if residual_mode == "ols":
        R = _residualize_ols(Xg, W)
        Sigma = (R.T @ R) / float(n)
        return _symmetrize(Sigma), R

    if residual_mode == "gls":
        if omega is None:
            raise ValueError("omega must be provided when residual_mode='gls'.")
        omega_inv = torch.linalg.inv(omega)
        R = _residualize_gls(Xg, W, omega_inv=omega_inv, eps=eps)
        Sigma = (R.T @ omega_inv @ R) / float(n)
        return _symmetrize(Sigma), R

    raise ValueError("residual_mode must be 'ols' or 'gls'.")


def _prepare_group_noise_basis(
    Zg: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build Q = (I - P_[W,R]) Zg (or GLS analogue), matching linear Group GM derivation.
    """
    WR = torch.cat([W, R], dim=1) if W.numel() > 0 else R
    if residual_mode == "ols":
        return _residualize_ols(Zg, WR)
    if residual_mode == "gls":
        if omega is None:
            raise ValueError("omega must be provided when residual_mode='gls'.")
        omega_inv = torch.linalg.inv(omega)
        return _residualize_gls(Zg, WR, omega_inv=omega_inv, eps=eps)
    raise ValueError("residual_mode must be 'ols' or 'gls'.")


def _kw_diagnostics(Kw: torch.Tensor) -> Dict[str, float]:
    n = Kw.shape[0]
    diag = torch.diagonal(Kw)
    off_mask = ~torch.eye(n, dtype=torch.bool, device=Kw.device)
    off_vals = Kw[off_mask]

    eigvals = torch.linalg.eigvalsh(_symmetrize(Kw))
    eigvals = torch.clamp(eigvals, min=0.0)
    eigsum = torch.sum(eigvals) + 1e-12
    p = eigvals / eigsum
    entropy = -torch.sum(p * torch.log(torch.clamp(p, min=1e-12)))
    eff_rank = torch.exp(entropy)

    return {
        "diag_mean": float(diag.mean().item()),
        "diag_std": float(diag.std(unbiased=False).item()),
        "off_mean": float(off_vals.mean().item()) if off_vals.numel() > 0 else 0.0,
        "off_std": float(off_vals.std(unbiased=False).item()) if off_vals.numel() > 0 else 0.0,
        "off_abs_mean": float(torch.mean(torch.abs(off_vals)).item()) if off_vals.numel() > 0 else 0.0,
        "effective_rank": float(eff_rank.item()),
        "n": int(n),
    }


def _group_geometry_diagnostics(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    G: torch.Tensor,
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Minimal geometry diagnostics for group mirror construction.
    Focuses on:
      1) residualization strength (R vs Xg, Q vs Zg)
      2) mirror separability / potential collapse (corr(U, V))
      3) noise-basis conditioning (QtQ spectrum)
    """
    _, R = _estimate_residual_covariance(Xg, W, residual_mode=residual_mode, omega=omega)
    Q = _prepare_group_noise_basis(Zg, W, R, residual_mode=residual_mode, omega=omega)
    QG = Q @ G
    U = Xg + QG
    V = Xg - QG

    xg_nrm = float(torch.linalg.norm(Xg).detach().item())
    zg_nrm = float(torch.linalg.norm(Zg).detach().item())
    r_nrm = float(torch.linalg.norm(R).detach().item())
    q_nrm = float(torch.linalg.norm(Q).detach().item())

    u_flat = U.reshape(-1)
    v_flat = V.reshape(-1)
    u_center = u_flat - u_flat.mean()
    v_center = v_flat - v_flat.mean()
    uv_denom = float(torch.linalg.norm(u_center).detach().item() * torch.linalg.norm(v_center).detach().item())
    uv_corr = float(torch.dot(u_center, v_center).detach().item() / (uv_denom + eps))

    QtQ = _symmetrize(Q.T @ Q)
    eig_q = torch.linalg.eigvalsh(QtQ)
    eig_q = torch.clamp(eig_q, min=0.0)
    eig_min = float(torch.min(eig_q).detach().item())
    eig_max = float(torch.max(eig_q).detach().item())
    cond_q = eig_max / (max(eig_min, 0.0) + 1e-8)
    rank_q = int(torch.linalg.matrix_rank(Q).detach().item())
    rank_r = int(torch.linalg.matrix_rank(R).detach().item())

    return {
        "ratio_R_over_Xg": r_nrm / (xg_nrm + eps),
        "ratio_Q_over_Zg": q_nrm / (zg_nrm + eps),
        "corr_UV_flat": uv_corr,
        "eigmin_QtQ": eig_min,
        "eigmax_QtQ": eig_max,
        "cond_QtQ": float(cond_q),
        "rank_R": rank_r,
        "rank_Q": rank_q,
    }


def _group_score_linear_surrogate(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Linear-kernel reduction surrogate:
      || (H U)^T (H V) / n ||_F^2
    """
    n = U.shape[0]
    Uc = U - U.mean(dim=0, keepdim=True)
    Vc = V - V.mean(dim=0, keepdim=True)
    C = (Uc.T @ Vc) / float(n)
    return (C * C).sum()


def _group_kernel_score_fixed_kw(
    U: torch.Tensor,
    V: torch.Tensor,
    Kw: torch.Tensor,
    sigma_u: float,
    sigma_v: float,
    tfm_u: Optional[Dict[str, torch.Tensor]] = None,
    tfm_v: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if tfm_u is not None:
        U = _apply_kernel_transform(U, tfm_u)
    if tfm_v is not None:
        V = _apply_kernel_transform(V, tfm_v)

    # Fast path: evaluate U/V kernels in one batched call to improve GPU utilization.
    UV = torch.stack([U, V], dim=0)  # (2, n, d)
    sigma_pair = torch.as_tensor([float(sigma_u), float(sigma_v)], dtype=UV.dtype, device=UV.device)
    K_pair = _rbf_kernel_matrix_batch_multivariate_per_sigma(UV, sigma_pair)  # (2, n, n)
    Kc_pair = _center_kernel_matrix_batch(K_pair)  # (2, n, n)
    Ku_c, Kv_c = Kc_pair[0], Kc_pair[1]

    n = U.shape[0]
    return (Ku_c * Kv_c * Kw).sum() / float(n * n)


def _pca_project_for_kw(
    W: torch.Tensor,
    n_components: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    PCA projection for W used only in Kw construction.
    Returns projected W and metadata.
    """
    if W.ndim != 2:
        raise ValueError("_pca_project_for_kw expects a 2D tensor.")
    n, d = W.shape
    if d == 0 or n <= 1:
        return W, {"kw_pca_used": 0, "kw_pca_components": 0, "kw_pca_input_dim": int(d)}

    k = int(max(1, min(int(n_components), d, n)))
    if k >= d:
        return W, {"kw_pca_used": 0, "kw_pca_components": int(d), "kw_pca_input_dim": int(d)}

    W_center = W - W.mean(dim=0, keepdim=True)
    # pca_lowrank is typically more stable/faster than full SVD in high-d W.
    U, S, _ = torch.pca_lowrank(W_center, q=k, center=False)
    U = U[:, :k]
    S = torch.clamp(S[:k], min=eps)
    W_pca = U * S.unsqueeze(0)
    return W_pca, {"kw_pca_used": 1, "kw_pca_components": int(k), "kw_pca_input_dim": int(d)}


def _precompute_group_kernel_terms(
    Xg: torch.Tensor,
    W: torch.Tensor,
    kw_use_pca: bool = False,
    kw_pca_components: Optional[int] = None,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_prefix: str = "",
) -> Dict[str, object]:
    tfm_u = _fit_kernel_transform(Xg)
    Xg_t = _apply_kernel_transform(Xg, tfm_u)
    sigma_u = _median_pairwise_distance(_subsample_rows_for_bw(Xg_t))
    sigma_v = sigma_u

    W_for_kw = W
    kw_pca_meta = {"kw_pca_used": 0, "kw_pca_components": int(W.shape[1]), "kw_pca_input_dim": int(W.shape[1])}
    if bool(kw_use_pca) and W.shape[1] > 0:
        if kw_pca_components is None:
            # Default: align with lag-block width m for time-series groups.
            kw_pca_components = int(Xg.shape[1])
        W_for_kw, kw_pca_meta = _pca_project_for_kw(W, n_components=int(kw_pca_components))

    tfm_w = _fit_kernel_transform(W_for_kw)
    W_t = _apply_kernel_transform(W_for_kw, tfm_w)
    sigma_w = _median_pairwise_distance(_subsample_rows_for_bw(W_t))
    Kw = _rbf_kernel_matrix_batch_multivariate(W_t.unsqueeze(0), sigma=sigma_w).squeeze(0)
    kw_diag = _kw_diagnostics(Kw)
    if print_kw_diagnostics:
        prefix = f"{kw_diagnostics_prefix} " if kw_diagnostics_prefix else ""
        print(
            f"{prefix}Kw diagnostics: "
            f"diag_mean={kw_diag['diag_mean']:.6f}, diag_std={kw_diag['diag_std']:.6f}, "
            f"off_mean={kw_diag['off_mean']:.6f}, off_std={kw_diag['off_std']:.6f}, "
            f"off_abs_mean={kw_diag['off_abs_mean']:.6f}, effective_rank={kw_diag['effective_rank']:.3f}, n={kw_diag['n']}",
            flush=True,
        )
    return {
        "sigma_u": float(sigma_u),
        "sigma_v": float(sigma_v),
        "sigma_w": float(sigma_w),
        "Kw": Kw,
        "tfm_u": tfm_u,
        "tfm_v": tfm_u,
        "tfm_w": tfm_w,
        "Kw_diagnostics": kw_diag,
        "Kw_pca": kw_pca_meta,
    }


def _build_G_benchmark(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    use_high_dim_ridge: bool = False,
    ridge_lambda_w: float = 1.0,
    ridge_lambda_z: float = 1.0,
    ridge_q_eps: float = 1e-6,
    ridge_g_delta: float = 1e-6,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Linear Group-GM closed-form benchmark (exactly matching derivation):
      R = M_W Xg
      Q = (I - P_[W,R]) Zg
      G = (Q^T Q)^(-1/2) (R^T R)^(1/2)
    """
    if use_high_dim_ridge:
        G0, ridge_pack = _build_group_ridge_benchmark(
            Xg=Xg,
            W=W,
            Zg=Zg,
            ridge_lambda_w=ridge_lambda_w,
            ridge_lambda_z=ridge_lambda_z,
            ridge_q_eps=ridge_q_eps,
            ridge_g_delta=ridge_g_delta,
        )
        info = dict(ridge_pack["base_info"])
        info["use_high_dim_ridge"] = True
        return G0, info

    _, R = _estimate_residual_covariance(Xg, W, residual_mode=residual_mode, omega=omega, eps=eps)
    Q = _prepare_group_noise_basis(Zg, W, R, residual_mode=residual_mode, omega=omega, eps=eps)

    if residual_mode == "ols":
        RtR = _symmetrize(R.T @ R)
        QtQ = _symmetrize(Q.T @ Q)
    else:
        omega_inv = torch.linalg.inv(omega)
        RtR = _symmetrize(R.T @ omega_inv @ R)
        QtQ = _symmetrize(Q.T @ omega_inv @ Q)

    eval_q, evec_q = torch.linalg.eigh(QtQ)
    eval_q = torch.clamp(eval_q, min=eps)
    invsqrt_q = evec_q @ torch.diag(1.0 / torch.sqrt(eval_q)) @ evec_q.T

    sqrt_r = _sqrtm_psd(RtR, eps=eps)
    G_bench = invsqrt_q @ sqrt_r
    info = {
        "RtR": RtR.detach().cpu().numpy(),
        "QtQ": QtQ.detach().cpu().numpy(),
        "residual_mode": residual_mode,
        "use_high_dim_ridge": False,
        "rank_R": int(torch.linalg.matrix_rank(R).item()),
        "rank_Q": int(torch.linalg.matrix_rank(Q).item()),
    }
    return G_bench, info


def _build_G_scheme_a_diag(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    objective: str = "kernel",
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    lambda_reg: float = 1e-2,
    max_iter: int = 100,
    lr: float = 1e-2,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_prefix: str = "",
    print_objective_every: int = 0,
    objective_print_prefix: str = "",
    kw_use_pca: bool = False,
    kw_pca_components: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    G_bench, base_info = _build_G_benchmark(Xg, W, Zg, residual_mode=residual_mode, omega=omega, eps=eps)
    RtR = torch.as_tensor(base_info["RtR"], dtype=Xg.dtype, device=Xg.device)
    _, R = _estimate_residual_covariance(Xg, W, residual_mode=residual_mode, omega=omega, eps=eps)
    Q = _prepare_group_noise_basis(Zg, W, R, residual_mode=residual_mode, omega=omega, eps=eps)

    evals, evecs = torch.linalg.eigh(_symmetrize(RtR))
    evals = torch.clamp(evals, min=eps)
    sqrt_evals = torch.sqrt(evals)
    m = Xg.shape[1]

    # d initialized near 1.0: softplus(raw)=1 => raw=log(exp(1)-1)
    init_raw = float(np.log(np.exp(1.0) - 1.0))
    d_raw = torch.full((m,), init_raw, dtype=Xg.dtype, device=Xg.device, requires_grad=True)
    opt = torch.optim.Adam([d_raw], lr=float(lr))

    kernel_terms = None
    if objective == "kernel":
        kernel_terms = _precompute_group_kernel_terms(
            Xg.detach(),
            W.detach(),
            kw_use_pca=kw_use_pca,
            kw_pca_components=kw_pca_components,
            print_kw_diagnostics=print_kw_diagnostics,
            kw_diagnostics_prefix=kw_diagnostics_prefix,
        )
        Kw = kernel_terms["Kw"].to(device=Xg.device, dtype=Xg.dtype)
        sigma_u = float(kernel_terms["sigma_u"])
        sigma_v = float(kernel_terms["sigma_v"])
        tfm_u = kernel_terms["tfm_u"]
        tfm_v = kernel_terms["tfm_v"]

    loss_hist = []
    eye_diag = torch.ones(m, dtype=Xg.dtype, device=Xg.device)

    iters = int(max_iter)
    print_every = int(max(0, print_objective_every))
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        d = torch.nn.functional.softplus(d_raw) + eps
        G = evecs @ torch.diag(sqrt_evals * d) @ evecs.T
        QG = Q @ G
        U = Xg + QG
        V = Xg - QG

        if objective == "kernel":
            dep = _group_kernel_score_fixed_kw(
                U,
                V,
                Kw=Kw,
                sigma_u=sigma_u,
                sigma_v=sigma_v,
                tfm_u=tfm_u,
                tfm_v=tfm_v,
            )
        elif objective == "linear_surrogate":
            dep = _group_score_linear_surrogate(U, V)
        else:
            raise ValueError("objective must be 'kernel' or 'linear_surrogate'.")

        reg = float(lambda_reg) * ((d - eye_diag) ** 2).sum()
        loss = dep + reg
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().item()))
        if print_every > 0 and (((it + 1) % print_every == 0) or (it == 0) or (it + 1 == iters)):
            prefix = f"{objective_print_prefix} " if objective_print_prefix else ""
            print(
                f"{prefix}[scheme_a_diag] iter={it + 1}/{iters}, "
                f"objective={float(dep.detach().item()):.6e}, "
                f"reg={float(reg.detach().item()):.6e}, "
                f"loss={float(loss.detach().item()):.6e}",
                flush=True,
            )

    with torch.no_grad():
        d_final = torch.nn.functional.softplus(d_raw) + eps
        G_final = evecs @ torch.diag(sqrt_evals * d_final) @ evecs.T

    info = {
        "method": "scheme_a_diag",
        "objective": objective,
        "loss_history": loss_hist,
        "d_final": d_final.detach().cpu().numpy(),
        "base_info": base_info,
        "kernel_terms": {
            "sigma_u": kernel_terms["sigma_u"],
            "sigma_v": kernel_terms["sigma_v"],
            "sigma_w": kernel_terms["sigma_w"],
            "kw_pca": kernel_terms["Kw_pca"],
        } if kernel_terms is not None else None,
    }
    return G_final.detach(), info


def _build_G_scheme_b_local(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    objective: str = "kernel",
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    lambda_reg: float = 1e-2,
    max_iter: int = 100,
    lr: float = 1e-2,
    rank_r: Optional[int] = None,
    enforce_symmetric_delta: bool = False,
    delta_norm_cap: Optional[float] = None,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_prefix: str = "",
    print_objective_every: int = 0,
    objective_print_prefix: str = "",
    kw_use_pca: bool = False,
    kw_pca_components: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    G0, base_info = _build_G_benchmark(Xg, W, Zg, residual_mode=residual_mode, omega=omega, eps=eps)
    _, R = _estimate_residual_covariance(Xg, W, residual_mode=residual_mode, omega=omega, eps=eps)
    Q = _prepare_group_noise_basis(Zg, W, R, residual_mode=residual_mode, omega=omega, eps=eps)
    m = Xg.shape[1]

    params = []
    A = B = Delta = None
    if rank_r is not None:
        r = int(max(1, min(int(rank_r), m)))
        A = torch.zeros((m, r), dtype=Xg.dtype, device=Xg.device, requires_grad=True)
        B = torch.zeros((m, r), dtype=Xg.dtype, device=Xg.device, requires_grad=True)
        params = [A, B]
    else:
        Delta = torch.zeros((m, m), dtype=Xg.dtype, device=Xg.device, requires_grad=True)
        params = [Delta]

    opt = torch.optim.Adam(params, lr=float(lr))

    kernel_terms = None
    if objective == "kernel":
        kernel_terms = _precompute_group_kernel_terms(
            Xg.detach(),
            W.detach(),
            kw_use_pca=kw_use_pca,
            kw_pca_components=kw_pca_components,
            print_kw_diagnostics=print_kw_diagnostics,
            kw_diagnostics_prefix=kw_diagnostics_prefix,
        )
        Kw = kernel_terms["Kw"].to(device=Xg.device, dtype=Xg.dtype)
        sigma_u = float(kernel_terms["sigma_u"])
        sigma_v = float(kernel_terms["sigma_v"])
        tfm_u = kernel_terms["tfm_u"]
        tfm_v = kernel_terms["tfm_v"]

    loss_hist = []

    iters = int(max_iter)
    print_every = int(max(0, print_objective_every))
    for it in range(iters):
        opt.zero_grad(set_to_none=True)

        if rank_r is not None:
            delta_cur = A @ B.T
        else:
            delta_cur = Delta

        if enforce_symmetric_delta:
            delta_cur = _symmetrize(delta_cur)

        G = G0 + delta_cur
        QG = Q @ G
        U = Xg + QG
        V = Xg - QG

        if objective == "kernel":
            dep = _group_kernel_score_fixed_kw(
                U,
                V,
                Kw=Kw,
                sigma_u=sigma_u,
                sigma_v=sigma_v,
                tfm_u=tfm_u,
                tfm_v=tfm_v,
            )
        elif objective == "linear_surrogate":
            dep = _group_score_linear_surrogate(U, V)
        else:
            raise ValueError("objective must be 'kernel' or 'linear_surrogate'.")

        reg = float(lambda_reg) * (delta_cur * delta_cur).sum()
        loss = dep + reg
        loss.backward()
        opt.step()

        # Optional trust-region projection on ||Delta||_F
        if delta_norm_cap is not None and rank_r is None:
            with torch.no_grad():
                nrm = torch.linalg.norm(Delta)
                cap = float(delta_norm_cap)
                if float(nrm.item()) > cap and cap > 0:
                    Delta.mul_(cap / (float(nrm.item()) + 1e-12))

        loss_hist.append(float(loss.detach().item()))
        if print_every > 0 and (((it + 1) % print_every == 0) or (it == 0) or (it + 1 == iters)):
            prefix = f"{objective_print_prefix} " if objective_print_prefix else ""
            print(
                f"{prefix}[scheme_b_local] iter={it + 1}/{iters}, "
                f"objective={float(dep.detach().item()):.6e}, "
                f"reg={float(reg.detach().item()):.6e}, "
                f"loss={float(loss.detach().item()):.6e}",
                flush=True,
            )

    with torch.no_grad():
        if rank_r is not None:
            delta_final = A @ B.T
        else:
            delta_final = Delta
        if enforce_symmetric_delta:
            delta_final = _symmetrize(delta_final)
        G_final = G0 + delta_final

    info = {
        "method": "scheme_b_local",
        "objective": objective,
        "loss_history": loss_hist,
        "delta_norm": float(torch.linalg.norm(delta_final).detach().item()),
        "base_info": base_info,
        "kernel_terms": {
            "sigma_u": kernel_terms["sigma_u"],
            "sigma_v": kernel_terms["sigma_v"],
            "sigma_w": kernel_terms["sigma_w"],
            "kw_pca": kernel_terms["Kw_pca"],
        } if kernel_terms is not None else None,
    }
    return G_final.detach(), info


def _build_G_scheme_c_free(
    Xg: torch.Tensor,
    W: torch.Tensor,
    Zg: torch.Tensor,
    objective: str = "kernel",
    residual_mode: str = "ols",
    omega: Optional[torch.Tensor] = None,
    lambda_reg: float = 1e-3,
    max_iter: int = 200,
    lr: float = 5e-3,
    enforce_symmetric: bool = False,
    g_norm_cap: Optional[float] = None,
    print_kw_diagnostics: bool = False,
    kw_diagnostics_prefix: str = "",
    print_objective_every: int = 0,
    objective_print_prefix: str = "",
    kw_use_pca: bool = False,
    kw_pca_components: Optional[int] = None,
    use_high_dim_ridge: bool = False,
    ridge_lambda_w: float = 1.0,
    ridge_lambda_z: float = 1.0,
    ridge_q_eps: float = 1e-6,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Fully free G optimization with identity initialization:
      G starts at I and is optimized directly (full matrix parameters).
    """
    if use_high_dim_ridge:
        geom = _build_group_ridge_geometry(
            Xg=Xg,
            W=W,
            Zg=Zg,
            ridge_lambda_w=ridge_lambda_w,
            ridge_lambda_z=ridge_lambda_z,
            ridge_q_eps=ridge_q_eps,
        )
        R = geom["R"]
        Q = geom["Q"]
    else:
        _, R = _estimate_residual_covariance(Xg, W, residual_mode=residual_mode, omega=omega, eps=eps)
        Q = _prepare_group_noise_basis(Zg, W, R, residual_mode=residual_mode, omega=omega, eps=eps)
    m = Xg.shape[1]

    G = torch.eye(m, dtype=Xg.dtype, device=Xg.device, requires_grad=True)
    opt = torch.optim.Adam([G], lr=float(lr))

    kernel_terms = None
    if objective == "kernel":
        kernel_terms = _precompute_group_kernel_terms(
            Xg.detach(),
            W.detach(),
            kw_use_pca=kw_use_pca,
            kw_pca_components=kw_pca_components,
            print_kw_diagnostics=print_kw_diagnostics,
            kw_diagnostics_prefix=kw_diagnostics_prefix,
        )
        Kw = kernel_terms["Kw"].to(device=Xg.device, dtype=Xg.dtype)
        sigma_u = float(kernel_terms["sigma_u"])
        sigma_v = float(kernel_terms["sigma_v"])
        tfm_u = kernel_terms["tfm_u"]
        tfm_v = kernel_terms["tfm_v"]

    eye_m = torch.eye(m, dtype=Xg.dtype, device=Xg.device)
    loss_hist = []
    objective_hist = []
    reg_hist = []

    iters = int(max_iter)
    print_every = int(max(0, print_objective_every))
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        G_cur = _symmetrize(G) if enforce_symmetric else G
        QG = Q @ G_cur
        U = Xg + QG
        V = Xg - QG

        if objective == "kernel":
            dep = _group_kernel_score_fixed_kw(
                U,
                V,
                Kw=Kw,
                sigma_u=sigma_u,
                sigma_v=sigma_v,
                tfm_u=tfm_u,
                tfm_v=tfm_v,
            )
        elif objective == "linear_surrogate":
            dep = _group_score_linear_surrogate(U, V)
        else:
            raise ValueError("objective must be 'kernel' or 'linear_surrogate'.")

        reg = float(lambda_reg) * ((G_cur - eye_m) * (G_cur - eye_m)).sum()
        loss = dep + reg
        loss.backward()
        opt.step()

        if g_norm_cap is not None:
            with torch.no_grad():
                nrm = torch.linalg.norm(G)
                cap = float(g_norm_cap)
                if float(nrm.item()) > cap and cap > 0:
                    G.mul_(cap / (float(nrm.item()) + 1e-12))

        objective_hist.append(float(dep.detach().item()))
        reg_hist.append(float(reg.detach().item()))
        loss_hist.append(float(loss.detach().item()))
        if print_every > 0 and (((it + 1) % print_every == 0) or (it == 0) or (it + 1 == iters)):
            prefix = f"{objective_print_prefix} " if objective_print_prefix else ""
            print(
                f"{prefix}[scheme_c_free] iter={it + 1}/{iters}, "
                f"objective={float(dep.detach().item()):.6e}, "
                f"reg={float(reg.detach().item()):.6e}, "
                f"loss={float(loss.detach().item()):.6e}",
                flush=True,
            )

    with torch.no_grad():
        G_final = _symmetrize(G) if enforce_symmetric else G

    info = {
        "method": "scheme_c_free",
        "objective": objective,
        "objective_history": objective_hist,
        "reg_history": reg_hist,
        "loss_history": loss_hist,
        "g_norm": float(torch.linalg.norm(G_final).detach().item()),
        "use_high_dim_ridge": bool(use_high_dim_ridge),
        "ridge_params": {
            "ridge_lambda_w": float(ridge_lambda_w),
            "ridge_lambda_z": float(ridge_lambda_z),
            "ridge_q_eps": float(ridge_q_eps),
        } if use_high_dim_ridge else None,
        "kernel_terms": {
            "sigma_u": kernel_terms["sigma_u"],
            "sigma_v": kernel_terms["sigma_v"],
            "sigma_w": kernel_terms["sigma_w"],
            "kw_pca": kernel_terms["Kw_pca"],
        } if kernel_terms is not None else None,
    }
    return G_final.detach(), info


def find_Gj(
    Xg,
    W,
    method: str = "benchmark",
    Zg=None,
    objective: str = "kernel",
    residual_mode: str = "ols",
    omega=None,
    lambda_reg: float = 1e-2,
    max_iter: Optional[int] = None,
    lr: float = 1e-2,
    rank_r: Optional[int] = None,
    enforce_symmetric_delta: bool = False,
    delta_norm_cap: Optional[float] = None,
    free_enforce_symmetric: bool = False,
    free_g_norm_cap: Optional[float] = None,
    kw_use_pca: bool = False,
    kw_pca_components: Optional[int] = None,
    high_dim_mode: str = "auto",  # auto | ridge | standard
    ridge_lambda_w: Optional[float] = None,
    ridge_lambda_z: Optional[float] = None,
    ridge_q_eps: Optional[float] = None,
    ridge_g_delta: Optional[float] = None,
    print_objective_every: int = 0,
    objective_print_prefix: str = "",
    random_state: int = 42,
    print_kw_diagnostics: bool = True,
    kw_diagnostics_prefix: str = "",
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    Group Neural Gaussian Mirror matrix estimator.

    Methods
    -------
    - benchmark:      G_bench = sqrt(Sigma_{g|W})
    - scheme_c_free:  G initialized at I and directly optimized (full matrix)

    Notes
    -----
    - scheme_a_diag / scheme_b_local are temporarily disabled.
    - high_dim_mode='auto' switches to ridge geometry when dim(W) > n.
    """
    device = _device()
    Xg_t = _to_2d_torch(Xg, device=device, dtype=torch.float32)
    W_t = _to_2d_torch(W, device=device, dtype=torch.float32)

    if Xg_t.shape[0] != W_t.shape[0]:
        raise ValueError("Xg and W must have the same number of rows.")

    if omega is not None:
        omega_t = _to_2d_torch(omega, device=device, dtype=torch.float32)
    else:
        omega_t = None

    if Zg is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(random_state))
        Zg_t = torch.randn(Xg_t.shape, dtype=Xg_t.dtype, device=device, generator=gen)
    else:
        Zg_t = _to_2d_torch(Zg, device=device, dtype=torch.float32)
        if Zg_t.shape != Xg_t.shape:
            raise ValueError("Zg must have the same shape as Xg.")

    method = str(method).lower()
    high_dim_mode = str(high_dim_mode).lower()
    d_w = int(W_t.shape[1])
    n = int(Xg_t.shape[0])
    if high_dim_mode == "auto":
        use_high_dim_ridge = bool(HIGH_DIM_AUTO_D_GT_N and (d_w > n))
    elif high_dim_mode == "ridge":
        use_high_dim_ridge = True
    elif high_dim_mode == "standard":
        use_high_dim_ridge = False
    else:
        raise ValueError("high_dim_mode must be 'auto', 'ridge', or 'standard'.")

    lam_w_res, lam_z_res, q_eps_res, g_delta_res = _resolve_ridge_params(
        n=n,
        d_w=d_w,
        m=int(Xg_t.shape[1]),
        ridge_lambda_w=ridge_lambda_w,
        ridge_lambda_z=ridge_lambda_z,
        ridge_q_eps=ridge_q_eps,
        ridge_g_delta=ridge_g_delta,
    )

    # Iteration budget:
    # - if user provides max_iter, respect it
    # - otherwise auto-select by dimensional regime
    #   low-dim (standard): 300
    #   high-dim (ridge):   1000
    if max_iter is None:
        max_iter_resolved = 1000 if use_high_dim_ridge else 300
    else:
        max_iter_resolved = int(max_iter)

    if method == "benchmark":
        G_t, info = _build_G_benchmark(
            Xg_t,
            W_t,
            Zg_t,
            residual_mode=residual_mode,
            omega=omega_t,
            use_high_dim_ridge=use_high_dim_ridge,
            ridge_lambda_w=lam_w_res,
            ridge_lambda_z=lam_z_res,
            ridge_q_eps=q_eps_res,
            ridge_g_delta=g_delta_res,
        )
    elif method == "scheme_a_diag":
        raise NotImplementedError("scheme_a_diag is temporarily disabled. Use 'benchmark' or 'scheme_c_free'.")
    elif method == "scheme_b_local":
        raise NotImplementedError("scheme_b_local is temporarily disabled. Use 'benchmark' or 'scheme_c_free'.")
    elif method == "scheme_c_free":
        free_max_iter = int(max(200, int(max_iter_resolved)))
        G_t, info = _build_G_scheme_c_free(
            Xg_t,
            W_t,
            Zg_t,
            objective=objective,
            residual_mode=residual_mode,
            omega=omega_t,
            lambda_reg=lambda_reg,
            max_iter=free_max_iter,
            lr=lr,
            enforce_symmetric=free_enforce_symmetric,
            g_norm_cap=free_g_norm_cap,
            kw_use_pca=kw_use_pca,
            kw_pca_components=kw_pca_components,
            use_high_dim_ridge=use_high_dim_ridge,
            ridge_lambda_w=lam_w_res,
            ridge_lambda_z=lam_z_res,
            ridge_q_eps=q_eps_res,
            print_objective_every=print_objective_every,
            objective_print_prefix=objective_print_prefix,
            print_kw_diagnostics=print_kw_diagnostics,
            kw_diagnostics_prefix=kw_diagnostics_prefix,
        )
    else:
        raise ValueError("method must be 'benchmark' or 'scheme_c_free'.")

    G_out = G_t.detach()
    if print_kw_diagnostics:
        if use_high_dim_ridge:
            geom_pack = _build_group_ridge_geometry(
                Xg=Xg_t,
                W=W_t,
                Zg=Zg_t,
                ridge_lambda_w=lam_w_res,
                ridge_lambda_z=lam_z_res,
                ridge_q_eps=q_eps_res,
            )
            R_dbg = geom_pack["R"]
            Q_dbg = geom_pack["Q"]
            U_dbg = Xg_t + Q_dbg @ G_out
            V_dbg = Xg_t - Q_dbg @ G_out
            xg_nrm = float(torch.linalg.norm(Xg_t).detach().item()) + 1e-12
            zg_nrm = float(torch.linalg.norm(Zg_t).detach().item()) + 1e-12
            u_flat = U_dbg.reshape(-1)
            v_flat = V_dbg.reshape(-1)
            u_center = u_flat - u_flat.mean()
            v_center = v_flat - v_flat.mean()
            uv_denom = float(torch.linalg.norm(u_center).detach().item() * torch.linalg.norm(v_center).detach().item()) + 1e-12
            QtQ = _symmetrize(Q_dbg.T @ Q_dbg)
            eig_q = torch.linalg.eigvalsh(QtQ)
            eig_q = torch.clamp(eig_q, min=0.0)
            geom = {
                "ratio_R_over_Xg": float(torch.linalg.norm(R_dbg).detach().item()) / xg_nrm,
                "ratio_Q_over_Zg": float(torch.linalg.norm(Q_dbg).detach().item()) / zg_nrm,
                "corr_UV_flat": float(torch.dot(u_center, v_center).detach().item()) / uv_denom,
                "eigmin_QtQ": float(torch.min(eig_q).detach().item()),
                "eigmax_QtQ": float(torch.max(eig_q).detach().item()),
                "cond_QtQ": float(torch.max(eig_q).detach().item() / (max(float(torch.min(eig_q).detach().item()), 0.0) + 1e-8)),
                "rank_R": int(torch.linalg.matrix_rank(R_dbg).detach().item()),
                "rank_Q": int(torch.linalg.matrix_rank(Q_dbg).detach().item()),
            }
        else:
            geom = _group_geometry_diagnostics(
                Xg_t,
                W_t,
                Zg_t,
                G_out,
                residual_mode=residual_mode,
                omega=omega_t,
            )
        prefix = f"{kw_diagnostics_prefix} " if kw_diagnostics_prefix else ""
        print(
            f"{prefix}Geometry diagnostics: "
            f"||R||/||Xg||={geom['ratio_R_over_Xg']:.6f}, "
            f"||Q||/||Zg||={geom['ratio_Q_over_Zg']:.6f}, "
            f"corr(U,V)={geom['corr_UV_flat']:.6f}, "
            f"rank(R)={geom['rank_R']}, rank(Q)={geom['rank_Q']}, "
            f"eigmin(QtQ)={geom['eigmin_QtQ']:.3e}, eigmax(QtQ)={geom['eigmax_QtQ']:.3e}, "
            f"cond(QtQ)={geom['cond_QtQ']:.3e}",
            flush=True,
        )
        info = dict(info)
        info["geometry_diagnostics"] = geom
    if return_info:
        info = dict(info)
        info["method"] = method
        info["objective"] = objective
        info["residual_mode"] = residual_mode
        info["max_iter_resolved"] = int(max_iter_resolved)
        info["high_dim_mode"] = high_dim_mode
        info["use_high_dim_ridge"] = bool(use_high_dim_ridge)
        info["ridge_params"] = {
            "ridge_lambda_w": float(lam_w_res),
            "ridge_lambda_z": float(lam_z_res),
            "ridge_q_eps": float(q_eps_res),
            "ridge_g_delta": float(g_delta_res),
        } if use_high_dim_ridge else None
        if return_torch:
            return G_out, info
        return G_out.cpu().numpy(), info
    if return_torch:
        return G_out
    return G_out.cpu().numpy()


def build_group_mirror_pair(
    Xg,
    W,
    Zg,
    G,
    residual_mode: str = "ols",
    omega=None,
    high_dim_mode: str = "auto",  # auto | ridge | standard
    ridge_lambda_w: Optional[float] = None,
    ridge_lambda_z: Optional[float] = None,
    ridge_q_eps: Optional[float] = None,
    return_q: bool = False,
    return_torch: bool = False,
):
    """
    Build group mirrors with the same Q geometry used by benchmark/scheme A/B:
      R = M_W Xg
      Q = (I - P_[W,R]) Zg
      U = Xg + QG
      V = Xg - QG
    """
    device = _device()
    Xg_t = _to_2d_torch(Xg, device=device, dtype=torch.float32)
    W_t = _to_2d_torch(W, device=device, dtype=torch.float32)
    Zg_t = _to_2d_torch(Zg, device=device, dtype=torch.float32)
    G_t = _to_2d_torch(G, device=device, dtype=torch.float32)

    if omega is not None:
        omega_t = _to_2d_torch(omega, device=device, dtype=torch.float32)
    else:
        omega_t = None

    high_dim_mode = str(high_dim_mode).lower()
    d_w = int(W_t.shape[1])
    n = int(Xg_t.shape[0])
    if high_dim_mode == "auto":
        use_high_dim_ridge = bool(HIGH_DIM_AUTO_D_GT_N and (d_w > n))
    elif high_dim_mode == "ridge":
        use_high_dim_ridge = True
    elif high_dim_mode == "standard":
        use_high_dim_ridge = False
    else:
        raise ValueError("high_dim_mode must be 'auto', 'ridge', or 'standard'.")

    lam_w_res, lam_z_res, q_eps_res, _ = _resolve_ridge_params(
        n=n,
        d_w=d_w,
        m=int(Xg_t.shape[1]),
        ridge_lambda_w=ridge_lambda_w,
        ridge_lambda_z=ridge_lambda_z,
        ridge_q_eps=ridge_q_eps,
        ridge_g_delta=None,
    )

    if use_high_dim_ridge:
        geom = _build_group_ridge_geometry(
            Xg=Xg_t,
            W=W_t,
            Zg=Zg_t,
            ridge_lambda_w=lam_w_res,
            ridge_lambda_z=lam_z_res,
            ridge_q_eps=q_eps_res,
        )
        Q = geom["Q"]
    else:
        _, R = _estimate_residual_covariance(Xg_t, W_t, residual_mode=residual_mode, omega=omega_t)
        Q = _prepare_group_noise_basis(Zg_t, W_t, R, residual_mode=residual_mode, omega=omega_t)
    QG = Q @ G_t
    U = Xg_t + QG
    V = Xg_t - QG

    U_out = U.detach()
    V_out = V.detach()
    Q_out = Q.detach()
    if return_torch:
        if return_q:
            return U_out, V_out, Q_out
        return U_out, V_out
    if return_q:
        return U_out.cpu().numpy(), V_out.cpu().numpy(), Q_out.cpu().numpy()
    return U_out.cpu().numpy(), V_out.cpu().numpy()


__all__ = [
    # legacy scalar APIs are commented out in current workflow
    # group APIs
    "find_Gj",
    "build_group_mirror_pair",
]
