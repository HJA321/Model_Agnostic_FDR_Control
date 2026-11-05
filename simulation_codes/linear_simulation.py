#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from group_lasso import GroupLasso
from sklearn.preprocessing import StandardScaler

# =========================
# Main parameters (edit here)
# =========================
SEED = 42
NUM_SIMULATIONS = 25

# Data generation
DATA_MODE = "iid_group"  # iid_group | lag_group
DATA_LENGTH = 2500
NUM_FEATURES = 150
NUM_NOISES = 1350
GROUP_SIZE = 5

# iid_group params
WITHIN_GROUP_CORR_VALUES = (0.1, 0.5, 0.9)
BETWEEN_GROUP_CORR_FACTORS = (0.2, 0.8)

# lag_group params
AR_PHI_VALUES = (0.1, 0.5, 0.9)
LAG_ENVELOPE_START = 1.0
LAG_ENVELOPE_END = 0.4

# beta and label noise
BETA_STD_SCALE = 20.0
LABEL_NOISE_STD = 1.0

# Grouped GM parameters
FDR_Q = 0.1
LOW_DIM_FACTOR = 2.0
SPLIT_RATIO = 1.0 / 3.0
SPLIT_GAP = 10
TIME_ORDERED_SPLIT = True
LAMBDA_GRID = np.geomspace(0.01, 0.5, 12)
NOISE_STD = 1.0
COEF_METHOD = "full_lstsq"  # closed_form | full_lstsq
CJ_METHOD = "grouped_matrix"  # grouped_matrix | original_scalar_first_feature

# Diagnostics
SHOW_PROGRESS = False
PRINT_CJ = False
CJ_PRINT_MAX_GROUPS = 5
CJ_PRINT_PRECISION = 4
RETURN_CJ_DICT = True

# Output controls
RESULTS_DIR = Path("linear_simulation_runs")
RESULT_CSV_PATH = None
SAVE_PER_SIM_DETAILS = False

assert DATA_MODE in {"iid_group", "lag_group"}
assert GROUP_SIZE > 0
assert NUM_FEATURES > 0 and NUM_NOISES >= 0
assert NUM_FEATURES % GROUP_SIZE == 0
assert NUM_NOISES % GROUP_SIZE == 0
assert DATA_LENGTH > 0
assert len(WITHIN_GROUP_CORR_VALUES) > 0
assert len(BETWEEN_GROUP_CORR_FACTORS) > 0
for _rho_w in WITHIN_GROUP_CORR_VALUES:
    assert 0.0 <= float(_rho_w) < 1.0
for _rho_b_factor in BETWEEN_GROUP_CORR_FACTORS:
    assert 0.0 <= float(_rho_b_factor) <= 1.0
assert len(AR_PHI_VALUES) > 0
for _phi in AR_PHI_VALUES:
    assert -1.0 < float(_phi) < 1.0


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
    out_path = RESULTS_DIR / f"linear_simulation_results_{run_id}.csv"
    return _ensure_non_conflicting_path(out_path)


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


def _symmetrize_torch(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.T)


def _sqrtm_psd_torch(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mat = _symmetrize_torch(mat)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=float(eps))
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def _invsqrtm_psd_torch(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mat = _symmetrize_torch(mat)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=float(eps))
    return evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T


def _residualize_torch(target: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[1] == 0:
        return target
    sol = torch.linalg.lstsq(basis, target).solution
    return target - basis @ sol


def _solve_spd_torch(mat: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mat_reg = _symmetrize_torch(mat) + float(eps) * torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
    return torch.linalg.solve(mat_reg, rhs)


def _scalar_cj_from_first_feature(
    x_minus_j: torch.Tensor,
    x_j: torch.Tensor,
    z_j: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    x_first = x_j[:, [0]]
    z_first = z_j[:, [0]]
    r_first = _residualize_torch(x_first, x_minus_j)
    u_first = _residualize_torch(z_first, x_minus_j)
    num = torch.sum(r_first * r_first)
    den = torch.sum(u_first * u_first) + float(eps)
    cj_scalar = torch.sqrt(torch.clamp(num / den, min=float(eps)))
    return cj_scalar


def _build_problem_structure() -> Tuple[Dict[str, list], Dict[str, bool], Dict[str, Any]]:
    num_relevant_groups = int(NUM_FEATURES // GROUP_SIZE)
    num_noise_groups = int(NUM_NOISES // GROUP_SIZE)
    num_total_groups = int(num_relevant_groups + num_noise_groups)

    relevant_group_names = [f"grp_rel_{g + 1:03d}" for g in range(num_relevant_groups)]
    noise_group_names = [f"grp_noise_{g + 1:03d}" for g in range(num_noise_groups)]

    feature_names = [
        f"feature_{i + 1:04d}_{relevant_group_names[i // GROUP_SIZE]}"
        for i in range(int(NUM_FEATURES))
    ]
    noise_names = [
        f"noise_{i + 1:04d}_{noise_group_names[i // GROUP_SIZE]}"
        for i in range(int(NUM_NOISES))
    ]
    all_column_names = feature_names + noise_names

    group_to_columns: Dict[str, list] = {}
    group_is_relevant: Dict[str, bool] = {}
    for g in relevant_group_names:
        cols = [c for c in feature_names if c.endswith(g)]
        group_to_columns[g] = cols
        group_is_relevant[g] = True
    for g in noise_group_names:
        cols = [c for c in noise_names if c.endswith(g)]
        group_to_columns[g] = cols
        group_is_relevant[g] = False

    info = {
        "num_relevant_groups": num_relevant_groups,
        "num_noise_groups": num_noise_groups,
        "num_total_groups": num_total_groups,
        "feature_names": feature_names,
        "noise_names": noise_names,
        "all_column_names": all_column_names,
        "relevant_group_names": relevant_group_names,
        "noise_group_names": noise_group_names,
    }
    return group_to_columns, group_is_relevant, info


def _generate_data_one_run(
    seed: int,
    *,
    ar_phi: float,
    within_group_corr: float | None = None,
    between_group_corr: float = 0.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    _, _, info = _build_problem_structure()
    num_total_groups = int(info["num_total_groups"])
    num_total_columns = int(NUM_FEATURES + NUM_NOISES)

    beta = np.zeros(num_total_columns, dtype=np.float64)
    std_dev = float(BETA_STD_SCALE) * np.sqrt(np.log(max(2, int(NUM_FEATURES))) / float(DATA_LENGTH))

    if str(DATA_MODE).lower() == "iid_group":
        rho_w = float(WITHIN_GROUP_CORR_VALUES[0] if within_group_corr is None else within_group_corr)
        rho_b = float(between_group_corr)
        if not (0.0 <= rho_b <= rho_w < 1.0):
            raise ValueError("iid_group requires 0 <= between_group_corr <= within_group_corr < 1.")

        common_factor = rng.normal(0.0, 1.0, size=(int(DATA_LENGTH), 1))
        group_factors = rng.normal(0.0, 1.0, size=(int(DATA_LENGTH), num_total_groups))
        x_blocks = []
        for g in range(num_total_groups):
            idiosyncratic = rng.normal(0.0, 1.0, size=(int(DATA_LENGTH), int(GROUP_SIZE)))
            block = (
                np.sqrt(rho_b) * common_factor
                + np.sqrt(rho_w - rho_b) * group_factors[:, [g]]
                + np.sqrt(1.0 - rho_w) * idiosyncratic
            )
            x_blocks.append(block)
        x = np.concatenate(x_blocks, axis=1)
        beta[: int(NUM_FEATURES)] = rng.normal(0.0, std_dev, size=int(NUM_FEATURES))
    else:
        lag_l = int(GROUP_SIZE - 1)
        rho_b = float(between_group_corr)
        if not (0.0 <= rho_b < 1.0):
            raise ValueError("lag_group requires 0 <= between_group_corr < 1.")

        common_s = np.zeros(int(DATA_LENGTH + lag_l), dtype=np.float64)
        common_eps = rng.normal(0.0, 1.0, size=int(DATA_LENGTH + lag_l))
        for t in range(1, int(DATA_LENGTH + lag_l)):
            common_s[t] = float(ar_phi) * common_s[t - 1] + common_eps[t]

        x_blocks = []
        for _ in range(num_total_groups):
            group_s = np.zeros(int(DATA_LENGTH + lag_l), dtype=np.float64)
            eps = rng.normal(0.0, 1.0, size=int(DATA_LENGTH + lag_l))
            for t in range(1, int(DATA_LENGTH + lag_l)):
                group_s[t] = float(ar_phi) * group_s[t - 1] + eps[t]
            s = np.sqrt(rho_b) * common_s + np.sqrt(1.0 - rho_b) * group_s
            cols = []
            for lag in range(int(GROUP_SIZE)):
                col = s[lag_l - lag: lag_l - lag + int(DATA_LENGTH)]
                cols.append(col)
            block = np.column_stack(cols)
            x_blocks.append(block)
        x = np.concatenate(x_blocks, axis=1)

        envelope = np.linspace(float(LAG_ENVELOPE_START), float(LAG_ENVELOPE_END), int(GROUP_SIZE))
        num_relevant_groups = int(NUM_FEATURES // GROUP_SIZE)
        for g in range(num_relevant_groups):
            start = int(g * GROUP_SIZE)
            end = int((g + 1) * GROUP_SIZE)
            signs = rng.choice([-1.0, 1.0], size=int(GROUP_SIZE))
            amps = rng.normal(0.0, std_dev, size=int(GROUP_SIZE))
            beta[start:end] = amps * envelope * signs

    y = x @ beta + rng.normal(0.0, float(LABEL_NOISE_STD), int(DATA_LENGTH))
    _, _, info = _build_problem_structure()
    x_df = pd.DataFrame(x, columns=info["all_column_names"])
    return x_df, y.astype(np.float64), beta


def compute_group_mirror_stats_ols(
    x_selected_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    group_is_relevant: Dict[str, bool],
    *,
    noise_std: float = 1.0,
    random_state: int = 42,
    device: str | None = None,
    print_cj: bool = False,
    cj_print_precision: int = 4,
    cj_print_max_groups: int | None = None,
    return_cj: bool = False,
    coef_method: str = "closed_form",
    cj_method: str = "grouped_matrix",
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any] | None]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    x_np = x_selected_df.values.astype(np.float64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    x_t = torch.tensor(x_np, dtype=torch.float64, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float64, device=device)

    _, p = x_t.shape
    col_names = list(x_selected_df.columns)
    col_to_idx = {c: i for i, c in enumerate(col_names)}

    if coef_method not in {"closed_form", "full_lstsq"}:
        raise ValueError("coef_method must be 'closed_form' or 'full_lstsq'.")
    if cj_method not in {"grouped_matrix", "original_scalar_first_feature"}:
        raise ValueError("cj_method must be 'grouped_matrix' or 'original_scalar_first_feature'.")

    mirror_stats: Dict[str, float] = {}
    group_details = []
    cj_dict = {} if return_cj else None

    for g_idx, (group_name, group_cols) in enumerate(group_to_columns.items()):
        idx_j = torch.tensor([col_to_idx[c] for c in group_cols], device=device, dtype=torch.long)
        mask = torch.ones(p, dtype=torch.bool, device=device)
        mask[idx_j] = False

        x_j = x_t[:, idx_j]
        x_minus_j = x_t[:, mask]

        r_j = _residualize_torch(x_j, x_minus_j)
        g_j = r_j.T @ r_j

        z_j = torch.randn((x_j.shape[0], x_j.shape[1]), dtype=torch.float64, device=device) * float(noise_std)

        if cj_method == "grouped_matrix":
            w_j = torch.cat([x_minus_j, r_j], dim=1)
            q_j = _residualize_torch(z_j, w_j)
            q_gram = q_j.T @ q_j
            c_j = _invsqrtm_psd_torch(q_gram) @ _sqrtm_psd_torch(g_j)
            delta_j = q_j @ c_j
            cj_obj: Any = c_j
        else:
            c_j_scalar = _scalar_cj_from_first_feature(x_minus_j, x_j, z_j)
            delta_j = c_j_scalar * z_j
            cj_obj = c_j_scalar

        if return_cj:
            if cj_method == "grouped_matrix":
                cj_dict[group_name] = cj_obj.detach().cpu().numpy()
            else:
                cj_dict[group_name] = float(cj_obj.detach().cpu().item())

        if print_cj and (cj_print_max_groups is None or g_idx < int(cj_print_max_groups)):
            print(f"c_j for {group_name} ({cj_method}):", flush=True)
            if cj_method == "grouped_matrix":
                c_np = cj_obj.detach().cpu().numpy()
                print(np.array2string(c_np, precision=int(cj_print_precision), suppress_small=True), flush=True)
            else:
                c_val = float(cj_obj.detach().cpu().item())
                print(f"{c_val:.{int(cj_print_precision)}f}", flush=True)
            print("-" * 60, flush=True)

        x_j_plus = x_j + delta_j
        x_j_minus = x_j - delta_j

        if coef_method == "closed_form":
            y_star = _residualize_torch(y_t, x_minus_j)
            v_plus = _residualize_torch(x_j_plus, x_minus_j)
            v_minus = _residualize_torch(x_j_minus, x_minus_j)
            two_g = 2.0 * g_j
            beta_plus = _solve_spd_torch(two_g, v_plus.T @ y_star).squeeze(-1)
            beta_minus = _solve_spd_torch(two_g, v_minus.T @ y_star).squeeze(-1)
        else:
            d_j = torch.cat([x_minus_j, x_j_plus, x_j_minus], dim=1)
            beta_hat = torch.linalg.lstsq(d_j, y_t).solution.squeeze(-1)
            m = x_j.shape[1]
            p_minus = x_minus_j.shape[1]
            beta_plus = beta_hat[p_minus:p_minus + m]
            beta_minus = beta_hat[p_minus + m:p_minus + 2 * m]

        signed_dot = torch.dot(beta_plus, beta_minus)
        sign_val = torch.sign(signed_dot)
        l_plus = torch.sum(torch.abs(beta_plus))
        l_minus = torch.sum(torch.abs(beta_minus))
        mirror_stat = sign_val * (l_plus + l_minus)

        mirror_stats[group_name] = float(mirror_stat.detach().cpu().item())
        group_details.append(
            {
                "group": group_name,
                "is_relevant": bool(group_is_relevant[group_name]),
                "group_size": int(x_j.shape[1]),
                "L1_plus": float(l_plus.detach().cpu().item()),
                "L1_minus": float(l_minus.detach().cpu().item()),
                "mirror_stat": float(mirror_stat.detach().cpu().item()),
                "coef_method": str(coef_method),
                "cj_method": str(cj_method),
            }
        )

    details_df = pd.DataFrame(group_details).sort_values("mirror_stat", ascending=False).reset_index(drop=True)
    return mirror_stats, details_df, cj_dict


def compute_cutoff(mirror_stats: Dict[str, float], q: float = 0.1, *, use_fallback: bool = True) -> Tuple[list, float]:
    """Return the mirror-FDR cutoff.

    If no positive threshold satisfies the mirror-FDR inequality, keep the
    simulation convention and choose the positive threshold whose estimated
    mirror-FDR ratio is closest to q. Set ``use_fallback=False`` only for a
    strict no-selection fallback.
    """
    sorted_mirror_stats = sorted(mirror_stats.items(), key=lambda x: x[1], reverse=False)
    cutoff = float("inf")
    fallback_cutoff = float("inf")
    fallback_gap = float("inf")

    for _, mirror_stat in sorted_mirror_stats:
        if float(mirror_stat) <= 0.0:
            continue
        num_j_less = np.sum(m[1] <= -float(mirror_stat) for m in sorted_mirror_stats)
        num_j_more = np.sum(m[1] >= float(mirror_stat) for m in sorted_mirror_stats)
        if num_j_more == 0:
            raise ValueError("No group is selected.")
        ratio = (num_j_less + 1) / num_j_more
        gap = abs(float(ratio) - float(q))
        if gap < fallback_gap:
            fallback_gap = gap
            fallback_cutoff = float(mirror_stat)
        if ratio <= float(q):
            cutoff = float(mirror_stat)
            break
    if not np.isfinite(cutoff) and bool(use_fallback) and np.isfinite(fallback_cutoff):
        cutoff = fallback_cutoff
    return sorted_mirror_stats, cutoff


def _oracle_from_mirror_stats(mirror_stats: Dict[str, float], relevant_groups: set, q: float) -> Dict[str, Any]:
    candidate_cutoffs = sorted({float(v) for v in mirror_stats.values() if float(v) > 0.0}) + [float("inf")]
    best = {
        "cutoff": float("inf"),
        "selected_names": set(),
        "fdr": 0.0,
        "power": 0.0,
    }
    denom_rel = max(1, len(relevant_groups))
    for cutoff in candidate_cutoffs:
        selected_names = {k for k, v in mirror_stats.items() if float(v) >= float(cutoff)}
        n_total = len(selected_names)
        tp = len(selected_names & relevant_groups)
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


def check_low_dim(n_obs: int, n_cols: int, factor: float = 2.0) -> bool:
    return float(factor) * int(n_cols) < int(n_obs)


def split_two_stage(
    x_df: pd.DataFrame,
    y: np.ndarray,
    *,
    split_ratio: float = 0.5,
    gap: int = 0,
    time_ordered: bool = True,
    random_state: int = 42,
):
    n = len(y)
    n1 = int(n * float(split_ratio))
    if n1 <= 0 or n1 >= n:
        raise ValueError("split_ratio must leave non-empty D1 and D2.")

    if bool(time_ordered):
        idx1 = np.arange(0, n1)
        start2 = n1 + int(gap)
        if start2 >= n:
            raise ValueError("gap is too large; D2 is empty.")
        idx2 = np.arange(start2, n)
    else:
        rng = np.random.default_rng(int(random_state))
        perm = rng.permutation(n)
        idx1 = np.sort(perm[:n1])
        idx2 = np.sort(perm[n1:])

    x1 = x_df.iloc[idx1].reset_index(drop=True)
    y1 = np.asarray(y)[idx1]
    x2 = x_df.iloc[idx2].reset_index(drop=True)
    y2 = np.asarray(y)[idx2]
    return x1, y1, x2, y2, idx1, idx2


def build_group_vector(x_df: pd.DataFrame, group_to_columns: Dict[str, list]):
    col_to_gid = {}
    ordered_groups = list(group_to_columns.keys())
    for gid, g in enumerate(ordered_groups):
        for c in group_to_columns[g]:
            col_to_gid[c] = gid
    groups_vec = np.array([col_to_gid[c] for c in x_df.columns], dtype=int)
    return groups_vec, ordered_groups


def selected_groups_from_model(model, ordered_groups):
    if hasattr(model, "chosen_groups_") and model.chosen_groups_ is not None:
        gids = sorted(list(model.chosen_groups_))
    else:
        coef = np.asarray(model.coef_).reshape(-1)
        groups = np.asarray(model.groups)
        gids = []
        for gid in np.unique(groups):
            if np.linalg.norm(coef[groups == gid], ord=2) > 1e-12:
                gids.append(int(gid))
        gids = sorted(gids)
    return [ordered_groups[g] for g in gids]


def subset_group_mapping(x_df: pd.DataFrame, group_to_columns: Dict[str, list], selected_groups: list):
    selected_cols = []
    sub_map = {}
    for g in selected_groups:
        cols = [c for c in group_to_columns[g] if c in x_df.columns]
        if len(cols) == 0:
            continue
        sub_map[g] = cols
        selected_cols.extend(cols)
    x_sub = x_df[selected_cols].copy() if selected_cols else x_df.iloc[:, :0].copy()
    return x_sub, sub_map, selected_cols


def run_group_lasso_screening(
    x1: pd.DataFrame,
    y1: np.ndarray,
    group_to_columns: Dict[str, list],
    *,
    n2: int,
    low_dim_factor: float = 2.0,
    lambda_grid: np.ndarray | None = None,
    random_state: int = 42,
    verbose: bool = True,
):
    if lambda_grid is None:
        lambda_grid = np.geomspace(0.01, 0.5, 12)

    groups_vec, ordered_groups = build_group_vector(x1, group_to_columns)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x1s = scaler.fit_transform(x1.values)

    screening_log = []
    best = None

    for lam in lambda_grid:
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
        gl.fit(x1s, y1)
        selected_groups = selected_groups_from_model(gl, ordered_groups)
        d2 = sum(len(group_to_columns[g]) for g in selected_groups)
        ok = check_low_dim(n2, d2, factor=float(low_dim_factor))
        screening_log.append(
            {
                "lambda": float(lam),
                "num_selected_groups": int(len(selected_groups)),
                "num_selected_columns": int(d2),
                "stage2_low_dim_ok": bool(ok),
                "selected_groups": selected_groups,
            }
        )
        if verbose:
            print(
                f"[Stage1] lambda={float(lam):.6f} -> groups={len(selected_groups):3d}, "
                f"cols={d2:4d}, Stage2 low-dim={ok}",
                flush=True,
            )
        if ok:
            best = screening_log[-1]
            if verbose:
                print(
                    f"[Stage1] Early stop at lambda={float(lam):.6f}: reached Stage2 low-dimensional criterion.",
                    flush=True,
                )
            break

    if best is None:
        best = screening_log[-1]
        if verbose:
            print(
                "[Stage1] Warning: no lambda reached Stage2 low-dimensional criterion; using the last result.",
                flush=True,
            )
    return best, screening_log


def run_group_gm_auto_pipeline(
    x_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list],
    group_is_relevant: Dict[str, bool],
    *,
    low_dim_factor: float = 2.0,
    split_ratio: float = 0.5,
    gap: int = 0,
    time_ordered: bool = True,
    lambda_grid: np.ndarray | None = None,
    noise_std: float = 1.0,
    random_state: int = 42,
    coef_method: str = "full_lstsq",
    cj_method: str = "grouped_matrix",
    print_cj: bool = False,
    cj_print_precision: int = 4,
    cj_print_max_groups: int = 5,
    return_cj: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    n_total, d_total = x_df.shape
    low_dim_full = check_low_dim(n_total, d_total, factor=float(low_dim_factor))

    if verbose:
        print("=" * 80, flush=True)
        print("[Auto Mode Check]", flush=True)
        print(f"n_total={n_total}, d_total={d_total}, low_dim_factor={float(low_dim_factor)}", flush=True)
        print(
            f"Criterion: factor * d < n -> {float(low_dim_factor) * d_total:.1f} < {n_total} == {low_dim_full}",
            flush=True,
        )
        print("=" * 80, flush=True)

    if low_dim_full:
        if verbose:
            print("[Route] Low-dimensional direct Grouped GM on full data.", flush=True)
        mirror_stats, group_details_df, cj_dict = compute_group_mirror_stats_ols(
            x_selected_df=x_df,
            y=y,
            group_to_columns=group_to_columns,
            group_is_relevant=group_is_relevant,
            noise_std=float(noise_std),
            random_state=int(random_state),
            print_cj=bool(print_cj),
            cj_print_precision=int(cj_print_precision),
            cj_print_max_groups=int(cj_print_max_groups) if cj_print_max_groups is not None else None,
            return_cj=bool(return_cj),
            coef_method=str(coef_method),
            cj_method=str(cj_method),
        )
        return {
            "mode": "low_dim_direct",
            "mirror_stats": mirror_stats,
            "group_details_df": group_details_df,
            "cj_dict": cj_dict if return_cj else None,
            "selected_groups_stage1": list(group_to_columns.keys()),
            "screening_log": [],
            "n2": int(n_total),
            "d2": int(d_total),
            "stage2_low_dim_ok": True,
        }

    if verbose:
        print("[Route] High-dimensional two-stage: Stage1 group-lasso screening + Stage2 Grouped GM.", flush=True)

    x1, y1, x2, y2, idx1, idx2 = split_two_stage(
        x_df,
        y,
        split_ratio=float(split_ratio),
        gap=int(gap),
        time_ordered=bool(time_ordered),
        random_state=int(random_state),
    )
    if verbose:
        print("-" * 80, flush=True)
        print("[Split Info]", flush=True)
        print(f"D1: n1={len(y1)}, D2: n2={len(y2)}, gap={int(gap)}, time_ordered={bool(time_ordered)}", flush=True)
        print(f"Index D1: [{idx1[0]}, ..., {idx1[-1]}], Index D2: [{idx2[0]}, ..., {idx2[-1]}]", flush=True)
        print("-" * 80, flush=True)

    best, screening_log = run_group_lasso_screening(
        x1=x1,
        y1=y1,
        group_to_columns=group_to_columns,
        n2=len(y2),
        low_dim_factor=float(low_dim_factor),
        lambda_grid=lambda_grid,
        random_state=int(random_state),
        verbose=bool(verbose),
    )

    selected_groups = best["selected_groups"]
    x2_sel, group_to_columns_sel, selected_cols = subset_group_mapping(x2, group_to_columns, selected_groups)
    d2 = int(x2_sel.shape[1])
    stage2_low_dim_ok = check_low_dim(len(y2), d2, factor=float(low_dim_factor))

    if verbose:
        print("-" * 80, flush=True)
        print("[Stage1 Final Selection]", flush=True)
        print(f"Chosen lambda={float(best['lambda']):.6f}", flush=True)
        print(
            f"Selected groups ({len(selected_groups)}): "
            f"{selected_groups[:20]}{' ...' if len(selected_groups) > 20 else ''}",
            flush=True,
        )
        print(f"Selected columns={len(selected_cols)}", flush=True)
        print(
            f"Stage2 criterion: factor * d2 < n2 -> {float(low_dim_factor) * d2:.1f} < {len(y2)} == {stage2_low_dim_ok}",
            flush=True,
        )
        print("-" * 80, flush=True)

    if len(selected_groups) == 0:
        if verbose:
            print("[Warning] Stage1 selected no groups. All mirror stats will be zero.", flush=True)
        mirror_stats_all = {g: 0.0 for g in group_to_columns.keys()}
        details_df = pd.DataFrame(
            columns=["group", "is_relevant", "group_size", "L1_plus", "L1_minus", "mirror_stat", "coef_method", "cj_method"]
        )
        return {
            "mode": "high_dim_two_stage",
            "mirror_stats": mirror_stats_all,
            "group_details_df": details_df,
            "cj_dict": {} if return_cj else None,
            "selected_groups_stage1": selected_groups,
            "screening_log": screening_log,
            "n2": int(len(y2)),
            "d2": int(d2),
            "stage2_low_dim_ok": bool(stage2_low_dim_ok),
        }

    if not stage2_low_dim_ok:
        raise RuntimeError(
            f"Stage2 still not low-dimensional after screening: factor*d2={float(low_dim_factor) * d2:.1f}, n2={len(y2)}. "
            "Increase lambda_grid upper range or adjust split_ratio."
        )

    group_is_relevant_sel = {g: group_is_relevant[g] for g in group_to_columns_sel.keys()}
    mirror_stats_sel, group_details_df, cj_dict = compute_group_mirror_stats_ols(
        x_selected_df=x2_sel,
        y=y2,
        group_to_columns=group_to_columns_sel,
        group_is_relevant=group_is_relevant_sel,
        noise_std=float(noise_std),
        random_state=int(random_state),
        print_cj=bool(print_cj),
        cj_print_precision=int(cj_print_precision),
        cj_print_max_groups=int(cj_print_max_groups) if cj_print_max_groups is not None else None,
        return_cj=bool(return_cj),
        coef_method=str(coef_method),
        cj_method=str(cj_method),
    )

    mirror_stats_all = {g: 0.0 for g in group_to_columns.keys()}
    mirror_stats_all.update(mirror_stats_sel)

    return {
        "mode": "high_dim_two_stage",
        "mirror_stats": mirror_stats_all,
        "group_details_df": group_details_df,
        "cj_dict": cj_dict if return_cj else None,
        "selected_groups_stage1": selected_groups,
        "screening_log": screening_log,
        "n2": int(len(y2)),
        "d2": int(d2),
        "stage2_low_dim_ok": bool(stage2_low_dim_ok),
    }


def main() -> None:
    run_id = _make_run_id()
    result_csv_path = _create_run_output_path(run_id)
    run_dir = result_csv_path.parent / f"{result_csv_path.stem}_artifacts"
    if SAVE_PER_SIM_DETAILS:
        run_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "RUN_ID": run_id,
        "RESULT_CSV_PATH": str(result_csv_path.resolve()),
        "SEED": int(SEED),
        "NUM_SIMULATIONS": int(NUM_SIMULATIONS),
        "DATA_MODE": str(DATA_MODE),
        "DATA_LENGTH": int(DATA_LENGTH),
        "NUM_FEATURES": int(NUM_FEATURES),
        "NUM_NOISES": int(NUM_NOISES),
        "GROUP_SIZE": int(GROUP_SIZE),
        "WITHIN_GROUP_CORR_VALUES": [float(v) for v in WITHIN_GROUP_CORR_VALUES],
        "BETWEEN_GROUP_CORR_FACTORS": [float(v) for v in BETWEEN_GROUP_CORR_FACTORS],
        "AR_PHI_VALUES": [float(v) for v in AR_PHI_VALUES],
        "LAG_ENVELOPE_START": float(LAG_ENVELOPE_START),
        "LAG_ENVELOPE_END": float(LAG_ENVELOPE_END),
        "BETA_STD_SCALE": float(BETA_STD_SCALE),
        "LABEL_NOISE_STD": float(LABEL_NOISE_STD),
        "FDR_Q": float(FDR_Q),
        "LOW_DIM_FACTOR": float(LOW_DIM_FACTOR),
        "SPLIT_RATIO": float(SPLIT_RATIO),
        "SPLIT_GAP": int(SPLIT_GAP),
        "TIME_ORDERED_SPLIT": bool(TIME_ORDERED_SPLIT),
        "LAMBDA_GRID": np.asarray(LAMBDA_GRID, dtype=float).tolist(),
        "NOISE_STD": float(NOISE_STD),
        "COEF_METHOD": str(COEF_METHOD),
        "CJ_METHOD": str(CJ_METHOD),
        "SHOW_PROGRESS": bool(SHOW_PROGRESS),
        "PRINT_CJ": bool(PRINT_CJ),
        "CJ_PRINT_MAX_GROUPS": int(CJ_PRINT_MAX_GROUPS),
        "CJ_PRINT_PRECISION": int(CJ_PRINT_PRECISION),
        "RETURN_CJ_DICT": bool(RETURN_CJ_DICT),
        "SAVE_PER_SIM_DETAILS": bool(SAVE_PER_SIM_DETAILS),
    }
    _write_header(result_csv_path, settings)
    wrote_results_header = False

    group_to_columns, group_is_relevant, info = _build_problem_structure()
    num_relevant_groups = int(info["num_relevant_groups"])
    num_noise_groups = int(info["num_noise_groups"])
    num_total_groups = int(info["num_total_groups"])

    global_sim_idx = 0
    if str(DATA_MODE).lower() == "iid_group":
        data_scenarios = [
            {
                "ar_phi": None,
                "within_group_corr": float(rho_w),
                "between_group_corr_factor": float(rho_b_factor),
                "between_group_corr": float(rho_w) * float(rho_b_factor),
            }
            for rho_w in WITHIN_GROUP_CORR_VALUES
            for rho_b_factor in BETWEEN_GROUP_CORR_FACTORS
        ]
    else:
        data_scenarios = [
            {
                "ar_phi": float(ar_phi),
                "within_group_corr": None,
                "between_group_corr_factor": float(rho_b_factor),
                "between_group_corr": float(rho_b_factor),
            }
            for ar_phi in AR_PHI_VALUES
            for rho_b_factor in BETWEEN_GROUP_CORR_FACTORS
        ]

    for scenario_idx, scenario in enumerate(data_scenarios):
        for sim_idx in range(int(NUM_SIMULATIONS)):
            run_seed = int(SEED) + int(scenario_idx) * int(NUM_SIMULATIONS) + int(sim_idx)
            _set_all_seeds(run_seed)

            x_df, y, beta = _generate_data_one_run(
                run_seed,
                ar_phi=float(scenario["ar_phi"]) if scenario["ar_phi"] is not None else 0.0,
                within_group_corr=scenario["within_group_corr"],
                between_group_corr=float(scenario["between_group_corr"] or 0.0),
            )
            with _maybe_suppress_output(enable_output=bool(SHOW_PROGRESS)):
                result = run_group_gm_auto_pipeline(
                    x_df=x_df,
                    y=y,
                    group_to_columns=group_to_columns,
                    group_is_relevant=group_is_relevant,
                    low_dim_factor=float(LOW_DIM_FACTOR),
                    split_ratio=float(SPLIT_RATIO),
                    gap=int(SPLIT_GAP),
                    time_ordered=bool(TIME_ORDERED_SPLIT),
                    lambda_grid=np.asarray(LAMBDA_GRID, dtype=float),
                    noise_std=float(NOISE_STD),
                    random_state=run_seed,
                    coef_method=str(COEF_METHOD),
                    cj_method=str(CJ_METHOD),
                    print_cj=bool(PRINT_CJ),
                    cj_print_precision=int(CJ_PRINT_PRECISION),
                    cj_print_max_groups=int(CJ_PRINT_MAX_GROUPS),
                    return_cj=bool(RETURN_CJ_DICT),
                    verbose=bool(SHOW_PROGRESS),
                )

            mirror_stats = result["mirror_stats"]
            sorted_mirror_stats, cutoff = compute_cutoff(mirror_stats, q=float(FDR_Q))
            relevant_groups = [g for g, v in group_is_relevant.items() if bool(v)]
            noise_groups = [g for g, v in group_is_relevant.items() if not bool(v)]
            selected_groups = [m[0] for m in sorted_mirror_stats if float(m[1]) >= float(cutoff)]
            num_selected_groups = int(len(selected_groups))
            num_selected_null_groups = int(sum(1 for g in noise_groups if float(mirror_stats[g]) >= float(cutoff)))
            num_total_relevant_groups = int(len(relevant_groups))

            power = (
                float(num_selected_groups - num_selected_null_groups) / float(num_total_relevant_groups)
                if num_total_relevant_groups > 0
                else 0.0
            )
            fdr = float(num_selected_null_groups / num_selected_groups) if num_selected_groups > 0 else 0.0

            oracle = _oracle_from_mirror_stats(mirror_stats, set(relevant_groups), float(FDR_Q))
            oracle_names = set(oracle["selected_names"])
            oracle_rel = len(oracle_names & set(relevant_groups))
            oracle_irr = len(oracle_names & set(noise_groups))

            stage1_selected_groups = list(result.get("selected_groups_stage1", []))
            stage1_selected_set = set(stage1_selected_groups)
            stage1_dropped_groups = int(num_total_groups - len(stage1_selected_groups))
            stage1_dropped_columns = int(
                sum(len(cols) for g, cols in group_to_columns.items() if g not in stage1_selected_set)
            )

            global_sim_idx += 1
            if str(DATA_MODE).lower() == "iid_group":
                result_key = (
                    f"{run_id}_rhoW{float(scenario['within_group_corr']):.1f}"
                    f"_rhoBfac{float(scenario['between_group_corr_factor']):.1f}_sim{sim_idx + 1}"
                )
            else:
                result_key = (
                    f"{run_id}_ar{float(scenario['ar_phi']):.1f}"
                    f"_rhoBfac{float(scenario['between_group_corr_factor']):.1f}_sim{sim_idx + 1}"
                )
            row = {
                "run_id": run_id,
                "result_key": result_key,
                "result_file": str(result_csv_path.resolve()),
                "simulation_id": int(global_sim_idx),
                "ar_simulation_id": int(sim_idx + 1),
                "seed": int(run_seed),
                "data_mode": str(DATA_MODE),
                "ar_phi": float(scenario["ar_phi"]) if scenario["ar_phi"] is not None else np.nan,
                "within_group_corr": (
                    float(scenario["within_group_corr"]) if scenario["within_group_corr"] is not None else np.nan
                ),
                "between_group_corr_factor": (
                    float(scenario["between_group_corr_factor"])
                    if scenario["between_group_corr_factor"] is not None
                    else np.nan
                ),
                "between_group_corr": (
                    float(scenario["between_group_corr"]) if scenario["between_group_corr"] is not None else np.nan
                ),
                "method": f"group_gm_{COEF_METHOD}_{CJ_METHOD}",
                "mode": str(result["mode"]),
                "fdr_q": float(FDR_Q),
                "cutoff": float(cutoff),
                "fdr": float(fdr),
                "power": float(power),
                "num_selected_groups": int(num_selected_groups),
                "num_selected_null_groups": int(num_selected_null_groups),
                "num_total_relevant_groups": int(num_total_relevant_groups),
                "num_relevant_groups": int(num_relevant_groups),
                "num_noise_groups": int(num_noise_groups),
                "num_total_groups": int(num_total_groups),
                "n_samples": int(DATA_LENGTH),
                "n_features_total": int(NUM_FEATURES + NUM_NOISES),
                "n_relevant_columns": int(NUM_FEATURES),
                "n_noise_columns": int(NUM_NOISES),
                "group_size": int(GROUP_SIZE),
                "beta_nonzero_count": int(np.sum(np.abs(beta) > 0.0)),
                "stage1_selected_groups": int(len(stage1_selected_groups)),
                "stage1_dropped_groups": int(stage1_dropped_groups),
                "stage1_dropped_columns": int(stage1_dropped_columns),
                "stage2_n2": int(result["n2"]),
                "stage2_d2": int(result["d2"]),
                "stage2_low_dim_ok": bool(result["stage2_low_dim_ok"]),
                "screening_log_len": int(len(result.get("screening_log", []))),
                "oracle_cutoff": float(oracle["cutoff"]),
                "oracle_fdr": float(oracle["fdr"]),
                "oracle_power": float(oracle["power"]),
                "oracle_selected_relevant_groups": int(oracle_rel),
                "oracle_selected_noise_groups": int(oracle_irr),
            }
            wrote_results_header = _append_row(result_csv_path, row, wrote_results_header)

            if SAVE_PER_SIM_DETAILS:
                details_df = result.get("group_details_df", pd.DataFrame())
                details_path = run_dir / f"{result_key}_group_details.csv"
                details_df.to_csv(details_path, index=False)
                screen = result.get("screening_log", [])
                if len(screen) > 0:
                    screen_df = pd.DataFrame([{k: v for k, v in row_i.items() if k != "selected_groups"} for row_i in screen])
                    screen_df.to_csv(run_dir / f"{result_key}_screening_log.csv", index=False)
                if RETURN_CJ_DICT and result.get("cj_dict") is not None:
                    cj_dict = result["cj_dict"]
                    cj_rows = [{"group": g, "cj_repr": repr(v)} for g, v in cj_dict.items()]
                    pd.DataFrame(cj_rows).to_csv(run_dir / f"{result_key}_cj_dict.csv", index=False)

    if bool(SHOW_PROGRESS):
        print(f"Saved results to: {result_csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
