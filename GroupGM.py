from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


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


def compute_group_mirror_stats_ols(
    x_selected_df: pd.DataFrame,
    y: np.ndarray,
    group_to_columns: Dict[str, list[str]],
    group_is_relevant: Dict[str, bool] | None = None,
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
            beta_plus = beta_hat[p_minus : p_minus + m]
            beta_minus = beta_hat[p_minus + m : p_minus + 2 * m]

        signed_dot = torch.dot(beta_plus, beta_minus)
        sign_val = torch.sign(signed_dot)
        l_plus = torch.sum(torch.abs(beta_plus))
        l_minus = torch.sum(torch.abs(beta_minus))
        mirror_stat = sign_val * (l_plus + l_minus)

        mirror_stats[group_name] = float(mirror_stat.detach().cpu().item())
        group_details.append(
            {
                "group": group_name,
                "is_relevant": None if group_is_relevant is None else bool(group_is_relevant[group_name]),
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


def compute_cutoff(
    mirror_stats: Dict[str, float],
    q: float = 0.1,
    *,
    use_fallback: bool = False,
) -> Tuple[list[tuple[str, float]], float]:
    """Return the mirror-FDR cutoff.

    The theoretical rule selects the smallest positive threshold satisfying
    (#{M_j <= -t} + 1) / (#{M_j >= t} or 1) <= q. If no positive threshold
    satisfies the rule, the FDR-control-safe behavior is to select nothing,
    represented by ``cutoff = inf``. ``use_fallback=True`` preserves the old
    exploratory behavior of returning the threshold whose ratio is closest to q.
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


def selected_groups_from_mirror_stats(mirror_stats: Dict[str, float], q: float = 0.1) -> tuple[list[str], float]:
    _, cutoff = compute_cutoff(mirror_stats, q=float(q))
    selected = [g for g, stat in mirror_stats.items() if float(stat) >= float(cutoff)]
    return selected, cutoff


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
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
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


def build_group_vector(x_df: pd.DataFrame, group_to_columns: Dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    col_to_gid = {}
    ordered_groups = list(group_to_columns.keys())
    for gid, g in enumerate(ordered_groups):
        for c in group_to_columns[g]:
            col_to_gid[c] = gid
    groups_vec = np.array([col_to_gid[c] for c in x_df.columns], dtype=int)
    return groups_vec, ordered_groups


def selected_groups_from_model(model: Any, ordered_groups: list[str]) -> list[str]:
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


def subset_group_mapping(
    x_df: pd.DataFrame,
    group_to_columns: Dict[str, list[str]],
    selected_groups: Iterable[str],
) -> tuple[pd.DataFrame, Dict[str, list[str]], list[str]]:
    selected_cols = []
    sub_map: Dict[str, list[str]] = {}
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
    group_to_columns: Dict[str, list[str]],
    *,
    n2: int,
    low_dim_factor: float = 2.0,
    lambda_grid: np.ndarray | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from group_lasso import GroupLasso
    except ImportError as exc:
        raise ImportError("group_lasso is required for high-dimensional stage-1 screening.") from exc

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
    group_to_columns: Dict[str, list[str]],
    group_is_relevant: Dict[str, bool] | None = None,
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

    group_is_relevant_sel = None if group_is_relevant is None else {g: group_is_relevant[g] for g in group_to_columns_sel.keys()}
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


__all__ = [
    "check_low_dim",
    "compute_cutoff",
    "compute_group_mirror_stats_ols",
    "run_group_gm_auto_pipeline",
    "run_group_lasso_screening",
    "selected_groups_from_mirror_stats",
    "selected_groups_from_model",
    "split_two_stage",
    "subset_group_mapping",
]
