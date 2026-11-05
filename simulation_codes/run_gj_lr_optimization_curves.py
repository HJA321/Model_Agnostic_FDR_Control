#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "catnet_matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

import simulation as sim  # noqa: E402
from catnet import _standardize_feature_df  # noqa: E402
from data_generation_sample import simulate_deeplink_t_dataset  # noqa: E402
from src.FDR.kernel_dependence import find_Gj  # noqa: E402


@dataclass(frozen=True)
class DimensionConfig:
    name: str
    n_relevant: int
    n_irrelevant: int
    lookback: int

    @property
    def n_features(self) -> int:
        return int(self.n_relevant) + int(self.n_irrelevant)


DIMENSION_CONFIGS: Dict[str, DimensionConfig] = {
    "lowdim": DimensionConfig("lowdim", n_relevant=15, n_irrelevant=285, lookback=5),
    "highdim": DimensionConfig("highdim", n_relevant=15, n_irrelevant=685, lookback=5),
}


def _parse_csv_floats(value: str) -> Tuple[float, ...]:
    out = tuple(float(v.strip()) for v in str(value).split(",") if v.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def _parse_csv_strings(value: str, *, allowed: Iterable[str]) -> Tuple[str, ...]:
    allowed_set = set(allowed)
    out = tuple(v.strip().lower() for v in str(value).split(",") if v.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    bad = [v for v in out if v not in allowed_set]
    if bad:
        raise argparse.ArgumentTypeError(f"unsupported value(s): {', '.join(bad)}")
    return tuple(dict.fromkeys(out))


def _build_sequence_array(data_df: pd.DataFrame, lookback: int) -> np.ndarray:
    rows = []
    lb = int(lookback)
    for i in range(lb, len(data_df)):
        rows.append(data_df.iloc[i - lb:i].values)
    return np.asarray(rows, dtype=np.float32)


def _make_dimension_data(
    cfg: DimensionConfig,
    *,
    seed: int,
    max_rows: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    data, _y, _relevant_columns, _irrelevant_columns = simulate_deeplink_t_dataset(
        n_samples=int(sim.N_SAMPLES),
        n_relevant=int(cfg.n_relevant),
        n_irrelevant=int(cfg.n_irrelevant),
        latent_dim=int(sim.LATENT_DIM),
        factor_model=sim.FACTOR_MODEL,
        data_generation_mode=getattr(sim, "DATA_GENERATION_MODE", "forecast"),
        latent_ar_phi=float(getattr(sim, "LATENT_AR_PHI", 0.8)),
        y_mode=sim.Y_MODE,
        lookback=int(cfg.lookback),
        lag_decay=float(sim.LAG_DECAY),
        lag_weight_norm=sim.LAG_WEIGHT_NORM,
        lag_weight_scheme=getattr(sim, "LAG_WEIGHT_SCHEME", None),
        lag_chunk_size=int(getattr(sim, "LAG_CHUNK_SIZE", 5)),
        lag_chunk_weight_low=float(getattr(sim, "LAG_CHUNK_WEIGHT_LOW", 0.8)),
        lag_chunk_weight_high=float(getattr(sim, "LAG_CHUNK_WEIGHT_HIGH", 1.2)),
        standardize_signal=getattr(sim, "STANDARDIZE_SIGNAL", None),
        link_scale=float(getattr(sim, "LINK_SCALE", 2.0)),
        signal_amplitude=float(sim.SIGNAL_AMPLITUDE),
        link_function=sim.Y_LINK_FUNCTION,
        noise_level=float(sim.LABEL_NOISE),
        random_state=int(seed),
    )
    if bool(sim.STANDARDIZE_INPUT):
        data = _standardize_feature_df(data)

    data_seq = _build_sequence_array(data, int(cfg.lookback))
    data_seq_t = torch.as_tensor(data_seq, dtype=torch.float32)
    gen = torch.Generator(device=data_seq_t.device)
    gen.manual_seed(int(seed) + 100_003 + int(cfg.n_features))
    noise_seq_t = torch.randn(data_seq_t.shape, dtype=data_seq_t.dtype, generator=gen)
    if max_rows is not None and int(max_rows) > 0 and data_seq_t.shape[0] > int(max_rows):
        rng = np.random.default_rng(int(seed) + 200_003 + int(cfg.n_features))
        idx_np = np.sort(rng.choice(data_seq_t.shape[0], size=int(max_rows), replace=False))
        idx_t = torch.as_tensor(idx_np, dtype=torch.long)
        data_seq_t = data_seq_t.index_select(0, idx_t)
        noise_seq_t = noise_seq_t.index_select(0, idx_t)
    return data_seq_t, noise_seq_t, [str(c) for c in data.columns]


def _normalize_trace(trace: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if trace.size == 0:
        return trace.astype(np.float32)
    denom = float(trace[0])
    if not np.isfinite(denom) or abs(denom) <= eps:
        denom = float(np.nanmax(np.abs(trace)))
    if not np.isfinite(denom) or abs(denom) <= eps:
        denom = 1.0
    norm = trace / denom
    norm[~np.isfinite(norm)] = np.nan
    return norm.astype(np.float32)


def _run_dimension_curves(
    cfg: DimensionConfig,
    *,
    lrs: Tuple[float, ...],
    max_iter: int,
    seed: int,
    metric: str,
    feature_limit: int | None,
    max_rows: int | None,
    show_progress: bool,
) -> Tuple[np.ndarray, List[str]]:
    data_seq_t, noise_seq_t, feature_names = _make_dimension_data(cfg, seed=seed, max_rows=max_rows)
    n_seq, lookback, p = data_seq_t.shape
    n_features_to_run = p if feature_limit is None else min(p, int(feature_limit))
    traces = np.full((len(lrs), n_features_to_run, int(max_iter)), np.nan, dtype=np.float32)
    history_key = f"{metric}_history"

    for j in range(n_features_to_run):
        Xg = data_seq_t[:, :, j]
        Zg = noise_seq_t[:, :, j]
        if p > 1:
            mask = [k for k in range(p) if k != j]
            W = data_seq_t[:, :, mask].reshape(n_seq, -1)
        else:
            W = torch.zeros((n_seq, 0), dtype=torch.float32)

        if show_progress:
            print(f"[{cfg.name}] feature {j + 1}/{n_features_to_run}: {feature_names[j]}", flush=True)

        for lr_idx, lr in enumerate(lrs):
            _G, info = find_Gj(
                Xg=Xg,
                W=W,
                method=sim.GROUP_METHOD,
                Zg=Zg,
                objective=sim.GROUP_OBJECTIVE,
                residual_mode=sim.GROUP_RESIDUAL_MODE,
                lambda_reg=float(sim.GROUP_LAMBDA_REG),
                max_iter=int(max_iter),
                lr=float(lr),
                rank_r=sim.GROUP_RANK_R,
                enforce_symmetric_delta=bool(sim.GROUP_ENFORCE_SYMMETRIC_DELTA),
                delta_norm_cap=sim.GROUP_DELTA_NORM_CAP,
                free_enforce_symmetric=bool(sim.GROUP_FREE_ENFORCE_SYMMETRIC),
                free_g_norm_cap=sim.GROUP_FREE_G_NORM_CAP,
                kw_use_pca=bool(sim.GROUP_KW_USE_PCA),
                kw_pca_components=int(cfg.lookback) * 2,
                high_dim_mode=sim.GROUP_HIGH_DIM_MODE,
                ridge_lambda_w=float(sim.N_SAMPLES) * 0.1,
                ridge_lambda_z=float(sim.N_SAMPLES) * 0.1,
                ridge_q_eps=sim.GROUP_RIDGE_Q_EPS,
                ridge_g_delta=sim.GROUP_RIDGE_G_DELTA,
                print_objective_every=0,
                random_state=int(seed) + j,
                print_kw_diagnostics=False,
                return_info=True,
                return_torch=True,
            )
            history = np.asarray(info.get(history_key, info.get("loss_history", [])), dtype=np.float64)
            if history.size != int(max_iter):
                padded = np.full(int(max_iter), np.nan, dtype=np.float64)
                padded[: min(history.size, int(max_iter))] = history[: int(max_iter)]
                history = padded
            traces[lr_idx, j, :] = _normalize_trace(history)

    return traces, feature_names[:n_features_to_run]


def _summary_from_traces(
    traces: np.ndarray,
    *,
    lrs: Tuple[float, ...],
    cfg: DimensionConfig,
    metric: str,
) -> pd.DataFrame:
    rows = []
    epochs = np.arange(1, traces.shape[2] + 1, dtype=int)
    for lr_idx, lr in enumerate(lrs):
        arr = traces[lr_idx]
        mean = np.nanmean(arr, axis=0)
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanquantile(arr, 0.25, axis=0)
        q75 = np.nanquantile(arr, 0.75, axis=0)
        for k, epoch in enumerate(epochs):
            rows.append(
                {
                    "dimension": cfg.name,
                    "n_features": int(cfg.n_features),
                    "lookback": int(cfg.lookback),
                    "metric": str(metric),
                    "learning_rate": float(lr),
                    "epoch": int(epoch),
                    "mean_normalized": float(mean[k]),
                    "median_normalized": float(median[k]),
                    "q25_normalized": float(q25[k]),
                    "q75_normalized": float(q75[k]),
                    "n_feature_curves": int(traces.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def _plot_dimension(
    summary: pd.DataFrame,
    *,
    cfg: DimensionConfig,
    metric: str,
    out_path_png: Path,
    out_path_pdf: Path,
) -> None:
    sub = summary.loc[summary["dimension"] == cfg.name].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    for lr, grp in sub.groupby("learning_rate", sort=True):
        grp = grp.sort_values("epoch")
        x = grp["epoch"].to_numpy(dtype=float)
        y = grp["mean_normalized"].to_numpy(dtype=float)
        q25 = grp["q25_normalized"].to_numpy(dtype=float)
        q75 = grp["q75_normalized"].to_numpy(dtype=float)
        label = f"lr={float(lr):g}"
        ax.plot(x, y, linewidth=2.0, label=label)
        ax.fill_between(x, q25, q75, alpha=0.14, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Mean normalized {metric}")
    ax.set_title(f"{cfg.name}: G_j optimization")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model-agnostic G_j optimization learning-rate curves for low/high-dimensional settings."
    )
    parser.add_argument("--dimensions", type=lambda s: _parse_csv_strings(s, allowed=tuple(DIMENSION_CONFIGS)),
                        default=("lowdim", "highdim"))
    parser.add_argument("--lrs", type=_parse_csv_floats, default=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--metric", choices=("objective", "loss"), default="objective")
    parser.add_argument("--feature-limit", type=int, default=0,
                        help="Optional smoke-test cap; 0 means use all features.")
    parser.add_argument("--max-rows", type=int, default=512,
                        help="Rows used for the G_j optimization curves; 0 uses all sequence rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("gj_lr_optimization_runs"))
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    lrs = tuple(float(v) for v in args.lrs)
    dimensions = tuple(str(v) for v in args.dimensions)
    feature_limit = None if int(args.feature_limit) <= 0 else int(args.feature_limit)
    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)

    settings = {
        "run_id": run_id,
        "dimensions": list(dimensions),
        "lrs": list(lrs),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "metric": str(args.metric),
        "feature_limit": feature_limit,
        "max_rows": max_rows,
        "n_samples": int(sim.N_SAMPLES),
        "group_method": str(sim.GROUP_METHOD),
        "group_objective": str(sim.GROUP_OBJECTIVE),
        "group_lambda_reg": float(sim.GROUP_LAMBDA_REG),
        "group_high_dim_mode": str(sim.GROUP_HIGH_DIM_MODE),
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    summaries = []
    trace_paths = {}
    for dim_name in dimensions:
        cfg = DIMENSION_CONFIGS[dim_name]
        print(
            f"Running {cfg.name}: p={cfg.n_features}, lookback={cfg.lookback}, "
            f"features={'all' if feature_limit is None else feature_limit}, "
            f"rows={'all' if max_rows is None else max_rows}, lrs={lrs}",
            flush=True,
        )
        traces, feature_names = _run_dimension_curves(
            cfg,
            lrs=lrs,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
            metric=str(args.metric),
            feature_limit=feature_limit,
            max_rows=max_rows,
            show_progress=bool(args.show_progress),
        )
        trace_path = out_dir / f"gj_lr_{cfg.name}_raw_traces.npz"
        np.savez_compressed(
            trace_path,
            traces=traces,
            lrs=np.asarray(lrs, dtype=np.float64),
            feature_names=np.asarray(feature_names, dtype=object),
            dimension=np.asarray(cfg.name),
            metric=np.asarray(str(args.metric)),
        )
        trace_paths[cfg.name] = str(trace_path.resolve())
        summaries.append(_summary_from_traces(traces, lrs=lrs, cfg=cfg, metric=str(args.metric)))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_path = out_dir / "gj_lr_summary_by_dimension.csv"
    summary_df.to_csv(summary_path, index=False)

    plots_dir = out_dir / "plots"
    for dim_name in dimensions:
        cfg = DIMENSION_CONFIGS[dim_name]
        base = plots_dir / f"gj_lr_{cfg.name}"
        _plot_dimension(
            summary_df,
            cfg=cfg,
            metric=str(args.metric),
            out_path_png=base.with_suffix(".png"),
            out_path_pdf=base.with_suffix(".pdf"),
        )

    print(f"Saved settings: {out_dir / 'settings.json'}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    for dim_name, path in trace_paths.items():
        print(f"Saved raw traces ({dim_name}): {path}", flush=True)
    print(f"Saved plots: {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
