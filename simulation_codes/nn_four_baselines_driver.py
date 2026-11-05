#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "catnet_matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baseline_simulation as external_baselines  # noqa: E402


def _load_local_simulation_module():
    simulation_path = SCRIPT_DIR / "simulation.py"
    spec = importlib.util.spec_from_file_location("catnet_nn_four_baselines_simulation", simulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load simulation module from {simulation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


catnet_simulation = _load_local_simulation_module()


COMMON_CONFIG_NAMES = (
    "SEED",
    "NUM_SIMULATIONS",
    "N_SAMPLES",
    "N_RELEVANT",
    "N_IRRELEVANT",
    "LOOKBACK",
    "FACTOR_MODEL",
    "LATENT_DIM",
    "Y_MODE",
    "Y_LINK_FUNCTION",
    "SIGNAL_AMPLITUDE",
    "LABEL_NOISE",
    "LAG_DECAY",
    "LAG_WEIGHT_NORM",
    "MODEL_TYPE",
    "HIDDEN_DIM_MULTIPLIER",
    "NUM_LAYERS",
    "DROPOUT",
    "OPTIM_LR",
    "OPTIM_WEIGHT_DECAY",
    "TRANSFORMER_NHEAD",
    "TRANSFORMER_DIM_FEEDFORWARD",
    "CATNET_TRAIN_EPOCHS",
    "AUTO_TUNE_HPARAMS",
    "TUNE_N_TRIALS",
    "TUNE_N_FOLDS",
    "TUNE_VAL_RATIO",
    "TUNE_MAX_EPOCHS",
    "TUNE_PATIENCE",
    "TUNE_WEIGHT_DECAY_TARGET",
    "TUNE_LAMBDA_WD",
    "TUNE_LAMBDA_DROPOUT",
    "FDR_Q",
    "STANDARDIZE_INPUT",
    "SHOW_PROGRESS",
)


def _parse_mirror_modes(value: str) -> tuple[str, ...]:
    modes = tuple(m.strip() for m in str(value).split(",") if m.strip())
    if not modes:
        raise argparse.ArgumentTypeError("mirror modes cannot be empty")
    valid = {"max_l1", "max_l2", "sum_l1", "sum_l2", "multiple_l1", "multiple_l2"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"unsupported mirror mode(s): {', '.join(bad)}")
    return tuple(dict.fromkeys(modes))


def _noise_tag(label_noise: float) -> str:
    return str(float(label_noise)).replace(".", "p")


def _configure_external_baselines(result_path: Path) -> None:
    for name in COMMON_CONFIG_NAMES:
        if hasattr(external_baselines, name) and hasattr(catnet_simulation, name):
            setattr(external_baselines, name, getattr(catnet_simulation, name))

    external_baselines.RUN_AFS = True
    external_baselines.RUN_PROX_GL = False
    external_baselines.RUN_PROX_AGL = True
    external_baselines.RUN_MIRROR = True
    external_baselines.AFS_TRAIN_EPOCHS = int(external_baselines.CATNET_TRAIN_EPOCHS)
    external_baselines.PROX_TRAIN_EPOCHS = int(external_baselines.CATNET_TRAIN_EPOCHS)
    external_baselines.SAVE_PER_SIM_DETAILS = False
    external_baselines.RESULT_CSV_PATH = result_path


def _configure_lag_perturb_derivative(
    result_path: Path,
    *,
    derivative_n_avg: int,
    mirror_modes: tuple[str, ...],
) -> None:
    catnet_simulation.IMPORTANCE_MODES = ("derivative",)
    catnet_simulation.IMPORTANCE_MODE = "derivative"
    catnet_simulation.DERIVATIVE_METHOD = "lag_perturb"
    catnet_simulation.DERIVATIVE_N_AVG = int(derivative_n_avg)
    catnet_simulation.DERIVATIVE_N_STEPS = 1
    catnet_simulation.MIRROR_MAGNITUDE_MODES = tuple(mirror_modes)
    catnet_simulation.MIRROR_MAGNITUDE_MODE = tuple(mirror_modes)[0]
    catnet_simulation.DIAG_RUN_ORIGINAL_IMPORTANCE = False
    catnet_simulation.RESULT_CSV_PATH = result_path


def run_four_baselines(
    *,
    model_type: str,
    label_noise: float,
    signal_name: str,
    dimension_name: str,
    n_relevant: int,
    n_irrelevant: int,
    lookback: int,
    num_simulations: int = 25,
    seed: int = 2,
    derivative_n_avg: int = 2,
    mirror_modes: tuple[str, ...] = ("sum_l2",),
    results_dir: Path = Path("nn_signal_baseline_runs"),
    show_progress: bool = False,
) -> tuple[Path, Path]:
    catnet_simulation.SEED = int(seed)
    catnet_simulation.NUM_SIMULATIONS = int(num_simulations)
    catnet_simulation.MODEL_TYPE = str(model_type)
    catnet_simulation.LABEL_NOISE = float(label_noise)
    catnet_simulation.N_RELEVANT = int(n_relevant)
    catnet_simulation.N_IRRELEVANT = int(n_irrelevant)
    catnet_simulation.LOOKBACK = int(lookback)
    catnet_simulation.GROUP_KW_PCA_COMPONENTS = int(lookback) * 2
    catnet_simulation.SHOW_PROGRESS = bool(show_progress)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_tag = str(signal_name).lower().strip()
    dimension_tag = str(dimension_name).lower().strip()
    n_features = int(n_relevant) + int(n_irrelevant)
    result_dir = Path(results_dir)
    external_path = result_dir / (
        f"{model_type}_{dimension_tag}_{signal_tag}_p{n_features}_lag{int(lookback)}_"
        f"noise{_noise_tag(label_noise)}_afs_agl_mirror_{run_tag}.csv"
    )
    derivative_path = result_dir / (
        f"{model_type}_{dimension_tag}_{signal_tag}_p{n_features}_lag{int(lookback)}_"
        f"noise{_noise_tag(label_noise)}_lag_perturb_derivative_{run_tag}.csv"
    )

    _configure_external_baselines(external_path)
    _configure_lag_perturb_derivative(
        derivative_path,
        derivative_n_avg=int(derivative_n_avg),
        mirror_modes=tuple(mirror_modes),
    )

    print(
        f"Running {dimension_tag} {signal_tag} {model_type} baselines: afs, prox_agl, mirror",
        flush=True,
    )
    external_baselines.main()
    print(f"External baseline results: {external_path.resolve()}", flush=True)

    print(f"Running {dimension_tag} {signal_tag} {model_type} lag-perturb fitted derivative baseline", flush=True)
    catnet_simulation.main()
    print(f"Lag-perturb derivative results: {derivative_path.resolve()}", flush=True)

    return external_path, derivative_path


def main(
    *,
    default_model_type: str = "lstm",
    default_label_noise: float = 0.3,
    default_signal_name: str = "strong",
    default_dimension_name: str = "lowdim",
    default_n_relevant: int = 15,
    default_n_irrelevant: int = 285,
    default_lookback: int = 5,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the neural baseline comparison: AFS, prox AGL, mirror, "
            "and lag-perturb fitted derivative."
        )
    )
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--model-type",
        choices=("lstm", "rnn", "gru", "transformer"),
        default=default_model_type,
    )
    parser.add_argument("--label-noise", type=float, default=default_label_noise)
    parser.add_argument("--signal-name", type=str, default=default_signal_name)
    parser.add_argument("--dimension-name", type=str, default=default_dimension_name)
    parser.add_argument("--n-relevant", type=int, default=default_n_relevant)
    parser.add_argument("--n-irrelevant", type=int, default=default_n_irrelevant)
    parser.add_argument("--lookback", type=int, default=default_lookback)
    parser.add_argument("--derivative-n-avg", type=int, default=2)
    parser.add_argument("--mirror-modes", type=_parse_mirror_modes, default=("sum_l2",))
    parser.add_argument("--results-dir", type=Path, default=Path("nn_signal_baseline_runs"))
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    run_four_baselines(
        model_type=str(args.model_type),
        label_noise=float(args.label_noise),
        signal_name=str(args.signal_name),
        dimension_name=str(args.dimension_name),
        n_relevant=int(args.n_relevant),
        n_irrelevant=int(args.n_irrelevant),
        lookback=int(args.lookback),
        num_simulations=int(args.num_simulations),
        seed=int(args.seed),
        derivative_n_avg=int(args.derivative_n_avg),
        mirror_modes=tuple(args.mirror_modes),
        results_dir=Path(args.results_dir),
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
