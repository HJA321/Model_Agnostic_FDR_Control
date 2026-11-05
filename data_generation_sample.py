import numpy as np
import pandas as pd
from typing import Optional


def _sample_deeplink_t_latent_factors(
    n_samples: int,
    rng: np.random.Generator,
    latent_dim: int = 3,
    w0: float = 0.3,
    w1: float = 0.7,
    temporal_model: str = "forecast",
    ar_phi: float = 0.8,
):
    """
    DeepLINK-T latent factors:
      f_raw_t ~ N(0, Sigma), Sigma_ij = 0.9^{|i-j|}, i,j=1..3
      f_1 = f_raw_1
      f_t = w0 * f_raw_{t-1} + w1 * f_raw_t, t>=2

    For temporal_model='ar':
      f_t = ar_phi * f_{t-1} + sqrt(1 - ar_phi^2) * f_raw_t.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if abs((w0 + w1) - 1.0) > 1e-8:
        raise ValueError("Require w0 + w1 = 1.")

    r = int(latent_dim)
    if r <= 0:
        raise ValueError("latent_dim must be positive.")
    idx = np.arange(r)
    Sigma = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    f_raw = rng.multivariate_normal(
        mean=np.zeros(r, dtype=np.float64),
        cov=Sigma,
        size=n_samples,
    ).astype(np.float64)

    tm = str(temporal_model).lower()
    f = np.zeros_like(f_raw)
    f[0, :] = f_raw[0, :]
    if tm in {"forecast", "smooth", "deeplink_t"}:
        for t in range(1, n_samples):
            f[t, :] = w0 * f_raw[t - 1, :] + w1 * f_raw[t, :]
    elif tm in {"ar", "nlp"}:
        phi = float(ar_phi)
        if not (-1.0 < phi < 1.0):
            raise ValueError("ar_phi must be in (-1, 1).")
        innovation_scale = np.sqrt(max(0.0, 1.0 - phi ** 2))
        for t in range(1, n_samples):
            f[t, :] = phi * f[t - 1, :] + innovation_scale * f_raw[t, :]
    else:
        raise ValueError("temporal_model must be 'forecast'/'smooth' or 'ar'.")

    return f_raw.astype(np.float32), f.astype(np.float32)


def generate_deeplink_t_features(
    n_samples: int,
    n_relevant: int,
    n_irrelevant: int,
    latent_dim: int = 3,
    factor_model: str = "linear",  # linear | logistic
    data_generation_mode: str = "forecast",  # forecast | NLP
    latent_ar_phi: float = 0.8,
    random_state=None,
):
    """
    Generate X under DeepLINK-T style factor mechanism.

    All features share the same generation mechanism; relevance is only used to
    define a support set (non-zero beta indices) later for y generation.
    """
    rng = np.random.default_rng(random_state)
    p = int(n_relevant) + int(n_irrelevant)
    if p <= 0:
        raise ValueError("Total number of features must be positive.")
    if n_relevant < 0 or n_irrelevant < 0:
        raise ValueError("n_relevant and n_irrelevant must be nonnegative.")

    mode = str(data_generation_mode).lower()
    temporal_model = "ar" if mode == "nlp" else "forecast"
    _, f = _sample_deeplink_t_latent_factors(
        n_samples=n_samples,
        rng=rng,
        latent_dim=latent_dim,
        w0=0.3,
        w1=0.7,
        temporal_model=temporal_model,
        ar_phi=float(latent_ar_phi),
    )
    eps = rng.normal(0.0, 1.0, size=(n_samples, p)).astype(np.float32)

    fm = str(factor_model).lower()
    if fm == "linear":
        Lambda = rng.normal(0.0, 1.0, size=(p, int(latent_dim))).astype(np.float32)
        X = (f @ Lambda.T).astype(np.float32) + eps
    elif fm == "logistic":
        c = rng.normal(0.0, 1.0, size=p).astype(np.float32)
        lambda_mat = rng.normal(0.0, 1.0, size=(p, int(latent_dim) + 1)).astype(np.float32)
        design = np.concatenate([np.ones((n_samples, 1), dtype=np.float32), f], axis=1)
        eta = design @ lambda_mat.T
        X = (c[None, :] / (1.0 + np.exp(eta))).astype(np.float32) + eps
    else:
        raise ValueError("factor_model must be 'linear' or 'logistic'.")

    support_idx = rng.choice(p, size=n_relevant, replace=False)
    support_set = set(int(k) for k in support_idx.tolist())

    column_names = []
    relevant_columns = []
    irrelevant_columns = []
    rel_count = 0
    irr_count = 0
    for j in range(p):
        if j in support_set:
            name = f"relevant_feature_{rel_count}"
            rel_count += 1
            relevant_columns.append(name)
        else:
            name = f"noise_{irr_count}"
            irr_count += 1
            irrelevant_columns.append(name)
        column_names.append(name)

    data_df = pd.DataFrame(X, columns=column_names)
    data_df.attrs["deeplink_t_factor_model"] = fm
    data_df.attrs["deeplink_t_support_size"] = int(n_relevant)
    data_df.attrs["deeplink_t_latent_dim"] = int(latent_dim)
    data_df.attrs["deeplink_t_w0"] = 0.3
    data_df.attrs["deeplink_t_w1"] = 0.7
    data_df.attrs["data_generation_mode"] = mode
    data_df.attrs["latent_temporal_model"] = temporal_model
    data_df.attrs["latent_ar_phi"] = float(latent_ar_phi)
    return data_df, relevant_columns, irrelevant_columns


def _apply_link(
    eta: np.ndarray,
    link_function: str,
    *,
    signal_amplitude: float = 1.0,
    link_scale: float = 2.0,
    standardized_signal: bool = False,
) -> np.ndarray:
    lf = str(link_function).lower()
    if bool(standardized_signal):
        amp = float(signal_amplitude)
        scale = max(float(link_scale), 1e-12)
        if lf in {"linear", "identity", "standardized_linear"}:
            return amp * eta
        if lf in {"tanh", "scaled_tanh"}:
            return amp * np.tanh(eta / scale)
        if lf == "sin":
            return amp * np.sin(0.8 * eta)

    if lf in {"linear", "identity"}:
        return eta
    if lf in {"nonlinear", "sinexp"}:
        eta_clip = np.clip(eta, -20.0, 20.0)
        return np.sin(eta_clip) * np.exp(eta_clip)
    if lf == "tanh":
        return np.tanh(eta)
    if lf == "sin":
        return np.sin(0.8 * eta)
    raise ValueError(
        "Unsupported link_function. Use one of: "
        "'linear', 'nonlinear'/'sinexp', 'tanh', 'scaled_tanh', 'sin'."
    )


def _build_lag_weights(
    lookback: int,
    rng: np.random.Generator,
    *,
    lag_decay: float = 0.8,
    lag_weight_norm: str = "l2",
    lag_weight_scheme: str = "decay",
    lag_chunk_size: int = 5,
    lag_chunk_weight_low: float = 0.8,
    lag_chunk_weight_high: float = 1.2,
) -> np.ndarray:
    L = int(lookback)
    scheme = str(lag_weight_scheme).lower()
    if scheme == "decay":
        raw_w = (float(lag_decay) ** np.arange(L)).astype(np.float64)
    elif scheme in {"uniform", "flat"}:
        raw_w = np.ones(L, dtype=np.float64)
    elif scheme in {"chunk_uniform", "chunk"}:
        chunk_size = int(lag_chunk_size)
        if chunk_size <= 0:
            raise ValueError("lag_chunk_size must be positive.")
        low = float(lag_chunk_weight_low)
        high = float(lag_chunk_weight_high)
        if high < low:
            raise ValueError("lag_chunk_weight_high must be >= lag_chunk_weight_low.")
        num_chunks = int(np.ceil(L / chunk_size))
        chunk_weights = rng.uniform(low, high, size=num_chunks)
        raw_w = np.repeat(chunk_weights, chunk_size)[:L].astype(np.float64)
    else:
        raise ValueError("lag_weight_scheme must be 'decay', 'uniform', or 'chunk_uniform'.")

    nrm_mode = str(lag_weight_norm).lower()
    if nrm_mode == "l1":
        return raw_w / (np.sum(np.abs(raw_w)) + 1e-12)
    if nrm_mode == "l2":
        return raw_w / np.sqrt(np.sum(raw_w ** 2) + 1e-12)
    raise ValueError("lag_weight_norm must be 'l1' or 'l2'.")


def _standardize_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-12)


def _build_beta_vector(
    columns,
    relevant_columns,
    rng: np.random.Generator,
    signal_amplitude: float = 10.0,
):
    col_to_idx = {col: j for j, col in enumerate(columns)}
    support_idx = [col_to_idx[col] for col in relevant_columns]
    p = len(columns)
    beta = np.zeros(p, dtype=np.float32)
    signs = rng.choice([-1.0, 1.0], size=len(support_idx)).astype(np.float32)
    beta[np.asarray(support_idx, dtype=int)] = float(signal_amplitude) * signs
    return beta


def generate_labels_for_lstm(
    data_df: pd.DataFrame,
    relevant_columns,
    noise_level: float = 1.0,
    signal_amplitude: float = 10.0,
    link_function: str = "nonlinear",
    y_mode: str = "current_x",  # current_x | lag_weighted
    lookback: int = 10,
    lag_decay: float = 0.8,
    lag_weight_norm: str = "l2",  # l1 | l2
    data_generation_mode: str = "forecast",  # forecast | NLP
    lag_weight_scheme: Optional[str] = None,  # decay | uniform | chunk_uniform
    lag_chunk_size: int = 5,
    lag_chunk_weight_low: float = 0.8,
    lag_chunk_weight_high: float = 1.2,
    standardize_signal: Optional[bool] = None,
    link_scale: float = 2.0,
    random_state=None,
):
    """
    Unified y generator with two options:
      1) current_x:
           eta_t = x_t^T beta
      2) lag_weighted:
           eta_t = sum_{l=0}^{L-1} w_l * (x_{t-1-l}^T beta), t>=L
           first L points are filled with 0.0 (CatNet-compatible alignment)
    """
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame.")

    rng = np.random.default_rng(random_state)
    X = data_df.to_numpy(dtype=np.float32)
    n_samples, _ = X.shape
    data_mode = str(data_generation_mode).lower()
    use_standardized_signal = (data_mode == "nlp") if standardize_signal is None else bool(standardize_signal)
    beta_amplitude = 1.0 if use_standardized_signal else float(signal_amplitude)
    beta = _build_beta_vector(
        columns=list(data_df.columns),
        relevant_columns=relevant_columns,
        rng=rng,
        signal_amplitude=beta_amplitude,
    )

    mode = str(y_mode).lower()
    if mode == "current_x":
        eta = X @ beta
        if use_standardized_signal:
            eta = _standardize_signal(eta)
        mu = _apply_link(
            eta,
            link_function,
            signal_amplitude=float(signal_amplitude),
            link_scale=float(link_scale),
            standardized_signal=use_standardized_signal,
        )
        y = mu.astype(np.float32) + rng.normal(0.0, float(noise_level), size=n_samples).astype(np.float32)
        return y.astype(np.float32)

    if mode != "lag_weighted":
        raise ValueError("y_mode must be 'current_x' or 'lag_weighted'.")

    L = int(lookback)
    if L <= 0:
        raise ValueError("lookback must be >= 1 for y_mode='lag_weighted'.")
    if n_samples <= L:
        raise ValueError("n_samples must be greater than lookback for y_mode='lag_weighted'.")

    effective_lag_weight_scheme = (
        "chunk_uniform" if data_mode == "nlp" else "decay"
    ) if lag_weight_scheme is None else str(lag_weight_scheme)
    w = _build_lag_weights(
        L,
        rng,
        lag_decay=float(lag_decay),
        lag_weight_norm=lag_weight_norm,
        lag_weight_scheme=effective_lag_weight_scheme,
        lag_chunk_size=int(lag_chunk_size),
        lag_chunk_weight_low=float(lag_chunk_weight_low),
        lag_chunk_weight_high=float(lag_chunk_weight_high),
    )

    proj = X @ beta  # (n_samples,)
    n_valid = n_samples - L
    eta_valid = np.zeros(n_valid, dtype=np.float64)
    for lag in range(L):
        eta_valid += w[lag] * proj[(L - 1 - lag):(n_samples - 1 - lag)]

    if use_standardized_signal:
        eta_valid = _standardize_signal(eta_valid)
    mu_valid = _apply_link(
        eta_valid,
        link_function,
        signal_amplitude=float(signal_amplitude),
        link_scale=float(link_scale),
        standardized_signal=use_standardized_signal,
    )
    eps = rng.normal(0.0, float(noise_level), size=n_valid).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    y[L:] = mu_valid.astype(np.float32) + eps
    return y.astype(np.float32)


def simulate_deeplink_t_dataset(
    n_samples: int = 500,
    n_relevant: int = 50,
    n_irrelevant: int = 150,
    latent_dim: int = 3,
    factor_model: str = "linear",
    data_generation_mode: str = "forecast",
    latent_ar_phi: float = 0.8,
    y_mode: str = "lag_weighted",
    lookback: int = 10,
    lag_decay: float = 0.8,
    lag_weight_norm: str = "l2",
    lag_weight_scheme: Optional[str] = None,
    lag_chunk_size: int = 5,
    lag_chunk_weight_low: float = 0.8,
    lag_chunk_weight_high: float = 1.2,
    standardize_signal: Optional[bool] = None,
    link_scale: float = 2.0,
    signal_amplitude: float = 10.0,
    link_function: str = "nonlinear",
    noise_level: float = 1.0,
    random_state=None,
):
    """
    High-level one-call API:
      1) generate DeepLINK-T style features
      2) generate y with selected y_mode/link
    """
    data_df, relevant_columns, irrelevant_columns = generate_deeplink_t_features(
        n_samples=n_samples,
        n_relevant=n_relevant,
        n_irrelevant=n_irrelevant,
        latent_dim=latent_dim,
        factor_model=factor_model,
        data_generation_mode=data_generation_mode,
        latent_ar_phi=float(latent_ar_phi),
        random_state=random_state,
    )
    y = generate_labels_for_lstm(
        data_df=data_df,
        relevant_columns=relevant_columns,
        noise_level=noise_level,
        signal_amplitude=signal_amplitude,
        link_function=link_function,
        y_mode=y_mode,
        lookback=lookback,
        lag_decay=lag_decay,
        lag_weight_norm=lag_weight_norm,
        data_generation_mode=data_generation_mode,
        lag_weight_scheme=lag_weight_scheme,
        lag_chunk_size=int(lag_chunk_size),
        lag_chunk_weight_low=float(lag_chunk_weight_low),
        lag_chunk_weight_high=float(lag_chunk_weight_high),
        standardize_signal=standardize_signal,
        link_scale=float(link_scale),
        random_state=random_state,
    )
    return data_df, y, relevant_columns, irrelevant_columns
