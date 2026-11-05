import numpy as np
import torch
import torch.nn as nn


def _build_plus_minus_pair_index(feature_names) -> torch.Tensor:
    """
    Build partner index map for mirror pairs:
      name_plus  <->  name_minus
    Unpaired players get -1.
    """
    names = [str(f) for f in feature_names]
    q = len(names)
    pair_idx = -1 * np.ones(q, dtype=np.int64)

    plus_map = {}
    minus_map = {}
    for i, name in enumerate(names):
        if name.endswith("_plus"):
            plus_map[name[:-5]] = i
        elif name.endswith("_minus"):
            minus_map[name[:-6]] = i

    for base, i_plus in plus_map.items():
        i_minus = minus_map.get(base, None)
        if i_minus is None:
            continue
        pair_idx[i_plus] = int(i_minus)
        pair_idx[i_minus] = int(i_plus)

    return torch.as_tensor(pair_idx, dtype=torch.long)


def calc_shap_values(
    model: nn.Module,
    X: torch.Tensor,
    feature_names,
    background: torch.Tensor = None,
    bg_size: int = 64,
    n_permutations: int = 10,
    eval_batch_size: int = None,
    perm_batch_size: int = None,
    sample_batch_size: int = None,
    path_chunk_size: int = 64,
    random_state: int = 0,
    player_mode: str = "feature_group",
    return_torch: bool = False,
    print_diagnostics: bool = False,
):
    """
    Monte Carlo permutation SHAP with feature groups as players.
    Each player is one feature's full lag block, i.e. count = p.

    For mirror-feature names ending with "_plus"/"_minus", we apply a
    pair-aware symmetric constraint when assigning marginal contributions:
    when evaluating one side, its paired mirror side is excluded from the
    coalition context (using forward/backward path selection per permutation).

    Optimized to reduce CPU<->GPU communication:
    - keep path construction/evaluation on torch tensors
    - only move final SHAP matrix back to CPU

    Returns:
        dict:
          - return_torch=False: {feature: np.ndarray(shape=(n,))}
          - return_torch=True:  {feature: torch.Tensor(shape=(n,), device=model_device)}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    X_dev = X.detach().to(device)
    n, T, p = X_dev.shape
    if len(feature_names) != p:
        raise ValueError("feature_names length must equal p")

    rng = np.random.default_rng(random_state)

    if background is None:
        bg_idx = rng.choice(n, size=min(bg_size, n), replace=False)
        background_dev = X_dev[torch.as_tensor(bg_idx, device=device, dtype=torch.long)]
    else:
        background_dev = background.detach().to(device=device, dtype=X.dtype)

    m_bg = int(background_dev.shape[0])
    if m_bg <= 0:
        raise ValueError("background must contain at least one sample.")

    # Auto-tuned defaults for better GPU utilization; still overridable by caller.
    if eval_batch_size is None:
        eval_batch_size = 16384 if device.type == "cuda" else 4096
    if perm_batch_size is None:
        perm_batch_size = min(n_permutations, 32 if device.type == "cuda" else 8)
    if sample_batch_size is None:
        sample_batch_size = min(n, 256 if device.type == "cuda" else 32)

    eval_batch_size = max(1, int(eval_batch_size))
    perm_batch_size = max(1, int(perm_batch_size))
    sample_batch_size = max(1, int(sample_batch_size))
    path_chunk_size = max(1, int(path_chunk_size))

    def predict_torch(arr_3d: torch.Tensor) -> torch.Tensor:
        """arr_3d: (m, T, p) -> (m,), output on device"""
        preds = []
        with torch.inference_mode():
            for k in range(0, arr_3d.shape[0], eval_batch_size):
                out = model(arr_3d[k:k + eval_batch_size])
                out = out.reshape(out.shape[0], -1)
                if out.shape[1] != 1:
                    raise ValueError("Model output must be scalar per sample.")
                preds.append(out[:, 0])
        return torch.cat(preds, dim=0)

    player_mode = str(player_mode).lower()
    if player_mode != "feature_group":
        raise ValueError(
            "Only group-player SHAP is supported now: player_mode must be 'feature_group'."
        )

    row_idx_cache = None
    q = p
    pair_idx_cpu = _build_plus_minus_pair_index(feature_names)  # (q,), -1 for unpaired
    pair_idx_dev = pair_idx_cpu.to(device=device)
    if bool(print_diagnostics):
        n_paired_players = int((pair_idx_cpu >= 0).sum().item())
        n_pairs = n_paired_players // 2
        print(
            f"SHAP(pair-aware): detected {n_pairs} plus/minus pairs "
            f"({n_paired_players}/{q} players paired, {q - n_paired_players} unpaired).",
            flush=True,
        )

    shap_vals_group = torch.zeros((n, p), dtype=torch.float32, device=device)
    for i_start in range(0, n, sample_batch_size):
        i_end = min(n, i_start + sample_batch_size)
        bs = i_end - i_start
        x_block = X_dev[i_start:i_end]  # (bs, T, p)
        phi_block = torch.zeros((bs, q), dtype=torch.float32, device=device)

        for s in range(0, n_permutations, perm_batch_size):
            b = min(perm_batch_size, n_permutations - s)
            total = bs * b

            perms_np = np.stack([rng.permutation(q) for _ in range(total)], axis=0)
            perms = torch.as_tensor(perms_np, device=device, dtype=torch.long)  # (total, q)
            x_rep = x_block.unsqueeze(1).expand(bs, b, T, p).reshape(total, T, p)  # (total, T, p)
            bg_sel = torch.as_tensor(rng.integers(0, m_bg, size=total), device=device, dtype=torch.long)
            baseline = background_dev.index_select(0, bg_sel)  # (total, T, p)

            if row_idx_cache is None or row_idx_cache.shape[0] < total:
                row_idx_cache = torch.arange(total, device=device)
            row_idx = row_idx_cache[:total]

            # Forward path: baseline -> x_rep (add one full feature-group each step)
            # Chunked state evaluation reduces model-call overhead significantly.
            f_state = baseline.clone()
            y_prev = predict_torch(f_state)  # (total,)
            contrib_f = torch.zeros((total, q), dtype=torch.float32, device=device)
            for k0 in range(0, q, path_chunk_size):
                k1 = min(q, k0 + path_chunk_size)
                state_list = []
                for k in range(k0, k1):
                    feat_idx = perms[:, k]
                    f_state[row_idx, :, feat_idx] = x_rep[row_idx, :, feat_idx]
                    state_list.append(f_state.clone())
                states = torch.cat(state_list, dim=0)  # ((k1-k0)*total, T, p)
                y_chunk = predict_torch(states).reshape(k1 - k0, total)  # (chunk, total)
                y_prev_row = y_prev
                for kk in range(k1 - k0):
                    y_cur = y_chunk[kk]
                    contrib_f[:, k0 + kk] = y_cur - y_prev_row
                    y_prev_row = y_cur
                y_prev = y_prev_row

            # Backward path: x_rep -> baseline (remove one full feature-group each step)
            b_state = x_rep.clone()
            y_prev = predict_torch(b_state)  # (total,)
            contrib_b = torch.zeros((total, q), dtype=torch.float32, device=device)
            for k0 in range(0, q, path_chunk_size):
                k1 = min(q, k0 + path_chunk_size)
                state_list = []
                for k in range(k0, k1):
                    feat_idx = perms[:, k]
                    b_state[row_idx, :, feat_idx] = baseline[row_idx, :, feat_idx]
                    state_list.append(b_state.clone())
                states = torch.cat(state_list, dim=0)  # ((k1-k0)*total, T, p)
                y_chunk = predict_torch(states).reshape(k1 - k0, total)  # (chunk, total)
                y_prev_row = y_prev
                for kk in range(k1 - k0):
                    y_cur = y_chunk[kk]
                    contrib_b[:, k0 + kk] = y_prev_row - y_cur
                    y_prev_row = y_cur
                y_prev = y_prev_row

            contrib = 0.5 * (contrib_f + contrib_b)  # (total, q), float32
            # Pair-aware symmetric exclusion for mirror variables:
            # for paired players, choose the direction where partner is absent
            # from coalition context; unpaired players keep standard averaging.
            if torch.any(pair_idx_dev >= 0):
                pos_idx = torch.arange(q, device=device, dtype=torch.long).view(1, q)  # (1, q)
                inv_perms = torch.empty_like(perms)
                inv_perms.scatter_(
                    1,
                    perms,
                    pos_idx.expand(total, q),
                )  # (total, q): player -> position

                player_at_pos = perms  # (total, q)
                partner_at_pos = pair_idx_dev[player_at_pos]  # (total, q), -1 if unpaired
                has_partner = partner_at_pos >= 0

                partner_safe = torch.clamp(partner_at_pos, min=0)
                partner_pos = inv_perms.gather(1, partner_safe)  # (total, q)
                use_forward = partner_pos > pos_idx.expand(total, q)

                contrib_pair = torch.where(use_forward, contrib_f, contrib_b)
                contrib = torch.where(has_partner, contrib_pair, contrib)

            phi_batch = torch.zeros((total, q), dtype=torch.float32, device=device)
            phi_batch.scatter_add_(1, perms, contrib)
            phi_block += phi_batch.reshape(bs, b, q).sum(dim=1)

        shap_vals_group[i_start:i_end] = (phi_block / n_permutations).to(torch.float32)

    out = {}
    if bool(return_torch):
        for feat_idx, feat_name in enumerate(feature_names):
            out[str(feat_name)] = shap_vals_group[:, feat_idx]
        return out

    shap_np = shap_vals_group.detach().cpu().numpy()  # (n, p)
    for feat_idx, feat_name in enumerate(feature_names):
        out[str(feat_name)] = shap_np[:, feat_idx]
    return out
