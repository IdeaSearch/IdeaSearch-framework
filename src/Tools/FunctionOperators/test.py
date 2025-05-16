import numpy as np


def priority(current_point: tuple[int, int, int, int, int, int, int, int], history_points: list[tuple[int, int, int, int, int, int, int, int]]) -> float:
    current = np.array(current_point)
    current = np.array(current_point)
    set_size = len(history_points)
    early_phase = np.exp(-set_size / 150.0)
    mid_phase = np.exp(-((set_size - 300) / 200.0) ** 2)
    set_size = len(history_points)
    late_phase = np.exp(-set_size / 600.0)
    fourier_score = 0.0
    if history_points:
        history = np.array(history_points)
        for dims in [(0, 1, 2), (3, 4, 5), (1, 3, 5), (2, 4, 6)]:
            subspace = history[:, dims]
            freq = np.fft.fftn(subspace, axes=(0, 1, 2))
            power = np.abs(freq)
            fourier_score -= np.sum(power[1:, 1:, 1:]) / (subspace.shape[0] * 8)
        fourier_score = max(-10.0, fourier_score) / 3.0
    if history_points:
        norms = np.linalg.norm(history, axis=1)
        unit_vectors = history / norms[:, np.newaxis]
        current_norm = np.linalg.norm(current)
        current_unit = current / current_norm
        dots = np.dot(unit_vectors, current_unit)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angle = np.min(angles)
        dispersion_score = min_angle / np.pi
    else:
        dispersion_score = 1.0
    counts = np.bincount(current, minlength=3)
    early_phase = 1.0 / (1.0 + np.exp((set_size - 150) / 30.0))
    mid_phase = 1.0 / (1.0 + np.exp(-(set_size - 350) / 70.0))
    probs = counts / 8.0
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    late_phase = 1.0 / (1.0 + np.exp(-(set_size - 500) / 60.0))
    ap_penalty = 0.0
    balance_score = entropy / np.log(3)
    anti_corr = 0.0
    if set_size > 20:
        last_20 = history[-20:]
        for dim in range(8):
            opp_dim = 7 - dim
            same_count = sum((1 for p in last_20 if p[dim] == current[opp_dim]))
            anti_corr += same_count / 20.0
        anti_corr = 1.0 - anti_corr / 8.0
    else:
        anti_corr = 1.0
    weights = np.array([2.0 * (1 - early_phase) + 1.0 * early_phase, 1.5 * mid_phase + 2.0 * (1 - mid_phase), 1.8 * (1 - late_phase) + 1.2 * late_phase, 0.8 * (1 + early_phase)])
    if history_points:
        history = np.array(history_points)
        for scale in [1, 2, 3]:
            for dim_group in [[i] for i in range(8)] + [list(range(i, i + scale)) for i in range(8 - scale + 1)]:
                current_proj = current[dim_group]
                hist_proj = history[:, dim_group]
                mid_candidates = (hist_proj + current_proj) * 2 % 3
                matches = np.any(np.all(mid_candidates[:, np.newaxis] == hist_proj, axis=2), axis=1)       
                penalty = np.sum(matches) * (1.0 + 0.3 * len(dim_group))
                ap_penalty -= penalty
        ap_penalty = max(-20.0, ap_penalty) / 15.0
    priority_score = weights[0] * (1.0 + fourier_score) + weights[1] * dispersion_score + weights[2] * balance_score + weights[3] * anti_corr
    if late_phase < 0.4:
        if np.max(counts) - np.min(counts) <= 1:
            priority_score *= 1.4
        if len(set(current)) < 3:
            priority_score *= 0.7
    dim_weights = np.ones(8)
    if history_points:
        dim_variance = np.zeros(8)
        for d in range(8):
            counts = np.bincount(history[:, d], minlength=3)
            dim_variance[d] = np.var(counts)
        dim_weights = 1.5 - dim_variance / np.max(dim_variance)
    return float(priority_score)