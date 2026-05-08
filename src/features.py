"""
Feature extraction for RUL prediction.

Every function is annotated with the mathematical definition it computes.
Vectorized where possible for speed over the full dataset.
"""
import numpy as np
from scipy import stats, signal as sp_signal

EPS = 1e-8


# ---------------------------------------------------------------------------
# 7.1 Basic statistics
# ---------------------------------------------------------------------------
def basic_stats(w):
    """
    Input: w of shape (W,) — values of one sensor in one window.
    Returns dict of: mean, std, min, max, range.

    Math:
        mean = (1/W) sum_i x_i
        std  = sqrt( (1/(W-1)) sum_i (x_i - mean)^2 )
    """
    return {
        "mean":  float(np.mean(w)),
        "std":   float(np.std(w, ddof=1)),
        "min":   float(np.min(w)),
        "max":   float(np.max(w)),
        "range": float(np.ptp(w)),
    }


# ---------------------------------------------------------------------------
# 7.2 Shape statistics
# ---------------------------------------------------------------------------
def shape_stats(w):
    """
    Skewness (3rd standardized moment) and excess kurtosis (4th - 3).

    Math:
        g_1 = (1/W) sum_i ((x_i - mean)/s)^3
        g_2 = (1/W) sum_i ((x_i - mean)/s)^4 - 3
    """
    return {
        "skew": float(stats.skew(w, bias=False)),
        "kurt": float(stats.kurtosis(w, bias=False)),  # excess by default
    }


# ---------------------------------------------------------------------------
# 7.3 Quantile features
# ---------------------------------------------------------------------------
def quantile_stats(w):
    """q10, q25, q75, q90, IQR = q75 - q25."""
    q10, q25, q75, q90 = np.quantile(w, [0.10, 0.25, 0.75, 0.90])
    return {"q10": float(q10), "q25": float(q25),
            "q75": float(q75), "q90": float(q90),
            "iqr": float(q75 - q25)}


# ---------------------------------------------------------------------------
# 7.4 Trend features
# ---------------------------------------------------------------------------
def trend_stats(w):
    """
    OLS slope, R^2, Theil-Sen slope.

    Math:
        slope_OLS = sum((t_i - t_bar)(x_i - x_bar)) / sum((t_i - t_bar)^2)
        R^2 = 1 - SS_res / SS_tot
        slope_TS = median over pairs (i,j) of (x_j - x_i)/(t_j - t_i)
    """
    W = len(w)
    t = np.arange(W, dtype=float)
    t_dev = t - t.mean()
    x_dev = w - w.mean()
    denom = np.sum(t_dev ** 2) + EPS
    slope = float(np.sum(t_dev * x_dev) / denom)
    intercept = float(w.mean() - slope * t.mean())
    pred = slope * t + intercept
    ss_tot = float(np.sum(x_dev ** 2) + EPS)
    ss_res = float(np.sum((w - pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)

    # Theil-Sen
    ts_slope = float(stats.theilslopes(w, t)[0])

    return {"slope": slope, "r2": r2, "theil_sen_slope": ts_slope}


# ---------------------------------------------------------------------------
# 7.5 Autocorrelation
# ---------------------------------------------------------------------------
def autocorr_stats(w, lags=(1, 5)):
    """
    Lag-k autocorrelation.

    Math:
        rho_k = sum_{i=1}^{W-k} (x_i - x_bar)(x_{i+k} - x_bar)
                / sum_{i=1}^{W} (x_i - x_bar)^2
    """
    out = {}
    w_dev = w - w.mean()
    denom = float(np.sum(w_dev ** 2) + EPS)
    for k in lags:
        if k >= len(w):
            out[f"ac_lag{k}"] = 0.0
            continue
        num = float(np.sum(w_dev[:-k] * w_dev[k:]))
        out[f"ac_lag{k}"] = num / denom
    return out


# ---------------------------------------------------------------------------
# 7.6 Approximate Entropy (ApEn)
# ---------------------------------------------------------------------------
def approximate_entropy(w, m: int = 2, r_factor: float = 0.2):
    """
    ApEn(m, r) = Phi^m(r) - Phi^{m+1}(r)

    where Phi^m(r) = (1/(W-m+1)) sum_i log C_i^m(r),
          C_i^m(r) = |{j : max_l |X_i[l] - X_j[l]| <= r}| / (W-m+1),
          X_i = (x_i, x_{i+1}, ..., x_{i+m-1}).
    """
    w = np.asarray(w, dtype=float)
    W = len(w)
    r = r_factor * np.std(w, ddof=1)
    if W < m + 2 or r < EPS:
        return 0.0

    def _phi(m):
        X = np.array([w[i:i + m] for i in range(W - m + 1)])
        # Chebyshev distance matrix via broadcasting
        d = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        C = np.sum(d <= r, axis=1) / (W - m + 1.0)
        return np.sum(np.log(C + EPS)) / (W - m + 1.0)

    return float(_phi(m) - _phi(m + 1))


# ---------------------------------------------------------------------------
# 7.7 Hurst exponent (R/S analysis)
# ---------------------------------------------------------------------------
def hurst_exponent(w):
    """
    Fit log(R(n)/S(n)) = H * log(n) + c over chunk sizes n.

    Returns H in [0, 1]. H=0.5 is random walk.
    """
    w = np.asarray(w, dtype=float)
    W = len(w)
    if W < 10:
        return 0.5
    ns = [n for n in [5, 10, 20, 40] if n <= W // 2]
    if len(ns) < 2:
        return 0.5
    rs = []
    for n in ns:
        n_chunks = W // n
        ratios = []
        for i in range(n_chunks):
            chunk = w[i * n:(i + 1) * n]
            mean = chunk.mean()
            dev = chunk - mean
            Z = np.cumsum(dev)
            R = Z.max() - Z.min()
            S = chunk.std(ddof=1) + EPS
            ratios.append(R / S)
        rs.append(np.mean(ratios))
    rs = np.array(rs)
    log_n = np.log(ns)
    log_rs = np.log(rs + EPS)
    H = float(np.polyfit(log_n, log_rs, 1)[0])
    return H


# ---------------------------------------------------------------------------
# 7.8 Dispersion features
# ---------------------------------------------------------------------------
def dispersion_stats(w):
    """
    CV  = std / |mean|
    DI  = var / |mean|      (dispersion index, variance-to-mean ratio)
    """
    mean = float(np.mean(w))
    std  = float(np.std(w, ddof=1))
    abs_mean = abs(mean) + EPS
    return {"cv": std / abs_mean,
            "dispersion_index": (std ** 2) / abs_mean}


# ---------------------------------------------------------------------------
# 7.9 CUSUM maximum
# ---------------------------------------------------------------------------
def cusum_max(w):
    """
    S_t = sum_{i=1}^{t} (x_i - mean).
    Returns max_t |S_t| — larger means a stronger monotone drift.
    """
    w = np.asarray(w, dtype=float)
    S = np.cumsum(w - w.mean())
    return float(np.max(np.abs(S)))


# ---------------------------------------------------------------------------
# 7.10 Spectral features (apply selectively)
# ---------------------------------------------------------------------------
def spectral_stats(w, fs: float = 1.0):
    """
    Power spectral density via periodogram.

    Math:
        P_k = |X_k|^2 where X_k is the DFT of w.
        Spectral centroid  C = sum(f_k * P_k) / sum(P_k)
        Spectral energy    E = sum(P_k)
        Spectral entropy   H = -sum(p_k log p_k), p_k = P_k / sum(P_k)
    """
    w = np.asarray(w, dtype=float)
    if len(w) < 4:
        return {"spec_centroid": 0.0, "spec_energy": 0.0, "spec_entropy": 0.0}
    freqs, psd = sp_signal.periodogram(w, fs=fs)
    total = np.sum(psd) + EPS
    centroid = float(np.sum(freqs * psd) / total)
    energy = float(total)
    p = psd / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    return {"spec_centroid": centroid,
            "spec_energy": energy,
            "spec_entropy": entropy}


# ---------------------------------------------------------------------------
# 8.1 Healthy-baseline comparison features (adapted from structural break project)
# ---------------------------------------------------------------------------
def baseline_comparison(w_now, w_base):
    """
    Compare a current 1-D window to a baseline 1-D window using:
        KS statistic, normalized mean shift, variance ratio, 1-Wasserstein.

    Math:
        D_KS       = sup_x |F_now(x) - F_base(x)|
        mean_shift = (mean_now - mean_base) / (std_base + eps)
        var_ratio  = var_now / (var_base + eps)
        W_1        = integral |F_now(x) - F_base(x)| dx
    """
    ks = float(stats.ks_2samp(w_base, w_now).statistic)
    mean_shift = float((w_now.mean() - w_base.mean()) / (w_base.std(ddof=1) + EPS))
    var_ratio = float(w_now.var(ddof=1) / (w_base.var(ddof=1) + EPS))
    wasser = float(stats.wasserstein_distance(w_base, w_now))
    return {"base_ks": ks, "base_mean_shift": mean_shift,
            "base_var_ratio": var_ratio, "base_wasser": wasser}


# ---------------------------------------------------------------------------
# 8.4 Tail variance ratio
# ---------------------------------------------------------------------------
def tail_variance_ratio(w, frac: float = 0.3):
    """
    Var(last frac of window) / Var(first frac of window).
    High value => window is becoming more volatile toward its end.
    """
    W = len(w)
    k = max(2, int(W * frac))
    head = np.var(w[:k], ddof=1) + EPS
    tail = np.var(w[-k:], ddof=1) + EPS
    return float(tail / head)


# ---------------------------------------------------------------------------
# 8.5 Additional features (see ANSWERS.md, Q6)
# ---------------------------------------------------------------------------
def sample_entropy(w, m: int = 2, r_factor: float = 0.2):
    """
    Sample Entropy (Richman & Moorman, 2000).

    SampEn(m, r) = -ln(A / B)
      A = # template-matching pairs of length m+1 within tolerance r
      B = # template-matching pairs of length m   within tolerance r

    Unlike ApEn, it excludes self-matches, giving more consistent results
    on short series.
    """
    w = np.asarray(w, dtype=float)
    W = len(w)
    r = r_factor * np.std(w, ddof=1)
    if W < m + 2 or r < EPS:
        return 0.0

    def _count_matches(m):
        X = np.array([w[i:i + m] for i in range(W - m + 1)])
        d = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        # Exclude self-matches (diagonal) and count only i < j
        np.fill_diagonal(d, np.inf)
        return float(np.sum(np.triu(d <= r, k=1)))

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if A == 0 or B == 0:
        return 0.0
    return float(-np.log(A / B))


def permutation_entropy(w, m: int = 3):
    """
    Permutation Entropy (Bandt & Pompe, 2002).

    H_P(m) = -sum_pi p(pi) log p(pi)

    pi ranges over all m! orderings of m consecutive values.
    Captures ordinal dynamics, ignores absolute magnitudes.
    """
    w = np.asarray(w, dtype=float)
    W = len(w)
    if W < m + 1:
        return 0.0
    # Each window of length m gets a pattern = ranks of its values
    patterns = {}
    for i in range(W - m + 1):
        chunk = w[i:i + m]
        ranks = tuple(np.argsort(np.argsort(chunk)))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    total = sum(patterns.values())
    probs = np.array([c / total for c in patterns.values()])
    return float(-np.sum(probs * np.log(probs + EPS)))


def extra_features(w):
    """
    Twelve extra features with direct interpretations.

    See ANSWERS.md, Q6 for the math behind each one.
    """
    w = np.asarray(w, dtype=float)
    W = len(w)
    mean = float(w.mean())
    std = float(w.std(ddof=1))
    abs_mean = abs(mean) + EPS

    # Energy, RMS
    energy = float(np.sum(w ** 2))
    rms = float(np.sqrt(energy / W))

    # Crest factor
    peak = float(np.max(np.abs(w)))
    crest = peak / (rms + EPS)

    # Zero crossing rate
    signs = np.sign(w)
    zcr = float(np.sum(signs[:-1] != signs[1:]) / max(1, W - 1))

    # MAD (mean absolute deviation)
    mad = float(np.mean(np.abs(w - mean)))

    # Median absolute deviation
    med = float(np.median(w))
    medad = float(np.median(np.abs(w - med)))

    # Signal-to-noise ratio
    snr = (mean ** 2) / (std ** 2 + EPS)

    # Peak-to-peak normalized
    p2p_norm = (float(w.max()) - float(w.min())) / (std + EPS)

    # Total variation (change rate)
    change_rate = float(np.mean(np.abs(np.diff(w))))

    # Last minus first
    delta_end = float(w[-1] - w[0])

    # Entropies (added as separate features since they are more expensive)
    sampen = sample_entropy(w)
    permen = permutation_entropy(w)

    return {
        "energy": energy,
        "rms": rms,
        "crest_factor": crest,
        "zcr": zcr,
        "mad": mad,
        "medad": medad,
        "snr": snr,
        "p2p_norm": p2p_norm,
        "change_rate": change_rate,
        "delta_end": delta_end,
        "sampen": sampen,
        "permen": permen,
    }


# ---------------------------------------------------------------------------
# Per-window, per-sensor feature extraction
# ---------------------------------------------------------------------------
def extract_sensor_features(w, prefix: str, include_spectral: bool = False,
                            include_extra: bool = True):
    """
    Input: w of shape (W,) — one sensor over one window.
    Returns dict of {prefix}_{featname}: value.

    include_extra: if True, also compute the 12 extra features (Q6 in ANSWERS.md).
    """
    feats = {}
    feats.update({f"{prefix}_{k}": v for k, v in basic_stats(w).items()})
    feats.update({f"{prefix}_{k}": v for k, v in shape_stats(w).items()})
    feats.update({f"{prefix}_{k}": v for k, v in quantile_stats(w).items()})
    feats.update({f"{prefix}_{k}": v for k, v in trend_stats(w).items()})
    feats.update({f"{prefix}_{k}": v for k, v in autocorr_stats(w).items()})
    feats[f"{prefix}_apen"]     = approximate_entropy(w)
    feats[f"{prefix}_hurst"]    = hurst_exponent(w)
    feats.update({f"{prefix}_{k}": v for k, v in dispersion_stats(w).items()})
    feats[f"{prefix}_cusum_max"] = cusum_max(w)
    feats[f"{prefix}_tail_var_ratio"] = tail_variance_ratio(w)
    if include_extra:
        feats.update({f"{prefix}_{k}": v for k, v in extra_features(w).items()})
    if include_spectral:
        feats.update({f"{prefix}_{k}": v for k, v in spectral_stats(w).items()})
    return feats


def extract_window_features(window, sensor_names,
                            baseline_window=None,
                            spectral_sensors=None):
    """
    window:           ndarray (W, n_sensors)
    sensor_names:     list of str, length n_sensors
    baseline_window:  optional ndarray (W_base, n_sensors) for baseline comparison
    spectral_sensors: list of sensor names to compute spectral features for
    """
    spectral_sensors = spectral_sensors or []
    feats = {}
    for i, name in enumerate(sensor_names):
        col = window[:, i]
        feats.update(
            extract_sensor_features(col, prefix=name,
                                    include_spectral=(name in spectral_sensors))
        )
        if baseline_window is not None:
            base_col = baseline_window[:, i]
            bc = baseline_comparison(col, base_col)
            feats.update({f"{name}_{k}": v for k, v in bc.items()})
    return feats


def _extract_one(idx, windows, sensor_names, meta, baselines, spectral_sensors):
    """Worker function for parallel extraction."""
    base = None
    if baselines is not None and meta is not None:
        uid = int(meta[idx, 0])
        base = baselines.get(uid, None)
    return extract_window_features(windows[idx], sensor_names,
                                   baseline_window=base,
                                   spectral_sensors=spectral_sensors)


def batch_extract(windows, sensor_names, meta=None, baselines=None,
                  spectral_sensors=None, n_jobs: int = 1, verbose: int = 0,
                  add_condition: bool = False):
    """
    Extract features for every window.

    Parameters
    ----------
    windows : ndarray (N, W, n_sensors)
    meta    : ndarray (N, 2) or (N, 3)
              columns [unit_id, cycle] or [unit_id, cycle, condition]
              (used to look up baselines and optionally add condition features).
    baselines : dict unit_id -> baseline window (if None, no baseline features)
    n_jobs  : number of parallel CPU workers.  1 = sequential,  -1 = all cores.
    verbose : joblib verbosity level (0 = silent, 10 = progress messages).
    add_condition : if True, append a one-hot 'condition' feature. Requires meta
                    to have a third column with the condition id. Use for
                    FD002/FD004.
    """
    import pandas as pd

    if n_jobs == 1:
        rows = []
        for idx in range(len(windows)):
            rows.append(_extract_one(idx, windows, sensor_names, meta,
                                     baselines, spectral_sensors))
        df = pd.DataFrame(rows)
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
            delayed(_extract_one)(i, windows, sensor_names, meta,
                                  baselines, spectral_sensors)
            for i in range(len(windows))
        )
        df = pd.DataFrame(rows)

    # Optionally append a one-hot condition indicator so the model can
    # distinguish regimes. This is meaningful only when the data has multiple
    # operating conditions and meta carries a condition column.
    if add_condition and meta is not None and meta.shape[1] >= 3:
        cond = meta[:, 2].astype(int)
        n_cond = int(cond.max()) + 1
        for c in range(n_cond):
            df[f"cond_{c}"] = (cond == c).astype(np.int8)

    return df
