"""
Shared analysis helpers: autocorrelation, steady-state metrics, FIM null
direction verification, and the plotting style used by all scenarios.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def acf(x, max_lag):
    """Sample autocorrelation of a 1-D series for lags 0..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.sum(x**2)
    if var <= 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = np.sum(x[: n - k] * x[k:]) / var
    return out


def steady_rms(x, frac=0.25):
    """RMS of the last `frac` of a series."""
    n = len(x)
    return float(np.sqrt(np.mean(x[int(n * (1 - frac)):] ** 2)))


def median_quantile(arr, axis=0):
    """median and [q25, q75] along axis 0."""
    med = np.median(arr, axis=axis)
    lo = np.percentile(arr, 25, axis=axis)
    hi = np.percentile(arr, 75, axis=axis)
    return med, lo, hi


def check_null_direction(features, K, T_true, sigma, v_line, p0):
    """Verify that the FIM null direction matches the analytic prediction.

    For a single point tracing a line along v_line through p0, the unobservable
    family is rotation about v_line (with a coupled translation), i.e. the null
    vector u ~ (v, -v x (p0 - c)). Returns (cosine_similarity, u_numeric, u_an).
    """
    H = np.concatenate(
        [measurement_jacobian(K, T_true[0], T_true[1], f) for f in features], axis=0
    )
    F = H.T @ H / sigma**2
    w, V = np.linalg.eigh(F)
    u_num = V[:, 0]
    c = T_true[1]
    u_an = np.concatenate([v_line, -np.cross(v_line, p0 - c)])
    u_an /= np.linalg.norm(u_an)
    return abs(u_num @ u_an), u_num, u_an, w[0] / w[-1]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def savefig(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


STYLE = {
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def set_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v
