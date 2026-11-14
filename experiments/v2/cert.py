"""Occupancy-TV certificate (v2). All functions are pure numpy and unit-tested in tests/test_cert.py.

Setting: fixed horizon T; each episode yields a sequence z_1..z_T over a finite alphabet Z
(|Z| = K, an absorbing symbol pads terminated episodes). Per-episode occupancy x_i(z) = (1/T) #{t: z_t=z}.
The occupancy measure of a policy is d(z) = E[x_i(z)].
"""
from __future__ import annotations
import math
import numpy as np


def occupancy(counts: np.ndarray) -> np.ndarray:
    """counts: (n_episodes, K) integer visit counts with row sums T. Returns pooled occupancy d_hat."""
    counts = np.asarray(counts, dtype=float)
    return counts.sum(0) / counts.sum()


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def jsd_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv_from_jsd_bits(j: float) -> float:
    """TV <= sqrt(2 ln2 * JSD_bits)  (Pinsker applied to the two KL terms against the mixture)."""
    return math.sqrt(2.0 * math.log(2.0) * max(j, 0.0))


def tv_slack(K: int, nA: int, nB: int, delta: float) -> float:
    """Finite-sample slack s such that, w.p. >= 1-delta,  TV(d_A,d_B) <= TV(dhat_A,dhat_B) + s.
    s = 0.5*sqrt(K/nA) + 0.5*sqrt(K/nB) + sqrt(0.5*(1/nA+1/nB)*ln(1/delta))
    (episodes are the independent units; McDiarmid with bounded difference 1/n per episode)."""
    return 0.5 * math.sqrt(K / nA) + 0.5 * math.sqrt(K / nB) + math.sqrt(0.5 * (1 / nA + 1 / nB) * math.log(1 / delta))


def certificate(countsA, countsB, costB, T: int, c_range: float = 1.0, delta: float = 0.025):
    """Returns dict with plug-in and finite-sample upper bounds on J_A = E_A[sum_t c_t], given
    cost-agnostic worst case over cost functions with range <= c_range on Z.
    costB: per-episode total costs of the baseline (shape (nB,)). Costs of A are NOT used."""
    countsA = np.asarray(countsA); countsB = np.asarray(countsB)
    K = countsA.shape[1]; nA, nB = len(countsA), len(countsB)
    dA, dB = occupancy(countsA), occupancy(countsB)
    tv_hat = tv(dA, dB)
    jb = float(np.mean(costB))
    cost_hi = T * c_range  # per-episode cost range [0, T*c_range]
    jb_hi = jb + cost_hi * math.sqrt(math.log(1 / delta) / (2 * nB))
    s = tv_slack(K, nA, nB, delta)
    return dict(
        tv_hat=tv_hat, jsd_bits=jsd_bits(dA, dB), tv_from_jsd=tv_from_jsd_bits(jsd_bits(dA, dB)),
        jb_hat=jb, plugin_bound=jb + T * c_range * tv_hat,
        plugin_bound_via_jsd=jb + T * c_range * tv_from_jsd_bits(jsd_bits(dA, dB)),
        finite_sample_bound=jb_hi + T * c_range * (tv_hat + s), slack=s, K=K, nA=nA, nB=nB,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Monotone-cost certificate: one-sided Kolmogorov-Smirnov distance on an ordinal risk level (no sqrt(K) term)
# ---------------------------------------------------------------------------------------------------------------------
def level_counts(counts: np.ndarray, level_of_symbol: np.ndarray) -> np.ndarray:
    """Aggregate (n_episodes, K) symbol counts into (n_episodes, L+1) counts per ordinal level."""
    L = int(level_of_symbol.max()) + 1
    out = np.zeros((counts.shape[0], L), dtype=float)
    for z, l in enumerate(level_of_symbol):
        out[:, l] += counts[:, z]
    return out


def ks_plus(dA_level: np.ndarray, dB_level: np.ndarray, onset=None) -> float:
    """max over tau in `onset` (default: all tau = 0..L-1) of [ d_A(level > tau) - d_B(level > tau) ]_+ ."""
    tailA = np.cumsum(dA_level[::-1])[::-1][1:]
    tailB = np.cumsum(dB_level[::-1])[::-1][1:]
    diff = tailA - tailB
    if onset is not None: diff = diff[list(onset)]
    return float(max(0.0, diff.max())) if len(diff) else 0.0


def ks_slack(L: int, nA: int, nB: int, delta: float) -> float:
    """Slack s such that, w.p. >= 1-delta,  KS+(d_A,d_B) <= KS+(dhat_A,dhat_B) + s, with L thresholds:
    s = sqrt(ln(2L/delta)/(2 nA)) + sqrt(ln(2L/delta)/(2 nB))   (Hoeffding on per-episode tail fractions + union bound)."""
    m = math.log(2 * L / delta)
    return math.sqrt(m / (2 * nA)) + math.sqrt(m / (2 * nB))


def ks_certificate(countsA, countsB, costB, T: int, level_of_symbol, delta: float = 0.025, onset=None, eta: float = 0.0):
    """Upper bound on J_A when the per-symbol cost is a nondecreasing function of the ordinal level (range <= 1) whose total rise
    outside the onset set is at most `eta`  (Theorem 4 with slack: J_A - J_B <= T (KS+_onset + eta KS+_all)).
    Valid w.p. >= 1-2*delta. With eta>0 the union bound runs over all L thresholds."""
    cA = level_counts(np.asarray(countsA), level_of_symbol); cB = level_counts(np.asarray(countsB), level_of_symbol)
    dA, dB = cA.sum(0) / cA.sum(), cB.sum(0) / cB.sum()
    Lall = len(dA) - 1
    m = Lall if (onset is None or eta > 0) else len(list(onset))
    nA, nB = len(cA), len(cB)
    ks = ks_plus(dA, dB, onset); ks_all = ks_plus(dA, dB); s = ks_slack(max(m, 1), nA, nB, delta)
    jb = float(np.mean(costB)); jb_hi = jb + T * math.sqrt(math.log(1 / delta) / (2 * nB))
    return dict(ks_hat=ks, ks_slack=s, ks_plugin=jb + T * (ks + eta * ks_all), ks_finite=jb_hi + T * (ks + s + eta * min(1.0, ks_all + s)))


def eb_radius(x: np.ndarray, delta: float) -> float:
    """Empirical-Bernstein radius (Maurer & Pontil 2009, Thm 4) for i.i.d. x in [0,1]: with prob >= 1-delta,
    |E x - mean(x)| one-sided <= sqrt(2 V ln(2/delta)/n) + 7 ln(2/delta) / (3 (n-1)),  V = unbiased sample variance."""
    n = len(x); V = float(np.var(x, ddof=1)); l = math.log(2 / delta)
    return math.sqrt(2 * V * l / n) + 7 * l / (3 * (n - 1))


def ks_certificate_eb(countsA, countsB, costB, T: int, level_of_symbol, delta: float = 0.025, onset=None, eta: float = 0.0):
    """Variance-adaptive version of ks_certificate: empirical-Bernstein radii for the baseline cost and for every tail fraction."""
    cA = level_counts(np.asarray(countsA), level_of_symbol) / T; cB = level_counts(np.asarray(countsB), level_of_symbol) / T
    L = cA.shape[1] - 1; allt = list(range(L)); taus = allt if onset is None else list(onset)
    m = L if (onset is None or eta > 0) else max(len(taus), 1)
    tailA = np.cumsum(cA[:, ::-1], axis=1)[:, ::-1][:, 1:]; tailB = np.cumsum(cB[:, ::-1], axis=1)[:, ::-1][:, 1:]
    dl = delta / (2 * m)                                      # failure budget per (policy, threshold); total delta
    gap = {}
    for tau in (allt if eta > 0 else taus):
        gap[tau] = float(tailA[:, tau].mean() + eb_radius(tailA[:, tau], dl)) - float(tailB[:, tau].mean() - eb_radius(tailB[:, tau], dl))
    best = max([0.0] + [gap[tau] for tau in taus]); best_all = max([0.0] + list(gap.values()))
    xb = np.asarray(costB, dtype=float) / T
    jb_hi = T * float(xb.mean() + eb_radius(xb, delta))
    return dict(ks_eb_bound=jb_hi + T * (best + eta * min(1.0, best_all)))
