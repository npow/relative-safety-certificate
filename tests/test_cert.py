import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "v2"))
from cert import tv, jsd_bits, tv_from_jsd_bits, tv_slack, certificate


def test_pinsker_jsd_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = int(rng.integers(2, 12))
        p, q = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 2)), rng.dirichlet(np.ones(k) * rng.uniform(0.1, 2))
        assert tv(p, q) <= tv_from_jsd_bits(jsd_bits(p, q)) + 1e-12


def test_old_cp_inequality_is_false():
    # disjoint point masses: JSD_bits = 1 but TV^2/(2 ln 2) = 0.72 < 1  => JSD <= TV^2/(2ln2) is false
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert jsd_bits(p, q) > tv(p, q) ** 2 / (2 * math.log(2))


def test_cost_difference_le_tv():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        k = 6; p, q = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k)); c = rng.uniform(0, 1, k)
        assert (p - q) @ c <= tv(p, q) * (c.max() - c.min()) + 1e-12


def test_finite_sample_slack_coverage():
    """Monte-Carlo: the slack should cover TV(d_A,d_B) at the nominal rate (it is conservative)."""
    rng = np.random.default_rng(2); K, T, n, delta = 6, 20, 40, 0.05
    # two Markov chains over K symbols -> dependent within episode
    def chain(P0):
        P = rng.dirichlet(np.ones(K), size=K) if P0 is None else P0
        return P
    PA, PB = chain(None), chain(None)
    def stat(P):  # stationary
        w, v = np.linalg.eig(P.T); s = np.real(v[:, np.argmax(np.real(w))]); return s / s.sum()
    def sim(P, n):
        C = np.zeros((n, K), int)
        for e in range(n):
            z = rng.integers(K)
            for t in range(T):
                C[e, z] += 1; z = rng.choice(K, p=P[z])
        return C
    # true occupancy via exact finite-horizon marginals from uniform start
    def occ(P):
        d = np.ones(K) / K; tot = np.zeros(K)
        for t in range(T): tot += d; d = d @ P
        return tot / T
    true = tv(occ(PA), occ(PB)); fails = 0
    for _ in range(100):
        cA, cB = sim(PA, n), sim(PB, n)
        c = certificate(cA, cB, np.zeros(n), T)
        fails += true > c["tv_hat"] + tv_slack(K, n, n, delta)
    assert fails <= 5


def test_ks_le_tv_and_dominates_monotone_costs():
    from cert import ks_plus
    rng = np.random.default_rng(3)
    for _ in range(2000):
        L = int(rng.integers(2, 8)); p, q = rng.dirichlet(np.ones(L)), rng.dirichlet(np.ones(L))
        g = np.sort(rng.uniform(0, 1, L)); ks = ks_plus(p, q)
        assert (p - q) @ g <= ks * (g[-1] - g[0]) + 1e-12      # Abel summation bound
        assert ks <= tv(p, q) + 1e-12
        tau = int(rng.integers(0, L - 1)); step = (np.arange(L) > tau).astype(float)
        assert (p - q) @ step <= ks + 1e-12


def test_ks_finite_sample_coverage():
    from cert import ks_plus, ks_slack, level_counts
    rng = np.random.default_rng(4); L1, T, n, delta = 5, 20, 40, 0.05
    P_A, P_B = rng.dirichlet(np.ones(L1), size=L1), rng.dirichlet(np.ones(L1), size=L1)
    def occ(P):
        d = np.ones(L1) / L1; tot = np.zeros(L1)
        for t in range(T): tot += d; d = d @ P
        return tot / T
    def sim(P, n):
        C = np.zeros((n, L1), int)
        for e in range(n):
            z = rng.integers(L1)
            for t in range(T): C[e, z] += 1; z = rng.choice(L1, p=P[z])
        return C
    true = ks_plus(occ(P_A), occ(P_B)); fails = 0
    for _ in range(100):
        cA, cB = sim(P_A, n), sim(P_B, n)
        dA, dB = cA.sum(0) / cA.sum(), cB.sum(0) / cB.sum()
        fails += true > ks_plus(dA, dB) + ks_slack(L1 - 1, n, n, delta)
    assert fails <= 5


def test_ks_onset_restriction():
    from cert import ks_plus
    rng = np.random.default_rng(5)
    for _ in range(2000):
        L = 8; p, q = rng.dirichlet(np.ones(L)), rng.dirichlet(np.ones(L)); on = [5, 6]
        inc = np.zeros(L - 1); inc[on] = rng.dirichlet(np.ones(2)) * rng.uniform(0, 1)   # cost increases only at onset thresholds
        g = np.concatenate([[0.0], np.cumsum(inc)])
        assert (p - q) @ g <= ks_plus(p, q, on) + 1e-12
        assert ks_plus(p, q, on) <= ks_plus(p, q) + 1e-12


def test_eb_radius_coverage():
    from cert import eb_radius
    rng = np.random.default_rng(6); fails = 0
    for _ in range(2000):
        x = rng.beta(0.3, 3.0, size=50)                      # skewed, small-variance variable in [0,1]
        fails += abs(x.mean() - 0.3 / 3.3) > eb_radius(x, 0.05)
    assert fails <= 0.05 * 2000 * 0.5                         # conservative: at most half the nominal failure rate


def test_ks_slack_eta():
    """Theorem 4 with slack: costs rising by eta outside the onset set add at most eta * KS+_all."""
    from cert import ks_plus
    rng = np.random.default_rng(7)
    for _ in range(3000):
        L = 8; p, q = rng.dirichlet(np.ones(L)), rng.dirichlet(np.ones(L)); on = [6]
        eta = rng.uniform(0, 0.2)
        inc = rng.dirichlet(np.ones(L - 1)) * eta
        inc[on] = 0.0; inc[on[0]] = rng.uniform(0, 1 - inc.sum())          # total rise <= 1
        g = np.concatenate([[0.0], np.cumsum(inc)])
        eta_true = inc.sum() - inc[on[0]]
        assert (p - q) @ g <= ks_plus(p, q, on) + eta_true * ks_plus(p, q) + 1e-12
