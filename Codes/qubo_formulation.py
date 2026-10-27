"""Exact weighted beam-assignment QUBO with symmetric x.T @ Q @ x storage.

Each decision x[k,s] selects the *s*-th candidate beam for user k.  If user
candidate sets differ, ``candidate_beams[k,s]`` contains the global codebook
beam ID and ``cross_gains[k,b]`` contains |H_k F_b|**2.  No scaling, gain
normalisation, logarithms, or objective shifts are applied.

The RF-cardinality penalty is included, with its constant lambda_R*R**2
OMITTED, exactly as in the supplied reference formulation. Consequently a
feasible weight-R solution has energy physical_objective - lambda_R*R**2.
"""

from itertools import combinations
import numpy as np

LAMBDA_G = 1.0
LAMBDA_I = 0.5
LAMBDA_F = 0.2
LAMBDA_U = 6.0
LAMBDA_B = 6.0
LAMBDA_R = 6.0


def fairness_matrix(candidate_gains):
    """Unweighted gain-deviation fairness proxy, computed over candidate gains."""
    gains = np.asarray(candidate_gains, dtype=float)
    if gains.ndim != 2 or not np.all(np.isfinite(gains)):
        raise ValueError("candidate_gains must be a finite 2D power-gain matrix")
    return 0.01 * (gains - np.mean(gains)) ** 2


def build_qubo_matrix(
    candidate_gains, n_rf, *, candidate_beams=None, cross_gains=None,
    lam_G=LAMBDA_G, lam_I=LAMBDA_I, lam_F=LAMBDA_F,
    lam_U=LAMBDA_U, lam_B=LAMBDA_B, lam_R=LAMBDA_R,
):
    """Return (symmetric_Q, 0.0), matching the original QUBO coefficients.

    ``candidate_gains``: shape (K, M), |H_k F_{candidate[k,s]}|**2.
    ``candidate_beams``: shape (K, M), global beam ID per variable; defaults
        to common IDs 0..M-1, matching the standalone reference example.
    ``cross_gains``: shape (K, n_global_beams), needed when candidate beam IDs
        differ between users to calculate actual cross-user leakage.
    """
    g = np.asarray(candidate_gains, dtype=float)
    if g.ndim != 2 or g.size == 0 or np.any(~np.isfinite(g)) or np.any(g < 0):
        raise ValueError("candidate_gains must be a nonnegative, finite K x M matrix")
    K, M = g.shape
    if not 0 <= n_rf <= K:
        raise ValueError("n_rf must be between zero and the number of users")
    bids = (np.tile(np.arange(M), (K, 1)) if candidate_beams is None
            else np.asarray(candidate_beams, dtype=int))
    if bids.shape != g.shape or np.any(bids < 0):
        raise ValueError("candidate_beams must contain nonnegative IDs with shape (K,M)")
    full = g if cross_gains is None else np.asarray(cross_gains, dtype=float)
    if (full.ndim != 2 or full.shape[0] != K or np.max(bids) >= full.shape[1]
            or np.any(~np.isfinite(full)) or np.any(full < 0)):
        raise ValueError("cross_gains must provide nonnegative powers for every candidate beam ID")
    # Check that the two views of signal power are consistent.
    if not np.allclose(np.take_along_axis(full, bids, axis=1), g, rtol=1e-8, atol=1e-10):
        raise ValueError("candidate_gains do not match cross_gains at candidate_beams")
    if any(len(set(row)) != M for row in bids):
        raise ValueError("Repeated candidate beam ID within one user")

    N = K * M
    Q = np.zeros((N, N), dtype=float)

    def pair(i, j, coeff):
        if i == j:
            Q[i, i] += coeff
        else:
            Q[i, j] += coeff / 2.0
            Q[j, i] += coeff / 2.0

    # (4) Gain and (6) fairness: linear terms.
    Q[np.diag_indices(N)] += (-lam_G * g + lam_F * fairness_matrix(g)).reshape(-1)

    # (5) All different-user pairs, regardless of the local/global beam order.
    for k, j in combinations(range(K), 2):
        for s in range(M):
            for t in range(M):
                leak = full[k, bids[j, t]] + full[j, bids[k, s]]
                pair(k * M + s, j * M + t, lam_I * leak)
                if bids[k, s] == bids[j, t]:
                    pair(k * M + s, j * M + t, lam_B)

    # (7) At most one candidate per user.
    for k in range(K):
        for s, t in combinations(range(M), 2):
            pair(k * M + s, k * M + t, lam_U)

    # (9) lambda_R * [(sum x)^2 - 2 R sum x], with constant omitted.
    Q[np.diag_indices(N)] += lam_R * (1 - 2 * n_rf)
    for i, j in combinations(range(N), 2):
        pair(i, j, 2 * lam_R)
    return Q, 0.0


def physical_components(selected_local, candidate_gains, candidate_beams=None,
                        cross_gains=None):
    """Return unweighted (gain, interference, fairness) for selected (k,s) pairs."""
    g = np.asarray(candidate_gains, dtype=float)
    K, M = g.shape
    bids = np.tile(np.arange(M), (K, 1)) if candidate_beams is None else np.asarray(candidate_beams)
    full = g if cross_gains is None else np.asarray(cross_gains, dtype=float)
    selected = [(int(k), int(s)) for k, s in selected_local]
    gain = sum(float(g[k, s]) for k, s in selected)
    fairness = sum(float(fairness_matrix(g)[k, s]) for k, s in selected)
    interference = sum(float(full[k, bids[j, t]] + full[j, bids[k, s]])
                       for (k, s), (j, t) in combinations(selected, 2))
    return dict(gain_linear=gain, interference_linear=interference, fairness=fairness)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    K, Nt, M, n_rf = 3, 8, 3, 2
    H = (rng.standard_normal((K, Nt)) + 1j*rng.standard_normal((K, Nt))) / np.sqrt(2)
    angles = np.linspace(-np.pi/3, np.pi/3, M)
    F = np.exp(1j*np.pi*np.outer(np.arange(Nt), np.sin(angles))) / np.sqrt(Nt)
    G2 = abs(H @ F)**2
    Q, _ = build_qubo_matrix(G2, n_rf)
    print("Power gain matrix G2:\n", np.round(G2, 4))
    print("Unweighted fairness matrix:\n", np.round(fairness_matrix(G2), 6))
    print("Exact symmetric QUBO matrix:\n", np.round(Q, 4))
    print("QUBO objective: minimize x.T @ Q @ x (RF constant omitted)")
