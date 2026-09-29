"""
Quantum-Enhanced Hybrid Precoder: QUBO formulation of beam-to-RF-chain
assignment, solved with QAOA and compared against Greedy and Brute Force.

Pipeline:
  1. Generate a small clustered mmWave channel H (K users x Nt antennas)
  2. Build a static DFT beam codebook F (Nt x M)
  3. Compute the gain matrix G = |H F|          (K x M)
  4. Build the QUBO: reward (SINR proxy) - interference - fairness
     + penalty terms for the assignment constraints
  5. Solve the QUBO with QAOA (qiskit-optimization / qiskit-algorithms)
  6. Solve the same QUBO exactly with brute force (ground truth)
  7. Solve the assignment with Greedy (classical baseline)
  8. Decode every solution into (user, beam) pairs and report the
     real metrics: SINR, spectral efficiency, Jain's fairness index,
     interference power, and QUBO feasibility.

Keep K * M small (<= ~16-20) -- QAOA here uses a statevector simulator,
so one qubit is needed per (user, beam) binary variable.
"""

import itertools
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler

np.random.seed(7)

# ---------------------------------------------------------------------
# 1. Problem size (kept small on purpose so QAOA is simulable)
# ---------------------------------------------------------------------
Nt   = 8   # antennas
K    = 3   # users
M    = 3   # candidate beams in the codebook
N_RF = 2   # RF chains (beams that can be activated)

NOISE_VAR = 1.0
P_USER    = 1.0   # per-user transmit power


# ---------------------------------------------------------------------
# 2. Clustered mmWave channel  (simplified single-ray-per-cluster model)
# ---------------------------------------------------------------------
def steering_vector(Nt, phi):
    n = np.arange(Nt)
    return np.exp(1j * np.pi * n * np.sin(phi)) / np.sqrt(Nt)


def generate_channel(K, Nt, n_cl=3):
    H = np.zeros((K, Nt), dtype=complex)
    for k in range(K):
        h = np.zeros(Nt, dtype=complex)
        for _ in range(n_cl):
            phi = np.random.uniform(-np.pi / 2, np.pi / 2)
            alpha = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            h += alpha * steering_vector(Nt, phi)
        H[k, :] = np.sqrt(Nt / n_cl) * h
    return H


# ---------------------------------------------------------------------
# 3. Static DFT beam codebook  (built once, independent of CSI)
# ---------------------------------------------------------------------
def build_dft_codebook(Nt, M):
    n = np.arange(Nt).reshape(-1, 1)
    m = np.arange(M).reshape(1, -1)
    F = np.exp(1j * 2 * np.pi * n * m / M) / np.sqrt(Nt)
    return F


H = generate_channel(K, Nt)
F = build_dft_codebook(Nt, M)

# ---------------------------------------------------------------------
# 4. Gain matrix  G[k, m] = |h_k^H f_m|
# ---------------------------------------------------------------------
G = np.abs(H @ F)          # (K, M)
G2 = G ** 2                # squared gain, used as the SINR-proxy reward


def idx(k, m):
    """Flatten (user k, beam m) -> single QUBO variable index."""
    return k * M + m


N_VARS = K * M


# ---------------------------------------------------------------------
# 5. Build the QUBO
#    x_{k,m} = 1  if user k is assigned beam m
#
#    cost =  - sum_{k,m} G2[k,m] * x_{k,m}                 (maximize signal)
#            + lam_I * sum_{(k,m)!=(j,n), k!=j} leak * x_{k,m} x_{j,n}
#            + lam_F * fairness-variance-proxy (linearised, see note)
#            + lam_A * (row/col/count) penalty terms
# ---------------------------------------------------------------------
lam_I = 0.5     # interference weight
lam_F = 0.2     # fairness weight
lam_A = 6.0     # constraint penalty weight (must dominate the objective)

qp = QuadraticProgram("beam_rf_assignment")
for k in range(K):
    for m in range(M):
        qp.binary_var(name=f"x_{k}_{m}")

linear = {}
quadratic = {}


def add_quad(i, j, val):
    if val == 0:
        return
    key = (i, j) if i <= j else (j, i)
    quadratic[key] = quadratic.get(key, 0.0) + val


def add_lin(i, val):
    if val == 0:
        return
    linear[i] = linear.get(i, 0.0) + val


# --- (a) signal reward: encourage high-gain (user, beam) pairs ---------
for k in range(K):
    for m in range(M):
        add_lin(idx(k, m), -G2[k, m])

# --- (b) interference penalty: two DIFFERENT users on DIFFERENT beams
#         still leak into each other through cross-gain terms -----------
for k in range(K):
    for m in range(M):
        for j in range(K):
            if j == k:
                continue
            for n in range(M):
                if n == m:
                    continue
                leak = G2[k, n] + G2[j, m]      # cross-leakage proxy
                add_quad(idx(k, m), idx(j, n), lam_I * leak / 2)

# --- (c) fairness: penalise the squared deviation of each user's
#         served gain from the mean (variance proxy, linear in x since
#         x is binary and each user gets at most one beam) -------------
mean_gain = G2.mean()
for k in range(K):
    for m in range(M):
        add_lin(idx(k, m), lam_F * (G2[k, m] - mean_gain) ** 2 * 0.01)

# --- (d) constraint: each user gets AT MOST one beam -------------------
#     penalty = lam_A * (sum_m x_{k,m})(sum_m x_{k,m} - 1)  -> 0 at 0 or 1
for k in range(K):
    vars_k = [idx(k, m) for m in range(M)]
    for a in vars_k:
        add_lin(a, -lam_A)
    for a, b in itertools.combinations(vars_k, 2):
        add_quad(a, b, 2 * lam_A)
    for a in vars_k:
        add_quad(a, a, lam_A)   # x^2 = x for binary vars -> folds into linear

# --- (e) constraint: each beam goes to AT MOST one user ----------------
for m in range(M):
    vars_m = [idx(k, m) for k in range(K)]
    for a in vars_m:
        add_lin(a, -lam_A)
    for a, b in itertools.combinations(vars_m, 2):
        add_quad(a, b, 2 * lam_A)
    for a in vars_m:
        add_quad(a, a, lam_A)

# --- (f) constraint: exactly N_RF pairs are chosen ----------------------
all_vars = list(range(N_VARS))
for a in all_vars:
    add_lin(a, lam_A * (1 - 2 * N_RF))
for a, b in itertools.combinations(all_vars, 2):
    add_quad(a, b, 2 * lam_A)
for a in all_vars:
    add_quad(a, a, lam_A)

# fold any leftover diagonal quadratic terms into linear (x_i^2 = x_i)
for (i, j), val in list(quadratic.items()):
    if i == j:
        add_lin(i, val)
        del quadratic[(i, j)]

var_names = [f"x_{k}_{m}" for k in range(K) for m in range(M)]
linear_named = {var_names[i]: v for i, v in linear.items()}
quadratic_named = {(var_names[i], var_names[j]): v for (i, j), v in quadratic.items()}

qp.minimize(linear=linear_named, quadratic=quadratic_named)


# ---------------------------------------------------------------------
# helpers shared by every solver
# ---------------------------------------------------------------------
def decode(bitstring_or_array):
    """0/1 array of length K*M -> list of (user, beam) pairs, feasibility."""
    x = np.array(bitstring_or_array).reshape(K, M)
    pairs = []
    for k in range(K):
        chosen = np.where(x[k] == 1)[0]
        if len(chosen) == 1:
            pairs.append((k, int(chosen[0])))
    beams_used = [m for _, m in pairs]
    feasible = (
        all(x[k].sum() <= 1 for k in range(K))
        and len(beams_used) == len(set(beams_used))
        and len(pairs) == N_RF
    )
    return pairs, feasible


def evaluate(pairs):
    """Compute SINR, spectral efficiency, fairness, interference for a
    concrete beam assignment (list of (user, beam) pairs)."""
    if not pairs:
        return dict(sinr=[], se=0.0, jain=0.0, interference=0.0)

    served = {k: m for k, m in pairs}
    sinr_list = []
    interference_total = 0.0
    for k, mk in pairs:
        signal = P_USER * np.abs(H[k] @ F[:, mk]) ** 2
        interf = sum(
            P_USER * np.abs(H[k] @ F[:, mj]) ** 2
            for j, mj in pairs if j != k
        )
        interference_total += interf
        sinr_list.append(signal / (interf + NOISE_VAR))

    se = sum(np.log2(1 + s) for s in sinr_list)
    jain = (sum(sinr_list) ** 2) / (len(sinr_list) * sum(s ** 2 for s in sinr_list) + 1e-12)
    return dict(sinr=sinr_list, se=se, jain=jain, interference=interference_total)


def report(name, pairs, feasible):
    m = evaluate(pairs) if feasible else evaluate(pairs)
    avg_sinr_db = 10 * np.log10(np.mean(m["sinr"])) if m["sinr"] else float("-inf")
    print(f"\n[{name}]  feasible={feasible}  assignment={pairs}")
    print(f"  avg SINR      : {avg_sinr_db:7.3f} dB")
    print(f"  spectral eff. : {m['se']:7.4f} bits/s/Hz")
    print(f"  Jain fairness : {m['jain']:7.4f}")
    print(f"  interference  : {m['interference']:7.4f}")


# ---------------------------------------------------------------------
# 6. Brute force (ground truth for this small instance)
# ---------------------------------------------------------------------
best_cost, best_pairs = None, None
for combo in itertools.permutations(range(M), N_RF):
    for users in itertools.permutations(range(K), N_RF):
        pairs = list(zip(users, combo))
        x = np.zeros(N_VARS)
        for k, m in pairs:
            x[idx(k, m)] = 1
        cost = qp.objective.evaluate(x)
        if best_cost is None or cost < best_cost:
            best_cost, best_pairs = cost, pairs

report("Brute force (QUBO optimum)", best_pairs, True)

# ---------------------------------------------------------------------
# 7. Greedy baseline (Algorithm 1 style, direct on the gain matrix)
# ---------------------------------------------------------------------
remaining_users = set(range(K))
remaining_beams = set(range(M))
greedy_pairs = []
for _ in range(N_RF):
    best = max(
        ((k, m) for k in remaining_users for m in remaining_beams),
        key=lambda km: G[km[0], km[1]],
    )
    greedy_pairs.append(best)
    remaining_users.discard(best[0])
    remaining_beams.discard(best[1])

report("Greedy", greedy_pairs, True)

# ---------------------------------------------------------------------
# 8. QAOA
# ---------------------------------------------------------------------
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=60), reps=1)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
qaoa_result = qaoa_optimizer.solve(qp)

qaoa_pairs, qaoa_feasible = decode(qaoa_result.x)
report("QAOA", qaoa_pairs, qaoa_feasible)

print(f"\nQAOA raw objective value : {qaoa_result.fval:.4f}")
print(f"Brute force objective    : {best_cost:.4f}")
