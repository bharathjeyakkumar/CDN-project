"""
XY-mixer QAOA beam-to-RF-chain assignment compared with Greedy, OMP and the
exact optimum, with RF-chain mapping, gain / SINR / SE / EE, fairness and
interference analysis.

BACKEND = "pennylane"    gate-level PennyLane circuit (qaoa_pennylane.py), same gates as below.
BACKEND = "circuit"      gate-level Qiskit circuit (qaoa_circuit.py):
                         W-state preparation, RZ/RZZ cost layer, RXX/RYY XY mixer.
                         Limited to ~20 qubits (K_SUB * (M_CAND + 1)).
BACKEND = "statevector"  exact NumPy simulation inside the one-hot subspace.
                         Larger subproblems, no gate decomposition.

The QUBO objective is the linearised sum of -ln(SINR_k) with victim-weighted
leakage (see build_qubo); POOL_MODE controls which users form the subproblem.

In all backends the cost Hamiltonian comes from the QUBO -> Ising mapping
of Eqs. (10)-(12); the mapping is verified numerically on random bit strings,
on every feasible-subspace state, and (circuit backend) against the Pauli
operator that is compiled into the circuit.

Figures written (Greedy / OMP / QAOA-XY only):
    qaoa_sinr_<backend>.png          average SINR (dB)
    qaoa_interference_<backend>.png  mean interference power per user (dB)
    qaoa_fairness_<backend>.png      Jain's fairness index
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize

from beam_dictionary import generate_beam_dictionary
from channel_model_distance_aware import generate_stress_scenario
from step1_greedy import greedy_assignment
from step1_omp import omp_assignment
from compute_sinr import compute_sinr_for_assignment


# ======================================================
# PARAMETERS
# ======================================================

BACKEND = "pennylane"          # "pennylane", "circuit" (Qiskit) or "statevector"
USE_SHOT_SAMPLING = True       # quantum backends: final measurement with shots on the backend simulator
QUANTUM = BACKEND in ("pennylane", "circuit")

if BACKEND == "pennylane":
    import qaoa_pennylane as qc_mod
elif BACKEND == "circuit":
    import qaoa_circuit as qc_mod
else:
    qc_mod = None

N_ANTENNAS = 16
N_USERS = 20
OVERSAMPLING = 1
N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0

# subproblem solved by QAOA / Greedy / OMP / Optimal
QUBIT_PRESET = 12      # quantum backends: 12 or 18 qubits  (qubits = K_SUB * (M_CAND + 1))
PRESETS = {                # (K_SUB users, M_CAND beams/user, N_RF_SUB RF chains, QAOA restarts)
    12: (4, 2, 2, 4),      # 4 x 3 = 12 qubits
    18: (6, 2, 3, 3),      # 6 x 3 = 18 qubits  (one-hot subspace has 3^6 = 729 states)
}

if QUANTUM:
    K_SUB, M_CAND, N_RF_SUB, N_RESTARTS = PRESETS[QUBIT_PRESET]
    MAXITER = 200
    MIXER = "trotter"                      # the circuits are Trotterised XY mixers
else:
    K_SUB, M_CAND, N_RF_SUB = 6, 3, 3      # 6 x (3 + 1) = 24 qubits-equivalent subspace
    MIXER = "exact"                        # "exact" = expm of XY Hamiltonian, or "trotter"
    N_RESTARTS, MAXITER = 6, 300

# QUBO objective
OBJECTIVE = "logsinr"   # "logsinr": first-order linearisation of sum_k -ln(SINR_k)  (new)
                        # "legacy" : -gain + MU * leakage in absolute gains          (old)
MU = 3.0                # weight of the interference term (1 = plain linearisation).
                        # 3 sits at the SINR peak of the sweep; recalibrate with RUN_MU_SWEEP
PENALTY_FACTOR = 2.0    # starting weight of constraint penalties, in units of the objective
                        # scale; raised automatically until the QUBO ground state is feasible

# subproblem user pool
POOL_MODE = "mixed"     # "crowded": users from the busiest hotspot only (old behaviour)
                        # "mixed"  : half from the busiest hotspot, the rest from other hotspots
                        # "diverse": round-robin over all hotspots
CAND_MIN_SEP = 1        # candidate beams per user must be >= this many grid steps apart

# QAOA
P_DEPTH = 2
SHOTS = 4096

# power / noise (same as run_greedy_review)
TX_POWER_PER_STREAM_W = 1.0
P_RF_CHAIN_W = 0.1
NOISE_POWER = 1e-4

RUN_MU_SWEEP = False   # True: first run the cheap exact-QUBO trade-off sweep over MU
MU_GRID = [1, 2, 3, 5, 8, 15]

N_SEEDS = 15
METHODS = ["Greedy", "OMP", "QAOA-XY", "Optimal"]   # all methods that are evaluated
PLOT_METHODS = ["Greedy", "OMP", "QAOA-XY"]          # methods shown in the graphs
DRAW_CIRCUIT = False                                 # True: also save a drawing of the circuit
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange",
          "QAOA-XY": "tab:green", "Optimal": "tab:red"}


# ======================================================
# 1. SCENARIO AND SUBPROBLEM
# ======================================================

def select_pool(G, hot, k_sub, mode):
    """
    Choose which users form the subproblem. The same pool is given to every
    method (Greedy, OMP, QAOA, Optimal), so the comparison stays fair.

    crowded : strongest users of the busiest hotspot first (maximum contention,
              but every pool user is angularly close, so any 2 served users interfere)
    mixed   : ceil(k/2) strongest users of the busiest hotspot, the remaining slots
              filled round-robin from the other hotspots (contention AND alternatives)
    diverse : round-robin over all hotspots, busiest first
    """
    strength = G.max(1)
    ids, counts = np.unique(hot, return_counts=True)
    ids = ids[np.argsort(-counts)]                       # busiest hotspot first
    per_hot = {h: [u for u in np.argsort(-strength) if hot[u] == h] for h in ids}

    if mode == "crowded":
        order = per_hot[ids[0]] + [u for h in ids[1:] for u in per_hot[h]]
        return np.array(order[:k_sub])

    pool = []
    if mode == "mixed":
        n_big = int(np.ceil(k_sub / 2))
        pool += per_hot[ids[0]][:n_big]
        rotation = list(ids[1:]) + [ids[0]]
    else:                                                # diverse
        rotation = list(ids)
    ptr = {h: (n_big if (mode == "mixed" and h == ids[0]) else 0) for h in ids}
    while len(pool) < k_sub:
        progressed = False
        for h in rotation:
            if len(pool) < k_sub and ptr[h] < len(per_hot[h]):
                pool.append(per_hot[h][ptr[h]])
                ptr[h] += 1
                progressed = True
        if not progressed:
            break
    return np.array(pool[:k_sub])


def candidate_beams(G, users, m, min_sep):
    """Best m beams per user by gain, at least min_sep grid steps apart (circular grid)."""
    n_beams = G.shape[1]
    cand = []
    for u in users:
        chosen = []
        for b in np.argsort(-G[u]):
            if all(min(abs(b - c), n_beams - abs(b - c)) >= min_sep for c in chosen):
                chosen.append(int(b))
            if len(chosen) == m:
                break
        cand.append(chosen)
    return np.array(cand)


def build_subproblem(seed):
    sc = generate_stress_scenario(
        seed=seed, n_tx=N_ANTENNAS, n_users=N_USERS,
        oversampling=OVERSAMPLING, n_hotspots=N_HOTSPOTS,
        hotspot_angular_spread_deg=HOTSPOT_SPREAD_DEG,
    )
    H_users, info = sc["H_users"], sc["info_users"]
    beam_dict, _ = generate_beam_dictionary(N_ANTENNAS, OVERSAMPLING, 0.5)
    G = np.array([np.abs(h.flatten() @ beam_dict) ** 2 for h in H_users])

    hot = np.array([i["hotspot_id"] for i in info])
    users = select_pool(G, hot, K_SUB, POOL_MODE)
    cand = candidate_beams(G, users, M_CAND, CAND_MIN_SEP)
    return H_users, beam_dict, G, users, cand


# ======================================================
# 2. QUBO AND ISING
# ======================================================

def build_qubo(G, users, cand, lam=None):
    """
    One qubit per (user, candidate beam) plus one idle qubit per user.

    OBJECTIVE = "logsinr" (default)
        Minimise  sum_k -ln SINR_k  over the served users, where
        SINR_k = g_k / (I_k + n),  g_k = G[k, b_k],  I_k = sum_{k' != k} G[k, b_k'].
        Linearising ln(I_k + n) around a reference leakage level Ibar_k gives
            sum_k [ -ln g_k + I_k / (Ibar_k + n) ] + const,
        which is quadratic in the binary variables:
          * diagonal : -ln(g / g_ref)             (strong beams are cheap)
          * pair     : G[k, b'] / (Ibar_k + n) + G[k', b] / (Ibar_k' + n)
        Because leakage is divided by the VICTIM's own reference level, weak or
        already-interfered users are protected first (fairness), and the log
        rewards balanced SINRs instead of one very strong link.
        Ibar_k is the mean leakage into user k over all other pool candidates.
    OBJECTIVE = "legacy": -G/G.max() on the diagonal, MU * (leakage) on pairs.

    The objective is shifted by a per-served-user constant (does not change which
    feasible state is optimal) and scaled to max |coefficient| = 1, so the constraint
    penalties are expressed in units of the objective.
    """
    lam = PENALTY_FACTOR if lam is None else lam
    K, m = cand.shape
    S = m + 1
    N = K * S
    q = lambda k, s: k * S + s

    D = np.zeros(N)                                  # diagonal (signal) part
    P = np.zeros((N, N))                             # pair weights, w placed once per (i, j)

    if OBJECTIVE == "legacy":
        Gn = G / G.max()
        for k in range(K):
            for s in range(m):
                D[q(k, s)] = -Gn[users[k], cand[k, s]]
        for k, k2 in itertools.combinations(range(K), 2):
            for s in range(m):
                for s2 in range(m):
                    b, b2 = cand[k, s], cand[k2, s2]
                    P[q(k, s), q(k2, s2)] = MU * (Gn[users[k], b2] + Gn[users[k2], b])
    else:
        noise = NOISE_POWER / TX_POWER_PER_STREAM_W
        g_ref = max(G[users[k], cand[k, s]] for k in range(K) for s in range(m))
        Ibar = np.zeros(K)
        for k in range(K):
            leak = [G[users[k], cand[k2, s2]]
                    for k2 in range(K) if k2 != k for s2 in range(m)]
            Ibar[k] = np.mean(leak) if leak else 0.0
        wv = 1.0 / (Ibar + noise)                    # victim weights
        for k in range(K):
            for s in range(m):
                D[q(k, s)] = -np.log(G[users[k], cand[k, s]] / g_ref)
        for k, k2 in itertools.combinations(range(K), 2):
            for s in range(m):
                for s2 in range(m):
                    b, b2 = cand[k, s], cand[k2, s2]
                    P[q(k, s), q(k2, s2)] = MU * (wv[k] * G[users[k], b2]
                                                  + wv[k2] * G[users[k2], b])

    served = [q(k, s) for k in range(K) for s in range(m)]
    # typical cost of serving one user alongside N_RF_SUB - 1 others -> remove it
    # (a constant on every state that serves exactly N_RF_SUB users)
    pair_mean = np.mean([P[i, j] for i, j in itertools.combinations(served, 2)
                         if i // S != j // S] or [0.0])
    D[served] -= (D[served].mean() + (N_RF_SUB - 1) * pair_mean)
    scale = max(np.abs(D[served]).max(), np.abs(P).max(), 1e-12)
    D, P = D / scale, P / scale

    Q = np.zeros((N, N))
    Q[np.diag_indices(N)] = D
    Q += P / 2 + P.T / 2

    for k, k2 in itertools.combinations(range(K), 2):        # same beam twice
        for s in range(m):
            for s2 in range(m):
                if cand[k, s] == cand[k2, s2]:
                    Q[q(k, s), q(k2, s2)] += lam / 2
                    Q[q(k2, s2), q(k, s)] += lam / 2

    for i in served:                                # lam * (sum x - R)^2, using x^2 = x
        Q[i, i] += lam * (1 - 2 * N_RF_SUB)
    for i, j in itertools.combinations(served, 2):
        Q[i, j] += lam
        Q[j, i] += lam
    offset = lam * N_RF_SUB ** 2
    return Q, offset, S


def feasible_mask(states, users, cand):
    return np.array([is_feasible(decode(s, users, cand)) for s in states])


def penalties_sufficient(E, states, users, cand):
    """The QUBO ground state must be a feasible assignment (not a penalised one)."""
    mask = feasible_mask(states, users, cand)
    return bool(mask.all() or E[mask].min() <= E[~mask].min() + 1e-12)


def ising_from_qubo(Q):
    """h, J, c from Eqs. (10)-(12), using x = (1 - s) / 2 (s = +1 <-> |0>)."""
    N = len(Q)
    h = np.array([-0.5 * (Q[i, i] + sum(Q[i, j] for j in range(N) if j != i))
                  for i in range(N)])
    J = {(i, j): 0.5 * Q[i, j] for i in range(N) for j in range(i + 1, N)}
    c = 0.5 * np.trace(Q) + 0.5 * np.triu(Q, 1).sum()
    return h, J, c


def check_ising_random(Q, h, J, c, trials=20):
    rng = np.random.default_rng(0)
    for _ in range(trials):
        x = rng.integers(0, 2, len(Q))
        s = 1 - 2 * x
        e_q = x @ Q @ x
        e_i = h @ s + sum(v * s[i] * s[j] for (i, j), v in J.items()) + c
        assert np.isclose(e_q, e_i), (e_q, e_i)


def ising_energies_subspace(h, J, c, offset, states, K, S):
    """Ising energy of every one-hot state, to compare with the QUBO energy."""
    X = np.zeros((len(states), K * S))
    for k in range(K):
        X[np.arange(len(states)), k * S + states[:, k]] = 1
    s = 1 - 2 * X
    Jm = np.zeros((K * S, K * S))
    for (i, j), v in J.items():
        Jm[i, j] = v
    return s @ h + np.einsum("ni,ij,nj->n", s, Jm, s) + c + offset


# ======================================================
# 3. FEASIBLE-SUBSPACE STATEVECTOR SIMULATION (NumPy)
# ======================================================

def subspace_energies(Q, offset, K, S):
    states = np.array(list(itertools.product(range(S), repeat=K)))
    X = np.zeros((len(states), K * S))
    for k in range(K):
        X[np.arange(len(states)), k * S + states[:, k]] = 1
    E = np.einsum("ni,ij,nj->n", X, Q, X) + offset
    return states, E


def xy_pairs(S):
    """Must match qaoa_circuit.xy_pairs: fixed pair order of the XY mixer."""
    return list(itertools.combinations(range(S), 2))


def group_mixer(S, beta, mixer):
    """
    XY mixer restricted to the one-hot sector of one group (S x S matrix).
    exact   : exp(-i beta A), A = complete-graph adjacency
    trotter : product over pairs of exp(-i beta (|a><b| + |b><a|)), the same
              ordered product the RXX/RYY gates implement in the circuit.
    """
    if mixer == "exact":
        A = np.ones((S, S)) - np.eye(S)
        return expm(-1j * beta * A)
    U = np.eye(S, dtype=complex)
    for a, b in xy_pairs(S):
        e = np.zeros((S, S))
        e[a, b] = e[b, a] = 1
        U = expm(-1j * beta * e) @ U
    return U


def apply_mixer(psi, U, K, S):
    psi = psi.reshape((S,) * K)
    for ax in range(K):
        psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)
    return psi.reshape(-1)


def qaoa_state(params, E, K, S, mixer):
    p = len(params) // 2
    gam, bet = params[:p], params[p:]
    psi = np.ones(S ** K, dtype=complex) / np.sqrt(S ** K)   # product of W states
    for l in range(p):
        psi = psi * np.exp(-1j * gam[l] * E)                  # U_C
        psi = apply_mixer(psi, group_mixer(S, bet[l], mixer), K, S)
    return psi


def run_qaoa_statevector(E, K, S, p, restarts, rng):
    def f(th):
        return float(np.sum(np.abs(qaoa_state(th, E, K, S, MIXER)) ** 2 * E))

    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rng.uniform(0, 0.5, p),
                              rng.uniform(0, np.pi / 2, p)])
        res = minimize(f, th0, method="COBYLA", options={"maxiter": MAXITER})
        if best is None or res.fun < best.fun:
            best = res
    prob = np.abs(qaoa_state(best.x, E, K, S, MIXER)) ** 2
    return best, prob / prob.sum()


# ======================================================
# 4. DECODING, FEASIBILITY, RF CHAIN ASSIGNMENT
# ======================================================

def decode(state, users, cand):
    m = cand.shape[1]
    return [(int(users[k]), int(cand[k, s])) for k, s in enumerate(state) if s < m]


def is_feasible(assign):
    return len(assign) == N_RF_SUB and len({b for _, b in assign}) == N_RF_SUB


def measure_and_select(prob, E, states, users, cand, rng):
    """Sample SHOTS measurements, return the lowest-energy feasible outcome."""
    shots = rng.choice(len(prob), size=SHOTS, p=prob)
    uniq = np.unique(shots)
    feas = [(E[i], decode(states[i], users, cand)) for i in uniq]
    feas = [c for c in feas if is_feasible(c[1])]
    if feas:
        return min(feas, key=lambda c: c[0])[1], True
    for i in np.argsort(-prob):                     # fallback: most probable feasible
        a = decode(states[i], users, cand)
        if is_feasible(a):
            return a, False
    return [], False


def assign_to_rf_chains(assign, G, beam_dict):
    """
    Map the chosen (user, beam) pairs onto physical RF chains.

    RF chain 0 gets the pair with the highest gain, chain 1 the next, etc.
    Chain i drives the analog beam f_{m_i}; the columns of F_RF are the
    steering vectors of the selected beams.

    Returns
    -------
    table : list of dicts (rf_chain, user, beam, gain), ordered by RF chain
    F_RF  : np.ndarray (N_antennas, n_chains), analog precoder
    """
    ordered = sorted(assign, key=lambda ub: -G[ub[0], ub[1]])
    table = [dict(rf_chain=i, user=int(u), beam=int(b), gain=float(G[u, b]))
             for i, (u, b) in enumerate(ordered)]
    F_RF = beam_dict[:, [t["beam"] for t in table]]
    return table, F_RF


def print_rf_table(name, table):
    print(f"  {name}")
    for t in table:
        print(f"    RF chain {t['rf_chain']} -> user {t['user']:2d}, "
              f"beam {t['beam']:2d}, gain = {t['gain']:.4f}")


# ======================================================
# 5. METRICS: GAIN, SINR, SE, EE, FAIRNESS, INTERFERENCE
# ======================================================

def jain_index(x):
    x = np.asarray(x, dtype=float)
    return float(x.sum() ** 2 / (len(x) * np.sum(x ** 2) + 1e-30))


def evaluate(assign, H_users, beam_dict, G):
    table, F_RF = assign_to_rf_chains(assign, G, beam_dict)
    ordered = [(t["user"], t["beam"]) for t in table]

    sinr, sinr_db = compute_sinr_for_assignment(
        ordered, H_users, beam_dict,
        tx_power=TX_POWER_PER_STREAM_W, noise_power=NOISE_POWER)
    sinr = np.array(sinr)
    se_user = np.log2(1 + sinr)
    total_se = float(se_user.sum())

    n = len(ordered)
    p_total = n * TX_POWER_PER_STREAM_W + n * P_RF_CHAIN_W
    ee = total_se / p_total

    # effective channel |h_k . f_i|^2: row = user, column = RF chain
    Hs = np.array([H_users[u].flatten() for u, _ in ordered])
    L = TX_POWER_PER_STREAM_W * np.abs(Hs @ F_RF) ** 2
    signal = np.diag(L).copy()
    interference = L.sum(1) - signal
    # Interference ratio I / (S + I): the share of the received power that is
    # interference. Linear, bounded in [0, 1), and exactly 0 for zero interference.
    leak_frac = interference / np.maximum(signal + interference, 1e-30)

    return dict(
        table=table, F_RF=F_RF, leakage=L,
        gain=float(sum(t["gain"] for t in table)),
        sinr_db_users=np.array(sinr_db), sinr_db=float(np.mean(sinr_db)),
        se_users=se_user, se=total_se, ee=ee,
        jain=jain_index(se_user), min_se=float(se_user.min()),
        interference=interference, leak_frac=leak_frac,
        mean_leak=float(np.mean(leak_frac)), max_leak=float(np.max(leak_frac)),
        mean_interf_db=float(10 * np.log10(np.mean(interference) + 1e-12)),
    )


# ======================================================
# 6. ONE SEED, ALL METHODS
# ======================================================

def run_seed(seed):
    rng = np.random.default_rng(seed)
    H_users, beam_dict, G, users, cand = build_subproblem(seed)
    K = len(users)

    lam = PENALTY_FACTOR                              # raise penalties until the QUBO
    for _ in range(10):                               # ground state is a feasible assignment
        Q, offset, S = build_qubo(G, users, cand, lam)
        states, E = subspace_energies(Q, offset, K, S)
        if penalties_sufficient(E, states, users, cand):
            break
        lam *= 1.5

    h, J, c = ising_from_qubo(Q)
    check_ising_random(Q, h, J, c)

    E_ising = ising_energies_subspace(h, J, c, offset, states, K, S)
    assert np.allclose(E, E_ising), "Ising and QUBO energies differ on the subspace"

    extra = {"lam": lam}

    # ------------------ QAOA ------------------
    if QUANTUM:
        res, prob_full, E_full, qc, gam, bet, stats = qc_mod.run_qaoa_circuit(
            Q, offset, h, J, c, K, S, P_DEPTH, N_RESTARTS, rng, maxiter=MAXITER)

        full_idx = qc_mod.subspace_to_full_index(states, S)
        prob = prob_full[full_idx]                       # one-hot states only
        extra["leakage"] = float(1.0 - prob.sum())       # probability outside the feasible subspace
        prob = prob / prob.sum()

        # cross-check the circuit against the NumPy Trotterised simulation
        psi = qaoa_state(res.x, E, K, S, "trotter")
        extra["circuit_vs_numpy"] = float(
            np.max(np.abs(np.abs(psi) ** 2 - prob_full[full_idx])))
        extra["stats"] = stats

        # measurement with shots on the backend simulator
        observed = (qc_mod.sample_indices(qc, gam, bet, res.x, SHOTS, seed)
                    if USE_SHOT_SAMPLING else None)
        if observed is not None:
            idx_to_state = {int(f): i for i, f in enumerate(full_idx)}
            feas = []
            for f in observed:
                if int(f) in idx_to_state:
                    i = idx_to_state[int(f)]
                    a = decode(states[i], users, cand)
                    if is_feasible(a):
                        feas.append((E[i], a))
            if feas:
                qaoa_assign, measured_ok = min(feas, key=lambda t: t[0])[1], True
            else:
                qaoa_assign, measured_ok = measure_and_select(
                    prob, E, states, users, cand, rng)
            extra["sampler"] = qc_mod.SAMPLER_NAME
        else:
            qaoa_assign, measured_ok = measure_and_select(
                prob, E, states, users, cand, rng)
            extra["sampler"] = "statevector sampling"
        extra["circuit_obj"] = (qc if seed == 0 else None)
        extra["best_params"] = res.x
    else:
        res, prob = run_qaoa_statevector(E, K, S, P_DEPTH, N_RESTARTS, rng)
        qaoa_assign, measured_ok = measure_and_select(prob, E, states, users, cand, rng)
        extra["leakage"] = 0.0

    # ------------------ exact optimum in the subspace ------------------
    feas_idx = [i for i in range(len(E))
                if is_feasible(decode(states[i], users, cand))]
    opt_i = min(feas_idx, key=lambda i: E[i])
    opt_assign = decode(states[opt_i], users, cand)

    # ------------------ Greedy / OMP on the SAME subproblem ------------------
    ubeams = np.unique(cand)
    g_a, _ = greedy_assignment(G[users][:, ubeams], N_RF_SUB)
    greedy_assign = [(int(users[k]), int(ubeams[m])) for k, m in g_a]
    o_a, _ = omp_assignment([H_users[u].flatten() for u in users],
                            beam_dict[:, ubeams], N_RF_SUB)
    omp_assign = [(int(users[k]), int(ubeams[m])) for k, m in o_a]

    assigns = {"Greedy": greedy_assign, "OMP": omp_assign,
               "QAOA-XY": qaoa_assign, "Optimal": opt_assign}
    results = {n: evaluate(a, H_users, beam_dict, G) for n, a in assigns.items()}

    extra.update(
        energy=float(res.fun), e_opt=float(E[opt_i]),
        # mean energy gap to the optimum over the FEASIBLE part of the output
        # distribution (0 = QAOA concentrates on the optimum)
        gap=float(prob[feas_idx] @ E[feas_idx] / prob[feas_idx].sum() - E[opt_i]),
        p_opt=float(prob[opt_i]), p_feas=float(prob[feas_idx].sum()),
        p_uniform=1.0 / len(E),
        measured_feasible=measured_ok,
        qaoa_matches_opt=set(qaoa_assign) == set(opt_assign),
    )
    return results, extra


# ======================================================
# 6b. MU SWEEP (interference vs SINR / fairness trade-off)
# ======================================================

def mu_sweep(mus, seeds):
    """
    For every interference weight MU, solve the QUBO exactly inside the feasible
    subspace (no QAOA, so it is fast) and report the mean metrics over the seeds,
    next to Greedy. Use it to recalibrate MU on your own channel data.
    """
    global MU
    keep, rows, greedy_rows = MU, [], []
    for mu in mus:
        MU = mu
        recs = []
        for seed in seeds:
            H_users, beam_dict, G, users, cand = build_subproblem(seed)
            K = len(users)
            lam = PENALTY_FACTOR
            for _ in range(10):
                Q, offset, S = build_qubo(G, users, cand, lam)
                states, E = subspace_energies(Q, offset, K, S)
                if penalties_sufficient(E, states, users, cand):
                    break
                lam *= 1.5
            mask = feasible_mask(states, users, cand)
            best = np.where(mask)[0][np.argmin(E[mask])]
            r = evaluate(decode(states[best], users, cand), H_users, beam_dict, G)
            ub = np.unique(cand)
            g_a, _ = greedy_assignment(G[users][:, ub], N_RF_SUB)
            gr = evaluate([(int(users[k]), int(ub[b])) for k, b in g_a],
                          H_users, beam_dict, G)
            recs.append((r, gr))
        rows.append({k: np.mean([a[k] for a, _ in recs])
                     for k in ("mean_leak", "sinr_db", "se", "jain", "min_se")})
        greedy_rows.append({k: np.mean([b[k] for _, b in recs])
                            for k in ("mean_leak", "sinr_db", "se", "jain", "min_se")})
    MU = keep

    print("\nMU sweep (exact QUBO optimum vs Greedy, mean over seeds)")
    print(f"{'MU':>5s}{'Interf':>10s}{'SINR dB':>9s}{'SE':>8s}{'Jain':>7s}{'MinSE':>8s}")
    for mu, r in zip(mus, rows):
        print(f"{mu:5g}{r['mean_leak']:10.4f}{r['sinr_db']:9.2f}"
              f"{r['se']:8.3f}{r['jain']:7.3f}{r['min_se']:8.3f}")
    g = greedy_rows[0]
    print(f"{'Greedy':>5s}{g['mean_leak']:10.4f}{g['sinr_db']:9.2f}"
          f"{g['se']:8.3f}{g['jain']:7.3f}{g['min_se']:8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("mean_leak", "sinr_db", "jain"),
                              ("Interference ratio I/(S+I)  (0 = none)",
                               "Average SINR (dB)", "Jain's fairness index")):
        ax.plot(mus, [r[key] for r in rows], "o-", color=COLORS["Optimal"],
                label="QUBO optimum")
        ax.axhline(g[key], color=COLORS["Greedy"], linestyle="--", label="Greedy")
        ax.set_xscale("log")
        ax.set_xlabel("interference weight MU")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"qaoa_mu_tradeoff_{POOL_MODE}.png", dpi=300)


# ======================================================
# 7. PLOTS
# ======================================================

def _bar(ax, all_res, key, title, ylabel):
    means = [np.mean([r[mth][key] for r in all_res]) for mth in PLOT_METHODS]
    stds = [np.std([r[mth][key] for r in all_res]) for mth in PLOT_METHODS]
    bars = ax.bar(PLOT_METHODS, means, yerr=stds, capsize=5,
                  color=[COLORS[mth] for mth in PLOT_METHODS])
    for bar, v, s in zip(bars, means, stds):
        y = v + s if v >= 0 else v - s               # just beyond the error bar
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, y),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_three(all_res):
    """Three separate graphs: SINR, interference (dB), fairness. Greedy / OMP / QAOA-XY only."""
    setup = (f"{len(all_res)} seeds, {K_SUB} users, {M_CAND} beams/user, "
             f"{N_RF_SUB} RF chains, {BACKEND}")

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    _bar(ax1, all_res, "sinr_db", f"Average SINR (dB), higher is better\n{setup}", "SINR (dB)")
    fig1.tight_layout()
    fig1.savefig(f"qaoa_sinr_{BACKEND}.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    _bar(ax2, all_res, "mean_interf_db",
         f"Mean interference power per user (dB), lower is better\n{setup}",
         "Interference (dB)")
    fig2.tight_layout()
    fig2.savefig(f"qaoa_interference_{BACKEND}.png", dpi=300)

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    _bar(ax3, all_res, "jain", f"Jain's fairness index, higher is better\n{setup}",
         "Jain's index")
    ax3.set_ylim(0, 1.1)
    fig3.tight_layout()
    fig3.savefig(f"qaoa_fairness_{BACKEND}.png", dpi=300)


def draw_circuit(qc, params=None):
    try:
        qc_mod.save_drawing(qc, f"qaoa_circuit_{BACKEND}.png", params)
    except Exception as e:                       # e.g. Qiskit drawing needs pylatexenc
        print("Circuit drawing skipped:", e)


# ======================================================
# 8. MAIN
# ======================================================

if __name__ == "__main__":
    if RUN_MU_SWEEP:
        mu_sweep(MU_GRID, range(N_SEEDS))

    all_res, all_extra = [], []
    print(f"Backend: {BACKEND} | subproblem: {K_SUB} users x {M_CAND} beams, "
          f"{N_RF_SUB} RF chains | p = {P_DEPTH} | mixer = {MIXER}")

    for seed in range(N_SEEDS):
        print("\n" + "=" * 60)
        print("SEED:", seed)
        print("=" * 60)
        results, extra = run_seed(seed)
        all_res.append(results)
        all_extra.append(extra)

        if seed == 0 and QUANTUM:
            st = extra["stats"]
            print(f"Circuit ({BACKEND}): {st['n_qubits']} qubits, compiled depth "
                  f"{st['depth']}, {st['two_qubit_gates']} CNOTs, "
                  f"{st['n_pauli_terms']} Pauli terms in H_C")
            if "device" in st:
                print(f"         device: {st['device']}, optimizer: {st['optimizer']}")
            print(f"         gate counts {st['ops']}")
            print(f"         Pauli operator vs QUBO diagonal error: "
                  f"{st['operator_check_err']:.2e}")
            if DRAW_CIRCUIT:
                draw_circuit(extra["circuit_obj"], extra["best_params"])

        print("RF chain assignment:")
        for name in METHODS:
            print_rf_table(name, results[name]["table"])
        print(f"  QAOA: penalty={extra['lam']:.2f}  <E>={extra['energy']:.4f}  E_opt={extra['e_opt']:.4f}  "
              f"P(optimal)={extra['p_opt']:.4f} (uniform {extra['p_uniform']:.4f})  "
              f"P(feasible)={extra['p_feas']:.3f}  "
              f"matches optimum: {extra['qaoa_matches_opt']}")
        if QUANTUM:
            print(f"        leakage out of one-hot subspace = {extra['leakage']:.2e}  "
                  f"| circuit vs NumPy max prob. diff = {extra['circuit_vs_numpy']:.2e}  "
                  f"| sampler: {extra['sampler']}")
        for name in METHODS:
            r = results[name]
            print(f"  {name:8s} gain={r['gain']:.4f}  SINR={r['sinr_db']:.2f} dB  "
                  f"SE={r['se']:.3f}  EE={r['ee']:.3f}  Jain={r['jain']:.3f}  "
                  f"interf={r['mean_interf_db']:.2f} dB")

    # ---------- summary ----------
    print("\n" + "=" * 78)
    print(f"SUMMARY (averaged over {N_SEEDS} seeds)")
    print("=" * 78)
    print(f"{'Method':9s}{'Gain':>9s}{'SINR dB':>10s}{'SE':>9s}{'EE':>9s}"
          f"{'Jain':>8s}{'MinSE':>9s}{'Interf dB':>11s}")
    for m in METHODS:
        g = lambda k: np.mean([r[m][k] for r in all_res])
        print(f"{m:9s}{g('gain'):9.4f}{g('sinr_db'):10.2f}{g('se'):9.3f}"
              f"{g('ee'):9.3f}{g('jain'):8.3f}{g('min_se'):9.3f}"
              f"{g('mean_interf_db'):11.2f}")

    print(f"\nQAOA mean energy gap to optimum (feasible): "
          f"{np.mean([e['gap'] for e in all_extra]):.3f}")
    print(f"QAOA mean P(optimal):          "
          f"{np.mean([e['p_opt'] for e in all_extra]):.4f}  "
          f"(uniform guess {all_extra[0]['p_uniform']:.4f})")
    print(f"QAOA mean P(feasible):         "
          f"{np.mean([e['p_feas'] for e in all_extra]):.3f}")
    print(f"QAOA found exact optimum in    "
          f"{sum(e['qaoa_matches_opt'] for e in all_extra)}/{N_SEEDS} seeds")
    if QUANTUM:
        print(f"Max leakage out of subspace:   "
              f"{max(e['leakage'] for e in all_extra):.2e}")
        print(f"Max circuit-vs-NumPy deviation: "
              f"{max(e['circuit_vs_numpy'] for e in all_extra):.2e}")

    plot_three(all_res)
    plt.show()