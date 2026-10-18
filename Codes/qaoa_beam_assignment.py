"""
XY-mixer QAOA beam-to-RF-chain assignment compared with Greedy, OMP and the
exact optimum, with RF-chain mapping, gain / SINR / SE / EE, fairness and
interference analysis.

BACKEND = "circuit"      gate-level Qiskit circuit (qaoa_circuit.py):
                         W-state preparation, RZ/RZZ cost layer, RXX/RYY XY mixer.
                         Limited to ~20 qubits (K_SUB * (M_CAND + 1)).
BACKEND = "statevector"  exact NumPy simulation inside the one-hot subspace.
                         Larger subproblems, no gate decomposition.

In both backends the cost Hamiltonian comes from the QUBO -> Ising mapping
of Eqs. (10)-(12); the mapping is verified numerically on random bit strings,
on every feasible-subspace state, and (circuit backend) against the Pauli
operator that is compiled into the circuit.

Figures written:
    qaoa_results_<backend>.png        gain / SINR / SE / EE
    qaoa_fairness_<backend>.png       per-user SE histogram, Jain index, worst user
    qaoa_interference_<backend>.png   interference, SIR histogram, leakage map
    qaoa_circuit_<backend>.png        drawing of the parametrised circuit (circuit backend)
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

BACKEND = "circuit"            # "circuit" or "statevector"
USE_AER_SAMPLING = True        # circuit backend: sample final shots with Qiskit Aer if installed

N_ANTENNAS = 16
N_USERS = 20
OVERSAMPLING = 1
N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0

# subproblem solved by QAOA / Greedy / OMP / Optimal
if BACKEND == "circuit":
    K_SUB, M_CAND, N_RF_SUB = 4, 2, 2      # 4 users x (2 beams + idle) = 12 qubits
    MIXER = "trotter"                      # the circuit is a Trotterised XY mixer
    N_RESTARTS, MAXITER = 4, 200
else:
    K_SUB, M_CAND, N_RF_SUB = 6, 3, 3      # 6 x (3 + 1) = 24 qubits-equivalent subspace
    MIXER = "exact"                        # "exact" = expm of XY Hamiltonian, or "trotter"
    N_RESTARTS, MAXITER = 6, 300

# QUBO weights (gains are normalised to max 1)
MU = 1.0           # interference weight
LAM_B = 2.0        # same-beam penalty
LAM_C = 2.0        # "exactly N_RF_SUB served users" penalty

# QAOA
P_DEPTH = 2
SHOTS = 4096

# power / noise (same as run_greedy_review)
TX_POWER_PER_STREAM_W = 1.0
P_RF_CHAIN_W = 0.1
NOISE_POWER = 1e-4

N_SEEDS = 15
METHODS = ["Greedy", "OMP", "QAOA-XY", "Optimal"]
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange",
          "QAOA-XY": "tab:green", "Optimal": "tab:red"}


# ======================================================
# 1. SCENARIO AND SUBPROBLEM
# ======================================================

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
    big = np.bincount(hot).argmax()                 # most crowded hotspot
    order = np.argsort(-G.max(1))
    users = [u for u in order if hot[u] == big]
    users += [u for u in order if hot[u] != big]
    users = np.array(users[:K_SUB])
    cand = np.array([np.argsort(-G[u])[:M_CAND] for u in users])
    return H_users, beam_dict, G, users, cand


# ======================================================
# 2. QUBO AND ISING
# ======================================================

def build_qubo(Gn, users, cand):
    """One qubit per (user, candidate beam) plus one idle qubit per user."""
    K, m = cand.shape
    S = m + 1
    N = K * S
    q = lambda k, s: k * S + s
    Q = np.zeros((N, N))

    for k in range(K):                              # signal (minimise -gain)
        for s in range(m):
            Q[q(k, s), q(k, s)] -= Gn[users[k], cand[k, s]]

    for k, k2 in itertools.combinations(range(K), 2):
        for s in range(m):
            for s2 in range(m):
                b, b2 = cand[k, s], cand[k2, s2]
                w = MU * (Gn[users[k], b2] + Gn[users[k2], b])   # leakage
                if b == b2:
                    w += LAM_B                                   # beam clash
                Q[q(k, s), q(k2, s2)] += w / 2
                Q[q(k2, s2), q(k, s)] += w / 2

    served = [q(k, s) for k in range(K) for s in range(m)]
    for i in served:                                # LAM_C (sum x - R)^2
        Q[i, i] += LAM_C * (1 - 2 * N_RF_SUB)
    for i, j in itertools.combinations(served, 2):
        Q[i, j] += LAM_C
        Q[j, i] += LAM_C
    offset = LAM_C * N_RF_SUB ** 2
    return Q, offset, S


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


def counts_from_aer(qc_bound, shots, seed):
    from qaoa_circuit import sample_with_aer
    return sample_with_aer(qc_bound, shots, seed)


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
    sir_db = 10 * np.log10(np.maximum(signal, 1e-12)
                           / np.maximum(interference, 1e-12))

    return dict(
        table=table, F_RF=F_RF, leakage=L,
        gain=float(sum(t["gain"] for t in table)),
        sinr_db_users=np.array(sinr_db), sinr_db=float(np.mean(sinr_db)),
        se_users=se_user, se=total_se, ee=ee,
        jain=jain_index(se_user), min_se=float(se_user.min()),
        interference=interference, sir_db=sir_db,
        mean_interf_db=float(10 * np.log10(np.mean(interference) + 1e-12)),
    )


# ======================================================
# 6. ONE SEED, ALL METHODS
# ======================================================

def run_seed(seed):
    rng = np.random.default_rng(seed)
    H_users, beam_dict, G, users, cand = build_subproblem(seed)
    Gn = G / G.max()

    Q, offset, S = build_qubo(Gn, users, cand)
    K = len(users)
    h, J, c = ising_from_qubo(Q)
    check_ising_random(Q, h, J, c)

    states, E = subspace_energies(Q, offset, K, S)
    E_ising = ising_energies_subspace(h, J, c, offset, states, K, S)
    assert np.allclose(E, E_ising), "Ising and QUBO energies differ on the subspace"

    extra = {}

    # ------------------ QAOA ------------------
    if BACKEND == "circuit":
        import qaoa_circuit as qc_mod
        res, prob_full, E_full, qc, gam, bet, stats = qc_mod.run_qaoa_circuit(
            Q, offset, h, J, c, K, S, P_DEPTH, N_RESTARTS, rng, maxiter=MAXITER)

        full_idx = qc_mod.subspace_to_full_index(states, S)
        prob = prob_full[full_idx]                       # one-hot states only
        extra["leakage"] = float(1.0 - prob.sum())       # probability outside the feasible subspace
        prob = prob / prob.sum()

        # cross-check circuit against the NumPy Trotterised simulation
        psi = qaoa_state(res.x, E, K, S, "trotter")
        extra["circuit_vs_numpy"] = float(
            np.max(np.abs(np.abs(psi) ** 2 - prob_full[full_idx])))
        extra["stats"] = stats

        # measurement
        bound = qc_mod.bound_circuit(qc, gam, bet, res.x, measure=True)
        counts = counts_from_aer(bound, SHOTS, seed) if USE_AER_SAMPLING else None
        if counts is not None:
            idx_to_state = {int(f): i for i, f in enumerate(full_idx)}
            feas = []
            for bits in counts:
                f = int(bits.replace(" ", ""), 2)        # little-endian: qubit i = bit i
                if f in idx_to_state:
                    i = idx_to_state[f]
                    a = decode(states[i], users, cand)
                    if is_feasible(a):
                        feas.append((E[i], a))
            if feas:
                qaoa_assign, measured_ok = min(feas, key=lambda t: t[0])[1], True
            else:
                qaoa_assign, measured_ok = measure_and_select(
                    prob, E, states, users, cand, rng)
            extra["sampler"] = "Qiskit Aer"
        else:
            qaoa_assign, measured_ok = measure_and_select(
                prob, E, states, users, cand, rng)
            extra["sampler"] = "statevector sampling"
        extra["circuit_obj"] = (qc if seed == 0 else None)
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
# 7. PLOTS
# ======================================================

def plot_main(all_res):
    specs = [("gain", "Total gain"), ("sinr_db", "Average SINR (dB)"),
             ("se", "Total SE (bits/s/Hz)"), ("ee", "EE (bits/s/Hz/W)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, specs):
        means = [np.mean([r[m][key] for r in all_res]) for m in METHODS]
        stds = [np.std([r[m][key] for r in all_res]) for m in METHODS]
        ax.bar(METHODS, means, yerr=stds, capsize=4,
               color=[COLORS[m] for m in METHODS])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Beam assignment performance over {len(all_res)} seeds "
                 f"({K_SUB} users, {M_CAND} beams/user, {N_RF_SUB} RF chains, "
                 f"{BACKEND} backend)")
    fig.tight_layout()
    fig.savefig(f"qaoa_results_{BACKEND}.png", dpi=300)


def plot_fairness(all_res):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    pooled = {m: np.concatenate([r[m]["se_users"] for r in all_res])
              for m in METHODS}
    bins = np.linspace(0, max(v.max() for v in pooled.values()) * 1.05, 20)
    for m in METHODS:
        ax1.hist(pooled[m], bins=bins, histtype="step", linewidth=2,
                 color=COLORS[m], label=m)
    ax1.set_xlabel("Per-user spectral efficiency (bits/s/Hz)")
    ax1.set_ylabel("Number of users (all seeds)")
    ax1.set_title("Per-user SE distribution")
    ax1.legend()

    jm = [np.mean([r[m]["jain"] for r in all_res]) for m in METHODS]
    js = [np.std([r[m]["jain"] for r in all_res]) for m in METHODS]
    ax2.bar(METHODS, jm, yerr=js, capsize=4, color=[COLORS[m] for m in METHODS])
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Jain's fairness index (1 = perfectly fair)")
    ax2.tick_params(axis="x", rotation=30)

    mm = [np.mean([r[m]["min_se"] for r in all_res]) for m in METHODS]
    ms = [np.std([r[m]["min_se"] for r in all_res]) for m in METHODS]
    ax3.bar(METHODS, mm, yerr=ms, capsize=4, color=[COLORS[m] for m in METHODS])
    ax3.set_title("Worst-user SE (bits/s/Hz)")
    ax3.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(f"qaoa_fairness_{BACKEND}.png", dpi=300)


def plot_interference(all_res):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    im = [np.mean([r[m]["mean_interf_db"] for r in all_res]) for m in METHODS]
    isd = [np.std([r[m]["mean_interf_db"] for r in all_res]) for m in METHODS]
    ax1.bar(METHODS, im, yerr=isd, capsize=4, color=[COLORS[m] for m in METHODS])
    ax1.set_title("Mean interference power per user (dB)")
    ax1.tick_params(axis="x", rotation=30)

    pooled = {m: np.concatenate([r[m]["sir_db"] for r in all_res])
              for m in METHODS}
    lo = min(v.min() for v in pooled.values())
    hi = max(v.max() for v in pooled.values())
    bins = np.linspace(lo, hi, 20)
    for m in METHODS:
        ax2.hist(pooled[m], bins=bins, histtype="step", linewidth=2,
                 color=COLORS[m], label=m)
    ax2.set_xlabel("Signal-to-interference ratio (dB)")
    ax2.set_ylabel("Number of users (all seeds)")
    ax2.set_title("Per-user SIR distribution")
    ax2.legend()

    L = all_res[0]["QAOA-XY"]["leakage"]
    t = all_res[0]["QAOA-XY"]["table"]
    img = ax3.imshow(10 * np.log10(np.maximum(L, 1e-12)), cmap="viridis")
    ax3.set_xticks(range(len(t)))
    ax3.set_xticklabels([f"RF{x['rf_chain']}" for x in t])
    ax3.set_yticks(range(len(t)))
    ax3.set_yticklabels([f"user {x['user']}" for x in t])
    ax3.set_title("QAOA-XY leakage map, seed 0 (dB)\n"
                  "diagonal = signal, off-diagonal = interference")
    fig.colorbar(img, ax=ax3)

    fig.tight_layout()
    fig.savefig(f"qaoa_interference_{BACKEND}.png", dpi=300)


def draw_circuit(qc):
    try:
        fig = qc.draw(output="mpl", fold=60, scale=0.5)
        fig.savefig(f"qaoa_circuit_{BACKEND}.png", dpi=200, bbox_inches="tight")
    except Exception as e:                       # drawing needs pylatexenc
        print("Circuit drawing skipped:", e)


# ======================================================
# 8. MAIN
# ======================================================

if __name__ == "__main__":
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

        if seed == 0 and BACKEND == "circuit":
            st = extra["stats"]
            print(f"Circuit: {st['n_qubits']} qubits, depth {st['depth']}, "
                  f"{st['two_qubit_gates']} two-qubit gates, "
                  f"{st['n_pauli_terms']} Pauli terms in H_C")
            print(f"         gate counts {st['ops']}")
            print(f"         Pauli operator vs QUBO diagonal error: "
                  f"{st['operator_check_err']:.2e}")
            draw_circuit(extra["circuit_obj"])

        print("RF chain assignment:")
        for name in METHODS:
            print_rf_table(name, results[name]["table"])
        print(f"  QAOA: <E>={extra['energy']:.4f}  E_opt={extra['e_opt']:.4f}  "
              f"P(optimal)={extra['p_opt']:.4f} (uniform {extra['p_uniform']:.4f})  "
              f"P(feasible)={extra['p_feas']:.3f}  "
              f"matches optimum: {extra['qaoa_matches_opt']}")
        if BACKEND == "circuit":
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
    if BACKEND == "circuit":
        print(f"Max leakage out of subspace:   "
              f"{max(e['leakage'] for e in all_extra):.2e}")
        print(f"Max circuit-vs-NumPy deviation: "
              f"{max(e['circuit_vs_numpy'] for e in all_extra):.2e}")

    plot_main(all_res)
    plot_fairness(all_res)
    plot_interference(all_res)
    plt.show()