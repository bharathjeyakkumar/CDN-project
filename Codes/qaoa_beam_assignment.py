"""
QAOA (XY mixer) beam-to-RF-chain assignment, compared against Greedy, OMP
and the exact optimum, with RF-chain mapping, gain / SINR / SE / EE,
fairness and interference analysis.

Pipeline per seed
-----------------
1. Generate the stress scenario, build the gain matrix G[k, m] = |h_k . f_m|^2
2. Reduce to a subproblem: K_SUB users (most crowded hotspot first) x
   M_CAND best beams per user, served by N_RF_SUB RF chains
3. Encode as QUBO (one-hot group per user: M_CAND beams + 1 idle slot),
   convert to Ising (Eqs. 10-12), unit-test the conversion
4. Simulate XY-mixer QAOA exactly inside the feasible (one-hot) subspace
5. Measure, keep the lowest-energy feasible bitstring, assign each chosen
   beam to an RF chain (analog precoder F_RF)
6. Evaluate all methods with the SAME metrics and produce 3 figures:
      qaoa_results.png       gain / SINR / SE / EE
      qaoa_fairness.png      per-user SE histogram + Jain's index
      qaoa_interference.png  interference power, SIR histogram, leakage map
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

N_ANTENNAS = 16
N_USERS = 20
OVERSAMPLING = 1
N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0

# subproblem solved by QAOA / Greedy / OMP / Optimal
K_SUB = 6          # users in the subproblem
M_CAND = 3         # candidate beams per user
N_RF_SUB = 3       # RF chains in the subproblem

# QUBO weights (gains are normalised to max 1)
MU = 1.0           # interference weight
LAM_B = 2.0        # same-beam penalty
LAM_C = 2.0        # "exactly N_RF_SUB served users" penalty

# QAOA
P_DEPTH = 2
N_RESTARTS = 6
MAXITER = 300
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
    """h, J, c from Eqs. (10)-(12), using x = (1 - s) / 2."""
    N = len(Q)
    h = -0.5 * Q.sum(1)
    J = {(i, j): 0.5 * Q[i, j] for i in range(N) for j in range(i + 1, N)}
    c = 0.5 * np.trace(Q) + 0.5 * np.triu(Q, 1).sum()
    return h, J, c


def check_ising(Q, trials=20):
    h, J, c = ising_from_qubo(Q)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        x = rng.integers(0, 2, len(Q))
        s = 1 - 2 * x
        e_q = x @ Q @ x
        e_i = h @ s + sum(v * s[i] * s[j] for (i, j), v in J.items()) + c
        assert np.isclose(e_q, e_i), (e_q, e_i)


# ======================================================
# 3. FEASIBLE-SUBSPACE QAOA SIMULATION
# ======================================================

def subspace_energies(Q, offset, K, S):
    states = np.array(list(itertools.product(range(S), repeat=K)))
    X = np.zeros((len(states), K * S))
    for k in range(K):
        X[np.arange(len(states)), k * S + states[:, k]] = 1
    E = np.einsum("ni,ij,nj->n", X, Q, X) + offset
    return states, E


def mixer_adjacency(S):
    """XY mixer restricted to the one-hot sector = complete-graph adjacency."""
    A = np.ones((S, S)) - np.eye(S)
    return A


def apply_mixer(psi, U, K, S):
    psi = psi.reshape((S,) * K)
    for ax in range(K):
        psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)
    return psi.reshape(-1)


def qaoa_state(params, E, A, K, S):
    p = len(params) // 2
    gam, bet = params[:p], params[p:]
    psi = np.ones(S ** K, dtype=complex) / np.sqrt(S ** K)   # product of W states
    for l in range(p):
        psi = psi * np.exp(-1j * gam[l] * E)                  # U_C
        psi = apply_mixer(psi, expm(-1j * bet[l] * A), K, S)  # U_M (XY)
    return psi


def run_qaoa(E, K, S, p, restarts, rng):
    A = mixer_adjacency(S)

    def f(th):
        return float(np.sum(np.abs(qaoa_state(th, E, A, K, S)) ** 2 * E))

    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rng.uniform(0, 0.5, p),
                              rng.uniform(0, np.pi / 2, p)])
        res = minimize(f, th0, method="COBYLA", options={"maxiter": MAXITER})
        if best is None or res.fun < best.fun:
            best = res
    prob = np.abs(qaoa_state(best.x, E, A, K, S)) ** 2
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
    # fallback: most probable feasible state in the full distribution
    for i in np.argsort(-prob):
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
    check_ising(Q)
    K = len(users)
    states, E = subspace_energies(Q, offset, K, S)

    # --- QAOA ---
    res, prob = run_qaoa(E, K, S, P_DEPTH, N_RESTARTS, rng)
    qaoa_assign, measured_ok = measure_and_select(prob, E, states, users, cand, rng)

    # --- exact optimum in the subspace ---
    feas_idx = [i for i in range(len(E))
                if is_feasible(decode(states[i], users, cand))]
    opt_i = min(feas_idx, key=lambda i: E[i])
    opt_assign = decode(states[opt_i], users, cand)

    # --- Greedy / OMP on the SAME subproblem ---
    ubeams = np.unique(cand)
    g_a, _ = greedy_assignment(G[users][:, ubeams], N_RF_SUB)
    greedy_assign = [(int(users[k]), int(ubeams[m])) for k, m in g_a]
    o_a, _ = omp_assignment([H_users[u].flatten() for u in users],
                            beam_dict[:, ubeams], N_RF_SUB)
    omp_assign = [(int(users[k]), int(ubeams[m])) for k, m in o_a]

    assigns = {"Greedy": greedy_assign, "OMP": omp_assign,
               "QAOA-XY": qaoa_assign, "Optimal": opt_assign}
    results = {n: evaluate(a, H_users, beam_dict, G) for n, a in assigns.items()}

    extra = dict(
        energy=float(res.fun), e_opt=float(E[opt_i]),
        # mean energy gap to the optimum over the FEASIBLE part of the output
        # distribution (0 = QAOA concentrates on the optimum)
        approx_ratio=float(prob[feas_idx] @ E[feas_idx] / prob[feas_idx].sum()
                           - E[opt_i]),
        p_opt=float(prob[opt_i]), p_feas=float(prob[feas_idx].sum()),
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
                 f"({K_SUB} users, {M_CAND} beams/user, {N_RF_SUB} RF chains)")
    fig.tight_layout()
    fig.savefig("qaoa_results.png", dpi=300)


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
    fig.savefig("qaoa_fairness.png", dpi=300)


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
    fig.savefig("qaoa_interference.png", dpi=300)


# ======================================================
# 8. MAIN
# ======================================================

if __name__ == "__main__":
    all_res, all_extra = [], []

    for seed in range(N_SEEDS):
        print("\n" + "=" * 60)
        print("SEED:", seed)
        print("=" * 60)
        results, extra = run_seed(seed)
        all_res.append(results)
        all_extra.append(extra)

        print("RF chain assignment:")
        for name in METHODS:
            print_rf_table(name, results[name]["table"])
        print(f"  QAOA: <E>={extra['energy']:.4f}  E_opt={extra['e_opt']:.4f}  "
              f"P(optimal)={extra['p_opt']:.3f}  P(feasible)={extra['p_feas']:.3f}  "
              f"matches optimum: {extra['qaoa_matches_opt']}")
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
          f"{np.nanmean([e['approx_ratio'] for e in all_extra]):.3f}")
    print(f"QAOA mean P(optimal):          "
          f"{np.mean([e['p_opt'] for e in all_extra]):.3f}")
    print(f"QAOA mean P(feasible):         "
          f"{np.mean([e['p_feas'] for e in all_extra]):.3f}")
    print(f"QAOA found exact optimum in    "
          f"{sum(e['qaoa_matches_opt'] for e in all_extra)}/{N_SEEDS} seeds")

    plot_main(all_res)
    plot_fairness(all_res)
    plot_interference(all_res)
    plt.show()