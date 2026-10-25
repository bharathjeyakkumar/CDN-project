"""
XY-mixer QAOA beam-to-RF-chain assignment compared with Greedy, OMP and the
exact optimum, with RF-chain mapping, gain / SINR / SE / EE, fairness and
interference analysis.

ENCODING (K*M qubits, no idle qubit)
    One qubit per (user, candidate beam): qubit i = k * M_CAND + s is 1 when user k
    uses its s-th candidate beam.  Number of qubits = K_SUB * M_CAND.
    * "at most one beam per user" and "one user per beam" are QUBO penalties,
    * "exactly N_RF_SUB assignments" is enforced by the ansatz itself:
      Dicke initial state + XY mixer (Hamming weight is conserved), so no RF penalty
      is needed (USE_RF_PENALTY = False).
    The search space is the weight-N_RF_SUB subspace, C(K*M, N_RF) states.

BACKEND = "pennylane"    gate-level PennyLane circuit (qaoa_pennylane.py).
BACKEND = "circuit"      gate-level Qiskit circuit (qaoa_circuit.py).
                         Both are limited to ~20 qubits (K_SUB * M_CAND).
BACKEND = "statevector"  exact NumPy simulation inside the weight-N_RF subspace.
                         Much larger subproblems, no gate decomposition.

The QUBO objective is the linearised sum of -ln(SINR_k) with victim-weighted
leakage (see build_qubo); POOL_MODE controls which users form the subproblem.

In all backends the cost Hamiltonian comes from the QUBO -> Ising mapping
of Eqs. (10)-(12); the mapping is verified numerically on random bit strings
and on every state of the feasible-weight subspace.

Figures written (Greedy / OMP / QAOA-XY only), always into a NEW sub-folder
results/run_<date>_<time>_<backend>_<tag>/ so earlier runs are never overwritten:
    plot_paper_figures : one PDF per group (e.g. per data set), the three panels
                         gain | interference | fairness side by side, 7 in wide
    plot_three         : three separate PNG line charts (seed mode)
"""

import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
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
QUBIT_PRESET = 12      # quantum backends: 12 or 18 qubits  (qubits = K_SUB * M_CAND)
PRESETS = {                # (K_SUB users, M_CAND beams/user, N_RF_SUB RF chains, QAOA restarts)
    12: (4, 3, 2, 4),      # 4 x 3 = 12 qubits, C(12,2) = 66 states in the weight-2 subspace
    18: (6, 3, 3, 3),      # 6 x 3 = 18 qubits, C(18,3) = 816 states in the weight-3 subspace
}

if QUANTUM:
    K_SUB, M_CAND, N_RF_SUB, N_RESTARTS = PRESETS[QUBIT_PRESET]
    MAXITER = 200
    MIXER = "trotter"                      # the circuits are Trotterised XY mixers
else:
    K_SUB, M_CAND, N_RF_SUB = 8, 3, 3      # 8 x 3 = 24 qubits-equivalent, C(24,3) = 2024 states
    MIXER = "exact"                        # "exact" = expm of XY Hamiltonian, or "trotter"
    N_RESTARTS, MAXITER = 6, 300

MIXER_GRAPH = "ring"    # XY mixer coupling graph: "ring" (N edges) or "complete" (N(N-1)/2 edges)
USE_RF_PENALTY = False  # False: the Dicke state + XY mixer already fix the number of served users.
                        # True : also add lam*(sum x - N_RF)^2 (constant on the subspace, kept only for tests)

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
OUTPUT_TAG = ""                                      # appended to the graph file names
RUN_LABEL = "seeds"                                  # what one run is called in graph titles
GAIN_IN_DB = True                                    # gain graph in dB (10 log10 of the summed beam gain) or linear
GROUP_LABELS = None                                  # optional: one label per run (e.g. the data set); a dotted
                                                     # line is drawn wherever the label changes
INTERF_YLABEL = "Interference (dB)"
INTERF_TITLE = "Mean interference power per user (dB), lower is better"
INTERF_YLABEL_SHORT = "Interference (dB)"            # y label of the narrow paper panel

# ---- output folder: every run gets its own sub-folder, nothing is ever overwritten ----
RESULTS_ROOT = "results"        # created next to this script
RUN_TAG = ""                    # extra text in the run-folder name (the dataset script puts users/beams/RF there)

# ---- paper figures (ICC / IEEE two-column paper) ----
X_LABELS = None                 # one x tick label per run (the dataset script uses the scenario names)
SAVE_FORMATS = ("pdf",)         # add "png" to also get a PNG of every paper figure
PAPER_FIG_WIDTH_IN = 7.0        # full text width of a two-column paper (use figure* in LaTeX)
PAPER_FIG_HEIGHT_IN = 2.35      # three equal panels of about 2.2 in each side by side
PAPER_MARKER_SIZE = 2.5         # explicit marker size (rcParams value of 0.75 pt would be invisible)
MARKERS = {"Greedy": "o", "OMP": "s", "QAOA-XY": "^", "Optimal": "D"}
LINESTYLES = {"Greedy": "--", "OMP": ":", "QAOA-XY": "-", "Optimal": "-."}
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange",
          "QAOA-XY": "tab:green", "Optimal": "tab:red"}


# ======================================================
# OUTPUT FOLDER AND PAPER STYLE
# ======================================================

_RUN_DIR = None


def run_dir():
    """
    Folder of the current run, created on first use:
        <script folder>/results/run_<YYYYmmdd_HHMMSS>_<backend>[_<RUN_TAG>]/
    The name carries the start time, so a new run never overwrites an old one
    (a numeric suffix is added in the unlikely case the name already exists).
    """
    global _RUN_DIR
    if _RUN_DIR is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{stamp}_{BACKEND}" + (f"_{RUN_TAG}" if RUN_TAG else "")
        root = Path(__file__).resolve().parent / RESULTS_ROOT
        d, k = root / name, 1
        while d.exists():
            d = root / f"{name}_{k}"
            k += 1
        d.mkdir(parents=True)
        _RUN_DIR = d
    return _RUN_DIR


def paper_style():
    """Matplotlib settings for figures that go into the paper (Times New Roman, small fonts, 600 dpi)."""
    mpl.rcParams.update({
        # Base font (falls back to the next installed Times-like font if Times New Roman is missing)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Axes
        "axes.linewidth": 0.5,

        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,

        # Legend
        "legend.framealpha": 0.9,

        # Lines and markers
        "lines.linewidth": 0.75,
        "lines.markersize": 0.75,

        # Saving
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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
    One qubit per (user, candidate beam): N = K * m qubits, qubit index i = k * m + s.
    There is NO idle qubit: a user that is not served simply has all m of its qubits at 0.

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

    Constraint penalties
        same user, two beams : lam * x_{k,s} x_{k,s'}      (at most one beam per user)
        same beam, two users : lam * x_{k,s} x_{k',s'}     (one user per beam)
        RF cardinality       : only if USE_RF_PENALTY (otherwise the Dicke state / XY mixer fix it)
    Returns Q and the constant offset.
    """
    lam = PENALTY_FACTOR if lam is None else lam
    K, m = cand.shape
    N = K * m
    q = lambda k, s: k * m + s

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

    # typical cost of serving one user alongside N_RF_SUB - 1 others -> remove it
    # (a constant on every state that serves exactly N_RF_SUB users)
    pair_mean = np.mean([P[i, j] for i, j in itertools.combinations(range(N), 2)
                         if i // m != j // m] or [0.0])
    D -= (D.mean() + (N_RF_SUB - 1) * pair_mean)
    scale = max(np.abs(D).max(), np.abs(P).max(), 1e-12)
    D, P = D / scale, P / scale

    Q = np.zeros((N, N))
    Q[np.diag_indices(N)] = D
    Q += P / 2 + P.T / 2

    for k in range(K):                                       # at most one beam per user
        for s, s2 in itertools.combinations(range(m), 2):
            Q[q(k, s), q(k, s2)] += lam / 2
            Q[q(k, s2), q(k, s)] += lam / 2

    for k, k2 in itertools.combinations(range(K), 2):        # same beam twice
        for s in range(m):
            for s2 in range(m):
                if cand[k, s] == cand[k2, s2]:
                    Q[q(k, s), q(k2, s2)] += lam / 2
                    Q[q(k2, s2), q(k, s)] += lam / 2

    offset = 0.0
    if USE_RF_PENALTY:                              # lam * (sum x - R)^2, using x^2 = x
        for i in range(N):
            Q[i, i] += lam * (1 - 2 * N_RF_SUB)
        for i, j in itertools.combinations(range(N), 2):
            Q[i, j] += lam
            Q[j, i] += lam
        offset = lam * N_RF_SUB ** 2
    return Q, offset


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


def states_to_bits(states, N):
    """states: (n, R) array of occupied qubit indices -> (n, N) 0/1 matrix."""
    X = np.zeros((len(states), N))
    X[np.arange(len(states))[:, None], states] = 1
    return X


def ising_energies_subspace(h, J, c, offset, states, N):
    """Ising energy of every weight-R state, to compare with the QUBO energy."""
    s = 1 - 2 * states_to_bits(states, N)
    Jm = np.zeros((N, N))
    for (i, j), v in J.items():
        Jm[i, j] = v
    return s @ h + np.einsum("ni,ij,nj->n", s, Jm, s) + c + offset


# ======================================================
# 3. FEASIBLE-WEIGHT SUBSPACE STATEVECTOR SIMULATION (NumPy)
# ======================================================

def subspace_energies(Q, offset, N, R):
    """All C(N, R) bit strings with exactly R ones, and their QUBO energies."""
    states = np.array(list(itertools.combinations(range(N), R)))
    X = states_to_bits(states, N)
    E = np.einsum("ni,ij,nj->n", X, Q, X) + offset
    return states, E


def xy_edges(N, graph=None):
    """
    Edges of the XY mixer, in a FIXED order. The circuit backends must apply their
    RXX/RYY gates on exactly these pairs in exactly this order (they receive it as `edges`).
    """
    graph = MIXER_GRAPH if graph is None else graph
    if graph == "complete":
        return list(itertools.combinations(range(N), 2))
    if N == 2:
        return [(0, 1)]
    return [(i, (i + 1) % N) for i in range(N)]              # ring


def build_mixer_data(states, N, graph=None):
    """
    Action of the XY mixer inside the weight-R subspace.
    For an edge (a, b), (XX + YY)/2 swaps the occupation of a and b. Within the subspace it
    pairs state i with partner p[i] (only when a and b are occupied differently).
    """
    edges = xy_edges(N, graph)
    n = len(states)
    X = states_to_bits(states, N).astype(int)
    index = {tuple(s): i for i, s in enumerate(states)}
    partners = []
    for a, b in edges:
        p = np.arange(n)
        differ = X[:, a] != X[:, b]
        for i in np.flatnonzero(differ):
            y = X[i].copy()
            y[a], y[b] = y[b], y[a]
            p[i] = index[tuple(np.flatnonzero(y))]
        partners.append((p, differ))
    rows = np.concatenate([np.flatnonzero(d) for _, d in partners])
    cols = np.concatenate([p[d] for p, d in partners])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return dict(edges=edges, partners=partners, A=A)


def apply_mixer(psi, beta, mix, mixer):
    """
    exact   : exp(-i beta A),  A = sum over edges of (XX + YY)/2 in the subspace
    trotter : ordered product over edges of exp(-i beta (XX + YY)/2), i.e. what the
              RXX(beta) RYY(beta) gate sequence implements; each edge is a 2x2 rotation
              [[cos b, -i sin b], [-i sin b, cos b]] on the pairs (i, partner(i)).
    """
    if mixer == "exact":
        return expm_multiply(-1j * beta * mix["A"], psi)
    c, s = np.cos(beta), -1j * np.sin(beta)
    for p, differ in mix["partners"]:
        psi = np.where(differ, c * psi + s * psi[p], psi)
    return psi


def qaoa_state(params, E, mix, mixer):
    p = len(params) // 2
    gam, bet = params[:p], params[p:]
    psi = np.ones(len(E), dtype=complex) / np.sqrt(len(E))   # Dicke state |D_N^R>
    for l in range(p):
        psi = psi * np.exp(-1j * gam[l] * E)                  # U_C
        psi = apply_mixer(psi, bet[l], mix, mixer)            # U_XY
    return psi


def run_qaoa_statevector(E, mix, p, restarts, rng):
    def f(th):
        return float(np.sum(np.abs(qaoa_state(th, E, mix, MIXER)) ** 2 * E))

    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rng.uniform(0, 0.5, p),
                              rng.uniform(0, np.pi / 2, p)])
        res = minimize(f, th0, method="COBYLA", options={"maxiter": MAXITER})
        if best is None or res.fun < best.fun:
            best = res
    prob = np.abs(qaoa_state(best.x, E, mix, MIXER)) ** 2
    return best, prob / prob.sum()


# ======================================================
# 4. DECODING, FEASIBILITY, RF CHAIN ASSIGNMENT
# ======================================================

def decode(state, users, cand):
    """state: occupied qubit indices -> list of (user, beam). Qubit i = k * m + s."""
    m = cand.shape[1]
    return [(int(users[i // m]), int(cand[i // m, i % m])) for i in state]


def is_feasible(assign):
    """N_RF assignments, every user at most once, every beam at most once."""
    return (len(assign) == N_RF_SUB
            and len({u for u, _ in assign}) == N_RF_SUB
            and len({b for _, b in assign}) == N_RF_SUB)


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
    N = K * cand.shape[1]                              # qubits = users x beams

    lam = PENALTY_FACTOR                              # raise penalties until the QUBO
    for _ in range(10):                               # ground state is a feasible assignment
        Q, offset = build_qubo(G, users, cand, lam)
        states, E = subspace_energies(Q, offset, N, N_RF_SUB)
        if penalties_sufficient(E, states, users, cand):
            break
        lam *= 1.5

    h, J, c = ising_from_qubo(Q)
    check_ising_random(Q, h, J, c)

    E_ising = ising_energies_subspace(h, J, c, offset, states, N)
    assert np.allclose(E, E_ising), "Ising and QUBO energies differ on the subspace"

    mix = build_mixer_data(states, N)
    extra = {"lam": lam}

    # ------------------ QAOA ------------------
    if QUANTUM:
        # the circuit backend now receives (N, R, edges) instead of (K, S):
        # Dicke state of weight R, cost layer from (h, J), XY mixer on `edges` in this order
        res, prob_full, E_full, qc, gam, bet, stats = qc_mod.run_qaoa_circuit(
            Q, offset, h, J, c, N, N_RF_SUB, mix["edges"], P_DEPTH, N_RESTARTS, rng,
            maxiter=MAXITER)

        full_idx = qc_mod.subspace_to_full_index(states, N)
        prob = prob_full[full_idx]                       # weight-R states only
        extra["leakage"] = float(1.0 - prob.sum())       # probability outside the weight-R subspace
        prob = prob / prob.sum()

        # cross-check the circuit against the NumPy Trotterised simulation
        psi = qaoa_state(res.x, E, mix, "trotter")
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
        res, prob = run_qaoa_statevector(E, mix, P_DEPTH, N_RESTARTS, rng)
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
            N = len(users) * cand.shape[1]
            lam = PENALTY_FACTOR
            for _ in range(10):
                Q, offset = build_qubo(G, users, cand, lam)
                states, E = subspace_energies(Q, offset, N, N_RF_SUB)
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
    fig.savefig(run_dir() / f"qaoa_mu_tradeoff_{POOL_MODE}.png", dpi=300)


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


def _line(ax, all_res, key, title, ylabel, fn=None):
    """One line per method, one point per run (seed / snapshot). fn optionally transforms the values."""
    n = len(all_res)
    x = np.arange(1, n + 1)
    for mth in PLOT_METHODS:
        y = [(fn(r[mth][key]) if fn else r[mth][key]) for r in all_res]
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.4, color=COLORS[mth],
                label=f"{mth} (mean {np.mean(y):.2f})")
    if GROUP_LABELS is not None and len(GROUP_LABELS) == n:
        for i in range(1, n):
            if GROUP_LABELS[i] != GROUP_LABELS[i - 1]:
                ax.axvline(i + 0.5, color="gray", linestyle=":", linewidth=1)
        top = ax.get_ylim()[1]
        start = 0
        for i in range(1, n + 1):
            if i == n or GROUP_LABELS[i] != GROUP_LABELS[start]:
                ax.text((start + 1 + i) / 2, top, str(GROUP_LABELS[start]), ha="center",
                        va="bottom", fontsize=9, color="dimgray")
                start = i
    step = max(1, int(np.ceil(n / 30)))
    ax.set_xticks(x[::step])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(f"{RUN_LABEL.rstrip('s').capitalize()} index")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18 if GROUP_LABELS is not None else 6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def plot_three(all_res):
    """Three separate line charts, one point per run: gain, interference (dB), fairness.
    Greedy / OMP / QAOA-XY only."""
    setup = (f"{len(all_res)} {RUN_LABEL}, {K_SUB} users, {M_CAND} beams/user, "
             f"{N_RF_SUB} RF chains, {BACKEND}")
    size = (11, 4.8)

    fig1, ax1 = plt.subplots(figsize=size)
    if GAIN_IN_DB:
        _line(ax1, all_res, "gain",
              f"Total beamforming gain of the selected beams (dB) per {RUN_LABEL.rstrip('s')}, higher is better\n{setup}",
              "Total gain (dB)", fn=lambda g: 10 * np.log10(max(g, 1e-30)))
    else:
        _line(ax1, all_res, "gain",
              f"Total beamforming gain of the selected beams per {RUN_LABEL.rstrip('s')}, higher is better\n{setup}",
              "Total gain (linear)")
    fig1.tight_layout()
    fig1.savefig(run_dir() / f"qaoa_gain_{BACKEND}{OUTPUT_TAG}.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=size)
    _line(ax2, all_res, "mean_interf_db", f"{INTERF_TITLE}\n{setup}", INTERF_YLABEL)
    fig2.tight_layout()
    fig2.savefig(run_dir() / f"qaoa_interference_{BACKEND}{OUTPUT_TAG}.png", dpi=300)

    fig3, ax3 = plt.subplots(figsize=size)
    _line(ax3, all_res, "jain", f"Jain's fairness index per {RUN_LABEL.rstrip('s')}, higher is better\n{setup}",
          "Jain's index")
    ax3.set_ylim(0, 1.1)
    fig3.tight_layout()
    fig3.savefig(run_dir() / f"qaoa_fairness_{BACKEND}{OUTPUT_TAG}.png", dpi=300)

def plot_paper_figures(all_res, groups=None, xlabels=None):
    """
    Generate three separate PDF graphs per group:
        1. Gain
        2. Interference
        3. Fairness

    Each graph contains one point per snapshot and three methods:
        Greedy, OMP, QAOA-XY

    The mean of each method is displayed in the legend.

    For 3 groups, this produces 9 PDFs.
    """
    paper_style()

    n = len(all_res)

    groups = GROUP_LABELS if groups is None else groups
    xlabels = X_LABELS if xlabels is None else xlabels

    if groups is None or len(groups) != n:
        groups = [RUN_LABEL.rstrip("s").capitalize()] * n

    written = []

    # One graph per metric.
    metrics = [
        {
            "key": "gain",
            "name": "Gain",
            "ylabel": "Total gain (dB)" if GAIN_IN_DB else "Total gain",
            "title": "Total beamforming gain",
            "transform": (
                lambda g: 10 * np.log10(max(g, 1e-30))
                if GAIN_IN_DB else g
            ),
        },
        {
            "key": "mean_interf_db",
            "name": "Interference",
            "ylabel": INTERF_YLABEL_SHORT,
            "title": INTERF_TITLE,
            "transform": None,
        },
        {
            "key": "jain",
            "name": "Fairness",
            "ylabel": "Jain's fairness index",
            "title": "Jain's fairness index",
            "transform": None,
        },
    ]

    # Process each group independently.
    for group in dict.fromkeys(groups):

        # Select snapshots belonging to this group.
        idx = [i for i in range(n) if groups[i] == group]

        res = [all_res[i] for i in idx]

        labels = (
            [str(xlabels[i]) for i in idx]
            if xlabels is not None and len(xlabels) == n
            else [str(k + 1) for k in range(len(idx))]
        )

        # Produce three separate graphs for this group.
        for metric in metrics:

            fig, ax = plt.subplots(figsize=(7.0, 3.8))

            x = np.arange(1, len(res) + 1)

            # Plot Greedy, OMP and QAOA-XY.
            for method in PLOT_METHODS:

                values = np.array([
                    (
                        metric["transform"](r[method][metric["key"]])
                        if metric["transform"] is not None
                        else r[method][metric["key"]]
                    )
                    for r in res
                ], dtype=float)

                # Mean across the snapshots in this group.
                mean_value = float(np.mean(values))

                ax.plot(
                    x,
                    values,
                    color=COLORS[method],
                    marker=MARKERS.get(method, "o"),
                    markersize=4,
                    linestyle=LINESTYLES.get(method, "-"),
                    linewidth=1.2,
                    label=f"{method} (mean = {mean_value:.3f})",
                )

            # X-axis: individual snapshots.
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlim(0.5, len(res) + 0.5)
            ax.set_xlabel("Snapshot")
            ax.set_ylabel(metric["ylabel"])

            # Title identifies the metric and group.
            ax.set_title(f"{metric['title']} — {group}")

            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(loc="best", fontsize=8)

            # Jain's fairness index lies between 0 and 1.
            if metric["key"] == "jain":
                ax.set_ylim(0, 1.05)

            fig.tight_layout()

            # Create a unique PDF filename for this group and metric.
            safe_group = (
                str(group)
                .replace(" ", "")
                .replace("/", "_")
                .replace("\\", "_")
            )

            stem = (
                f"{safe_group}_{metric['name']}_"
                f"{K_SUB}u{M_CAND}b{N_RF_SUB}rf_{BACKEND}"
            )

            for ext in SAVE_FORMATS:
                path = run_dir() / f"{stem}.{ext}"
                fig.savefig(path, pad_inches=0.03)
                written.append(path)

            plt.close(fig)

    return written



def _paper_figure(res, labels):
    """Three panels (gain | interference | fairness), PAPER_FIG_WIDTH_IN wide, one third each."""
    fig, axes = plt.subplots(1, 3, figsize=(PAPER_FIG_WIDTH_IN, PAPER_FIG_HEIGHT_IN),
                             gridspec_kw=dict(wspace=0.42))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.30)
    x = np.arange(1, len(res) + 1)
    to_db = (lambda g: 10 * np.log10(max(g, 1e-30))) if GAIN_IN_DB else None
    panels = [("gain", "(a) Total beam gain", "Gain (dB)" if GAIN_IN_DB else "Gain", to_db),
              ("mean_interf_db", "(b) Interference", INTERF_YLABEL_SHORT, None),
              ("jain", "(c) Fairness", "Jain's index", None)]
    for ax, (key, title, ylabel, fn) in zip(axes, panels):
        for mth in PLOT_METHODS:
            y = [(fn(r[mth][key]) if fn else r[mth][key]) for r in res]
            ax.plot(x, y, color=COLORS[mth], marker=MARKERS.get(mth, "o"),
                    markersize=PAPER_MARKER_SIZE, linestyle=LINESTYLES.get(mth, "-"), label=mth)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlim(0.5, len(res) + 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[2].set_ylim(top=1.02)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=len(names),
               frameon=True, borderpad=0.3, columnspacing=1.5, handlelength=2.0)
    return fig


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
    print(f"Backend: {BACKEND} | subproblem: {K_SUB} users x {M_CAND} beams "
          f"= {K_SUB * M_CAND} qubits, {N_RF_SUB} RF chains | p = {P_DEPTH} | "
          f"mixer = {MIXER} ({MIXER_GRAPH})")

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
            print(f"        leakage out of weight-{N_RF_SUB} subspace = {extra['leakage']:.2e}  "
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