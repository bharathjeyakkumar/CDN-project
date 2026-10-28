"""PennyLane-only XY-QAOA beam assignment versus Greedy, OMP, and exact ILP.

The cost is EXACTLY the supplied weighted QUBO (qubo_formulation.py):
  -lambda_G * signal_gain + lambda_I * pairwise_cross_leakage
  +lambda_F * unweighted_fairness_proxy + user/beam/RF penalties.
The RF-penalty constant is omitted as in the original reference code.
No log-SINR, SE, EE, Jain index, rescaling, or NumPy statevector simulator.
Candidate beams have user-specific global IDs for QAOA-XY and ILP.
Legacy Greedy/OMP use the original full-codebook selection logic; their
search space is wider. All methods report the same linear-power metrics.
"""

import itertools
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

import qaoa_pennylane as qc_mod
from qubo_formulation import (
    build_qubo_matrix, fairness_matrix, physical_components,
    LAMBDA_G, LAMBDA_I, LAMBDA_F, LAMBDA_U, LAMBDA_B, LAMBDA_R,
)
from step1_greedy import greedy_assignment
from step1_omp import omp_assignment

BACKEND = "pennylane"  # Sole quantum backend
QUANTUM = True
USE_SHOT_SAMPLING = True

N_ANTENNAS = 16
N_USERS = 20
OVERSAMPLING = 1
N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0

QUBIT_PRESET = 12
PRESETS = {12: (4, 3, 2, 4), 18: (6, 3, 3, 3)}
K_SUB, M_CAND, N_RF_SUB, N_RESTARTS = PRESETS[QUBIT_PRESET]
MAXITER = 200
MIXER = "trotter"
MIXER_GRAPH = "ring"
P_DEPTH = 2
SHOTS = 1024
N_SEEDS = 15
METHODS = ["Greedy", "OMP", "QAOA-XY", "ILP"]
DRAW_CIRCUIT = False
POOL_MODE = "mixed"
CAND_MIN_SEP = 1
RESULTS_ROOT = "results"
RUN_TAG = ""
_RUN_DIR = None


def run_dir():
    global _RUN_DIR
    if _RUN_DIR is None:
        root = Path(__file__).resolve().parent / RESULTS_ROOT
        root.mkdir(parents=True, exist_ok=True)
        name = f"run_{datetime.now():%Y%m%d_%H%M%S}_pennylane" + (f"_{RUN_TAG}" if RUN_TAG else "")
        path, counter = root / name, 1
        while path.exists():
            path = root / f"{name}_{counter}"
            counter += 1
        path.mkdir(parents=True)
        _RUN_DIR = path
    return _RUN_DIR


def select_pool(G, hotspot_ids, k_sub, mode="mixed"):
    """Choose hotspot-balanced users in synthetic (non-dataset) mode."""
    hot = np.asarray(hotspot_ids)
    ids = sorted(set(hot), key=lambda h: -int(np.sum(hot == h)))
    per_hot = {h: sorted(np.flatnonzero(hot == h), key=lambda k: -G[k].max()) for h in ids}
    if mode == "crowded":
        return np.asarray(per_hot[ids[0]][:k_sub], dtype=int)
    if mode == "diverse":
        groups = [per_hot[h] for h in ids]
        return np.asarray(list(itertools.islice((v for group in itertools.zip_longest(*groups)
                                                  for v in group if v is not None), k_sub)), int)
    first = per_hot[ids[0]][:max(1, k_sub // 2)]
    rest = [k for h in ids[1:] for k in per_hot[h]]
    rest += [k for k in per_hot[ids[0]] if k not in first]
    return np.asarray((first + rest)[:k_sub], int)


def candidate_beams(G, users, m, min_sep=1):
    n = G.shape[1]
    chosen_all = []
    for u in users:
        chosen = []
        for b in np.argsort(-G[u]):
            b = int(b)
            if all(min(abs(b - t), n - abs(b - t)) >= min_sep for t in chosen):
                chosen.append(b)
                if len(chosen) == m:
                    break
        if len(chosen) != m:
            raise ValueError("Not enough distinct candidate beams")
        chosen_all.append(chosen)
    return np.asarray(chosen_all, int)


def build_subproblem(seed):
    """Optional synthetic driver; dataset driver replaces this function."""
    try:
        from beam_dictionary import generate_beam_dictionary
        from channel_model_distance_aware import generate_stress_scenario
    except ImportError as e:
        raise ImportError("Synthetic mode needs beam_dictionary.py and channel_model_distance_aware.py; "
                          "use the dataset driver for the provided spreadsheet") from e
    scenario = generate_stress_scenario(
        seed=seed, n_tx=N_ANTENNAS, n_users=N_USERS, oversampling=OVERSAMPLING,
        n_hotspots=N_HOTSPOTS, hotspot_angular_spread_deg=HOTSPOT_SPREAD_DEG)
    H = scenario["H_users"]
    F, _ = generate_beam_dictionary(N_ANTENNAS, OVERSAMPLING, 0.5)
    G = np.asarray([abs(h.flatten() @ F)**2 for h in H])
    hot = [info["hotspot_id"] for info in scenario["info_users"]]
    users = select_pool(G, hot, K_SUB, POOL_MODE)
    cand = candidate_beams(G, users, M_CAND, CAND_MIN_SEP)
    return H, F, G, users, cand


def candidate_gain_matrix(G, users, cand):
    return np.take_along_axis(np.asarray(G)[np.asarray(users, dtype=int)], np.asarray(cand, int), axis=1)


def build_qubo(G, users, cand, *, lam_G=LAMBDA_G, lam_I=LAMBDA_I,
               lam_F=LAMBDA_F, lam_U=LAMBDA_U, lam_B=LAMBDA_B, lam_R=LAMBDA_R):
    """EXACT source QUBO, using cross-gain from actual global candidate beam IDs."""
    local_full = np.asarray(G, float)[np.asarray(users, int)]
    return build_qubo_matrix(candidate_gain_matrix(G, users, cand), N_RF_SUB,
                             candidate_beams=cand, cross_gains=local_full,
                             lam_G=lam_G, lam_I=lam_I, lam_F=lam_F,
                             lam_U=lam_U, lam_B=lam_B, lam_R=lam_R)


def decode(state, users, cand):
    M = cand.shape[1]
    return [(int(users[i // M]), int(cand[i // M, i % M])) for i in state]


def is_feasible(assign, n_rf=None):
    n_rf = N_RF_SUB if n_rf is None else int(n_rf)
    return (len(assign) == n_rf and len({u for u, _ in assign}) == n_rf
            and len({b for _, b in assign}) == n_rf)


def states_to_bits(states, N):
    X = np.zeros((len(states), N), dtype=np.float64)
    if len(states):
        X[np.arange(len(states))[:, None], states] = 1
    return X


def subspace_energies(Q, offset, N, R):
    """Classical enumeration FOR VALIDATION only; NOT a quantum backend."""
    states = np.array(list(itertools.combinations(range(N), R)), dtype=int).reshape(-1, R)
    X = states_to_bits(states, N)
    return states, np.einsum("ni,ij,nj->n", X, Q, X) + offset


def ising_from_qubo(Q):
    """For symmetric Q and x=(1-Z)/2, derive H=h.Z + J.ZZ + c."""
    diagonal = np.diag(Q)
    h = -0.5 * Q.sum(axis=1)  # Q_ii x_i and off-diagonal i,j with BOTH Q_ij and Q_ji
    J = {(i, j): 0.5 * Q[i, j] for i in range(len(Q))
         for j in range(i + 1, len(Q)) if Q[i, j] != 0}
    c = 0.5 * diagonal.sum() + 0.5 * np.triu(Q, 1).sum()
    return h, J, float(c)


def verify_ising(Q, h, J, c, states=None):
    rng = np.random.default_rng(17)
    X = (states_to_bits(states, len(Q)) if states is not None
         else rng.integers(0, 2, (50, len(Q))).astype(float))
    Z = 1 - 2 * X
    E1 = np.einsum("ni,ij,nj->n", X, Q, X)
    E2 = Z @ h + c + sum(v * Z[:, i] * Z[:, j] for (i, j), v in J.items())
    if not np.allclose(E1, E2, atol=1e-6, rtol=1e-9):
        raise AssertionError(f"Ising mismatch: {np.max(abs(E1-E2))}")


def xy_edges(N, graph=None):
    graph = MIXER_GRAPH if graph is None else graph
    if graph == "complete":
        return list(itertools.combinations(range(N), 2))
    if graph != "ring":
        raise ValueError("MIXER_GRAPH must be 'ring' or 'complete'")
    if N == 1:
        return []
    if N == 2:
        return [(0, 1)]
    return [(i, (i + 1) % N) for i in range(N)]


def ilp_assignment(G, users, cand, n_rf):
    """Solve EXACT SAME QUBO under HARD feasibility constraints with scipy.milp.

    Penalties are retained in Q (although zero on the feasible constrained set,
    except the omitted RF constant). The objective uses auxiliary y_ij=x_i*x_j.
    """
    K, M = cand.shape
    N = K * M
    if n_rf != N_RF_SUB:
        raise ValueError("n_rf must match N_RF_SUB for QUBO consistency")
    Q, offset = build_qubo(G, users, cand)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if Q[i, j] != 0]
    size = N + len(pairs)
    costs = np.zeros(size)
    costs[:N] = np.diag(Q)
    for t, (i, j) in enumerate(pairs):
        costs[N + t] = 2 * Q[i, j]
    rows, lows, highs = [], [], []
    def constrain(row, lo, hi):
        rows.append(row)
        lows.append(lo)
        highs.append(hi)
    constrain({i: 1 for i in range(N)}, n_rf, n_rf)
    for k in range(K):
        constrain({k * M + s: 1 for s in range(M)}, 0, 1)
    for b in np.unique(cand):
        members = [k * M + s for k in range(K) for s in range(M) if cand[k, s] == b]
        if len(members) > 1:
            constrain({i: 1 for i in members}, 0, 1)
    for t, (i, j) in enumerate(pairs):
        y = N + t
        # With 0<=y<=1, this is sufficient for minimum-cost exact linearization.
        if Q[i, j] >= 0:
            constrain({y: 1, i: -1, j: -1}, -1, np.inf)
        else:
            constrain({y: 1, i: -1}, -np.inf, 0)
            constrain({y: 1, j: -1}, -np.inf, 0)
    matrix = sparse.lil_matrix((len(rows), size), dtype=float)
    for ridx, row in enumerate(rows):
        for col, coeff in row.items():
            matrix[ridx, col] = coeff
    start = time.perf_counter()
    sol = milp(costs, integrality=np.r_[np.ones(N), np.zeros(len(pairs))],
               bounds=Bounds(np.zeros(size), np.ones(size)),
               constraints=LinearConstraint(matrix.tocsr(), lows, highs),
               options={"mip_rel_gap": 0.0})
    duration = time.perf_counter() - start
    if not sol.success:
        raise RuntimeError(f"Exact ILP failed: {sol.message}")
    inds = np.flatnonzero(sol.x[:N] > 0.5)
    assignment = decode(inds, users, cand)
    if not is_feasible(assignment, n_rf):
        raise AssertionError("ILP produced infeasible assignment")
    return assignment, float(sol.fun + offset), duration


def measure_and_select(probs, energies, states, users, cand, rng, observed=None):
    """Best-energy feasible bit string among actual sampled results, else fallback."""
    indices = (np.unique(observed) if observed is not None else
               np.unique(rng.choice(len(probs), size=SHOTS, p=probs)))
    sampled = [(energies[i], decode(states[i], users, cand)) for i in indices]
    sampled = [v for v in sampled if is_feasible(v[1])]
    if sampled:
        return min(sampled, key=lambda x: x[0])[1], True
    for i in np.argsort(-probs):
        a = decode(states[i], users, cand)
        if is_feasible(a):
            return a, False
    raise RuntimeError("No feasible assignment in the weight-R subspace")


def select_without_feasibility(probs, energies, states, users, cand, rng, observed=None):
    """QAOA_FCN: minimum-QUBO-energy SAMPLE, without feasibility filtering.

    Uses exactly the same PennyLane circuit and measurement outcomes as QAOA-XY.
    Never substitutes the exact enumerated ground state for a sampled outcome.
    """
    if observed is not None:
        sampled = np.unique(np.asarray(observed, dtype=int))
    else:
        sampled = np.unique(rng.choice(len(probs), size=SHOTS, p=probs))
    if not len(sampled):
        raise RuntimeError("No weight-R samples available for QAOA_FCN")
    idx = int(sampled[np.argmin(energies[sampled])])
    return decode(states[idx], users, cand), float(energies[idx])


def evaluate_unchecked(assign, H_users, beam_dict, G, users, cand):
    """Physical metrics for QAOA_FCN, including infeasible beam selections.

    Same per-selection gain and fairness terms as QUBO; cross-user interference
    only for DISTINCT users. Multiple beams assigned to one user accumulate in
    that user's row, preserving one row per workbook source user. Reports raw
    physical objective, NOT the constraint-penalized energy.
    """
    selected = [(int(u), int(b)) for u, b in assign]
    mean_gain = float(np.mean(candidate_gain_matrix(G, users, cand)))
    gains = [float(G[u, b]) for u, b in selected]
    fairs = [float(.01 * (v - mean_gain) ** 2) for v in gains]
    interference_by_selection = [float(sum(G[u, bb] for vv, bb in selected
                                           if vv != u)) for u, b in selected]
    gain = float(sum(gains))
    interference = float(sum(interference_by_selection))
    fairness = float(sum(fairs))
    table, RF = assign_to_rf_chains(selected, G, beam_dict)
    candidate_users = set(map(int, users))
    contributions = []
    for u in range(G.shape[0]):
        positions = [i for i, (v, _) in enumerate(selected) if v == u]
        beams = [selected[i][1] for i in positions]
        contributions.append(dict(
            user=int(u), is_candidate_pool=u in candidate_users,
            served=bool(positions), beam="|".join(map(str, beams)) if beams else None,
            gain_linear=float(sum(gains[i] for i in positions)),
            interference_linear=float(sum(interference_by_selection[i] for i in positions)),
            fairness=float(sum(fairs[i] for i in positions)),
        ))
    for key, expected in (("gain_linear", gain), ("interference_linear", interference),
                          ("fairness", fairness)):
        if not np.isclose(sum(v[key] for v in contributions), expected, rtol=1e-10, atol=1e-8):
            raise AssertionError(f"FCN user {key} does not reconcile with total")
    score = float(-LAMBDA_G * gain + LAMBDA_I * interference + LAMBDA_F * fairness)
    return dict(table=table, F_RF=RF, gain_linear=gain,
                interference_linear=interference, fairness=fairness,
                objective_score=score, user_contributions=contributions)


def assign_to_rf_chains(assign, G, beam_dict):
    ordered = sorted(assign, key=lambda ub: -G[ub[0], ub[1]])
    table = [dict(rf_chain=i, user=int(u), beam=int(b), gain=float(G[u, b]))
             for i, (u, b) in enumerate(ordered)]
    return table, np.asarray(beam_dict)[:, [t["beam"] for t in table]]


def print_rf_table(name, table):
    print(f"  {name}")
    for t in table:
        print(f"    RF chain {t['rf_chain']} -> user {t['user']:2d}, "
              f"beam {t['beam']:3d}, gain = {t['gain']:.6g}")


def evaluate(assign, H_users, beam_dict, G, users, cand):
    """Raw unweighted metrics aligned to the exact QUBO physical objective.

    Gain = sum selected |h_k f_b|^2.
    Interference = sum over unordered selected pairs (G[k,b_j] + G[j,b_k]).
    Fairness = sum selected 0.01*(candidate gain - mean(all candidate gains))^2.
    All are LINEAR power metrics, with no decibels or weight lambda_F in plots.
    """
    table, RF = assign_to_rf_chains(assign, G, beam_dict)
    user_to_local = {int(u): k for k, u in enumerate(users)}
    # The legacy baselines can select any full-codebook beam, including beams
    # outside a user's QAOA candidate list. Compute metrics on the same raw
    # physical powers and use the SAME candidate-reference mean for fairness.
    # This keeps the reporting metric consistent without altering old baselines.
    selected = [(int(u), int(b)) for u, b in assign]
    mean_candidate_gain = float(np.mean(candidate_gain_matrix(G, users, cand)))
    gain = float(sum(G[u, b] for u, b in selected))
    interference = float(sum(G[u, b_other] + G[u_other, b]
                             for (u, b), (u_other, b_other)
                             in itertools.combinations(selected, 2)))
    fairness = float(sum(0.01 * (G[u, b] - mean_candidate_gain) ** 2
                         for u, b in selected))
    # Per-source-user contributions; rows not assigned an RF chain contribute zero.
    # Interference received by user u is sum of the powers from OTHER selected beams.
    # Summing over selected users exactly recovers the symmetric pairwise proxy.
    contributions = []
    chosen_beams = dict(selected)
    candidate_users = set(map(int, users))
    for u in range(G.shape[0]):
        beam = chosen_beams.get(u)
        served = beam is not None
        g_u = float(G[u, beam]) if served else 0.0
        i_u = (float(sum(G[u, other_beam] for other_user, other_beam in selected
                         if other_user != u)) if served else 0.0)
        f_u = (float(0.01 * (g_u - mean_candidate_gain)**2) if served else 0.0)
        contributions.append(dict(user=int(u), is_candidate_pool=u in candidate_users,
                                  served=served, beam=int(beam) if served else None,
                                  gain_linear=g_u, interference_linear=i_u, fairness=f_u))
    for key, expected in (("gain_linear", gain), ("interference_linear", interference),
                          ("fairness", fairness)):
        if not np.isclose(sum(v[key] for v in contributions), expected,
                          rtol=1e-10, atol=1e-8):
            raise AssertionError(f"Per-user {key} does not reconcile with total {expected}")
    # The same feasible physical score is reported for every method.
    # RF-cardinality contributes only the omitted constant -lambda_R*R**2.
    objective_score = float(-LAMBDA_G * gain + LAMBDA_I * interference
                            + LAMBDA_F * fairness)
    return dict(table=table, F_RF=RF, gain_linear=gain,
                interference_linear=interference, fairness=fairness,
                objective_score=objective_score,
                user_contributions=contributions)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    H, F, G, users, cand = build_subproblem(seed)
    users, cand = np.asarray(users, int), np.asarray(cand, int)
    K, M = cand.shape
    N = K * M
    if N > qc_mod.MAX_QUBITS:
        raise ValueError(f"{N} qubits exceed PennyLane limit {qc_mod.MAX_QUBITS}")
    Q, offset = build_qubo(G, users, cand)
    states, energies = subspace_energies(Q, offset, N, N_RF_SUB)
    feasible = np.asarray([is_feasible(decode(st, users, cand)) for st in states])
    if not np.any(feasible):
        raise ValueError(
            f"No feasible assignment for {N_RF_SUB} RF chains from the "
            f"{K} user-specific beam shortlists. Candidate lists may overlap "
            "too much: fewer than RF distinct user-to-beam matches exist. "
            "The dataset runner now repairs this before QAOA; synthetic "
            "or custom subproblems must provide feasible candidates."
        )
    opt_idx = np.flatnonzero(feasible)[np.argmin(energies[feasible])]
    opt_assign = decode(states[opt_idx], users, cand)
    h, J, c = ising_from_qubo(Q)
    verify_ising(Q, h, J, c, states)

    circuit_result, prob_full, _, circuit, gam, bet, stats = qc_mod.run_qaoa_circuit(
        Q, offset, h, J, c, N, N_RF_SUB, xy_edges(N), P_DEPTH, N_RESTARTS,
        rng, maxiter=MAXITER)
    full_indices = qc_mod.subspace_to_full_index(states, N)
    probs = np.asarray(prob_full[full_indices], dtype=float)
    outside_prob = float(max(0, 1 - probs.sum()))
    if probs.sum() <= 0:
        raise RuntimeError("PennyLane returned no weight-R probability")
    probs /= probs.sum()
    observed = None
    if USE_SHOT_SAMPLING:
        full_observed = qc_mod.sample_indices(circuit, gam, bet, circuit_result.x, SHOTS, seed)
        idx_map = {int(f): i for i, f in enumerate(full_indices)}
        observed = [idx_map[int(v)] for v in full_observed if int(v) in idx_map]
    qaoa_assign, measured = measure_and_select(probs, energies, states, users, cand, rng,
                                               observed=observed)

    fcn_assign, fcn_qubo_energy = select_without_feasibility(
        probs, energies, states, users, cand, rng, observed=observed)

    ilp_assign, ilp_objective, ilp_duration = ilp_assignment(G, users, cand, N_RF_SUB)
    if not np.isclose(ilp_objective, energies[opt_idx], atol=1e-5, rtol=1e-7):
        raise AssertionError(f"ILP energy {ilp_objective} != exact QUBO energy {energies[opt_idx]}")

    # Original Greedy and OMP baselines, unchanged selection procedures.
    # They search the FULL beam codebook rather than QAOA's per-user shortlists.
    # The original functions return local-user indices and global beam indices.
    local_full = np.asarray(G)[users]
    local_H = np.asarray(H)[users]
    greedy_local, _ = greedy_assignment(
        local_full, N_RF_SUB, lambda_I=LAMBDA_I, lambda_F=LAMBDA_F,
        lambda_A=LAMBDA_U)
    omp_local, _ = omp_assignment(
        local_H, F, N_RF_SUB, lambda_I=LAMBDA_I,
        lambda_F=LAMBDA_F, lambda_A=LAMBDA_U)
    assignments = {
        "Greedy": [(int(users[k]), int(b)) for k, b in greedy_local],
        "OMP": [(int(users[k]), int(b)) for k, b in omp_local],
        "QAOA-XY": qaoa_assign, "ILP": ilp_assign,
    }
    for name, a in assignments.items():
        if not is_feasible(a, N_RF_SUB):
            raise AssertionError(f"{name} produced an infeasible assignment: {a}")
        if name in ("QAOA-XY", "ILP") and not all(
            b in set(cand[np.where(users == u)[0][0]]) for u, b in a
        ):
            raise AssertionError(f"{name} selected a beam outside the user's candidates")
    results = {name: evaluate(a, H, F, G, users, cand) for name, a in assignments.items()}
    results["QAOA_FCN"] = evaluate_unchecked(fcn_assign, H, F, G, users, cand)
    for name, assignment in [*assignments.items(), ("QAOA_FCN", fcn_assign)]:
        results[name]["feasibility_check"] = "Yes" if is_feasible(assignment, N_RF_SUB) else "No"
    results["QAOA_FCN"]["qubo_energy"] = fcn_qubo_energy
    feasible_inds = np.flatnonzero(feasible)
    extra = dict(
        energy=float(circuit_result.fun), e_opt=float(energies[opt_idx]),
        p_opt=float(probs[opt_idx]), p_feas=float(probs[feasible_inds].sum()),
        p_uniform=1 / len(energies), measured_feasible=measured,
        qaoa_matches_opt=set(qaoa_assign) == set(opt_assign),
        gap=float(np.dot(probs[feasible_inds], energies[feasible_inds]) / probs[feasible_inds].sum()
                  - energies[opt_idx]) if probs[feasible_inds].sum() > 0 else float("nan"),
        leakage=outside_prob, stats=stats,
        ilp_obj=ilp_objective, ilp_time=ilp_duration,
        penalties_sufficient=(np.min(energies[feasible]) <= np.min(energies[~feasible])
                              if np.any(~feasible) else True),
        sampler=qc_mod.SAMPLER_NAME if USE_SHOT_SAMPLING else "PennyLane probabilities",
        circuit_obj=circuit if seed == 0 else None, best_params=circuit_result.x,
    )
    return results, extra


def draw_circuit(circuit, params=None):
    if circuit is not None:
        qc_mod.save_drawing(circuit, run_dir() / "qaoa_circuit.png", params=params)


if __name__ == "__main__":
    import csv
    results_rows = []
    print(f"PennyLane QAOA, K={K_SUB}, candidate beams={M_CAND}, RF={N_RF_SUB}")
    for seed in range(N_SEEDS):
        results, extra = run_seed(seed)
        print(f"\nSeed {seed}, QAOA P(feasible)={extra['p_feas']:.3f}, "
              f"P(optimal)={extra['p_opt']:.4f}, matched ILP={extra['qaoa_matches_opt']}")
        for name, r in results.items():
            print_rf_table(name, r["table"])
            print(f"  gain={r['gain_linear']:.6g}, interference={r['interference_linear']:.6g}, "
                  f"fairness penalty={r['fairness']:.6g}")
            results_rows.append({"seed": seed, "method": name,
                                 **{k: r[k] for k in ("gain_linear", "interference_linear", "fairness")}})
        if seed == 0 and DRAW_CIRCUIT:
            draw_circuit(extra["circuit_obj"], extra["best_params"])
    path = run_dir() / "qaoa_seed_results.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results_rows[0]))
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"Saved result metrics to {path}; use plotter.py for figures.")
