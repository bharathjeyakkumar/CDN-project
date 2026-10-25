"""
qaoa_beam_assignment_dataset.py -- QAOA / Greedy / OMP comparison on channels built from
channel_test_suite_organized.xlsx.

ENCODING: one qubit per (user, candidate beam)  ->  qubits = users x beams.
The number of served users (RF chains) is enforced by the ansatz (Dicke state + XY mixer),
"at most one beam per user" and "one user per beam" by QUBO penalties.

HOW TO RUN
----------
    python qaoa_beam_assignment_dataset.py
        asks you to type the number of users, candidate beams per user and RF chains, prints what
        that costs in qubits (and rough memory / time), asks for confirmation, then runs.

    python qaoa_beam_assignment_dataset.py --users 6 --beams 3 --rf 2 --set "Set 1" --snapshots 5 --yes
        same thing without questions.   --set: all | "Set 1" | "Set 2" | "Set 3"

Needs, in the same folder:
    qaoa_beam_assignment.py   (QUBO, QAOA circuit, metrics, plots; nothing of it is duplicated here)
    plotter.py                (figures from the results CSV; also runnable on its own)
    dataset_channels.py       (spreadsheet loader + channel generation)
    channels.py               (your channel model and codebooks)
    qaoa_pennylane.py (or qaoa_circuit.py), step1_greedy.py, step1_omp.py, compute_sinr.py
    channel_test_suite_organized.xlsx

What one run does
-----------------
1. Take the chosen set(s) of the spreadsheet in file order. A snapshot = the next SNAPSHOT_SIZE (10) consecutive
   rows. Every scenario of a set has exactly 10 rows, so snapshot 1 = the first scenario (UMi-LOS), snapshot 2 =
   the next scenario (UMi-NLOS), ... : 10 snapshots per set, 30 for all three sets, one scenario per snapshot.
2. Build the noise-normalised channels from those rows (dataset_channels.py).
3. From the 10 users of the snapshot choose the QAOA subproblem: the number of users, beams and RF chains you
   typed. Greedy, OMP, QAOA and the exact optimum all solve exactly this subproblem.
4. Write everything into a NEW folder  results/run_<date>_<time>_<backend>_<users>u<beams>b<rf>rf/
   (older runs are never overwritten):
       qaoa_dataset_results.csv           every number of every snapshot and method (gain, SINR, interference, ...)
       Set1_..., Set2_..., Set3_... .pdf   one 7-inch wide figure per set: gain | interference | fairness,
                                          drawn by plotter.py from the CSV above (NOT from variables of this script)
       run_info.txt                       the settings of the run
   To change the look of the figures, edit plotter.py and run it again on the CSV; no simulation is repeated.

Units: the channel is noise-normalised (noise power = 1, transmit power already inside H), so the
interference graph is interference-to-noise ratio in dB: 0 dB = interference equals the noise.
"""

import argparse
import sys
from math import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qaoa_beam_assignment as base
import dataset_channels as dc
import plotter


# ======================================================
# CONFIGURATION (defaults; the prompts / arguments override users, beams and RF chains)
# ======================================================

DATA_FILE = "channel_test_suite_organized.xlsx"
DATA_SELECTION = "all"        # "all", "Set 1", "Set 2" or "Set 3"
SNAPSHOTS_PER_SET = 10        # snapshots per set, taken in file order (10 = the whole set; "all" = 3 sets -> 30)
SNAPSHOT_SIZE = 10            # consecutive spreadsheet rows (= users) in one snapshot; a scenario has 10 rows
POOL_MODE = "mixed"           # which users of the snapshot form the subproblem:
                              #   "mixed"  : strongest user + its nearest-angle neighbours, rest far away in angle
                              #   "crowded": strongest user + nearest-angle neighbours only
                              #   "random" : random users
CODEBOOK = "polar"            # "polar": your polar_codebook (DFT atoms + near-field atoms), "dft": DFT only
CODEBOOK_OVERSAMPLE = 2       # your default
CAND_RHO = 0.7                # candidate beams of one user must have |f_a^H f_b| < CAND_RHO (1.01 = plain top-m)
SHADOWING = True              # the spreadsheet path loss has no shadow fading, so add it (row sigma)
PATHLOSS_SOURCE = "dataset"   # "dataset" (spreadsheet column) or "model" (your path_loss_db())
TX_DBM, BW_HZ, NF_DB = 30.0, 100e6, 9.0
MASTER_SEED = 0
RESULTS_CSV = "qaoa_dataset_results.csv"      # written into the run folder, never overwritten

MAX_QUBITS = 20               # dense simulation limit of the circuit backends
MAX_FEASIBLE_STATES = 2_000_000   # limit of the "statevector" backend (weight-R subspace size C(users*beams, R))


# ======================================================
# COST OF A CHOICE: qubits, memory, rough time
# ======================================================

def retries_for(qubits):
    """Number of QAOA restarts used for a given size (fewer restarts for bigger circuits)."""
    return 4 if qubits <= 12 else 3 if qubits <= 18 else 2


def estimate_cost(users, beams, rf, n_snapshots):
    """
    qubits = users x beams: one qubit per (user, candidate beam), no idle qubit.
    The ansatz (Dicke state + XY mixer) lives in the weight-rf subspace, C(qubits, rf) states;
    'feasible' is the upper bound on fully valid assignments, C(users, rf) x beams^rf.
    Time is a ROUGH guess scaled from my own timings (about 0.02 s per optimiser step at 12 qubits and
    0.6 s at 18, growing about x1.76 per extra qubit); your machine will differ.
    """
    q = users * beams
    feasible = comb(users, rf) * beams ** rf
    subspace = comb(q, rf)
    full_states = 2 ** q
    mem_mb = 16 * full_states / 1e6                         # one complex128 state vector
    restarts = retries_for(q)
    steps = getattr(getattr(base, "qc_mod", None), "ADAM_STEPS", base.MAXITER)
    step_s = 0.02 * 1.76 ** (q - 12)
    per_snapshot = restarts * steps * step_s + 2.0 * step_s * 20 + 1.0
    return dict(qubits=q, feasible=feasible, subspace=subspace, full_states=full_states,
                mem_mb=mem_mb, restarts=restarts, per_snapshot_s=per_snapshot,
                total_s=per_snapshot * n_snapshots, n_snapshots=n_snapshots)


def check_choice(users, beams, rf):
    """Returns (ok, message)."""
    if users < 1 or beams < 1 or rf < 1:
        return False, "users, beams and RF chains must all be at least 1."
    if rf > users:
        return False, f"RF chains ({rf}) cannot be more than users ({users})."
    if users > SNAPSHOT_SIZE:
        return False, (f"{users} users is more than the {SNAPSHOT_SIZE} users (spreadsheet rows) "
                       f"of one snapshot. Use at most {SNAPSHOT_SIZE}.")
    q = users * beams
    if base.QUANTUM and q > MAX_QUBITS:
        return False, (f"{users} users x {beams} beams = {q} qubits is more than the {MAX_QUBITS}-qubit "
                       f"limit of the circuit backend. Reduce users or beams.")
    if not base.QUANTUM and comb(q, rf) > MAX_FEASIBLE_STATES:
        return False, (f"C({q},{rf}) = {comb(q, rf):,} states is too many for the statevector backend. "
                       f"Reduce users, beams or RF chains.")
    return True, ""


def print_cost(users, beams, rf, c):
    t = c["total_s"]
    t_txt = f"{t:.0f} s" if t < 120 else f"{t / 60:.1f} min" if t < 7200 else f"{t / 3600:.1f} h"
    print("\n" + "-" * 62)
    print("COST OF THIS CHOICE")
    print("-" * 62)
    print(f"  users in the subproblem        : {users}")
    print(f"  candidate beams per user       : {beams}")
    print(f"  RF chains (users served)       : {rf}")
    print(f"  QUBITS = {users} x {beams}                 : {c['qubits']}")
    print(f"  valid assignments (upper bound): {c['feasible']:,}   (= C({users},{rf}) x {beams}^{rf})")
    print(f"  search space (weight-{rf} states): {c['subspace']:,}   (= C({c['qubits']},{rf}))")
    if base.QUANTUM:
        mem = f"{c['mem_mb'] * 1000:.0f} KB" if c["mem_mb"] < 1 else f"{c['mem_mb']:.0f} MB"
        print(f"  full state vector              : 2^{c['qubits']} = {c['full_states']:,} amplitudes (~{mem})")
        print(f"  QAOA restarts                  : {c['restarts']}")
        print(f"  rough time                     : ~{t_txt} for {c['n_snapshots']} snapshots (very approximate)")
    if c["qubits"] > 16 and base.QUANTUM:
        print("  note: above 16 qubits every optimiser step becomes slow; test with --snapshots 1 first.")
    print("-" * 62)


# ======================================================
# INPUT (typed answers or command-line arguments)
# ======================================================

def ask_int(text, default):
    while True:
        try:
            s = input(f"{text} [{default}]: ").strip()
        except EOFError:                       # no keyboard (script / pipe closed): use the default
            return default
        if s == "":
            return default
        try:
            return int(s)
        except ValueError:
            print("  please type a whole number")


def get_choice(args, n_snapshots_fn):
    users, beams, rf = args.users, args.beams, args.rf
    interactive = users is None or beams is None or rf is None
    if interactive:
        print("Choose the QAOA subproblem (press Enter to keep the value in brackets).")
    while True:
        users = args.users if args.users is not None else ask_int("  Number of users", base.K_SUB)
        beams = args.beams if args.beams is not None else ask_int("  Candidate beams per user", base.M_CAND)
        rf = args.rf if args.rf is not None else ask_int("  RF chains (users served)", min(base.N_RF_SUB, users))
        ok, msg = check_choice(users, beams, rf)
        print_cost(users, beams, rf, estimate_cost(users, beams, rf, n_snapshots_fn(users))) if ok else None
        if not ok:
            print(f"\n  Not possible: {msg}\n")
            if not interactive:
                sys.exit(1)
            args.users = args.beams = args.rf = None
            continue
        if args.yes:
            return users, beams, rf
        try:
            answer = input("Proceed with this choice? [Y/n]: ").strip().lower()
        except EOFError:
            answer = "y"
        if answer in ("", "y", "yes"):
            return users, beams, rf
        args.users = args.beams = args.rf = None          # ask again


# ======================================================
# SNAPSHOTS AND SUBPROBLEM
# ======================================================

def build_snapshots(selection, per_set):
    """
    Sequential snapshots. For every chosen set the rows are taken in file order, SNAPSHOT_SIZE at a time:
    snapshot 1 = rows 1-10 (first scenario), snapshot 2 = rows 11-20 (next scenario), ...
    """
    rng = np.random.default_rng(MASTER_SEED)
    df = dc.load_dataset(DATA_FILE)
    for col in ("set", "N", "R", "scenario", "fc_GHz"):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found in the dataset; columns are: {list(df.columns)}")
    if "instance_id" in df.columns:
        df = df.sort_values("instance_id", kind="stable")
    df = df.reset_index(drop=True)

    names = list(dict.fromkeys(df["set"]))                       # sets in file order
    if selection != "all":
        key = lambda t: str(t).lower().replace(" ", "")
        names = [n for n in names if key(n) == key(selection)]
        if not names:
            raise ValueError(f"set '{selection}' not found; available: {list(dict.fromkeys(df['set']))}")

    snaps = []
    for name in names:
        sub = df[df["set"] == name].reset_index(drop=True)
        n_max = len(sub) // SNAPSHOT_SIZE
        n_take = min(per_set, n_max)
        if per_set > n_max:
            print(f"  note: {name} has only {n_max} snapshots of {SNAPSHOT_SIZE} rows, using {n_max}.")
        for j in range(n_take):
            rows = sub.iloc[j * SNAPSHOT_SIZE:(j + 1) * SNAPSHOT_SIZE].reset_index(drop=True)
            scen = list(dict.fromkeys(rows["scenario"]))
            snaps.append(dict(set=name, N=int(rows["N"].iloc[0]), R=int(rows["R"].iloc[0]), rows=rows,
                              scenario=scen[0] if len(scen) == 1 else "/".join(map(str, scen))))
    for i, s in enumerate(snaps):
        H, info = dc.generate_channels_from_rows(
            s["rows"], s["N"], rng, n_chains=base.N_RF_SUB, tx_dbm=TX_DBM, bw_hz=BW_HZ,
            nf_db=NF_DB, shadowing=SHADOWING, pathloss_source=PATHLOSS_SOURCE)
        s.update(H=H, info=info, index=i, fc=float(s["rows"]["fc_GHz"].iloc[0]))
    return snaps


def select_pool(G, sin_phi, k_sub, mode, rng):
    """Choose the subproblem users. The same pool is given to every method."""
    K = G.shape[0]
    if k_sub >= K:
        return np.arange(K)
    if mode == "random":
        return rng.choice(K, size=k_sub, replace=False)
    anchor = int(np.argmax(G.max(1)))
    order = np.argsort(np.abs(sin_phi - sin_phi[anchor]))        # anchor first, then by angular distance
    if mode == "crowded":
        return order[:k_sub]
    n_near = int(np.ceil(k_sub / 2))
    near = list(order[:n_near])
    far = [u for u in order[::-1] if u not in near][: k_sub - n_near]
    return np.array(near + far)


def candidate_beams_distinct(G, F, users, m, rho):
    """
    For every user the m strongest beams that are mutually distinct: |f_a^H f_b| < rho.
    Works for any codebook (including the stacked polar one). rho >= 1.01 gives plain top-m.
    """
    cand = []
    for u in users:
        chosen = []
        for b in np.argsort(-G[u]):
            if all(abs(np.vdot(F[:, c], F[:, b])) < rho for c in chosen):
                chosen.append(int(b))
            if len(chosen) == m:
                break
        if len(chosen) < m:                                       # not enough distinct beams: fill by gain
            for b in np.argsort(-G[u]):
                if int(b) not in chosen:
                    chosen.append(int(b))
                if len(chosen) == m:
                    break
        cand.append(chosen)
    return np.array(cand)


_CODEBOOKS = {}


def codebook(N, fc):
    key = (N, fc, CODEBOOK, CODEBOOK_OVERSAMPLE)
    if key not in _CODEBOOKS:
        _CODEBOOKS[key] = dc.get_codebook(CODEBOOK, N, fc, CODEBOOK_OVERSAMPLE)
    return _CODEBOOKS[key]


def make_build_subproblem(snaps):
    """Replacement for base.build_subproblem: 'seed' is the snapshot index."""
    def build_subproblem(i):
        s = snaps[i]
        F = codebook(s["N"], s["fc"])
        H = s["H"]
        G = np.abs(H @ F) ** 2                                     # (SNAPSHOT_SIZE, n_beams)
        rng = np.random.default_rng(MASTER_SEED + i)
        users = select_pool(G, s["info"]["sin_los"], base.K_SUB, POOL_MODE, rng)
        cand = candidate_beams_distinct(G, F, users, base.M_CAND, CAND_RHO)
        return [h for h in H], F, G, np.asarray(users), cand
    return build_subproblem


# ======================================================
# MAIN
# ======================================================

def parse_args():
    p = argparse.ArgumentParser(description="QAOA beam assignment on the spreadsheet channels")
    p.add_argument("--users", type=int, help="users in the QAOA subproblem")
    p.add_argument("--beams", type=int, help="candidate beams per user")
    p.add_argument("--rf", type=int, help="RF chains (users served)")
    p.add_argument("--set", dest="set_", default=None, help='all | "Set 1" | "Set 2" | "Set 3"')
    p.add_argument("--snapshots", type=int, default=None, help="snapshots per set")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selection = args.set_ or DATA_SELECTION
    per_set = args.snapshots or SNAPSHOTS_PER_SET
    n_sets = 3 if selection == "all" else 1

    users, beams, rf = get_choice(args, lambda u: per_set * n_sets)
    base.RUN_TAG = f"{users}u{beams}b{rf}rf"                    # goes into the run-folder name

    # apply the choice to the shared QAOA code
    base.K_SUB, base.M_CAND, base.N_RF_SUB = users, beams, rf
    base.N_RESTARTS = retries_for(users * beams)

    snaps = build_snapshots(selection, per_set)

    # noise-normalised channel: noise power 1, transmit power already inside H
    base.NOISE_POWER = 1.0
    base.TX_POWER_PER_STREAM_W = 1.0
    base.build_subproblem = make_build_subproblem(snaps)

    qubits = users * beams
    print(f"\nData: {DATA_FILE} | selection: {selection} | {len(snaps)} sequential snapshots of "
          f"{SNAPSHOT_SIZE} users | pool: {POOL_MODE} | codebook: {CODEBOOK} | shadowing: {SHADOWING}")
    out_dir = base.run_dir()
    print(f"Results of this run go to: {out_dir}")
    with open(out_dir / "run_info.txt", "w") as f:
        f.write(f"data file       : {DATA_FILE}\nselection       : {selection}\n"
                f"snapshots       : {len(snaps)} x {SNAPSHOT_SIZE} users (sequential)\n"
                f"subproblem      : {users} users x {beams} beams = {qubits} qubits, {rf} RF chains\n"
                f"backend         : {base.BACKEND}, p = {base.P_DEPTH}, mixer = {base.MIXER} ({base.MIXER_GRAPH})\n"
                f"pool / codebook : {POOL_MODE} / {CODEBOOK}, shadowing {SHADOWING}\n"
                f"objective       : {base.OBJECTIVE}, MU = {base.MU}, penalty = {base.PENALTY_FACTOR}\n")
    print(f"Backend: {base.BACKEND} | subproblem: {users} users x {beams} beams, {rf} RF chains "
          f"= {qubits} qubits | p = {base.P_DEPTH} | mixer = {base.MIXER} ({base.MIXER_GRAPH})")

    all_res, all_extra, rows = [], [], []
    for i, s in enumerate(snaps):
        print("\n" + "=" * 60)
        print(f"SNAPSHOT {i + 1}/{len(snaps)}: {s['set']}, scenario {s['scenario']} "
              f"(N = {s['N']} antennas, R = {s['R']} RF chains in the dataset)")
        print("  scenarios:", ", ".join(s["info"]["scenario"]))
        results, extra = base.run_seed(i)
        all_res.append(results)
        all_extra.append(extra)

        print("RF chain assignment:")
        for name in base.METHODS:
            base.print_rf_table(name, results[name]["table"])
        print(f"  QAOA: P(optimal)={extra['p_opt']:.4f}  P(feasible)={extra['p_feas']:.3f}  "
              f"matches optimum: {extra['qaoa_matches_opt']}")
        for name in base.METHODS:
            r = results[name]
            print(f"  {name:8s} SINR={r['sinr_db']:.2f} dB  interf={r['mean_interf_db']:.2f} dB  "
                  f"Jain={r['jain']:.3f}  SE={r['se']:.3f}")
            rows.append(dict(snapshot=i + 1, set=s["set"], scenario=s["scenario"], N=s["N"], users=users, beams=beams, rf_chains=rf,
                             qubits=qubits, backend=base.BACKEND, method=name,
                             gain_linear=r["gain"], gain_db=10 * np.log10(max(r["gain"], 1e-30)),
                             sinr_db=r["sinr_db"],
                             interference_db=r["mean_interf_db"], jain=r["jain"], sum_se=r["se"],
                             min_user_se=r["min_se"], scenarios="|".join(s["info"]["scenario"])))

    # ---------- summary ----------
    print("\n" + "=" * 70)
    print(f"SUMMARY (averaged over {len(snaps)} snapshots, {qubits} qubits)")
    print("=" * 70)
    print(f"{'Method':9s}{'SINR dB':>10s}{'Interf dB':>11s}{'Jain':>8s}{'SE':>9s}{'MinSE':>9s}")
    for m in base.METHODS:
        g = lambda k: np.mean([r[m][k] for r in all_res])
        print(f"{m:9s}{g('sinr_db'):10.2f}{g('mean_interf_db'):11.2f}{g('jain'):8.3f}"
              f"{g('se'):9.3f}{g('min_se'):9.3f}")
    print(f"\nQAOA mean P(optimal): {np.mean([e['p_opt'] for e in all_extra]):.4f}  "
          f"(uniform guess {all_extra[0]['p_uniform']:.4f})")
    print(f"QAOA mean P(feasible): {np.mean([e['p_feas'] for e in all_extra]):.3f}")
    print(f"QAOA found the exact optimum in "
          f"{sum(e['qaoa_matches_opt'] for e in all_extra)}/{len(snaps)} snapshots")

    csv_path = out_dir / RESULTS_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Per-snapshot results saved to {csv_path}")

    # ---------- figures: drawn by plotter.py from the CSV that was just stored ----------
    # (the data comes back from the file, not from the variables of this script)
    paths, _ = plotter.plot_from_csv(csv_path, out_dir=out_dir, close=False)
    for path in paths:
        print(f"Figure saved to {path}")
    plt.show()