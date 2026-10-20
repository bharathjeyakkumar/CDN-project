"""
qaoa_beam_assignment_dataset.py -- the same QAOA / Greedy / OMP comparison as qaoa_beam_assignment.py,
but the channels come from channel_test_suite_organized.xlsx (via dataset_channels.py).

Needs, in the same folder:
    qaoa_beam_assignment.py   (the QUBO, QAOA circuit, metrics and plots; nothing in it is duplicated here)
    dataset_channels.py       (spreadsheet loader + channel generation)
    your channel file         (defines SCEN, C0, path_loss_db, _lsp, _nf_user, _ff_user; see CHANNEL_MODULE)
    qaoa_pennylane.py, beam_dictionary.py, step1_greedy.py, step1_omp.py, compute_sinr.py
    channel_test_suite_organized.xlsx

What one run does
-----------------
1. Take the chosen set(s): DATA_SELECTION = "all" or "Set 1" / "Set 2" / "Set 3".
2. For each set draw SNAPSHOTS_PER_SET snapshots. A snapshot = K_DROP users drawn at random from that
   set (scenarios are mixed). All users of a snapshot share the set's antenna count N (64 or 128).
3. Generate the noise-normalised channels from the spreadsheet rows.
4. From the K_DROP users pick the QAOA subproblem (K_SUB users, N_RF_SUB RF chains, 2 candidate
   beams each, 12 qubits with the default preset). Greedy, OMP, QAOA and the exact optimum all
   solve exactly this subproblem.
5. Evaluate, average over all snapshots, save the three graphs and a CSV.

Units: the channel is noise-normalised (noise power = 1, transmit power already inside H).
So the interference graph is interference-to-noise ratio in dB: 0 dB means interference equals the
noise, lower is better, and below about -10 dB interference is negligible.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qaoa_beam_assignment as base
import dataset_channels as dc
from beam_dictionary import generate_beam_dictionary
from channels import SCEN, C0, path_loss_db, _lsp, _nf_user, _ff_user

# ======================================================
# CONFIGURATION
# ======================================================

DATA_FILE = "channel_test_suite_organized.xlsx"
DATA_SELECTION = "all"        # "all", "Set 1", "Set 2" or "Set 3"
SNAPSHOTS_PER_SET = 10        # snapshots drawn per set ("all" = 3 sets -> 30 snapshots)
K_DROP = 8                    # users drawn per snapshot (must be >= base.K_SUB)
POOL_MODE = "mixed"           # which K_SUB of the K_DROP users form the subproblem:
                              #   "mixed"  : strongest user + its nearest-angle neighbours, rest far away in angle
                              #   "crowded": strongest user + nearest-angle neighbours only
                              #   "random" : random users
SHADOWING = True              # spreadsheet path loss has no shadow fading, so add it (row sigma)
PATHLOSS_SOURCE = "dataset"   # "dataset" (spreadsheet column) or "model" (your path_loss_db())
TX_DBM, BW_HZ, NF_DB = 30.0, 100e6, 9.0
MASTER_SEED = 0
OVERSAMPLING = 1              # codebook size = OVERSAMPLING * N beams
RESULTS_CSV = "qaoa_dataset_results.csv"


# ======================================================
# SNAPSHOTS
# ======================================================

def build_snapshots():
    rng = np.random.default_rng(MASTER_SEED)
    df = dc.load_dataset(DATA_FILE)
    sets = dc.choose_sets(df, DATA_SELECTION)
    snaps = dc.draw_snapshots(sets, SNAPSHOTS_PER_SET, K_DROP, rng)
    for i, s in enumerate(snaps):
        H, info = dc.generate_channels_from_rows(
            s["rows"], s["N"], rng, n_chains=base.N_RF_SUB, tx_dbm=TX_DBM, bw_hz=BW_HZ,
            nf_db=NF_DB, shadowing=SHADOWING, pathloss_source=PATHLOSS_SOURCE)
        s.update(H=H, info=info, index=i)
    return snaps


def select_pool(G, sin_phi, k_sub, mode, rng):
    """Choose the K_SUB users of the subproblem. The same pool is given to every method."""
    K = G.shape[0]
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


_BEAMS = {}


def beam_dictionary(N):
    if N not in _BEAMS:
        _BEAMS[N] = generate_beam_dictionary(N, OVERSAMPLING, 0.5)[0]
    return _BEAMS[N]


def make_build_subproblem(snaps):
    """Replacement for base.build_subproblem: 'seed' is the snapshot index."""
    def build_subproblem(i):
        s = snaps[i]
        beam_dict = beam_dictionary(s["N"])
        H = s["H"]
        G = np.abs(H @ beam_dict) ** 2                            # (K_DROP, n_beams)
        rng = np.random.default_rng(MASTER_SEED + i)
        users = select_pool(G, s["info"]["sin_los"], base.K_SUB, POOL_MODE, rng)
        cand = base.candidate_beams(G, users, base.M_CAND, base.CAND_MIN_SEP)
        return [h for h in H], beam_dict, G, np.asarray(users), cand
    return build_subproblem


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    if K_DROP < base.K_SUB:
        raise ValueError(f"K_DROP ({K_DROP}) must be at least K_SUB ({base.K_SUB})")

    snaps = build_snapshots()

    # noise-normalised channel: noise power 1, transmit power already inside H
    base.NOISE_POWER = 1.0
    base.TX_POWER_PER_STREAM_W = 1.0
    base.build_subproblem = make_build_subproblem(snaps)
    base.OUTPUT_TAG = "_dataset"
    base.RUN_LABEL = "snapshots"
    base.INTERF_YLABEL = "Interference-to-noise ratio (dB)"
    base.INTERF_TITLE = "Mean interference per user vs noise (dB, 0 = noise level), lower is better"

    print(f"Data: {DATA_FILE} | selection: {DATA_SELECTION} | {len(snaps)} snapshots of "
          f"{K_DROP} users | pool: {POOL_MODE} | shadowing: {SHADOWING}")
    print(f"Backend: {base.BACKEND} | subproblem: {base.K_SUB} users x {base.M_CAND} beams, "
          f"{base.N_RF_SUB} RF chains | p = {base.P_DEPTH}")

    all_res, all_extra, rows = [], [], []
    for i, s in enumerate(snaps):
        print("\n" + "=" * 60)
        print(f"SNAPSHOT {i}: {s['set']} (N = {s['N']} antennas, R = {s['R']} RF chains in the dataset)")
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
            rows.append(dict(snapshot=i, set=s["set"], N=s["N"], method=name,
                             sinr_db=r["sinr_db"], interference_db=r["mean_interf_db"],
                             jain=r["jain"], sum_se=r["se"], min_user_se=r["min_se"],
                             scenarios="|".join(s["info"]["scenario"])))

    # ---------- summary ----------
    print("\n" + "=" * 70)
    print(f"SUMMARY (averaged over {len(snaps)} snapshots)")
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

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"Per-snapshot results saved to {RESULTS_CSV}")

    base.plot_three(all_res)
    plt.show()