"""Run beam assignment on the supplied 200-row, 2-set, 10-scenario workbook.

Rows 1..100: Set 1, ten groups of ten source users (scenarios).
Rows 101..200: Set 2, same ten scenarios and ten source users each.
Each group of 10 rows forms ONE multiuser beam-assignment snapshot; therefore
there are 20 optimisation runs, or TWO snapshot-level metric observations per
scenario, each representing ten source rows. This preserves the RF assignment
problem's multiuser nature without incorrectly treating individual users as
independent RF-assignment runs.

Files per run:
- qaoa_dataset_results.csv: 20 snapshot x 4 method rows = 80 result records.
- qaoa_scenario_means.csv: 10 scenario x 4 method = 40 mean records; each
  scenario mean combines TWO snapshot results covering 20 input rows.
- Exactly three plots by plotter.py: linear gain, linear interference, fairness.

The supplied channels.py computes path loss, shadow fading and channel rays
using each dataset row's actual distance, angle, frequency and antenna heights.
The simplified fallback is used only if channels.py is unavailable.

Example (quiet, smallest PennyLane benchmark):
    python qaoa_beam_assignment_datasets.py --users 4 --beams 3 --rf 2 --yes
"""

import argparse
import sys
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd

import qaoa_beam_assignment as base
import dataset_channels as dc
import plotter
from qubo_formulation import (LAMBDA_G, LAMBDA_I, LAMBDA_F,
                               LAMBDA_U, LAMBDA_B, LAMBDA_R)

DATA_FILE = Path(__file__).with_name("channel_test_suiteupdated.xlsx")
DATA_SELECTION = "all"
SNAPSHOTS_PER_SET = 10
SNAPSHOT_SIZE = 10
POOL_MODE = "mixed"
CODEBOOK = "polar"
CODEBOOK_OVERSAMPLE = 2
CAND_RHO = 0.7
SHADOWING = True
PATHLOSS_SOURCE = "computed"
TX_DBM, BW_HZ, NF_DB = 30.0, 100e6, 9.0
MASTER_SEED = 0
RESULTS_CSV = "qaoa_dataset_results.csv"
MEANS_CSV = "qaoa_scenario_means.csv"
MAX_QUBITS = 20


def retries_for(qubits):
    return 4 if qubits <= 12 else 3 if qubits <= 18 else 2


def check_choice(users, beams, rf):
    if min(users, beams, rf) < 1:
        return False, "users, beams and RF chains must each be >= 1"
    if users > SNAPSHOT_SIZE:
        return False, f"At most {SNAPSHOT_SIZE} users are available in one scenario block"
    if rf > users:
        return False, "RF chains cannot exceed number of users"
    if users * beams > MAX_QUBITS:
        return False, f"PennyLane circuit limited to {MAX_QUBITS} qubits"
    return True, ""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users", type=int, default=None, help="number of users (interactive if omitted)")
    p.add_argument("--beams", type=int, default=None, help="candidate beams per user (interactive if omitted)")
    p.add_argument("--rf", type=int, default=None, help="number of RF chains (interactive if omitted)")
    p.add_argument("--set", dest="set_", default="all", help='all | "Set 1" | "Set 2"')
    p.add_argument("--snapshots", type=int, default=SNAPSHOTS_PER_SET,
                   help="number of sequential scenario blocks per set, maximum 10")
    p.add_argument("--channel-model", choices=("auto", "external", "simplified"), default="auto")
    p.add_argument("--codebook", choices=("dft", "polar"), default=CODEBOOK)
    p.add_argument("--dataset", type=Path, default=DATA_FILE)
    p.add_argument("--yes", action="store_true", help="run without interactive confirmation")
    p.add_argument("--no-plot", action="store_true", help="save CSV files only; plotter can run later")
    return p.parse_args()


def build_snapshots(selection, per_set, dataset=DATA_FILE, channel_model="auto"):
    """Preserve the workbook's scenario order, grouping ten user rows per snapshot."""
    if per_set < 1 or per_set > 10:
        raise ValueError("--snapshots must be from 1 to 10")
    df = dc.load_dataset(dataset)
    if "instance_id" in df.columns:
        df = df.sort_values("instance_id", kind="stable")
    df = df.reset_index(drop=True)
    names = list(dict.fromkeys(df["set"]))
    if selection != "all":
        selected = [name for name in names if str(name).replace(" ", "").lower() ==
                    str(selection).replace(" ", "").lower()]
        if not selected:
            raise ValueError(f"Unknown set '{selection}'; available: {names}")
        names = selected
    rng = np.random.default_rng(MASTER_SEED)
    snapshots = []
    for name in names:
        rows = df[df["set"] == name].reset_index(drop=True)
        if len(rows) % SNAPSHOT_SIZE:
            raise ValueError(f"{name} has {len(rows)} rows, not divisible by {SNAPSHOT_SIZE}")
        blocks = len(rows) // SNAPSHOT_SIZE
        for j in range(min(per_set, blocks)):
            batch = rows.iloc[j*SNAPSHOT_SIZE:(j+1)*SNAPSHOT_SIZE].copy()
            scen_names = batch["scenario"].unique()
            if len(scen_names) != 1:
                raise ValueError(f"{name} scenario block {j + 1} mixes scenarios: {scen_names}")
            if batch["N"].nunique() != 1 or batch["fc_GHz"].nunique() != 1:
                raise ValueError("Within-scenario antenna/frequency mismatch")
            N, fc = int(batch["N"].iloc[0]), float(batch["fc_GHz"].iloc[0])
            H, info = dc.generate_channels_from_rows(batch, N, rng,
                n_chains=base.N_RF_SUB, tx_dbm=TX_DBM, bw_hz=BW_HZ, nf_db=NF_DB,
                shadowing=SHADOWING, pathloss_source=PATHLOSS_SOURCE, model=channel_model)
            snapshots.append(dict(index=len(snapshots), set=name, scenario=str(scen_names[0]),
                                  scenario_order=j+1, N=N, fc=fc, H=H, info=info,
                                  row_count=len(batch), rows=batch))
    return snapshots


def select_pool(G, sin_phi, k_sub, mode, rng):
    K = G.shape[0]
    if k_sub >= K:
        return np.arange(K)
    if mode == "random":
        return rng.choice(K, size=k_sub, replace=False)
    anchor = int(np.argmax(G.max(1)))
    near_order = np.argsort(abs(sin_phi - sin_phi[anchor]))
    if mode == "crowded":
        return near_order[:k_sub]
    n_near = int(np.ceil(k_sub/2))
    near = list(map(int, near_order[:n_near]))
    far = [int(u) for u in near_order[::-1] if int(u) not in near]
    return np.asarray(near + far[:k_sub-n_near], int)


def candidate_beams_distinct(G, F, users, m, rho):
    choices = []
    for u in users:
        beam_ids = []
        for b in np.argsort(-G[u]):
            b = int(b)
            if all(abs(np.vdot(F[:, c], F[:, b])) < rho for c in beam_ids):
                beam_ids.append(b)
            if len(beam_ids) == m:
                break
        if len(beam_ids) < m:
            beam_ids.extend(int(b) for b in np.argsort(-G[u])
                            if int(b) not in beam_ids and len(beam_ids) < m)
        if len(beam_ids) != m:
            raise ValueError("Codebook has fewer distinct beams than requested")
        choices.append(beam_ids)
    return np.asarray(choices, int)



def max_candidate_matching(cand):
    """Maximum matching: one unique global beam per selected user.

    Returns a dictionary {local_user_index: global_beam_id}.  A feasible
    R-chain assignment exists iff len(result) >= R.
    """
    cand = np.asarray(cand, dtype=int)
    if cand.ndim != 2:
        raise ValueError("Candidate beam array must be two-dimensional")
    beam_to_user = {}

    def augment(user, seen):
        for beam in cand[user]:
            beam = int(beam)
            if beam in seen:
                continue
            seen.add(beam)
            if beam not in beam_to_user or augment(beam_to_user[beam], seen):
                beam_to_user[beam] = user
                return True
        return False

    for user in range(len(cand)):
        augment(user, set())
    return {user: beam for beam, user in beam_to_user.items()}


def ensure_feasible_candidates(G, users, cand, n_rf):
    """Minimally augment candidate lists to permit an R-chain assignment.

    Per-user shortlist generation sometimes picks the same few popular beam
    IDs for many users.  With more RF chains than unique user-beam matches,
    the QUBO has no feasible state.  Keep existing shortlisted choices where
    possible. Add one currently unused high-gain global beam to an unmatched
    user at a time, replacing that user's lowest-priority shortlisted beam.
    The QUBO objective and its weights are not modified.

    Returns (repaired_candidates, count_of_replacements).
    """
    gains = np.asarray(G, dtype=float)
    users = np.asarray(users, dtype=int)
    result = np.array(cand, dtype=int, copy=True)
    K, M = result.shape
    B = gains.shape[1]
    if n_rf > K:
        raise ValueError(f"{n_rf} RF chains require at least {n_rf} users; got {K}")
    if n_rf > B:
        raise ValueError(f"{n_rf} distinct RF beams requested, but the codebook has only {B} beams")
    if M > B:
        raise ValueError(f"{M} distinct candidates per user require at least {M} codebook beams")
    changed = 0
    matching = max_candidate_matching(result)
    while len(matching) < n_rf:
        used = set(matching.values())
        unmatched = [k for k in range(K) if k not in matching]
        if not unmatched:
            raise RuntimeError("Candidate matching repair reached unexpected full matching")
        # Prefer highest available power among not-yet-matched users and beams;
        # does not disturb any already matched user.
        replacements = [(float(gains[users[k], b]), k, b)
                        for k in unmatched for b in range(B)
                        if b not in used and b not in result[k]]
        if not replacements:
            raise ValueError("Cannot repair candidate shortlists: insufficient distinct beams")
        _, k, beam = max(replacements)
        result[k, -1] = beam
        changed += 1
        new_matching = max_candidate_matching(result)
        if len(new_matching) <= len(matching):
            raise RuntimeError("Candidate repair failed to increase unique user-beam matching")
        matching = new_matching
    return result, changed


_CODEBOOKS = {}

def codebook(N, fc, mode="auto", kind=CODEBOOK):
    key = (N, fc, mode, kind, CODEBOOK_OVERSAMPLE)
    if key not in _CODEBOOKS:
        _CODEBOOKS[key] = dc.get_codebook(kind, N, fc, CODEBOOK_OVERSAMPLE, mode=mode)
    return _CODEBOOKS[key]


def make_build_subproblem(snaps, codebook_kind="polar"):
    def build_subproblem(i):
        s = snaps[i]
        F = codebook(s["N"], s["fc"], mode=s["info"]["channel_model"], kind=codebook_kind)
        H = np.asarray(s["H"])
        G = abs(H @ F)**2
        rng = np.random.default_rng(MASTER_SEED + i)
        users = select_pool(G, s["info"]["sin_los"], base.K_SUB, POOL_MODE, rng)
        cand = candidate_beams_distinct(G, F, users, base.M_CAND, CAND_RHO)
        cand, repaired = ensure_feasible_candidates(G, users, cand, base.N_RF_SUB)
        if repaired:
            print(f"   Candidate feasibility repair: replaced {repaired} "
                  f"beam shortlist entr{'y' if repaired == 1 else 'ies'} "
                  f"to support {base.N_RF_SUB} RF chains.")
        return H, F, G, users, cand
    return build_subproblem


def aggregate_scenario_means(snapshot_results, out_path=None):
    """Write the two-set scenario means using stored snapshot metric values.

    10 rows from each set -> one RF assignment snapshot per set. Each scenario
    therefore averages two per-snapshot results and represents 20 input rows.
    """
    df = pd.read_csv(snapshot_results) if isinstance(snapshot_results, (str, Path)) else snapshot_results.copy()
    cols = ("gain_linear", "interference_linear", "fairness")
    required = {"scenario_order", "scenario", "method", "set", "source_rows", *cols}
    if required - set(df.columns):
        raise ValueError(f"Missing snapshot CSV columns: {sorted(required-set(df.columns))}")
    groups = ["scenario_order", "scenario", "method"]
    # The ONLY aggregation (arithmetic means) happens here, NEVER in plotter.py.
    averages = df.groupby(groups, sort=False, as_index=False).agg(
        gain_linear=("gain_linear", "mean"),
        interference_linear=("interference_linear", "mean"),
        fairness=("fairness", "mean"),
        snapshot_count=("snapshot", "nunique"),
        source_rows=("source_rows", "sum"),
        sets_combined=("set", lambda s: " | ".join(dict.fromkeys(s))),
    ).sort_values(["scenario_order", "method"], kind="stable")
    if out_path is not None:
        averages.to_csv(out_path, index=False)
    return averages


def get_experiment_parameters(args):
    """Ask for missing values, or use defaults in non-interactive / --yes mode."""
    defaults = {"users": 4, "beams": 3, "rf": 2}
    labels = {
        "users": "Number of users",
        "beams": "Number of candidate beams per user",
        "rf": "Number of RF chains",
    }
    interactive = sys.stdin.isatty() and not args.yes
    for key in ("users", "beams", "rf"):
        if getattr(args, key) is not None:
            continue
        if not interactive:
            setattr(args, key, defaults[key])
            continue
        while True:
            raw = input(f"{labels[key]} [{defaults[key]}]: ").strip()
            try:
                value = defaults[key] if not raw else int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue
            if value < 1:
                print("Please enter a positive whole number.")
                continue
            if key == "users" and value > SNAPSHOT_SIZE:
                print(f"The dataset supports at most {SNAPSHOT_SIZE} users per snapshot.")
                continue
            if key == "beams" and value * args.users > MAX_QUBITS:
                print(f"The selected number of users allows at most {MAX_QUBITS // args.users} "
                      "beams per user (qubit limit).")
                continue
            if key == "rf" and value > args.users:
                print("RF chains cannot exceed the number of users.")
                continue
            setattr(args, key, value)
            break
    return args


def main():
    args = get_experiment_parameters(parse_args())
    ok, msg = check_choice(args.users, args.beams, args.rf)
    if not ok:
        sys.exit(msg)
    # PennyLane only, with explicit option to lower execution cost.
    base.K_SUB, base.M_CAND, base.N_RF_SUB = args.users, args.beams, args.rf
    base.N_RESTARTS = retries_for(args.users * args.beams)
    base.RUN_TAG = f"{args.users}u{args.beams}b{args.rf}rf"
    if not args.yes:
        q = args.users * args.beams
        print(f"QAOA: {q} qubits, {comb(q, args.rf)} fixed-weight states, "
              f"{base.N_RESTARTS} restarts, {args.snapshots} scenario snapshots per set")
        answer = input("Proceed? [Y/n]: ").strip().lower()
        if answer not in ("", "y", "yes"):
            return
    snaps = build_snapshots(args.set_, args.snapshots, args.dataset, args.channel_model)
    if not snaps:
        raise ValueError("No snapshots selected")
    base.build_subproblem = make_build_subproblem(snaps, args.codebook)
    output = base.run_dir()
    print(f"\n{len(snaps)} multiuser snapshots | output: {output}")
    with (output / "run_info.txt").open("w") as file:
        file.write(f"dataset: {args.dataset}\n"
                   f"source rows selected: {sum(s['row_count'] for s in snaps)}\n"
                   f"snapshots: {len(snaps)}; 10 user rows per snapshot\n"
                   f"channel model requested: {args.channel_model}\n"
                   f"channel model actually used: " +
                   ", ".join(dict.fromkeys(s["info"]["channel_model"] for s in snaps)) + "\n"
                   f"original workbook lacks R, path_loss_dB, shadow_fading_sigma_dB; "
                   f"simplified mode is NOT the original 3GPP generator\n"
                   f"subproblem: {args.users} users, {args.beams} beams, {args.rf} RF chains\n"
                   f"QUBO coefficients: gain={LAMBDA_G}, interference={LAMBDA_I}, "
                   f"fairness={LAMBDA_F}, user={LAMBDA_U}, beam={LAMBDA_B}, RF={LAMBDA_R}\n"
                   f"fairness plot is unweighted\n"
                   f"optimiser: PennyLane, depth={base.P_DEPTH}, restarts={base.N_RESTARTS}\n"
                   f"measurement shots: {base.SHOTS}\n"
                   f"candidate shortlist matching feasibility repair: enabled\n")
    rows, diagnostics = [], []
    for snap in snaps:
        i = snap["index"]
        print(f"\nSnapshot {i+1}/{len(snaps)}: {snap['set']} / {snap['scenario']}")
        results, extra = base.run_seed(i)
        diagnostics.append({"snapshot": i+1, "set": snap["set"], "scenario": snap["scenario"],
                            "p_opt": extra["p_opt"], "p_feas": extra["p_feas"],
                            "matches_ilp": extra["qaoa_matches_opt"],
                            "penalties_sufficient": extra["penalties_sufficient"]})
        for method in base.METHODS:
            r = results[method]
            base.print_rf_table(method, r["table"])
            print(f"   gain={r['gain_linear']:.6g} | interference={r['interference_linear']:.6g} "
                  f"| fairness={r['fairness']:.6g}")
            rows.append(dict(snapshot=i+1, set=snap["set"], scenario=snap["scenario"],
                             scenario_order=snap["scenario_order"],
                             source_rows=snap["row_count"], model=snap["info"]["channel_model"],
                             N=snap["N"], users=args.users, beams=args.beams,
                             rf_chains=args.rf, qubits=args.users*args.beams,
                             method=method, gain_linear=r["gain_linear"],
                             interference_linear=r["interference_linear"], fairness=r["fairness"]))
        if not extra["penalties_sufficient"]:
            print("   Warning: fixed QUBO constraint penalties do NOT make the unrestricted "
                  "weight-R ground state feasible for this snapshot; feasibility is enforced "
                  "during selection/ILP, not by changing the specified weights.")
    individual = output / RESULTS_CSV
    pd.DataFrame(rows).to_csv(individual, index=False)
    meanfile = output / MEANS_CSV
    means = aggregate_scenario_means(individual, meanfile)
    pd.DataFrame(diagnostics).to_csv(output / "qaoa_diagnostics.csv", index=False)
    print(f"\nSaved {len(rows)} snapshot x method records: {individual}")
    print(f"Saved {len(means)} scenario x method mean records: {meanfile}")
    if not args.no_plot:
        paths, _ = plotter.plot_from_csv(meanfile, out_dir=output)
        print("Three figures (PDF and PNG each):")
        for p in paths:
            print(" ", p)
    print("Channel model:", ", ".join(dict.fromkeys(s["info"]["channel_model"] for s in snaps)))


if __name__ == "__main__":
    main()
