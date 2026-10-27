"""Reproducible classical checks independent of the PennyLane installation."""
import itertools
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import pandas as pd

# This stub merely allows importing the classical QUBO/ILP functions without
# PennyLane installed. IT IS NEVER USED TO GENERATE QAOA RESULTS.
sys.modules.setdefault("qaoa_pennylane", types.SimpleNamespace(MAX_QUBITS=20))
from qubo_formulation import build_qubo_matrix, fairness_matrix, physical_components
from step1_greedy import greedy_assignment
from step1_omp import omp_assignment
import qaoa_beam_assignment as base
import qaoa_beam_assignment_datasets as datasets
import plotter


def check_exact_coefficients():
    rng = np.random.default_rng(42)
    g = abs(rng.normal(size=(3, 3)))**2
    q, offset = build_qubo_matrix(g, 2)
    expected = np.zeros((9, 9))
    def pair(i, j, coefficient):
        expected[i, j] += coefficient/2
        expected[j, i] += coefficient/2
    for k in range(3):
        for m in range(3):
            i = 3*k+m
            expected[i, i] += -g[k, m] + 0.2*.01*(g[k,m]-g.mean())**2
            for n in range(m+1, 3):
                pair(i, 3*k+n, 6)
    for k,j in itertools.combinations(range(3), 2):
        for m in range(3):
            for n in range(3):
                pair(3*k+m, 3*j+n, .5*(g[k,n]+g[j,m]))
                if m==n:
                    pair(3*k+m, 3*j+n, 6)
    for i in range(9):
        expected[i,i] += 6*(1-4)
        for j in range(i+1,9):
            pair(i,j,12)
    np.testing.assert_allclose(q, expected, atol=1e-12)
    assert offset == 0


def check_methods():
    rng = np.random.default_rng(12)
    H = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    F = rng.normal(size=(5, 10)) + 1j * rng.normal(size=(5, 10))
    G = abs(H@F)**2
    users = np.arange(4)
    candidate = np.array([[1,2,3], [2,4,5], [1,6,7], [3,5,8]])
    base.N_RF_SUB = 2
    Q, offset = base.build_qubo(G, users, candidate)
    h,J,c=base.ising_from_qubo(Q)
    base.verify_ising(Q,h,J,c)
    states, E = base.subspace_energies(Q, offset, 12, 2)
    best = min(E[i] for i, st in enumerate(states)
               if base.is_feasible(base.decode(st, users, candidate)))
    ilp, objective, _ = base.ilp_assignment(G,users,candidate,2)
    assert abs(best-objective) < 1e-7
    candidate_gain=base.candidate_gain_matrix(G,users,candidate)
    gr,_=greedy_assignment(G,2)
    om,_=omp_assignment(H,F,2)
    for a in (gr,om,ilp):
        assert base.is_feasible(a,2)
        metrics=base.evaluate(a,H,F,G,users,candidate)
        chosen=list(a)
        np.testing.assert_allclose(metrics["gain_linear"],sum(G[k,b] for k,b in chosen))
        np.testing.assert_allclose(metrics["fairness"],sum(
            0.01*(G[k,b]-candidate_gain.mean())**2 for k,b in chosen))
    assert all(b in candidate[k] for k,b in ilp)
    physical=physical_components([(k,int(np.where(candidate[k]==b)[0][0])) for k,b in ilp],
             candidate_gain,candidate_beams=candidate,cross_gains=G)
    metrics=base.evaluate(ilp,H,F,G,users,candidate)
    for m in ("gain_linear","interference_linear","fairness"):
        np.testing.assert_allclose(metrics[m],physical[m])


def check_spreadsheet_and_plots():
    df=datasets.dc.load_dataset(datasets.DATA_FILE)
    assert len(df) == 200
    assert len(df["set"].unique()) == 2
    assert list(dict.fromkeys(df["scenario"])) == [
        "UMi-LOS", "UMi-NLOS", "UMa-LOS", "UMa-NLOS", "InH-LOS", "InH-NLOS",
        "InF-LOS", "InF-NLOS", "NearField", "SNS"]
    # Use stored sample values (no simulations) to check the aggregation.
    rows=[]
    for set_idx in range(2):
        for i, scenario in enumerate(dict.fromkeys(df["scenario"])):
            for name in datasets.base.METHODS:
                rows.append(dict(snapshot=10*set_idx+i+1,set=f"Set {set_idx+1}",
                    scenario=scenario,scenario_order=i+1,source_rows=10,method=name,
                    gain_linear=10*set_idx+i, interference_linear=2*set_idx+i,
                    fairness=3*set_idx+i))
    with tempfile.TemporaryDirectory() as temp:
        mean_path=Path(temp)/"means.csv"
        means=datasets.aggregate_scenario_means(pd.DataFrame(rows),mean_path)
        assert len(means)==40 and set(means["source_rows"])=={20}
        assert set(means["snapshot_count"])=={2}
        first=means[(means.scenario=="UMi-LOS")&(means.method=="OMP")].iloc[0]
        assert first["gain_linear"]==5 and first["interference_linear"]==1
        paths,_=plotter.plot_from_csv(mean_path,out_dir=temp)
        assert len(paths)==6 and all(p.exists() for p in paths)


if __name__ == "__main__":
    check_exact_coefficients()
    check_methods()
    check_spreadsheet_and_plots()
    print("PASS: exact QUBO, Ising mapping, ILP optimum, original Greedy/OMP, 200-row dataset, scenario aggregation, 3 figures.")
    print("NOTE: Real PennyLane circuit execution requires the installed package.")
