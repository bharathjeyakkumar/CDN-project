# Beam assignment: exact weighted QUBO + PennyLane XY-QAOA

## What changed

- The gain/interference/fairness/constraints matrix is implemented in
  `qubo_formulation.py`, preserving **exactly** the source formula and default
  weights λG=1, λI=0.5, λF=0.2, λU=6, λB=6, λR=6. The constant
  λR × RF² is omitted, exactly as in the supplied reference code.
- `qaoa_beam_assignment.py` runs **PennyLane only**; it does not import Qiskit,
  use an alternate NumPy statevector QAOA, calculate SINR, or use a
  log-SINR objective. `qaoa_pennylane.py` retains PennyLane's own optional
  state-preparation method, which is not an alternate quantum backend.
- QAOA-XY and ILP use per-user candidate beam lists, while the **original**
  Greedy and OMP implementations retain their full-codebook search and original
  scoring. This is a difference in search spaces.
- Three metrics (all **unweighted** and **linear-power**):
  - **Gain**: sum of selected signal powers, higher is better.
  - **Interference**: sum over *unordered selected user pairs* of both
    cross-user powers, lower is better.
  - **Fairness penalty**: sum of the selected `0.01 * (power - mean_candidate_power)²`,
    lower is better. **λF multiplies this only inside the optimization objective.**
  - `mean_candidate_power` is the mean of the K × M candidate power matrix
    in that subproblem, not the mean of the whole large codebook.
- The plotter reads saved, precomputed scenario means and creates **exactly
  3 graphs**, as both PNG and PDF.

## The 200-source-row / 10-plot-point layout

The supplied workbook has 2 sets, each with 10 scenarios × 10 input rows.

- Set 1: rows 1–10 = UMi-LOS; 11–20 = UMi-NLOS; ...; 91–100 = SNS.
- Set 2: rows 101–110 = UMi-LOS; 111–120 = UMi-NLOS; ...; 191–200 = SNS.
- Each 10-user group is **one multiuser RF-assignment snapshot**. Running
  algorithms once per user row would not represent the requested multiuser RF
  assignment, so there are 20 snapshot-level solver runs total: one for each
  scenario in each set.
- Every scenario-level graph point is the arithmetic mean of the **two
  snapshot-level metrics**, i.e. results computed with all **20 source rows**
  for that scenario. This is *not* 20 independent optimizer outputs.
- `qaoa_dataset_results.csv`: 20 snapshots × 4 algorithms = 80 records.
- `qaoa_scenario_means.csv`: 10 scenarios × 4 algorithms = 40 records; each
  record includes `source_rows = 20` and `snapshot_count = 2` for a full run.
- `qaoa_diagnostics.csv`: QAOA feasibility and optimum diagnostic columns.

## Original channel model integrated (updated)

`channels.py` is now bundled, copied from the source provided by the user.
The default `--channel-model auto` selects **external** and invokes its
`_ff_user`, `_nf_user`, `_lsp`, `path_loss_db`, `dft_codebook` and
`polar_codebook` as appropriate. The adapter takes the scenario, frequency,
antenna heights, *actual distance*, and *actual LOS azimuth sine* **from each
of the 200 workbook rows**. These row-specific values replace the random user
drops that `channels.generate_channels` would ordinarily draw. The original
channel LSPs and stochastic scattering are retained; shadow standard deviation
comes from `_lsp` when it is not present in the spreadsheet.

`channels.py` uses per-user transmit power `P_tx / K` where `K` is the number
of users in the channel generation snapshot (10 here). The adapter retains
this scaling, so its output is consistent with the source channel model.
No path-loss or shadow-fading input columns need to be added to the workbook:
`channels.py` computes them based on the scenario and geometry.

For an explicit no-fallback run:

```bash
python qaoa_beam_assignment_datasets.py --channel-model external --users 4 --beams 3 --rf 2 --yes
```

The run metadata and CSV `model` field should now display `external`.
`--channel-model simplified` remains available only for deliberately
illustrative tests. The supplied channel model is a **reduced 3GPP-style
approximation** as documented by its author, not a fully validated TR 38.901
implementation.

The PennyLane operator consistency check also now uses absolute and relative
numerical tolerances, preventing harmless floating-point differences on
high-dynamic-range channel snapshots from ending the run.

## Setup / run

Put the entire zip contents in the **same directory**. Python 3.10+ recommended:

```bash
python -m pip install -r requirements.txt
python qaoa_beam_assignment_datasets.py --users 4 --beams 3 --rf 2 --yes
```

Full run solves 20 multiuser snapshots and saves its CSV files and three
comparison graphs into a new `results/run_.../` directory. PennyLane
simulation can be computationally expensive. For a short smoke test:

```bash
python qaoa_beam_assignment_datasets.py --users 3 --beams 2 --rf 2 --set "Set 1" --snapshots 1 --yes
```

To regenerate figures **without rerunning optimization**:

```bash
python plotter.py results/run_.../qaoa_scenario_means.csv
```

To run source-QUBO demonstration:

```bash
python qubo_formulation.py
```

Optional smoke tests, which test all the non-quantum parts and do **not** need
PennyLane installed:

```bash
python test_classical.py
```

## Verification boundary

QUBO coefficients were compared term-for-term with the supplied formula.
ILP was checked against exact feasible-state enumeration, and Greedy/OMP were
checked for candidate/constraint feasibility. The 200-row layout, source-row
counts, 10-scenario averaging, and three-chart generation were checked.

**The actual PennyLane circuit/optimizer has not been executed in this
workspace because PennyLane is not installed and its package index is not
reachable.** Run on a machine with PennyLane to validate quantum performance
and circuit statistics. No fabricated QAOA output figures are included.

Note: the fixed penalty weights λU=λB=λR=6 may not force the *unconstrained*
QUBO ground state to be feasible for all raw channel powers. They are **not**
automatically adjusted. The code flags such cases and always enforces hard
feasibility in reported assignments. If physical gain scales are very large,
a future science-driven penalty tuning experiment may be needed.

## Legacy Greedy / OMP restored (this package)

`step1_greedy.py` and `step1_omp.py` contain the **original** scoring,
interference-index conditions, residual update, and selection loops from your
initial uploads. Only the Greedy module's demo-only `step0_gain_matrix` import
was moved under `if __name__ == "__main__"` so production imports work without
old optional beam-dictionary/channel-model dependencies.

**Scientific comparability warning:** Original Greedy/OMP operate over the
**full codebook**, while QAOA-XY/ILP operate over the user-specific candidate
beam shortlists (`--beams`). Therefore they solve different feasible search
spaces. This is intentional to preserve the original baselines, but differences
in their metrics cannot be attributed solely to algorithm quality.
All four methods are evaluated with the same gain, cross-user leakage, and
unweighted fairness metric (fairness reference mean is the QAOA candidate-gain
matrix). The original Greedy/OMP's restrictive beam-index interference
condition is **not fixed**, as requested.

## Candidate feasibility safeguard and 1024 shots

- `SHOTS = 1024` in `qaoa_beam_assignment.py`. QAOA reporting still chooses
  the best feasible state among sampled outcomes; this is *best-of-1024*, not
  a single-shot success rate.
- At user/beam/RF configurations with overlapping shortlists, it is possible
  that no assignment uses `R` different users and `R` different beams. This
  previously stopped the experiment with `No feasible assignment`.
- Dataset generation now checks the **maximum bipartite matching** between
  local users and candidate beam IDs. If fewer than `R` users can be matched,
  the runner adds an unused high-power beam for an unmatched user, replacing
  that user's final shortlist position. It repeats until `R` matches exist.
  This ensures the QAOA and ILP domains remain feasible without modifying the
  weighted QUBO or the original Greedy/OMP algorithms.
- Each repaired snapshot prints the count of replaced shortlist entries.
  The repair changes that snapshot's allowed QAOA/ILP candidates and should
  be documented if reporting comparative results.
- A hard error is still produced if RF chains exceed users or distinct global
  codebook beams, since no valid RF assignment exists in that situation.

## NEW: Individual source-row results and true /20 scenario averages

After a successful full 20-snapshot run, the output directory contains:

- `qaoa_per_user_results.csv` — **800 records** = 200 original workbook rows × 4 algorithms. Columns include set, scenario, original `instance_id`, user index (0-based within the 10-row scenario block), `is_candidate_pool`, `served`, RF chain, `assigned_beam`, `gain_linear`, `interference_linear`, and `fairness`. Each record corresponds to an original dataset input row for one method. An unserved user has all three **contributions** set to zero and a blank assigned beam. Zero means "not contributing to the selected assignment", not that the underlying radio channel is zero.
- `qaoa_dataset_results.csv` — **80 records** = 20 multiuser snapshots × 4 methods; these are *totals*, computed over the selected RF assignments.
- `qaoa_scenario_means.csv` — **40 records** = 10 scenarios × 4 methods. For every metric, this now calculates `(Set1_snapshot_total + Set2_snapshot_total)/20`, **NOT** `/2`. It is the *mean contribution per original input row*, including non-served users as zeros. For one set only, it divides by its 10 original source rows. It is **not** the mean over only RF-served users.
- Three `scenario_mean_*.png` and three `scenario_mean_*.pdf` charts — gain, interference, fairness. Log10 y axes for strictly positive data; symmetric log for data containing zeros (preserves zeros). No dB conversion or change to recorded metric values.

Per-served-user metric definitions with assigned beam `b[k]` and candidate-reference mean `mu`:

```
g_k = |H[k] @ F[:,b[k]]|**2
i_k = sum(|H[k] @ F[:,b[j]]|**2 for each other SELECTED user j)
f_k = 0.01 * (g_k - mu)**2
mu  = mean(|H[selected_candidate_users] @ F[user_candidates]|**2)
```

Only selected users have a defined assigned-beam metric; for unserved rows, recorded contributions are 0. Per-user values sum to snapshot totals, including symmetric pairwise interference exactly. The fairness term is unweighted in CSV and plots; lambda_F=0.2 is retained inside the QUBO optimization.

The original Greedy and OMP selection algorithms are preserved. They still search the **full beam codebook**, unlike QAOA and ILP, which search user-specific candidate shortlists: this is a deliberate baseline difference that should be disclosed when reporting a controlled comparison.

Run with the original `channels.py` (which is included in this package):

```powershell
pip install -r requirements.txt
python qaoa_beam_assignment_datasets.py --channel-model external
```

Or non-interactive:

```powershell
python qaoa_beam_assignment_datasets.py --users 4 --beams 3 --rf 2 --yes --channel-model external
```

Existing 1024-shot PennyLane setting, exact QUBO, interactive parameters, channel-model integration, and candidate-feasibility repair remain unchanged. The full quantum run was not executed in the packaging environment.


## Added: weighted multi-objective score (all four methods)

No optimization, constraint, selection, or plotting logic is changed by this addition.
For each **feasible assignment**, the reported physical objective is:

```text
objective_score = -1.0 * gain_linear + 0.5 * interference_linear + 0.2 * fairness
```

Lower values are better. This score is computed identically for Greedy, OMP,
QAOA-XY, and ILP from their evaluated assignments. It appears in each
method/snapshot row of `qaoa_dataset_results.csv` and is printed in the console.
The scenario-summary file `qaoa_scenario_means.csv` also contains
`objective_score`, aggregated according to the **existing** per-source-row
formula (sum of snapshot scores divided by source rows, normally 20).
The project still writes its original per-user metrics and three graphs.

The score excludes constraint penalties and the omitted RF cardinality constant.
For feasible QAOA/ILP assignments the raw QUBO energy equals
`objective_score - LAMBDA_R * (RF_chains**2)`. The score is valid as a common
**evaluation** of original Greedy and OMP, even when they choose global beams
outside the shorter QAOA candidate sets; do not interpret that as an equal
search-space comparison. The score is on the raw, unnormalized metric scale;
changing the candidate mean, model, or weights changes its interpretation.
