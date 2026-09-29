# QUBO Formulation and QAOA Simulation for Beam-to-RF-Chain Assignment

A learning document for the *Quantum-Enhanced Hybrid Precoder Design for 6G
Massive MIMO Systems* project.

---

## 1. What problem are we solving?

In hybrid precoding, a base station has many antennas (`Nt`) but only a few
RF chains (`N_RF`). Each RF chain must be connected to one beam, chosen from
a fixed set of candidate beams (the codebook). Choosing **which beam goes to
which RF chain, for which user**, is the beam-to-RF-chain assignment
problem.

This is a **combinatorial optimisation problem**: as the number of users,
beams and RF chains grows, the number of valid assignments explodes. It is
NP-hard, which is why classical Greedy/OMP methods only find "good enough"
answers, and why the project investigates a **QUBO + QAOA** approach that
searches the space more globally.

---

## 2. Concepts used, in order

| # | Concept | Full form | What it is |
|---|---|---|---|
| 1 | CSI | Channel State Information | Numbers describing how the wireless channel changes a signal for each user |
| 2 | Channel matrix `H` | — | `K x Nt` matrix; row `k` is user `k`'s channel vector `h_k` |
| 3 | Beam codebook `F` | — | `Nt x M` matrix of `M` candidate beams, built once from antenna geometry (DFT beams here) |
| 4 | DFT | Discrete Fourier Transform | The formula used to build orthogonal beams: `F[n,m] = exp(j2π n m / M) / √Nt` |
| 5 | Gain matrix `G` | — | `G[k,m] = |h_k^H f_m|`, how well user `k` matches beam `m` |
| 6 | Precoder | — | The matrix applied at the transmitter to shape signals; here it is hybrid: `F_RF` (analog, phase-only, beam selection) and `F_BB` (digital, interference cancellation) |
| 7 | QUBO | Quadratic Unconstrained Binary Optimization | A cost function over 0/1 variables, with linear and pairwise (quadratic) terms |
| 8 | QAOA | Quantum Approximate Optimization Algorithm | A hybrid quantum-classical algorithm that searches for a low-cost bitstring for a QUBO/Ising problem |

If any of these is unclear, they build on each other in this order: CSI is
measured → codebook is fixed in advance → gain matrix combines the two →
the gain matrix feeds the QUBO → QAOA solves the QUBO → the solution
configures the analog precoder `F_RF`.

---

## 3. The pipeline, step by step

```
CSI (H)  +  Codebook (F)
        |
        v
  Gain matrix  G = |H . F|          (K x M)
        |
        v
  QUBO built from G                  (K*M binary variables)
        |
        v
  Solved by: Brute force | Greedy | QAOA
        |
        v
  Decode -> list of (user, beam) pairs
        |
        v
  Real metrics: SINR, spectral efficiency, fairness, interference
```

### 3.1 Channel generation
A simplified clustered mmWave model: each user's channel is a sum of a few
random "clusters", each cluster contributing a steering vector at a random
angle with a random complex gain. This mimics multipath reflections.

### 3.2 Codebook
Built once, independent of the channel. Column `m` is a DFT steering vector
pointing at a fixed angle. Because DFT columns are **orthogonal**, the
beams do not overlap in the ideal case — this is why DFT codebooks are a
common choice.

### 3.3 Gain matrix
`G[k, m]` is large when beam `m` points close to user `k`'s dominant
direction. This single `K x M` matrix is the only thing that depends on
CSI; everything downstream (QUBO, Greedy, metrics) is built from it.

---

## 4. The QUBO formulation

### 4.1 Variables
`x_{k,m} = 1` if user `k` is assigned beam `m`, otherwise `0`. There are
`K * M` binary variables in total, flattened into one array.

### 4.2 Objective terms

| Term | Formula | Purpose |
|---|---|---|
| Signal reward | `- Σ G[k,m]^2 * x_{k,m}` | Rewards high-gain pairs (a proxy for SINR's numerator) |
| Interference penalty | `+ λ_I Σ (G[k,n]^2 + G[j,m]^2) * x_{k,m} * x_{j,n}` for `k≠j, m≠n` | Penalises picking two pairs whose beams leak into each other's users |
| Fairness penalty | `+ λ_F Σ (G[k,m]^2 - mean)^2 * x_{k,m}` | Discourages solutions that serve some users far better than others |

SINR itself is a **ratio** (signal ÷ interference+noise), which is not
directly expressible in a QUBO (QUBOs only have linear and pairwise binary
terms). So the QUBO uses the **numerator and denominator as separate
additive terms** instead of the ratio — reward the signal, penalise the
interference — and `λ_I` controls the trade-off between them.

### 4.3 Constraint terms (as penalties)
QUBOs are *unconstrained*, so constraints are added as penalty terms that
cost `0` when satisfied and a large positive value when violated:

| Constraint | Penalty shape |
|---|---|
| Each user gets **at most one** beam | `λ_A (Σ_m x_{k,m})(Σ_m x_{k,m} - 1)` for every user `k` |
| Each beam goes to **at most one** user | Same shape, summed over `m` |
| **Exactly `N_RF`** pairs are chosen | `λ_A (Σ_{k,m} x_{k,m} - N_RF)^2` |

`λ_A` (the constraint weight) must be **large enough** that no invalid
assignment ever beats a valid one, but not so large that it drowns out the
real objective. Tuning this by hand is exactly the "penalty-tuning"
weakness your literature review identifies in plain QUBO formulations —
it's the motivation for a constraint-preserving mixer (QAOA+) later.

### 4.4 Why the interference and fairness terms are quadratic/nonlinear
Interference genuinely depends on **two** chosen pairs together (leakage
between them), so it is naturally quadratic — this is exactly the kind of
term a classical Greedy method (which decides one pair at a time) cannot
see in advance. Fairness (variance across users) is not naturally linear
either; the code uses a simplified per-pair penalty as an approximation.

---

## 5. The three solvers

| Solver | How it works | Role |
|---|---|---|
| **Brute force** | Tries every valid combination of users/beams, keeps the one with the lowest QUBO cost | Ground truth — tells us the true optimum for small instances, so we can measure how close Greedy/QAOA get |
| **Greedy** | Repeatedly picks the single highest-gain `(user, beam)` pair, removes that user and beam, repeats `N_RF` times | Fast classical baseline (your Algorithm 1) — works on `G` directly, not on the QUBO |
| **QAOA** | Encodes the QUBO as a quantum circuit, alternates problem and mixer layers, and a classical optimiser (COBYLA) tunes the circuit's angles to minimise the expected cost | The method under investigation — searches the space more globally, encoding interference and fairness explicitly |

### 5.1 What QAOA is doing, briefly
1. The QUBO is converted into an **Ising Hamiltonian** (via `x = (1-Z)/2`).
2. A quantum circuit alternates two layers: a **cost layer** (rotates qubits
   based on the Hamiltonian) and a **mixer layer** (spreads out the
   probability across bitstrings), repeated `reps` times.
3. Running the circuit gives a probability distribution over all possible
   bitstrings (assignments).
4. A classical optimiser adjusts the circuit's angles (`γ`, `β`) to push
   the distribution toward low-cost bitstrings.
5. After optimisation, the most likely bitstring is read out and decoded
   into `(user, beam)` pairs.

More layers (`reps`) and more optimiser iterations (`maxiter`) generally
give better solutions, at the cost of a slower simulation.

---

## 6. From assignment to real metrics

The QUBO cost is only a **proxy**. Once any solver returns an assignment,
the code recomputes the actual formulas so every method is judged fairly
on the same scale:

| Metric | Formula | Meaning |
|---|---|---|
| SINR (per user) | `P|h_k^H f_{m_k}|^2 / (Σ_{j≠k} P|h_k^H f_{m_j}|^2 + σ^2)` | Signal strength vs. interference + noise |
| Spectral efficiency | `Σ log2(1 + SINR_k)` | Total data rate in bits/s/Hz |
| Jain's fairness index | `(Σ SINR_k)^2 / (n Σ SINR_k^2)` | 1.0 = perfectly fair, closer to `1/n` = very unfair |
| Interference | `Σ_{j≠k} P|h_k^H f_{m_j}|^2` | Total leaked power across served users |

---

## 7. What the test run showed

With 3 users, 3 beams and 2 RF chains:

- **Brute force and Greedy agreed** on the same optimal pair — expected for
  a very small, easy instance.
- **QAOA found a feasible but suboptimal assignment.** This is expected
  with a shallow circuit (`reps=1`) and few optimiser iterations
  (`maxiter=60`) — the run was meant to prove the *pipeline* works
  end-to-end, not to be a tuned result.

This gap (QAOA cost `-12.83` vs. brute force `-14.24`) is itself useful:
it is exactly the kind of number your Results chapter will report, scaled
up across many random seeds and instance sizes.

---

## 8. Parameters you control, and their effect

| Parameter | Where | Increasing it does |
|---|---|---|
| `reps` | QAOA circuit depth | Better solution quality, slower simulation |
| `maxiter` | COBYLA optimiser | Better convergence, slower run |
| `λ_A` | constraint penalty | Fewer infeasible solutions, but too large drowns the real objective |
| `λ_I` | interference weight | More conservative beam choices, possibly lower raw SINR reward |
| `λ_F` | fairness weight | More even service across users, possibly lower total spectral efficiency |
| `K, M, N_RF, Nt` | problem size | Larger, more realistic scenarios, but `K*M` qubits must stay simulable (roughly ≤ 20) |
| random seed | channel generation | Lets you average results over many channel realisations, like the 15-seed Greedy result in your deck |

---

## 9. Glossary (quick reference)

- **QUBO** — a way of writing an optimisation problem using only 0/1
  variables and quadratic (pairwise) terms, with no explicit constraints
  (constraints become penalty terms instead).
- **Ising Hamiltonian** — the same problem rewritten with ±1 variables
  (spins) instead of 0/1, which is the natural language for quantum
  circuits.
- **QAOA** — Quantum Approximate Optimization Algorithm; a shallow quantum
  circuit plus a classical optimiser, used to find good (not always
  optimal) solutions to QUBO/Ising problems.
- **Feasible solution** — one that satisfies every constraint (one beam per
  user, one user per beam, exactly `N_RF` pairs chosen).
- **Constraint-preserving mixer (QAOA+)** — a variant of QAOA whose mixer
  layer is designed so the circuit only ever visits feasible bitstrings,
  removing the need to tune penalty weights at all. This is the project's
  proposed improvement over plain QAOA.

---

## 10. Suggested next reading order

1. Re-read Section 4 (QUBO formulation) alongside the code's "Build the
   QUBO" section, line by line.
2. Re-read Section 5.1 (what QAOA does) alongside the "QAOA" section of the
   code.
3. Try changing one parameter at a time from Section 8 and re-run, to see
   how the gap between QAOA and brute force changes.
