"""
Gate-level XY-mixer QAOA in Qiskit.

Qubit layout
------------
User group k owns qubits  k*S ... k*S + S - 1   (S = M_CAND + 1 slots:
M_CAND candidate beams + 1 idle slot).  Exactly one qubit per group is |1>.
Qubit index of (user k, slot s) = k*S + s, identical to the QUBO index.

Sign convention
---------------
x = (1 - s) / 2 with s = +1 for |0>, s = -1 for |1>, which is Qiskit's Z
eigenvalue convention (Z|0> = +|0>).  Therefore the Ising Hamiltonian
H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + c has diagonal entries equal
to the QUBO energy x^T Q x (+ offset), with h, J, c from Eqs. (10)-(12).

Circuit per layer l
-------------------
  initial state : one W state per group (Dicke state with k = 1)
  U_C(gamma)    : RZ(2*gamma*h_i) on every qubit
                  RZZ(2*gamma*J_ij) on every coupled pair
  U_M(beta)     : for every group, for every pair (a, b) in xy_pairs(S):
                  RXX(beta) then RYY(beta)
                  = exp(-i*beta/2 * (XX + YY))   (XX and YY commute)
Pairs that share a qubit do not commute, so the mixer is a first-order
Trotterised XY mixer with the fixed pair order defined by xy_pairs().
"""

import itertools
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

MAX_QUBITS = 20     # dense statevector bound used by this module


# ------------------------------------------------------
# helpers
# ------------------------------------------------------

def xy_pairs(S):
    """Fixed pair order of the XY mixer inside one group (complete graph)."""
    return list(itertools.combinations(range(S), 2))


def full_energies(Q, offset):
    """Diagonal of H_C over ALL 2^N basis states (little-endian indexing)."""
    N = len(Q)
    assert N <= MAX_QUBITS, f"{N} qubits is too many for dense simulation"
    idx = np.arange(2 ** N)
    X = ((idx[:, None] >> np.arange(N)) & 1).astype(float)
    return np.einsum("ni,ij,nj->n", X, Q, X) + offset


def subspace_to_full_index(states, S):
    """Map subspace states (n, K) of slot indices to full basis indices."""
    K = states.shape[1]
    return sum((1 << (k * S + states[:, k])) for k in range(K))


# ------------------------------------------------------
# Hamiltonian as a Pauli operator
# ------------------------------------------------------

def build_cost_operator(h, J, c, offset, N):
    """H_C = sum h_i Z_i + sum J_ij Z_i Z_j + (c + offset) as a SparsePauliOp."""
    terms = [("Z", [i], float(h[i])) for i in range(N) if h[i] != 0]
    terms += [("ZZ", [i, j], float(v)) for (i, j), v in J.items() if v != 0]
    terms.append(("I", [0], float(c + offset)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N).simplify()


def verify_cost_operator(H, E_full, atol=1e-8):
    """Check that the Pauli operator really equals the QUBO energy diagonal."""
    diag = np.real(H.to_matrix(sparse=True).diagonal())
    err = float(np.max(np.abs(diag - E_full)))
    assert err < atol, f"Ising/Pauli operator mismatch: {err:.3e}"
    return err


# ------------------------------------------------------
# circuit building blocks
# ------------------------------------------------------

def prepare_w_state(qc, qubits):
    """
    Prepare the W state (1/sqrt(n)) sum_i |0..1_i..0> on `qubits`, i.e. the
    Dicke state D^n_1, with a cascade of controlled-RY and CNOT gates.
    """
    n = len(qubits)
    qc.x(qubits[0])
    for i in range(n - 1):
        theta = 2 * np.arcsin(np.sqrt((n - i - 1) / (n - i)))
        qc.cry(theta, qubits[i], qubits[i + 1])
        qc.cx(qubits[i + 1], qubits[i])


def cost_layer(qc, gamma, h, J):
    for i, hi in enumerate(h):
        if hi != 0:
            qc.rz(2 * gamma * float(hi), i)
    for (i, j), v in J.items():
        if v != 0:
            qc.rzz(2 * gamma * float(v), i, j)


def xy_mixer_layer(qc, beta, K, S):
    for k in range(K):
        base = k * S
        for a, b in xy_pairs(S):
            qc.rxx(beta, base + a, base + b)
            qc.ryy(beta, base + a, base + b)


def build_qaoa_circuit(h, J, K, S, p):
    """Parametrised depth-p XY-mixer QAOA circuit."""
    N = K * S
    gam = ParameterVector("gamma", p)
    bet = ParameterVector("beta", p)
    qc = QuantumCircuit(N, name=f"XY-QAOA(p={p})")

    for k in range(K):                                  # feasible start state
        prepare_w_state(qc, list(range(k * S, (k + 1) * S)))
    for l in range(p):
        qc.barrier()
        cost_layer(qc, gam[l], h, J)
        qc.barrier()
        xy_mixer_layer(qc, bet[l], K, S)
    return qc, gam, bet


def circuit_statistics(qc):
    """Depth and gate counts after compiling to the basis {rz, sx, x, cx}."""
    from qiskit import transpile
    tc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
    ops = {k: int(v) for k, v in tc.count_ops().items() if k != "barrier"}
    return dict(n_qubits=qc.num_qubits, depth=tc.depth(), ops=ops,
                two_qubit_gates=ops.get("cx", 0))


# ------------------------------------------------------
# simulation and optimisation
# ------------------------------------------------------

def circuit_probabilities(qc, gam, bet, params):
    p = len(gam)
    binding = {gam[i]: float(params[i]) for i in range(p)}
    binding.update({bet[i]: float(params[p + i]) for i in range(p)})
    sv = Statevector(qc.assign_parameters(binding))
    prob = sv.probabilities()
    return prob / prob.sum()


def run_qaoa_circuit(Q, offset, h, J, c, K, S, p, restarts, rng, maxiter=200):
    """
    Optimise (gamma, beta) of the gate-level circuit with COBYLA.

    Returns
    -------
    best      : scipy OptimizeResult (best.x = [gammas, betas], best.fun = <H_C>)
    prob_full : probabilities over all 2^N basis states at the optimum
    E_full    : diagonal of H_C over all 2^N basis states
    qc, gam, bet : the parametrised circuit
    stats     : circuit statistics
    """
    N = K * S
    E_full = full_energies(Q, offset)
    H = build_cost_operator(h, J, c, offset, N)
    op_err = verify_cost_operator(H, E_full)

    qc, gam, bet = build_qaoa_circuit(h, J, K, S, p)
    stats = circuit_statistics(qc)
    stats["operator_check_err"] = op_err
    stats["n_pauli_terms"] = len(H)

    def f(th):
        return float(circuit_probabilities(qc, gam, bet, th) @ E_full)

    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rng.uniform(0, 0.5, p),
                              rng.uniform(0, np.pi / 2, p)])
        res = minimize(f, th0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    prob_full = circuit_probabilities(qc, gam, bet, best.x)
    return best, prob_full, E_full, qc, gam, bet, stats


def bound_circuit(qc, gam, bet, params, measure=True):
    """Return the circuit with numeric parameters (optionally with measurements)."""
    p = len(gam)
    binding = {gam[i]: float(params[i]) for i in range(p)}
    binding.update({bet[i]: float(params[p + i]) for i in range(p)})
    out = qc.assign_parameters(binding)
    if measure:
        out.measure_all()
    return out


def sample_with_aer(qc_bound, shots, seed=0):
    """
    Optional: run the measured circuit on Qiskit Aer (shot-based simulator).
    Returns {bitstring: count}, or None if qiskit-aer is not installed.
    Bitstrings are little-endian: the rightmost character is qubit 0.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
    except ImportError:
        return None
    sim = AerSimulator(seed_simulator=seed)
    job = sim.run(transpile(qc_bound, sim), shots=shots)
    return job.result().get_counts()