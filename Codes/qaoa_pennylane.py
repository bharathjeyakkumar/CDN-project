"""
Gate-level XY-mixer QAOA in PennyLane (K*M-qubit encoding).

Dedicated PennyLane backend used by qaoa_beam_assignment.py.

Qubit layout
------------
One wire per (user k, candidate beam s): wire = k * M_CAND + s, identical to the
QUBO index. There is no idle qubit. The circuit keeps the Hamming weight at R
(= number of RF chains): the initial state is the Dicke state |D_N^R> and the
mixer is an XY mixer, so only bit strings with exactly R ones are ever reached.

Bit-order warning
-----------------
PennyLane is BIG-endian: wire 0 is the most significant bit of a basis-state
index (wire 0 is the most significant bit). All index helpers below use that convention,
so the main script needs no changes.

Sign convention
---------------
x = (1 - s) / 2, s = +1 for |0>, s = -1 for |1> (Z|0> = +|0>). Hence
H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + c has diagonal entries equal
to the QUBO energy x^T Q x (+ offset), with h, J, c from Eqs. (10)-(12).

Circuit per layer l
-------------------
  initial state : Dicke state |D_N^R>, Baertschi-Eidenbenz circuit
                  (DICKE_PREP = "circuit", O(N*R) gates, O(N) depth)
                  or a direct PennyLane StatePrep load (DICKE_PREP = "statevector",
                  faster, simulation only)
  U_C(gamma)    : RZ(2*gamma*h_i) on every wire
                  IsingZZ(2*gamma*J_ij) on every coupled pair
  U_XY(beta)    : for every edge (a, b) in `edges`, in the given order:
                  IsingXX(beta) then IsingYY(beta) = exp(-i*beta/2 (XX + YY))
`edges` comes from qaoa_beam_assignment.xy_edges(), and preserves Hamming weight at every mixer layer.

Optimiser
---------
"adam"   : gradient-based. Exact gradients come from the simulator
           (adjoint differentiation on lightning.qubit, back-propagation on
           default.qubit), so one step costs about one forward + one backward
           pass instead of the many forward passes COBYLA needs.
"cobyla" : gradient-free SciPy optimizer.
"""

import itertools
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

MAX_QUBITS = 20
SAMPLER_NAME = "PennyLane shots"

OPTIMIZER = "adam"        # "adam" or "cobyla"
ADAM_STEPS = 120
ADAM_LR = 0.08
DICKE_PREP = "circuit"    # "circuit" (gate-level) or "statevector" (simulation shortcut)


# ------------------------------------------------------
# device
# ------------------------------------------------------

def make_device(n_wires):
    """lightning.qubit if installed (faster, adjoint gradients), else default.qubit."""
    try:
        return qml.device("lightning.qubit", wires=n_wires), "adjoint"
    except Exception:
        return qml.device("default.qubit", wires=n_wires), "backprop"


# ------------------------------------------------------
# helpers (big-endian indexing)
# ------------------------------------------------------

def full_energies(Q, offset):
    """Diagonal of H_C over ALL 2^N basis states; wire 0 = most significant bit."""
    N = len(Q)
    assert N <= MAX_QUBITS, f"{N} qubits is too many for dense simulation"
    idx = np.arange(2 ** N)
    X = ((idx[:, None] >> (N - 1 - np.arange(N))) & 1).astype(float)
    return np.einsum("ni,ij,nj->n", X, Q, X) + offset


def subspace_to_full_index(states, N):
    """
    Map weight-R states, given as (n, R) arrays of occupied qubit indices,
    to full basis-state indices (wire 0 = most significant bit).
    """
    return np.sum(1 << (N - 1 - np.asarray(states)), axis=1)


def dicke_vector(N, R):
    """Dicke state |D_N^R> as a 2^N statevector (big-endian)."""
    v = np.zeros(2 ** N)
    for pos in itertools.combinations(range(N), R):
        v[sum(1 << (N - 1 - p) for p in pos)] = 1.0
    return v / np.linalg.norm(v)


# ------------------------------------------------------
# Hamiltonian
# ------------------------------------------------------

def build_cost_operator(h, J, N):
    """Non-constant part of H_C = sum h_i Z_i + sum J_ij Z_i Z_j as a PennyLane operator."""
    coeffs, ops = [], []
    for i in range(N):
        if h[i] != 0:
            coeffs.append(float(h[i]))
            ops.append(qml.Z(i))
    for (i, j), v in J.items():
        if v != 0:
            coeffs.append(float(v))
            ops.append(qml.Z(i) @ qml.Z(j))
    return qml.dot(coeffs, ops)


def verify_cost_operator(H, const, E_full, N, atol=1e-8, rtol=1e-10):
    """
    Rebuild the diagonal of H from its Pauli terms and compare with the QUBO
    energy over all 2^N basis states (checks the h/J mapping and bit order).
    """
    idx = np.arange(2 ** N)
    zval = lambda w: 1 - 2 * ((idx >> (N - 1 - w)) & 1)
    diag = np.full(2 ** N, float(const))
    coeffs, ops = H.terms()
    for cf, op in zip(coeffs, ops):
        term = np.ones(2 ** N)
        for w in op.wires:
            term = term * zval(w)
        diag += float(np.real(cf)) * term
    err = float(np.max(np.abs(diag - E_full)))
    scale = max(1.0, float(np.max(np.abs(E_full))), float(np.max(np.abs(diag))))
    tolerance = atol + rtol * scale
    assert err <= tolerance, (
        f"Ising/Pauli operator mismatch: {err:.3e}, allowed: {tolerance:.3e}"
    )
    return err


# ------------------------------------------------------
# circuit building blocks
# ------------------------------------------------------

def _scs(n, k, off):
    """
    Split-and-cyclic-shift block SCS_{n,k} of Baertschi & Eidenbenz on wires
    off ... off + n - 1 (it acts on the last k + 1 of them).
    """
    w = lambda i: off + i
    qml.CNOT(wires=[w(n - 2), w(n - 1)])
    qml.ctrl(qml.RY, control=[w(n - 1)])(2 * np.arccos(np.sqrt(1.0 / n)), wires=w(n - 2))
    qml.CNOT(wires=[w(n - 2), w(n - 1)])
    for l in range(2, k + 1):
        qml.CNOT(wires=[w(n - l - 1), w(n - 1)])
        qml.ctrl(qml.RY, control=[w(n - 1), w(n - l)])(
            2 * np.arccos(np.sqrt(float(l) / n)), wires=w(n - l - 1))
        qml.CNOT(wires=[w(n - l - 1), w(n - 1)])


def prepare_dicke_state(N, R):
    """
    Deterministic Dicke-state preparation |D_N^R> (Baertschi & Eidenbenz, FCT 2019):
    start from |0^{N-R} 1^R> and apply SCS_{N,R}, ..., SCS_{R+1,R}, SCS_{R,R-1}, ..., SCS_{2,1}.
    O(N*R) gates, O(N) depth.
    """
    for i in range(N - R, N):
        qml.PauliX(wires=i)
    for l in range(N, R, -1):
        _scs(l, R, 0)
    for l in range(R, 1, -1):
        _scs(l, l - 1, 0)


def qaoa_ops(params, h, J, N, R, edges, p):
    """Queue the full depth-p circuit. params = [gamma_1..gamma_p, beta_1..beta_p]."""
    if DICKE_PREP == "statevector":
        qml.StatePrep(dicke_vector(N, R), wires=range(N))
    else:
        prepare_dicke_state(N, R)
    for l in range(p):
        gamma, beta = params[l], params[p + l]
        for i, hi in enumerate(h):                       # U_C
            if hi != 0:
                qml.RZ(2 * gamma * float(hi), wires=i)
        for (i, j), v in J.items():
            if v != 0:
                qml.IsingZZ(2 * gamma * float(v), wires=[i, j])
        for a, b in edges:                               # U_XY (Trotterised, fixed edge order)
            qml.IsingXX(beta, wires=[a, b])
            qml.IsingYY(beta, wires=[a, b])


class PLCircuit:
    """Holds everything needed to run / sample / draw one QAOA circuit."""

    def __init__(self, h, J, N, R, edges, p, H, const, phase_scale=1.0):
        self.h, self.J, self.N, self.R, self.edges, self.p = h, J, N, R, list(edges), p
        self.H, self.const = H, float(const)
        # Scale only the *angle parameterisation*, never the QUBO/Ising costs.
        # gamma_physical = gamma_trainable / phase_scale avoids impractically
        # oscillatory loss landscapes on high-power channel realisations.
        self.phase_scale = float(phase_scale)
        self._initialise_qnodes()

    def physical_params(self, params):
        return qml.math.concatenate([params[:self.p] / self.phase_scale, params[self.p:]])

    def _initialise_qnodes(self):

        dev, diff = make_device(self.N)
        self.device_name = dev.name
        self.diff_method = diff

        @qml.qnode(dev, diff_method=diff)
        def energy(params):
            qaoa_ops(self.physical_params(params), self.h, self.J, self.N, self.R, self.edges, self.p)
            return qml.expval(self.H)

        dev_p = qml.device("default.qubit", wires=self.N)

        @qml.qnode(dev_p)
        def probs(params):
            qaoa_ops(self.physical_params(params), self.h, self.J, self.N, self.R, self.edges, self.p)
            return qml.probs(wires=range(self.N))

        self.energy_qnode = energy
        self.probs_qnode = probs

    def energy(self, params):
        return self.energy_qnode(params) + self.const

    def probabilities(self, params):
        pr = np.array(self.probs_qnode(pnp.array(params, requires_grad=False)))
        return pr / pr.sum()


def circuit_statistics(circ, params):
    """Gate counts and depth, logical and compiled to {RZ, RX, RY, H, CNOT, X}."""
    with qml.queuing.AnnotatedQueue() as q:
        qaoa_ops(circ.physical_params(params), circ.h, circ.J, circ.N, circ.R, circ.edges, circ.p)
    tape = qml.tape.QuantumScript(q.queue)

    def depth_of(ops):
        d = {}
        for op in ops:
            lvl = 1 + max(d.get(w, 0) for w in op.wires)
            for w in op.wires:
                d[w] = lvl
        return max(d.values()) if d else 0

    def count(ops):
        out = {}
        for op in ops:
            out[op.name] = out.get(op.name, 0) + 1
        return out

    stats = dict(n_qubits=circ.N, logical_depth=depth_of(tape.operations),
                 logical_ops=count(tape.operations))
    try:
        tapes, _ = qml.transforms.decompose(
            tape, gate_set={"RZ", "RX", "RY", "Hadamard", "CNOT", "PauliX"})
        comp = tapes[0].operations
        stats.update(depth=depth_of(comp), ops=count(comp),
                     two_qubit_gates=count(comp).get("CNOT", 0))
    except Exception:                                   # fall back to analytic CNOT count
        ops = stats["logical_ops"]
        cx = (2 * (ops.get("IsingZZ", 0) + ops.get("IsingXX", 0) + ops.get("IsingYY", 0))
              + ops.get("CNOT", 0))
        stats.update(depth=stats["logical_depth"], ops=ops, two_qubit_gates=cx)
        stats["note"] = "controlled rotations of the Dicke circuit are not included in the CNOT count"
    stats["device"] = f"{circ.device_name} ({circ.diff_method} gradients)"
    return stats


# ------------------------------------------------------
# optimisation
# ------------------------------------------------------

class _Result:
    def __init__(self, x, fun):
        self.x, self.fun = np.asarray(x, dtype=float), float(fun)


def _optimise_cobyla(circ, th0, maxiter):
    f = lambda th: float(circ.energy(pnp.array(th, requires_grad=False)))
    res = minimize(f, th0, method="COBYLA", options={"maxiter": maxiter})
    return _Result(res.x, res.fun)


def _optimise_adam(circ, th0, steps, lr):
    opt = qml.AdamOptimizer(stepsize=lr)
    params = pnp.array(th0, requires_grad=True)
    best_x, best_f = np.array(th0), np.inf
    for _ in range(steps):
        params, cost = opt.step_and_cost(circ.energy, params)
        if cost < best_f:
            best_f, best_x = float(cost), np.array(params)   # params before this update
    final = float(circ.energy(pnp.array(params, requires_grad=False)))
    if final < best_f:
        best_f, best_x = final, np.array(params)
    return _Result(best_x, best_f)


def run_qaoa_circuit(Q, offset, h, J, c, N, R, edges, p, restarts, rng, maxiter=200):
    """
    Optimise (gamma, beta) of the PennyLane circuit.

    Parameters
    ----------
    N, R  : number of qubits (users x beams) and Hamming weight (RF chains)
    edges : XY-mixer edges in application order (from qaoa_beam_assignment.xy_edges)

    Returns a PennyLane-specific circuit result tuple
    -------
    best      : object with .x = [gammas, betas] and .fun = <H_C>
    prob_full : probabilities over all 2^N basis states at the optimum
    E_full    : diagonal of H_C over all 2^N basis states
    circ      : PLCircuit object
    None, None: unused API slots; PennyLane stores angles in params
    stats     : circuit statistics
    """
    E_full = full_energies(Q, offset)
    H = build_cost_operator(h, J, N)
    const = c + offset
    op_err = verify_cost_operator(H, const, E_full, N)

    # Global phase is irrelevant: only the variation in energies sets gamma range.
    phase_scale = max(1.0, float(np.max(np.abs(E_full - np.mean(E_full)))))
    circ = PLCircuit(h, J, N, R, edges, p, H, const, phase_scale=phase_scale)

    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rng.uniform(0, 0.5, p),
                              rng.uniform(0, np.pi / 2, p)])
        if OPTIMIZER == "adam":
            res = _optimise_adam(circ, th0, ADAM_STEPS, ADAM_LR)
        else:
            res = _optimise_cobyla(circ, th0, maxiter)
        if best is None or res.fun < best.fun:
            best = res

    prob_full = circ.probabilities(best.x)
    stats = circuit_statistics(circ, best.x)
    stats["operator_check_err"] = op_err
    stats["n_pauli_terms"] = len(H.terms()[0]) + 1
    stats["optimizer"] = OPTIMIZER
    stats["gamma_phase_scale"] = phase_scale
    return best, prob_full, E_full, circ, None, None, stats


# ------------------------------------------------------
# sampling and drawing
# ------------------------------------------------------

def sample_indices(circ, gam, bet, params, shots, seed=0):
    """
    Measure the circuit with a shot-based device. Returns the distinct basis-state
    indices observed (big-endian, same convention as subspace_to_full_index).
    """
    dev = qml.device("default.qubit", wires=circ.N, seed=seed)

    @qml.qnode(dev)
    def sampler(params):
        qaoa_ops(circ.physical_params(params), circ.h, circ.J, circ.N, circ.R, circ.edges, circ.p)
        return qml.counts(wires=range(circ.N))

    sampler = qml.set_shots(sampler, shots=shots)
    counts = sampler(pnp.array(params, requires_grad=False))
    return np.array(sorted(int(bits, 2) for bits in counts))


def save_drawing(circ, path, params=None):
    """Save a drawing of the circuit (needs matplotlib)."""
    import matplotlib.pyplot as plt
    if params is None:
        params = np.full(2 * circ.p, 0.1)
    fig, _ = qml.draw_mpl(circ.probs_qnode, decimals=2, style="black_white")(
        pnp.array(params, requires_grad=False))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)