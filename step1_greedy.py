"""
Step 1 — Greedy beam-to-RF-chain assignment.

Formulated problem (repeated for t = 1 .. N_RF):

    (k*, m*) = argmax_{k not yet served, m not yet used}  G[k, m]

    Assign beam m* to the next free RF chain, mark user k* as served,
    mark beam m* as used. No backtracking -- once chosen, permanent.
"""

import numpy as np
from step0_gain_matrix import build_gain_matrix


def greedy_assignment(G: np.ndarray, n_rf_chains: int):
    """
    Parameters
    ----------
    G : np.ndarray, shape (K, M)
        Gain matrix from Step 0.
    n_rf_chains : int
        Number of RF chains available (<= K, <= M).

    Returns
    -------
    assignment : list of (user_index, beam_index) tuples, length n_rf_chains
    total_gain : float
        Sum of G[k, m] over the chosen assignment -- the objective value.
    """
    K, M = G.shape
    remaining_users = set(range(K))
    remaining_beams = set(range(M))
    assignment = []
    total_gain = 0.0

    for _ in range(n_rf_chains):
        best_gain = -np.inf
        best_pair = None

        for k in remaining_users:
            for m in remaining_beams:
                if G[k, m] > best_gain:
                    best_gain = G[k, m]
                    best_pair = (k, m)

        k_star, m_star = best_pair
        assignment.append((k_star, m_star))
        total_gain += best_gain
        remaining_users.remove(k_star)
        remaining_beams.remove(m_star)

    return assignment, total_gain


if __name__ == "__main__":
    G, beam_dict, H_users = build_gain_matrix(n_antennas=16, n_users=6, oversampling=4, seed=1)
    N_RF = 4  # fewer RF chains than users -> forces competition

    assignment, total_gain = greedy_assignment(G, n_rf_chains=N_RF)

    print(f"Greedy assignment (RF chains = {N_RF}):")
    for rf_chain_idx, (user, beam) in enumerate(assignment):
        print(f"  RF chain {rf_chain_idx} -> user {user}, beam {beam}, gain = {G[user, beam]:.4f}")
    print(f"Total objective (sum of gains): {total_gain:.4f}")
