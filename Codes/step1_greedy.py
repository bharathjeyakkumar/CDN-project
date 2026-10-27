import numpy as np


def greedy_assignment(
    G: np.ndarray,
    n_rf_chains: int,
    lambda_I: float = 0.5,
    lambda_F: float = 0.2,
    lambda_A: float = 6.0,
):
    """
    Multi-objective greedy beam-to-RF-chain assignment.

    Objectives:
        1. Maximize channel power gain.
        2. Minimize the QUBO interference surrogate.
        3. Include the QUBO gain-deviation fairness surrogate.

    Constraints:
        - Each user receives at most one beam.
        - Each beam serves at most one user.
        - Select exactly n_rf_chains pairs, if feasible.

    Returns:
        assignment: list of (user_index, beam_index) tuples
        total_gain: sum of original gains for selected pairs
    """

    G = np.asarray(G, dtype=float)
    K, M = G.shape

    if n_rf_chains < 0 or n_rf_chains > min(K, M):
        raise ValueError(
            "n_rf_chains must be between 0 and min(K, M)."
        )

    # Same power-gain values used by the QUBO.
    # G is assumed to contain power gains G_{k,m}^2.
    G2 = G

    # Same fairness surrogate as the QUBO.
    mean_gain = np.mean(G2) if G2.size else 0.0
    fairness_cost = (
        lambda_F * 0.01 * (G2 - mean_gain) ** 2
    )

    remaining_users = set(range(K))
    remaining_beams = set(range(M))

    assignment = []
    total_gain = 0.0

    for _ in range(n_rf_chains):

        best_score = -np.inf
        best_pair = None

        for k in remaining_users:
            for m in remaining_beams:

                # Objective 1: reward high gain.
                score = G2[k, m]

                # Objective 2: incremental interference
                # using the same pairwise index conditions
                # as the QUBO formulation.
                for j, n in assignment:

                    if k < j and m < n:
                        interference = G2[k, n] + G2[j, m]

                    elif j < k and n < m:
                        interference = G2[j, m] + G2[k, n]

                    else:
                        interference = 0.0

                    score -= lambda_I * interference

                # Objective 3: fairness surrogate penalty.
                score -= fairness_cost[k, m]

                if score > best_score:
                    best_score = score
                    best_pair = (k, m)

        if best_pair is None:
            break

        k_star, m_star = best_pair

        assignment.append((k_star, m_star))
        total_gain += G2[k_star, m_star]

        remaining_users.remove(k_star)
        remaining_beams.remove(m_star)

    return assignment, total_gain


if __name__ == "__main__":
    from step0_gain_matrix import build_gain_matrix

    G, beam_dict, H_users = build_gain_matrix(
        n_antennas=16,
        n_users=6,
        oversampling=4,
        seed=1,
    )

    N_RF = 4

    assignment, total_gain = greedy_assignment(
        G,
        n_rf_chains=N_RF,
        lambda_I=0.5,
        lambda_F=0.2,
        lambda_A=6.0,
    )

    print(f"Multi-objective greedy assignment (RF chains = {N_RF}):")

    for rf_chain_idx, (user, beam) in enumerate(assignment):
        print(
            f"  RF chain {rf_chain_idx} -> "
            f"user {user}, beam {beam}, "
            f"gain = {G[user, beam]:.4f}"
        )

    print(f"Total original gain: {total_gain:.4f}")