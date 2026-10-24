import numpy as np


def omp_assignment(
    H_users,
    beam_dict,
    n_rf_chains,
    lambda_I=0.5,
    lambda_F=0.2,
    lambda_A=6.0,
):
    """
    Multi-objective assignment-constrained OMP.

    Objectives:
        1. Maximize residual channel-beam correlation.
        2. Minimize the QUBO interference surrogate.
        3. Include the QUBO gain-deviation fairness surrogate.

    Constraints:
        - One beam can serve only one user.
        - One user can receive only one beam.
        - Number of selected pairs <= n_rf_chains.

    Returns:
        assignment, total_gain
    """

    H = np.asarray(H_users, dtype=complex)
    F = np.asarray(beam_dict, dtype=complex)

    K = H.shape[0]
    N = F.shape[0]
    M = F.shape[1]

    if H.ndim != 2 or F.ndim != 2:
        raise ValueError("H_users and beam_dict must be 2D arrays.")

    if H.shape[1] != N:
        raise ValueError(
            "Channel length must equal beam_dict.shape[0]."
        )

    if n_rf_chains < 0:
        raise ValueError("n_rf_chains cannot be negative.")

    # Original channel power gains: G^2[k, m]
    G2 = np.abs(H @ F) ** 2

    # Same fairness surrogate as the QUBO
    mean_gain = np.mean(G2)

    fairness_cost = (
        lambda_F * 0.01 * (G2 - mean_gain) ** 2
    )

    # Initial residual
    R = H.copy()

    selected_users = set()
    selected_beams = set()
    assignment = []

    total_gain = 0.0

    F_RF = np.empty((N, 0), dtype=complex)
    identity = np.eye(N, dtype=complex)

    # lambda_A is unnecessary here because the constraints
    # are enforced directly rather than through penalties.
    _ = lambda_A

    max_iterations = min(n_rf_chains, K, M)

    for _ in range(max_iterations):

        best_user = None
        best_beam = None
        best_score = np.inf

        for k in range(K):

            if k in selected_users:
                continue

            for m in range(M):

                if m in selected_beams:
                    continue

                # OMP residual correlation
                residual_gain = abs(R[k] @ F[:, m]) ** 2

                # Gain reward and fairness surrogate
                score = (
                    -residual_gain
                    + fairness_cost[k, m]
                )

                # Same pairwise interference indexing
                # as the QUBO formulation
                for j, n in assignment:

                    if k < j and m < n:
                        interference = G2[k, n] + G2[j, m]

                    elif j < k and n < m:
                        interference = G2[j, m] + G2[k, n]

                    else:
                        interference = 0.0

                    score += lambda_I * interference

                if score < best_score:
                    best_score = score
                    best_user = k
                    best_beam = m

        if best_user is None:
            break

        # Store selected pair
        assignment.append((best_user, best_beam))

        selected_users.add(best_user)
        selected_beams.add(best_beam)

        # Original channel gain, consistent with the
        # original function's total_gain calculation
        total_gain += G2[best_user, best_beam]

        # Add selected beam to RF beamformer
        selected_vector = F[:, best_beam].reshape(-1, 1)
        F_RF = np.hstack((F_RF, selected_vector))

        # Orthogonal projection and residual update
        projection = (
            identity - F_RF @ np.linalg.pinv(F_RF)
        )

        R = H @ projection

    # Same return type as the original function
    return assignment, total_gain