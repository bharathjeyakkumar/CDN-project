
import numpy as np


def omp_multi_objective(
    H_users,
    beam_dict,
    n_rf_chains,
    lambda_I=0.5,
    lambda_F=0.2,
    lambda_A=6.0,
):
    """
    Multi-objective, assignment-constrained OMP.

    Objectives:
        1. Maximize residual channel correlation.
        2. Minimize the QUBO-inspired interference surrogate.
        3. Include the QUBO-inspired gain-deviation fairness surrogate.

    Constraints:
        1. At most one beam per user.
        2. At most one user per beam.
        3. At most n_rf_chains selected user-beam pairs.

    Returns:
        assignment, total_gain, F_RF, objective_history
    """

    H = np.asarray(H_users, dtype=complex)
    F = np.asarray(beam_dict, dtype=complex)

    if H.ndim != 2 or F.ndim != 2:
        raise ValueError("H_users and beam_dict must be 2D arrays.")

    K, Nt = H.shape
    N, M = F.shape

    if Nt != N:
        raise ValueError(
            "Channel length must match beam vector length."
        )

    if n_rf_chains < 0:
        raise ValueError("n_rf_chains cannot be negative.")

    if lambda_I < 0 or lambda_F < 0 or lambda_A < 0:
        raise ValueError("Penalty weights must be non-negative.")

    # Original channel power gains: G^2[k, m]
    G2 = np.abs(H @ F) ** 2

    # Same gain-deviation fairness surrogate as the QUBO.
    mean_gain = np.mean(G2)
    fairness_cost = (
        lambda_F * 0.01 * (G2 - mean_gain) ** 2
    )

    # Initialize residual with original channels.
    R = H.copy()

    selected_users = set()
    selected_beams = set()
    assignment = []
    objective_history = []

    total_gain = 0.0
    F_RF = np.empty((Nt, 0), dtype=complex)
    identity = np.eye(Nt, dtype=complex)

    # lambda_A is not needed for feasibility penalties here:
    # feasibility is enforced directly by skipping selected users
    # and beams, and limiting the number of iterations.

    max_iterations = min(n_rf_chains, K, M)

    for _ in range(max_iterations):

        best_pair = None
        best_score = np.inf

        # Search all currently feasible user-beam pairs.
        for k in range(K):

            if k in selected_users:
                continue

            for m in range(M):

                if m in selected_beams:
                    continue

                # OMP residual correlation score.
                residual_gain = abs(R[k] @ F[:, m]) ** 2

                # Negative gain is a cost to be minimized.
                candidate_cost = (
                    -residual_gain + fairness_cost[k, m]
                )

                # Add incremental interference cost using
                # the same index conditions as the QUBO.
                for j, n in assignment:

                    if k < j and m < n:
                        interference = G2[k, n] + G2[j, m]

                    elif j < k and n < m:
                        interference = G2[j, m] + G2[k, n]

                    else:
                        interference = 0.0

                    candidate_cost += lambda_I * interference

                if candidate_cost < best_score:
                    best_score = candidate_cost
                    best_pair = (k, m)

        if best_pair is None:
            break

        k, m = best_pair

        # Record selected user-beam pair.
        assignment.append((k, m))
        selected_users.add(k)
        selected_beams.add(m)
        objective_history.append(best_score)

        # Report original channel gain.
        total_gain += G2[k, m]

        # Append the selected beam to the RF beamformer.
        selected_vector = F[:, m].reshape(-1, 1)
        F_RF = np.hstack((F_RF, selected_vector))

        # Orthogonal projection onto the complement of
        # the selected beam subspace.
        projection = identity - F_RF @ np.linalg.pinv(F_RF)

        # Update residual channels.
        R = H @ projection

    return assignment, total_gain, F_RF, objective_history


if __name__ == "__main__":

    np.random.seed(42)

    # Simulation parameters matching your QUBO setup.
    Nt = 8
    K = 3
    M = 3
    N_RF = 2

    lambda_I = 0.5
    lambda_F = 0.2
    lambda_A = 6.0

    # Generate synthetic complex channel matrix.
    H_users = (
        np.random.randn(K, Nt)
        + 1j * np.random.randn(K, Nt)
    ) / np.sqrt(2)

    # Generate synthetic candidate beam codebook.
    beam_dict = (
        np.random.randn(Nt, M)
        + 1j * np.random.randn(Nt, M)
    ) / np.sqrt(2)

    # Normalize each candidate beam to unit norm.
    beam_dict = beam_dict / np.linalg.norm(
        beam_dict, axis=0, keepdims=True
    )

    # Run multi-objective OMP.
    assignment, total_gain, F_RF, history = (
        omp_multi_objective(
            H_users=H_users,
            beam_dict=beam_dict,
            n_rf_chains=N_RF,
            lambda_I=lambda_I,
            lambda_F=lambda_F,
            lambda_A=lambda_A,
        )
    )

    print("\n===== Multi-Objective OMP Results =====")
    print("Number of antennas:", Nt)
    print("Number of users:", K)
    print("Number of candidate beams:", M)
    print("Number of RF chains:", N_RF)

    print("\nSelected user-beam pairs (0-based indices):")
    for user, beam in assignment:
        print(f"User {user} -> Beam {beam}")

    print("\nTotal original channel gain:", total_gain)
    print("RF beamformer shape:", F_RF.shape)
    print("Greedy selection costs:", history)
