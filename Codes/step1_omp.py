import numpy as np


def omp_assignment(
    H_users,
    beam_dict,
    n_rf_chains,
):
    """
    Assignment-constrained OMP beam selection.

    Constraints:
        - One beam can serve only one user.
        - One user can receive only one beam.

    OMP procedure:
        1. Start with original channel matrix as residual.
        2. Find user-beam pair with maximum residual correlation.
        3. Select that beam.
        4. Add beam to RF beamformer.
        5. Orthogonally project channels onto remaining subspace.
        6. Repeat.
    """

    K = len(H_users)
    N = beam_dict.shape[0]

    # --------------------------------------------------
    # Channel matrix
    # --------------------------------------------------

    H_concat = np.array(H_users, dtype=complex)

    # Residual starts as complete channel
    R = H_concat.copy()

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    selected_users = set()
    selected_beams = set()

    assignment = []

    total_gain = 0.0

    F_RF = np.empty(
        (N, 0),
        dtype=complex
    )

    # --------------------------------------------------
    # OMP iterations
    # --------------------------------------------------

    for iteration in range(
        min(n_rf_chains, K, beam_dict.shape[1])
    ):

        best_user = None
        best_beam = None
        best_correlation = -np.inf

        # ----------------------------------------------
        # Search over available user-beam pairs
        # ----------------------------------------------

        for k in range(K):

            if k in selected_users:
                continue

            residual_k = R[k]

            for m in range(
                beam_dict.shape[1]
            ):

                if m in selected_beams:
                    continue

                beam = beam_dict[:, m]

                correlation = abs(
                    residual_k @ beam
                ) ** 2

                if correlation > best_correlation:

                    best_correlation = correlation
                    best_user = k
                    best_beam = m

        # Safety check
        if best_user is None:
            break

        # ----------------------------------------------
        # Store assignment
        # ----------------------------------------------

        assignment.append(
            (best_user, best_beam)
        )

        selected_users.add(best_user)
        selected_beams.add(best_beam)

        # ----------------------------------------------
        # Original channel gain
        #
        # Used only for reporting so that Gain remains
        # directly comparable with Greedy.
        # ----------------------------------------------

        original_gain = abs(
            H_users[best_user]
            @ beam_dict[:, best_beam]
        ) ** 2

        total_gain += original_gain

        # ----------------------------------------------
        # Add selected beam to RF beamformer
        # ----------------------------------------------

        selected_vector = (
            beam_dict[:, best_beam]
            .reshape(-1, 1)
        )

        F_RF = np.hstack(
            [F_RF, selected_vector]
        )

        # ----------------------------------------------
        # Orthogonal projection
        # ----------------------------------------------

        F_RF_pinv = np.linalg.pinv(F_RF)

        projection = (
            np.eye(N)
            - F_RF @ F_RF_pinv
        )

        # ----------------------------------------------
        # Update residual
        # ----------------------------------------------

        R = H_concat @ projection

    return assignment, total_gain