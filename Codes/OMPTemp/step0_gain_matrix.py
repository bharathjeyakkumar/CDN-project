"""
Step 0 — Build the channel/beam gain matrix.

G[k, m] = |h_k f_m|^2

This matrix is useful for:
    - evaluating the final assignment
    - comparing Greedy / OMP / ILP / QAOA
    - calculating the total selected beam gain

OMP itself does NOT make its decisions solely from this matrix.
Instead, it repeatedly evaluates correlations against the updated
residual channel.
"""

import numpy as np


def build_gain_matrix(H_users, beam_dict):
    """
    Build the user-beam gain matrix.

    Parameters
    ----------
    H_users : list
        List of user channel matrices/vectors.
    beam_dict : np.ndarray
        Beam dictionary of shape (N_antennas, M).

    Returns
    -------
    G : np.ndarray
        Gain matrix of shape (K, M).
    """

    K = len(H_users)
    M = beam_dict.shape[1]

    G = np.zeros((K, M))

    for k in range(K):
        h_k = H_users[k].flatten()

        # Preserve the convention used by the existing pipeline:
        # gain = |h_k @ f_m|^2
        G[k, :] = np.abs(h_k @ beam_dict) ** 2

    return G


if __name__ == "__main__":

    from beam_dictionary import generate_beam_dictionary
    from channel_model_distance_aware import generate_multiuser_channels

    N_ANTENNAS = 16
    N_USERS = 20

    beam_dict, _ = generate_beam_dictionary(
        n_antennas=N_ANTENNAS,
        oversampling_factor=1
    )

    H_users, _ = generate_multiuser_channels(
        n_users=N_USERS,
        n_tx=N_ANTENNAS,
        n_rx_per_user=1,
        n_hotspots=3,
        hotspot_angular_spread_deg=3.0,
        seed=1
    )

    G = build_gain_matrix(H_users, beam_dict)

    print(f"Gain matrix shape: {G.shape}")
    print(
        f"Best beam for user 0: "
        f"{np.argmax(G[0])}, gain={G[0].max():.4f}"
    )