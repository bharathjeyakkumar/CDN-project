"""
Step 0 — Build the gain matrix G[k, m] shared by greedy, OMP, and QUBO.

g[k, m] = |h_k . f_m|^2  ->  how well beam m serves user k.
This is the ONLY thing all three algorithms below will look at.
"""

import numpy as np
from beam_dictionary import generate_beam_dictionary
from channel_model_distance_aware import generate_multiuser_channels


def build_gain_matrix(n_antennas: int, n_users: int, oversampling: int = 4, seed: int = 1):
    """
    Returns
    -------
    G : np.ndarray, shape (K, M)
        G[k, m] = |h_k . f_m|^2 for every user k and every candidate beam m.
    beam_dict : np.ndarray, shape (n_antennas, M)
    H_users : list of channel vectors, one per user
    """
    beam_dict, angles = generate_beam_dictionary(n_antennas, oversampling_factor=oversampling)
    H_users, info_users = generate_multiuser_channels(
        n_users=n_users, n_tx=n_antennas, n_rx_per_user=1, seed=seed,
    )

    K = n_users
    M = beam_dict.shape[1]
    G = np.zeros((K, M))

    for k in range(K):
        h_k = H_users[k].flatten()          # shape (n_antennas,)
        G[k, :] = np.abs(h_k @ beam_dict) ** 2

    return G, beam_dict, H_users


if __name__ == "__main__":
    G, beam_dict, H_users = build_gain_matrix(n_antennas=16, n_users=6, oversampling=4, seed=1)
    print(f"Gain matrix G shape: {G.shape}  (K users x M beams)")
    print(f"Best beam for user 0: beam index {np.argmax(G[0]):3d}, gain = {G[0].max():.4f}")
    print(f"Best beam for user 1: beam index {np.argmax(G[1]):3d}, gain = {G[1].max():.4f}")
