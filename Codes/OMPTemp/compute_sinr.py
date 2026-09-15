"""
Compute SINR (Signal-to-Interference-plus-Noise Ratio) for each method's
assignment. Accounts for desired signal power, inter-user interference leakage,
and thermal noise.
"""

import numpy as np


def compute_sinr_for_assignment(assignment, H_users, beam_dict,
                                 tx_power: float = 1.0, noise_power: float = 1e-4):
    """
    Parameters
    ----------
    assignment : list of (user_index, beam_index) tuples
    H_users : list of channel vectors
    beam_dict : np.ndarray, shape (N_tx, M)
    tx_power : float
        Transmit power per stream.
    noise_power : float
        Noise floor.

    Returns
    -------
    sinr_list : list of float, linear scale
    sinr_db_list : list of float, dB scale
    """
    served_users = [k for k, m in assignment]
    beams_used = {k: m for k, m in assignment}

    sinr_list = []
    for k in served_users:
        h_k = H_users[k].flatten()
        m_k = beams_used[k]

        signal = tx_power * np.abs(h_k @ beam_dict[:, m_k]) ** 2

        interference = 0.0
        for k_other in served_users:
            if k_other == k:
                continue
            m_other = beams_used[k_other]
            interference += tx_power * np.abs(h_k @ beam_dict[:, m_other]) ** 2

        sinr = signal / (interference + noise_power)
        sinr_list.append(sinr)

    sinr_db_list = [10 * np.log10(max(s, 1e-12)) for s in sinr_list]
    return sinr_list, sinr_db_list