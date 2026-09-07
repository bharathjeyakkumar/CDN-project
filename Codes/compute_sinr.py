"""
Compute SINR (Signal-to-Interference-plus-Noise Ratio) for each method's
assignment -- this is the metric your project actually claims to improve,
not just raw gain sum.

For a given assignment {user_k -> beam_m}, user k's SINR accounts for:
  - Signal power: |h_k . f_{m_k}|^2  (its own assigned beam)
  - Interference: sum over every OTHER served user k' of |h_k . f_{m_k'}|^2
    (leakage from other users' beams into user k's channel -- this is real
    and significant when beams aren't perfectly orthogonal, especially for
    users inside the same hotspot, which are angularly close by design)
  - Noise power: a fixed noise floor

SINR_k = signal_k / (interference_k + noise)
"""
import numpy as np


def compute_sinr_for_assignment(assignment, H_users, beam_dict,
                                  tx_power: float = 1.0, noise_power: float = 1e-4):
    """
    Parameters
    ----------
    assignment : list of (user_index, beam_index) tuples
    H_users : list of channel vectors (from your channel model)
    beam_dict : np.ndarray, shape (n_antennas, M)
    tx_power : float
        Transmit power per stream (equal power allocation assumed).
    noise_power : float
        Noise floor.

    Returns
    -------
    sinr_list : list of float, one SINR value per served user (linear, not dB)
    sinr_db_list : list of float, same values in dB
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
