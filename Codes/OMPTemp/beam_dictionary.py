"""
Beam Dictionary Generator — Oversampled DFT Codebook
------------------------------------------------------
Generates the oversampled DFT beam dictionary matrix used for
beam-to-RF-chain assignment in mmWave hybrid precoding architectures.
"""

import numpy as np


def generate_beam_dictionary(n_antennas: int = 16, oversampling_factor: int = 1,
                             spacing: float = 0.5):
    """
    Generate an oversampled DFT beam dictionary for a Uniform Linear Array (ULA).

    Parameters
    ----------
    n_antennas : int, default 16
        Number of transmit antennas (N_t).
    oversampling_factor : int, default 1
        Oversampling factor O. Total beams M = O * N_t.
    spacing : float, default 0.5
        Antenna element spacing in units of wavelength (lambda/2 = 0.5).

    Returns
    -------
    beam_dict : np.ndarray, shape (n_antennas, n_beams)
        Complex steering-vector matrix. Column m is candidate beam m.
    angles : np.ndarray, shape (n_beams,)
        Physical angles (radians) corresponding to each beam direction.
    """
    n_beams = oversampling_factor * n_antennas
    u_grid = -1 + 2 * np.arange(n_beams) / n_beams
    angles = np.arccos(np.clip(u_grid, -1, 1))

    n_idx = np.arange(n_antennas).reshape(-1, 1)
    beam_dict = np.exp(1j * 2 * np.pi * spacing * n_idx * u_grid.reshape(1, -1))
    beam_dict = beam_dict / np.sqrt(n_antennas)

    return beam_dict, angles