"""
Beam Dictionary Generator — Oversampled DFT Codebook
------------------------------------------------------
Generates the beam dictionary (analog steering-vector codebook) used for
beam-to-RF-chain assignment in hybrid precoding, following the standard
oversampled DFT codebook construction (El Ayach et al., 2014; 5G NR Type II).

Each column of the returned dictionary is one candidate beam: a steering
vector pointing at a specific angle, generated from the array geometry.
This dictionary is generated once and reused identically across every
baseline (greedy, OMP, ILP, RL) and QAOA/QAOA+ so comparisons stay fair.
"""

import numpy as np


def generate_beam_dictionary(n_antennas: int, oversampling_factor: int = 4,
                              spacing: float = 0.5):
    """
    Generate an oversampled DFT beam dictionary for a uniform linear array (ULA).

    Parameters
    ----------
    n_antennas : int
        Number of antenna elements in the array (N).
    oversampling_factor : int, default 4
        Oversampling factor O. 5G NR Type II codebooks use O = 4.
        Higher O -> finer angular resolution -> more candidate beams.
    spacing : float, default 0.5
        Antenna element spacing in units of wavelength (lambda).
        Standard half-wavelength spacing = 0.5.

    Returns
    -------
    beam_dict : np.ndarray, shape (n_antennas, n_beams)
        Complex steering-vector matrix. Column m is the m-th candidate beam.
        n_beams = oversampling_factor * n_antennas.
    angles : np.ndarray, shape (n_beams,)
        Physical angle (radians, measured from array boresight) that each
        column of beam_dict points toward. Useful for plotting / labeling.
    """
    n_beams = oversampling_factor * n_antennas

    # Oversampled angular grid in the "spatial frequency" domain u = cos(theta).
    # u ranges over [-1, 1); sampling it uniformly at n_beams points gives the
    # oversampled DFT grid used by 5G NR Type II codebooks.
    u_grid = -1 + 2 * np.arange(n_beams) / n_beams

    # Convert spatial frequency back to physical angle for reference/plotting.
    angles = np.arccos(np.clip(u_grid, -1, 1))

    # Antenna element index vector: 0, 1, ..., N-1
    n_idx = np.arange(n_antennas).reshape(-1, 1)  # column vector (N, 1)

    # Steering vector for element n at spatial frequency u:
    #   a_n(u) = exp(j * 2*pi * spacing * n * u)
    # Stacking over all n (rows) and all u in u_grid (columns) gives the
    # full (N x n_beams) DFT beam dictionary.
    beam_dict = np.exp(1j * 2 * np.pi * spacing * n_idx * u_grid.reshape(1, -1))

    # Normalize each beam (column) to unit norm — standard for a valid codebook.
    beam_dict = beam_dict / np.sqrt(n_antennas)

    return beam_dict, angles


def beam_gain_pattern(beam_dict: np.ndarray, angles: np.ndarray, beam_index: int,
                       n_points: int = 361):
    """
    Compute the array gain pattern |a(theta)^H * beam|^2 for one beam,
    swept across theta in [0, pi]. Useful for sanity-checking / plotting
    that a given beam actually points where you expect.

    Parameters
    ----------
    beam_dict : np.ndarray
        Output of generate_beam_dictionary().
    angles : np.ndarray
        Output of generate_beam_dictionary() (unused here, kept for API symmetry).
    beam_index : int
        Which column (candidate beam) of beam_dict to evaluate.
    n_points : int
        Number of angle samples to sweep across [0, pi].

    Returns
    -------
    theta_sweep : np.ndarray, shape (n_points,)
    gain_db : np.ndarray, shape (n_points,)
    """
    n_antennas = beam_dict.shape[0]
    theta_sweep = np.linspace(0, np.pi, n_points)
    u_sweep = np.cos(theta_sweep)

    n_idx = np.arange(n_antennas).reshape(-1, 1)
    steering = np.exp(1j * 2 * np.pi * 0.5 * n_idx * u_sweep.reshape(1, -1)) / np.sqrt(n_antennas)

    beam = beam_dict[:, beam_index].reshape(-1, 1)
    gain = np.abs(np.conj(beam).T @ steering).flatten() ** 2
    gain_db = 10 * np.log10(np.maximum(gain, 1e-12))

    return theta_sweep, gain_db


if __name__ == "__main__":
    # ---- Example usage / sanity check ----
    N_ANTENNAS = 32          # array size
    OVERSAMPLING = 4         # 5G NR Type II standard oversampling factor

    beam_dict, angles = generate_beam_dictionary(N_ANTENNAS, OVERSAMPLING)

    print(f"Antennas (N):        {N_ANTENNAS}")
    print(f"Oversampling (O):    {OVERSAMPLING}")
    print(f"Beam dictionary shape: {beam_dict.shape}  (N_antennas x N_beams)")
    print(f"Number of candidate beams: {beam_dict.shape[1]}")
    print()

    # Check that beams are (approximately) unit norm — validity check.
    norms = np.linalg.norm(beam_dict, axis=0)
    print(f"Beam norms — min: {norms.min():.4f}, max: {norms.max():.4f} (should be ~1.0)")

    # Show a few example beam angles (in degrees) from the dictionary.
    print("\nFirst 5 candidate beam angles (degrees):")
    print(np.round(np.degrees(angles[:5]), 2))
