"""
mmWave Clustered Channel Model — Saleh-Valenzuela (SV) Style
--------------------------------------------------------------
Generates the propagation channel H used to evaluate beam-to-RF-chain
assignment quality. This is independent of the beam dictionary — the
dictionary comes from array geometry, the channel comes from the
propagation environment. They are only combined later, when an
assignment algorithm (greedy, OMP, QAOA, ...) correlates channel paths
against dictionary columns.

Model follows the standard geometric clustered channel used in:
  O. El Ayach et al., "Spatially Sparse Precoding in Millimeter Wave
  MIMO Systems," IEEE Trans. Wireless Commun., 2014.

Includes a stress-scenario generator sized specifically to push greedy
and OMP toward degraded, sub-optimal solutions: large antenna count,
many simultaneous users/streams, and a limited RF-chain budget.
"""

import numpy as np


def steering_vector(n_antennas: int, theta: float, spacing: float = 0.5) -> np.ndarray:
    """
    ULA steering vector at angle theta (radians), matching the same
    convention used in beam_dictionary.py so channel paths and beam
    dictionary columns are directly comparable.
    """
    n_idx = np.arange(n_antennas)
    return np.exp(1j * 2 * np.pi * spacing * n_idx * np.cos(theta)) / np.sqrt(n_antennas)


def generate_mmwave_channel(n_tx: int, n_rx: int, n_clusters: int = 4,
                             rays_per_cluster: int = 1, angle_spread_deg: float = 7.5,
                             distance_m: float = 50.0,
                             path_loss_exponent: float = 2.0,
                             reference_distance_m: float = 1.0,
                             rng: np.random.Generator = None):
    """
    Generate one distance-aware clustered mmWave channel matrix H.

    In addition to the original Saleh-Valenzuela angular model, this version
    includes a simple distance-dependent large-scale path-loss factor.

    Parameters
    ----------
    n_tx, n_rx : int
        Number of transmit / receive antennas.
    n_clusters : int
        Number of scattering clusters.
    rays_per_cluster : int
        Sub-rays per cluster.
    angle_spread_deg : float
        Standard deviation of ray angles around each cluster center.
    distance_m : float
        UE-to-gNodeB distance in meters.
    path_loss_exponent : float
        Power-law path-loss exponent. 2.0 is free-space-like.
    reference_distance_m : float
        Reference distance for the power-law model.
    rng : np.random.Generator, optional
        Pass a seeded generator for reproducibility.

    Returns
    -------
    H : np.ndarray
        Channel matrix of shape (n_rx, n_tx).
    info : dict
        Path angles, gains, distance and path-loss factor.
    """
    if rng is None:
        rng = np.random.default_rng()

    if distance_m < reference_distance_m:
        raise ValueError(
            f"distance_m must be >= reference_distance_m ({reference_distance_m} m)"
        )

    n_paths = n_clusters * rays_per_cluster
    H = np.zeros((n_rx, n_tx), dtype=complex)

    aod_list, aoa_list, gain_list = [], [], []
    spread = np.deg2rad(angle_spread_deg)

    # Simple distance-dependent amplitude attenuation:
    # power loss = (d0 / d)^path_loss_exponent
    # amplitude factor = sqrt(power loss)
    power_loss = (reference_distance_m / distance_m) ** path_loss_exponent
    distance_amplitude = np.sqrt(power_loss)

    for c in range(n_clusters):
        # Central angle for this cluster, uniform over [0, pi]
        aod_center = rng.uniform(0, np.pi)
        aoa_center = rng.uniform(0, np.pi)

        for r in range(rays_per_cluster):
            aod = np.clip(aod_center + rng.normal(0, spread), 0, np.pi)
            aoa = np.clip(aoa_center + rng.normal(0, spread), 0, np.pi)

            # Complex small-scale path gain
            alpha = (
                (rng.normal(0, 1) + 1j * rng.normal(0, 1))
                / np.sqrt(2)
            )

            # Apply distance-dependent attenuation
            alpha *= distance_amplitude

            a_tx = steering_vector(n_tx, aod)
            a_rx = steering_vector(n_rx, aoa)

            H += alpha * np.outer(a_rx, np.conj(a_tx))

            aod_list.append(aod)
            aoa_list.append(aoa)
            gain_list.append(alpha)

    H *= np.sqrt(n_tx * n_rx / n_paths)

    info = {
        "aod": np.array(aod_list),
        "aoa": np.array(aoa_list),
        "gain": np.array(gain_list),
        "n_paths": n_paths,
        "distance_m": distance_m,
        "power_loss": power_loss,
        "distance_amplitude": distance_amplitude,
    }

    return H, info

def generate_multiuser_channels(n_users: int, n_tx: int, n_rx_per_user: int = 1,
                                  n_clusters: int = 4, rays_per_cluster: int = 1,
                                  angle_spread_deg: float = 7.5,
                                  min_distance_m: float = 10.0,
                                  max_distance_m: float = 100.0,
                                  path_loss_exponent: float = 2.0,
                                  seed: int = None):
    """
    Generate independent distance-aware channels for multiple users.

    Each UE is assigned a random distance from the gNodeB between
    min_distance_m and max_distance_m.
    """
    if min_distance_m <= 0:
        raise ValueError("min_distance_m must be > 0")
    if max_distance_m < min_distance_m:
        raise ValueError("max_distance_m must be >= min_distance_m")

    rng = np.random.default_rng(seed)
    H_users, info_users = [], []

    for u in range(n_users):
        # Random UE-to-gNodeB distance within the cell
        distance_m = rng.uniform(min_distance_m, max_distance_m)

        H, info = generate_mmwave_channel(
            n_tx=n_tx,
            n_rx=n_rx_per_user,
            n_clusters=n_clusters,
            rays_per_cluster=rays_per_cluster,
            angle_spread_deg=angle_spread_deg,
            distance_m=distance_m,
            path_loss_exponent=path_loss_exponent,
            rng=rng,
        )

        H_users.append(H)
        info_users.append(info)

    return H_users, info_users

def generate_stress_scenario(seed: int = 0):
    """
    A deliberately large-scale scenario, sized to push greedy/OMP toward
    degraded, sub-optimal assignments:

      - Large antenna count (N_tx) -> large, dense beam dictionary
      - Many simultaneous users -> many competing paths to serve
      - RF-chain budget kept small relative to users -> tight assignment
        constraint, exactly where greedy's irreversible per-step choices
        and OMP's per-user independence start costing solution quality

    This is the regime your project's benchmarking should target to show
    the classical-vs-QAOA quality gap widening, per your project's claim.

    Returns
    -------
    scenario : dict with keys:
        H_users, info_users, n_tx, n_rf_chains, n_users, oversampling
    """
    N_TX = 256          # large antenna array -> dense beam dictionary
    N_RF_CHAINS = 8     # tight RF-chain budget relative to N_TX and n_users
    N_USERS = 20        # more users than RF chains -> forced competition
    OVERSAMPLING = 4    # matches 5G NR Type II codebook convention
    MIN_DISTANCE_M = 10.0
    MAX_DISTANCE_M = 100.0
    PATH_LOSS_EXPONENT = 2.0

    H_users, info_users = generate_multiuser_channels(
        n_users=N_USERS, n_tx=N_TX, n_rx_per_user=1,
        n_clusters=5,
        rays_per_cluster=2,
        angle_spread_deg=7.5,
        min_distance_m=MIN_DISTANCE_M,
        max_distance_m=MAX_DISTANCE_M,
        path_loss_exponent=PATH_LOSS_EXPONENT,
        seed=seed,
    )

    n_beams = OVERSAMPLING * N_TX
    print("Stress scenario generated:")
    print(f"  Antennas (N_tx):        {N_TX}")
    print(f"  RF chains:              {N_RF_CHAINS}")
    print(f"  Users:                  {N_USERS}")
    print(f"  Beam dictionary size:   {n_beams}  (oversampling x N_tx)")
    print(f"  Users : RF chains ratio: {N_USERS / N_RF_CHAINS:.1f} : 1")
    print(f"  UE distance range:       {MIN_DISTANCE_M:.1f} - {MAX_DISTANCE_M:.1f} m")
    print(f"  Path-loss exponent:      {PATH_LOSS_EXPONENT:.1f}")
    print(f"  Exhaustive search space: {n_beams} ** {N_RF_CHAINS} "
          f"= {n_beams ** N_RF_CHAINS:.3e} combinations (infeasible)")
    print("  -> This is the regime where greedy/OMP's per-step, per-user")
    print("     shortcuts are expected to leave the largest quality gap")
    print("     versus a more globally-searched (QAOA/QAOA+) assignment.")

    return {
        "H_users": H_users, "info_users": info_users,
        "n_tx": N_TX, "n_rf_chains": N_RF_CHAINS,
        "n_users": N_USERS, "oversampling": OVERSAMPLING,
    }


if __name__ == "__main__":
    # ---- Small sanity-check scenario ----
    print("=== Small sanity-check channel (8 tx, 1 rx, 4 clusters) ===")
    H, info = generate_mmwave_channel(
        n_tx=8,
        n_rx=1,
        n_clusters=4,
        distance_m=50.0,
        rng=np.random.default_rng(1)
    )
    print(f"H shape: {H.shape}")
    print(f"Path AoDs (deg): {np.round(np.degrees(info['aod']), 1)}")
    print(f"UE distance: {info['distance_m']:.1f} m")
    print(f"Frobenius norm of H: {np.linalg.norm(H):.3f}")
    print()

    # ---- The actual stress scenario for benchmarking greedy/OMP vs QAOA ----
    print("=== Stress scenario (for Week 10-11 benchmarking) ===")
    scenario = generate_stress_scenario(seed=42)
