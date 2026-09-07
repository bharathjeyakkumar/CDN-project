"""
mmWave Clustered Channel Model — Saleh-Valenzuela (SV) Style, with
Distance-Aware Path Loss AND User Hotspot Clustering
-----------------------------------------------------------------
Generates the propagation channel H used to evaluate beam-to-RF-chain
assignment quality. This is independent of the beam dictionary -- the
dictionary comes from array geometry, the channel comes from the
propagation environment. They are only combined later, when an
assignment algorithm (greedy, OMP, QAOA, ...) correlates channel paths
against dictionary columns.

Model follows the standard geometric clustered channel used in:
  O. El Ayach et al., "Spatially Sparse Precoding in Millimeter Wave
  MIMO Systems," IEEE Trans. Wireless Commun., 2014.

NEW in this version: user hotspot clustering. Instead of every user's
dominant angle-of-departure being fully independent and uniform over
[0, pi], users can be grouped into a small number of spatial hotspots
(e.g. a dense crowd, a lecture hall, a transit platform) sharing a
common central direction with a small angular spread. This is what
creates GENUINE beam contention: when several users' channels are
dominated by nearly the same steering direction, they compete for the
same one or two candidate beams in the dictionary, which is exactly
the condition under which greedy's irreversible per-step choice and
OMP's independent beam selection can leave gain on the table that a
globally-searched (QAOA/QAOA+) assignment would capture.

Without hotspot clustering (n_hotspots=None), behavior is identical to
the original fully-independent-user model.
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
                             aod_center_override: float = None,
                             aoa_center_override: float = None,
                             rng: np.random.Generator = None):
    """
    Generate one distance-aware clustered mmWave channel matrix H.

    Parameters
    ----------
    n_tx, n_rx : int
        Number of transmit / receive antennas.
    n_clusters : int
        Number of scattering clusters.
    rays_per_cluster : int
        Sub-rays per cluster.
    angle_spread_deg : float
        Standard deviation of ray angles around each cluster center
        (multipath spread -- small-scale angular variation).
    distance_m : float
        UE-to-gNodeB distance in meters.
    path_loss_exponent : float
        Power-law path-loss exponent. 2.0 is free-space-like.
    reference_distance_m : float
        Reference distance for the power-law model.
    aod_center_override, aoa_center_override : float, optional (radians)
        If given, ALL cluster centers are drawn around this angle instead
        of independently uniform over [0, pi]. This is what lets a caller
        place a user inside a specific hotspot direction -- see
        generate_multiuser_channels(..., n_hotspots=...).
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

    power_loss = (reference_distance_m / distance_m) ** path_loss_exponent
    distance_amplitude = np.sqrt(power_loss)

    for c in range(n_clusters):
        # Cluster center: either around the hotspot-assigned direction
        # (if provided) or fully independent, as in the original model.
        if aod_center_override is not None:
            aod_center = np.clip(aod_center_override + rng.normal(0, spread), 0, np.pi)
        else:
            aod_center = rng.uniform(0, np.pi)

        if aoa_center_override is not None:
            aoa_center = np.clip(aoa_center_override + rng.normal(0, spread), 0, np.pi)
        else:
            aoa_center = rng.uniform(0, np.pi)

        for r in range(rays_per_cluster):
            aod = np.clip(aod_center + rng.normal(0, spread), 0, np.pi)
            aoa = np.clip(aoa_center + rng.normal(0, spread), 0, np.pi)

            alpha = (rng.normal(0, 1) + 1j * rng.normal(0, 1)) / np.sqrt(2)
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
                                  n_hotspots: int = None,
                                  hotspot_angular_spread_deg: float = 5.0,
                                  seed: int = None):
    """
    Generate independent distance-aware channels for multiple users.

    Parameters
    ----------
    n_hotspots : int, optional
        If None (default): every user's dominant direction is fully
        independent and uniform over [0, pi] -- the original behavior.
        If set (e.g. 3): users are split across this many spatial
        hotspots (dense clusters of users -- a crowd, a platform, a
        lecture hall). All users in the same hotspot share a common
        central angle-of-departure, so their channels are strongly
        correlated -- this is what creates genuine beam contention.
    hotspot_angular_spread_deg : float
        How tightly users within one hotspot cluster around the
        hotspot's central angle (degrees). Smaller = more users forced
        toward the same nearest beam = harder assignment problem.
        This is the "channel area / diameter" knob: physically, a
        hotspot's angular width as seen from the base station is
        roughly (physical diameter / distance), so a small value here
        represents a tighter physical crowd, not just an abstraction.

    Returns
    -------
    H_users, info_users : as before, plus each info dict now also
        includes "hotspot_id" (or None if n_hotspots not used).
    """
    if min_distance_m <= 0:
        raise ValueError("min_distance_m must be > 0")
    if max_distance_m < min_distance_m:
        raise ValueError("max_distance_m must be >= min_distance_m")

    rng = np.random.default_rng(seed)
    H_users, info_users = [], []

    # Pre-assign hotspot centers and user-to-hotspot membership if requested.
    hotspot_centers = None
    user_hotspot_id = None
    if n_hotspots is not None:
        hotspot_centers = rng.uniform(0, np.pi, size=n_hotspots)
        user_hotspot_id = rng.integers(0, n_hotspots, size=n_users)

    for u in range(n_users):
        distance_m = rng.uniform(min_distance_m, max_distance_m)

        aod_override = None
        hotspot_id = None
        if n_hotspots is not None:
            hotspot_id = int(user_hotspot_id[u])
            # user's personal center = hotspot center + small jitter,
            # using hotspot_angular_spread_deg (NOT angle_spread_deg,
            # which stays reserved for multipath spread within a cluster)
            jitter = rng.normal(0, np.deg2rad(hotspot_angular_spread_deg))
            aod_override = np.clip(hotspot_centers[hotspot_id] + jitter, 0, np.pi)

        H, info = generate_mmwave_channel(
            n_tx=n_tx,
            n_rx=n_rx_per_user,
            n_clusters=n_clusters,
            rays_per_cluster=rays_per_cluster,
            angle_spread_deg=angle_spread_deg,
            distance_m=distance_m,
            path_loss_exponent=path_loss_exponent,
            aod_center_override=aod_override,
            rng=rng,
        )
        info["hotspot_id"] = hotspot_id

        H_users.append(H)
        info_users.append(info)

    return H_users, info_users


def generate_stress_scenario(seed: int = 0, n_tx: int = 16, n_rf_chains: int = 8,
                              n_users: int = 20, oversampling: int = 1,
                              n_hotspots: int = 3, hotspot_angular_spread_deg: float = 5.0,
                              min_distance_m: float = 10.0, max_distance_m: float = 100.0,
                              path_loss_exponent: float = 2.0):
    """
    Scenario generator sized to push greedy/OMP toward degraded,
    sub-optimal assignments using REAL beam contention via user hotspot
    clustering, not just raw problem size.

    Defaults now match a fixed-hardware setup (16 antennas, 8 RF chains)
    with 3 user hotspots -- multiple users per hotspot genuinely compete
    for the same one or two nearby beams.
    """
    H_users, info_users = generate_multiuser_channels(
        n_users=n_users, n_tx=n_tx, n_rx_per_user=1,
        n_clusters=5, rays_per_cluster=2, angle_spread_deg=7.5,
        min_distance_m=min_distance_m, max_distance_m=max_distance_m,
        path_loss_exponent=path_loss_exponent,
        n_hotspots=n_hotspots, hotspot_angular_spread_deg=hotspot_angular_spread_deg,
        seed=seed,
    )

    n_beams = oversampling * n_tx
    print("Stress scenario generated:")
    print(f"  Antennas (N_tx):         {n_tx}")
    print(f"  RF chains:               {n_rf_chains}")
    print(f"  Users:                   {n_users}")
    print(f"  Beam dictionary size:    {n_beams}  (oversampling x N_tx)")
    print(f"  Users : RF chains ratio: {n_users / n_rf_chains:.1f} : 1")
    print(f"  User hotspots:           {n_hotspots}  (spread {hotspot_angular_spread_deg} deg)")
    print(f"  UE distance range:       {min_distance_m:.1f} - {max_distance_m:.1f} m")
    print(f"  Path-loss exponent:      {path_loss_exponent:.1f}")
    print("  -> Users sharing a hotspot have strongly correlated channels,")
    print("     forcing genuine competition for the same nearby beams --")
    print("     this is where greedy/OMP's shortcuts cost real gain versus")
    print("     a globally-searched (QAOA/QAOA+) assignment.")

    return {
        "H_users": H_users, "info_users": info_users,
        "n_tx": n_tx, "n_rf_chains": n_rf_chains,
        "n_users": n_users, "oversampling": oversampling,
        "n_hotspots": n_hotspots,
    }


if __name__ == "__main__":
    print("=== Small sanity-check channel (no hotspots, original behavior) ===")
    H, info = generate_mmwave_channel(
        n_tx=8, n_rx=1, n_clusters=4, distance_m=50.0, rng=np.random.default_rng(1)
    )
    print(f"H shape: {H.shape}")
    print(f"Path AoDs (deg): {np.round(np.degrees(info['aod']), 1)}")
    print()

    print("=== Fixed-hardware stress scenario (16 antennas, 8 RF chains, 3 hotspots) ===")
    scenario = generate_stress_scenario(seed=42)
