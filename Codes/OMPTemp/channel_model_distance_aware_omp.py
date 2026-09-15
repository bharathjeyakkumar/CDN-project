import numpy as np


def steering_vector(n_antennas, theta, spacing=0.5):
    """
    ULA steering vector.
    """
    n = np.arange(n_antennas)

    return (
        np.exp(
            1j * 2 * np.pi * spacing * n * np.cos(theta)
        )
        / np.sqrt(n_antennas)
    )


def generate_mmwave_channel(
    n_tx=16,
    n_rx=1,
    n_clusters=3,
    n_rays_per_cluster=5,
    path_loss_exponent=2.0,
    reference_distance=1.0,
    distance=10.0,
    spacing=0.5,
    aod_center_override=None,
):
    """
    Distance-aware Saleh-Valenzuela mmWave channel.
    """

    n_paths = n_clusters * n_rays_per_cluster

    H = np.zeros((n_rx, n_tx), dtype=complex)

    # Random AoA/AoD
    if aod_center_override is None:
        aod_center = np.random.uniform(0, np.pi)
    else:
        aod_center = aod_center_override

    aoa_center = np.random.uniform(0, np.pi)

    for _ in range(n_paths):

        # Small angular variations around cluster center
        aod = aod_center + np.random.normal(0, np.deg2rad(3.0))
        aoa = aoa_center + np.random.normal(0, np.deg2rad(3.0))

        aod = np.clip(aod, 0, np.pi)
        aoa = np.clip(aoa, 0, np.pi)

        a_tx = steering_vector(
            n_tx,
            aod,
            spacing
        )

        a_rx = steering_vector(
            n_rx,
            aoa,
            spacing
        )

        alpha = (
            np.random.randn() + 1j * np.random.randn()
        ) / np.sqrt(2)

        H += alpha * np.outer(
            a_rx,
            np.conj(a_tx)
        )

    # Distance-dependent path loss
    path_loss = (
        reference_distance / distance
    ) ** path_loss_exponent

    H *= np.sqrt(path_loss)

    # Saleh-Valenzuela normalization
    H *= np.sqrt(
        (n_tx * n_rx) / n_paths
    )

    return H


def generate_multiuser_channels(
    n_users=20,
    n_tx=16,
    n_hotspots=3,
    hotspot_angular_spread_deg=3.0,
    spacing=0.5,
    seed=None,
):
    """
    Generate channels for multiple users distributed
    around several angular hotspots.
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate hotspot centers
    hotspot_centers = np.random.uniform(
        0,
        np.pi,
        n_hotspots
    )

    H_users = []

    # Assign each user to a hotspot
    user_hotspots = np.random.randint(
        0,
        n_hotspots,
        n_users
    )

    for k in range(n_users):

        hotspot_id = user_hotspots[k]

        center = hotspot_centers[hotspot_id]

        user_angle = (
            center
            + np.random.normal(
                0,
                np.deg2rad(
                    hotspot_angular_spread_deg
                )
            )
        )

        user_angle = np.clip(
            user_angle,
            0,
            np.pi
        )

        # Give users different distances
        distance = np.random.uniform(
            5.0,
            20.0
        )

        H = generate_mmwave_channel(
            n_tx=n_tx,
            n_rx=1,
            n_clusters=3,
            n_rays_per_cluster=5,
            path_loss_exponent=2.0,
            reference_distance=1.0,
            distance=distance,
            spacing=spacing,
            aod_center_override=user_angle,
        )

        H_users.append(
            H.flatten()
        )

    return np.array(H_users)


def generate_stress_scenario(
    seed,
    n_tx=16,
    n_users=20,
    n_hotspots=3,
    hotspot_angular_spread_deg=3.0,
    spacing=0.5,
):
    """
    Generate the 20-user / 3-hotspot stress scenario
    used by the OMP pipeline.
    """

    return generate_multiuser_channels(
        n_users=n_users,
        n_tx=n_tx,
        n_hotspots=n_hotspots,
        hotspot_angular_spread_deg=hotspot_angular_spread_deg,
        spacing=spacing,
        seed=seed,
    )


if __name__ == "__main__":

    H_users = generate_stress_scenario(
        seed=0,
        n_tx=16,
        n_users=20,
        n_hotspots=3,
        hotspot_angular_spread_deg=3.0,
        spacing=0.5,
    )

    print("Channel shape:", H_users.shape)
    print("Number of users:", len(H_users))