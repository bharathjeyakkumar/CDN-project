import numpy as np
import matplotlib.pyplot as plt

from beam_dictionary import generate_beam_dictionary

from channel_model_distance_aware_omp import (
    generate_stress_scenario
)

from step0_gain_matrix import (
    build_gain_matrix
)

from step1_omp import (
    omp_assignment
)

from compute_sinr import (
    compute_sinr_for_assignment
)


# ======================================================
# SYSTEM PARAMETERS
# ======================================================

N_ANTENNAS = 16
N_USERS = 20

OVERSAMPLING = 1

N_RF_CHAINS = 8

N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0

N_SEEDS = 15


# ======================================================
# POWER PARAMETERS
# ======================================================

TX_POWER_PER_STREAM_W = 1.0
P_RF_CHAIN_W = 0.1

NOISE_POWER = 1e-4


# ======================================================
# BEAM DICTIONARY
# ======================================================

beam_dict, angles = generate_beam_dictionary(
    n_antennas=N_ANTENNAS,
    oversampling_factor=OVERSAMPLING,
    spacing=0.5,
)


# ======================================================
# STORAGE
# ======================================================

all_gains = []
all_sinr_db = []
all_total_se = []
all_ee = []


# ======================================================
# RUN EXPERIMENTS
# ======================================================

for seed in range(N_SEEDS):

    print("\n" + "=" * 50)
    print("SEED:", seed)
    print("=" * 50)

    # --------------------------------------------------
    # Generate OMP channel
    # --------------------------------------------------

    H_users = generate_stress_scenario(
        seed=seed,
        n_tx=N_ANTENNAS,
        n_users=N_USERS,
        n_hotspots=N_HOTSPOTS,
        hotspot_angular_spread_deg=HOTSPOT_SPREAD_DEG,
        spacing=0.5,
    )

    # --------------------------------------------------
    # Gain matrix
    # --------------------------------------------------

    G = build_gain_matrix(
        H_users,
        beam_dict
    )

    # --------------------------------------------------
    # OMP assignment
    # --------------------------------------------------

    assignment, total_gain = omp_assignment(
        H_users,
        beam_dict,
        N_RF_CHAINS,
    )

    print("OMP assignment:")
    print(assignment)

    print("OMP total gain:", total_gain)

    # --------------------------------------------------
    # SINR
    # --------------------------------------------------

    sinr_linear, sinr_db = compute_sinr_for_assignment(
        assignment,
        H_users,
        beam_dict,
        tx_power=TX_POWER_PER_STREAM_W,
        noise_power=NOISE_POWER
    )

    # --------------------------------------------------
    # Spectral efficiency
    # --------------------------------------------------

    se = [
        np.log2(1 + s)
        for s in sinr_linear
    ]

    total_se = np.sum(se)

    # --------------------------------------------------
    # Energy efficiency
    # --------------------------------------------------

    P_total = (
        N_RF_CHAINS
        * TX_POWER_PER_STREAM_W
        +
        N_RF_CHAINS
        * P_RF_CHAIN_W
    )

    ee = total_se / P_total

    # --------------------------------------------------
    # Store
    # --------------------------------------------------

    all_gains.append(total_gain)

    all_sinr_db.extend(sinr_db)

    all_total_se.append(total_se)

    all_ee.append(ee)

    print(
        "Average SINR:",
        np.mean(sinr_db),
        "dB"
    )

    print(
        "Total SE:",
        total_se,
        "bits/s/Hz"
    )

    print(
        "EE:",
        ee,
        "bits/s/Hz/W"
    )


# ======================================================
# FINAL RESULTS
# ======================================================

average_gain = np.mean(
    all_gains
)

average_sinr_db = np.mean(
    all_sinr_db
)

average_total_se = np.mean(
    all_total_se
)

average_ee = np.mean(
    all_ee
)


print("\n")
print("=" * 60)
print(
    "SUMMARY (OMP only, averaged over",
    N_SEEDS,
    "seeds)"
)
print("=" * 60)

print(
    "Average total gain:",
    average_gain
)

print(
    "Average SINR:",
    average_sinr_db,
    "dB"
)

print(
    "Average total spectral eff:",
    average_total_se,
    "bits/s/Hz"
)

print(
    "Average energy efficiency:",
    average_ee,
    "bits/s/Hz/W"
)


# ======================================================
# PLOT
# ======================================================

metrics = [
    average_gain,
    average_sinr_db,
    average_total_se,
    average_ee,
]

labels = [
    "Average Gain",
    "Average SINR (dB)",
    "Average SE",
    "Average EE",
]

plt.figure(figsize=(10, 6))

plt.bar(
    labels,
    metrics
)

plt.ylabel("Value")
plt.title(
    "OMP Performance over 15 Seeds"
)

plt.tight_layout()

plt.savefig(
    "omp_only_results.png",
    dpi=300
)

plt.show()