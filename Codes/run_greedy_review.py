"""
FIRST REVIEW -- Greedy-only pipeline.

Runs ONLY the greedy beam-to-RF-chain assignment on a real channel
scenario (your channel_model_distance_aware.py + beam_dictionary.py)
and reports the four metrics a review typically wants:

  1. Gain            -- from step1_greedy.py (sum of |h_k . f_m|^2)
  2. SINR             -- from compute_sinr.py (per-user, dB)
  3. Spectral efficiency (SE)  -- log2(1 + SINR) per user, bits/s/Hz
  4. Energy efficiency (EE)    -- total SE / total power consumed

Requires, in the same folder:
  beam_dictionary.py
  channel_model_distance_aware.py
  step0_gain_matrix.py
  step1_greedy.py
  compute_sinr.py

NOTE ON EE: spectral/energy efficiency are NOT part of your existing
scripts, so the formulas below are added fresh here. Power model is a
standard hybrid-beamforming assumption from the literature
(e.g. Mendez-Rial et al., "Hybrid MIMO Architectures for Millimeter
Wave Communications," IEEE Access 2016) -- adjust P_TX_TOTAL_W and
P_RF_CHAIN_W to match whatever your report cites.
"""

import numpy as np
import matplotlib.pyplot as plt

from beam_dictionary import generate_beam_dictionary
from channel_model_distance_aware import generate_multiuser_channels
from step1_greedy import greedy_assignment
from compute_sinr import compute_sinr_for_assignment

# ---------------- Scenario knobs ----------------
# NOTE: matches the stress scenario used in your original sinr_comparison.py
# (16 antennas, 8 RF chains, 20 users, 3 hotspots, averaged over 15 seeds)
# so results are directly comparable to that 0.31 dB figure.
N_ANTENNAS = 16
N_USERS = 20
OVERSAMPLING = 1
N_RF_CHAINS = 8
N_HOTSPOTS = 3
HOTSPOT_SPREAD_DEG = 3.0
N_SEEDS = 15


def build_gain_matrix_hotspot(seed):
    beam_dict, _ = generate_beam_dictionary(N_ANTENNAS, oversampling_factor=OVERSAMPLING)
    H_users, _ = generate_multiuser_channels(
        n_users=N_USERS, n_tx=N_ANTENNAS, n_rx_per_user=1,
        n_hotspots=N_HOTSPOTS, hotspot_angular_spread_deg=HOTSPOT_SPREAD_DEG,
        seed=seed,
    )
    K, M = N_USERS, beam_dict.shape[1]
    G = np.zeros((K, M))
    for k in range(K):
        G[k, :] = np.abs(H_users[k].flatten() @ beam_dict) ** 2
    return G, beam_dict, H_users

# ---------------- Power model (for EE) ----------------
# Total power = transmit power (sum over active streams) + circuit power
# of the RF chains actually turned on.
TX_POWER_PER_STREAM_W = 1.0      # matches tx_power used in compute_sinr.py
P_RF_CHAIN_W = 0.1               # 100 mW per active RF chain (typical mmWave figure)
NOISE_POWER = 1e-4               # matches compute_sinr.py default


def compute_spectral_efficiency(sinr_list):
    """SE_k = log2(1 + SINR_k), SINR_k linear (not dB). Returns per-user SE list."""
    return [np.log2(1 + s) for s in sinr_list]


def compute_energy_efficiency(se_list, n_rf_chains,
                               tx_power_per_stream=TX_POWER_PER_STREAM_W,
                               p_rf_chain=P_RF_CHAIN_W):
    """
    EE = total spectral efficiency / total power consumed (bits/s/Hz per Watt).
    Total power = (tx power summed over active streams) + (RF chain circuit power).
    """
    total_se = sum(se_list)
    total_tx_power = tx_power_per_stream * n_rf_chains
    total_circuit_power = p_rf_chain * n_rf_chains
    total_power = total_tx_power + total_circuit_power
    return total_se / total_power, total_power


def main():
    print(f"Scenario: {N_USERS} users, {N_ANTENNAS} antennas, {N_RF_CHAINS} RF chains, "
          f"{N_HOTSPOTS} hotspots (spread {HOTSPOT_SPREAD_DEG} deg), averaged over {N_SEEDS} seeds\n")

    per_seed_gain, per_seed_sinr_db, per_seed_se, per_seed_ee = [], [], [], []
    all_sinr_db = []

    for seed in range(N_SEEDS):
        G, beam_dict, H_users = build_gain_matrix_hotspot(seed)

        assignment, total_gain = greedy_assignment(G, n_rf_chains=N_RF_CHAINS)
        sinr_list, sinr_db_list = compute_sinr_for_assignment(
            assignment, H_users, beam_dict, tx_power=TX_POWER_PER_STREAM_W, noise_power=NOISE_POWER
        )
        se_list = compute_spectral_efficiency(sinr_list)
        ee, total_power = compute_energy_efficiency(se_list, N_RF_CHAINS)

        per_seed_gain.append(total_gain)
        per_seed_sinr_db.append(np.mean(sinr_db_list))
        per_seed_se.append(sum(se_list))
        per_seed_ee.append(ee)
        all_sinr_db.extend(sinr_db_list)

    avg_gain = np.mean(per_seed_gain)
    avg_sinr_db = np.mean(all_sinr_db)          # averaged over every served user, every seed
    avg_se = np.mean(per_seed_se)
    avg_ee = np.mean(per_seed_ee)

    print("=== SUMMARY (Greedy only, averaged over 15 seeds) ===")
    print(f"Average total gain:        {avg_gain:.4f}")
    print(f"Average SINR:              {avg_sinr_db:.2f} dB")
    print(f"Average total spectral eff: {avg_se:.4f} bits/s/Hz")
    print(f"Average energy efficiency: {avg_ee:.4f} bits/s/Hz/W")

    return {
        "avg_gain": avg_gain,
        "avg_sinr_db": avg_sinr_db,
        "avg_se": avg_se,
        "avg_ee": avg_ee,
        "per_seed_gain": per_seed_gain,
        "per_seed_sinr_db": per_seed_sinr_db,
        "per_seed_se": per_seed_se,
        "per_seed_ee": per_seed_ee,
    }


def plot_results(results, save_path="greedy_only_results.png"):
    labels = ["Gain", "SINR (dB)", "Spectral Eff.\n(bits/s/Hz)", "Energy Eff.\n(bits/s/Hz/W)"]
    values = [results["avg_gain"], results["avg_sinr_db"], results["avg_se"], results["avg_ee"]]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Value")
    ax.set_title(f"Greedy-only results\n({N_USERS} users, {N_ANTENNAS} antennas, "
                 f"{N_RF_CHAINS} RF chains, {N_HOTSPOTS} hotspots, avg over {N_SEEDS} seeds)")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    results = main()
    plot_results(results)
