# Optimization-Based Beam Selection for SINR, Spectral Efficiency and Energy Efficiency in 6G Massive MIMO Systems

## 📁 Project Drive Folder

📎 [Project Drive Folder — Datasets, Results & Documents](https://drive.google.com/drive/folders/1g1j3vRdJsSh5KAm-Jqj7oOJR_bh--43b?usp=sharing)

---

## 📌 Project Overview

This project focuses on **beam selection and beam-to-RF-chain assignment** in a multi-user 6G massive MIMO system.

The main objective is to investigate whether optimization-based beam-selection techniques can improve:

* **SINR (Signal-to-Interference-plus-Noise Ratio)**
* **Spectral Efficiency (SE)**
* **Energy Efficiency (EE)**

The project considers a **hybrid beamforming architecture**, where the number of available RF chains is smaller than the number of transmit antennas. Therefore, only a limited number of user-beam combinations can be selected simultaneously.

The overall project will compare three beam-selection approaches:

1. **Greedy Algorithm** — ✅ Implemented
2. **OMP (Orthogonal Matching Pursuit)** — 🔄 Planned
3. **QAOA (Quantum Approximate Optimization Algorithm)** — 🔄 Planned

The current implementation establishes the **Greedy algorithm as the classical baseline** for the subsequent OMP and QAOA implementations.

---

# 📂 Project Folder Structure

The planned GitHub repository is organized as follows:

```text
6G-Beam-Selection/
│
├── README.md
│
├── architecture/
│   
├── references/
│
├── codes/
│   ├── channel_model/
│   ├── beam_codebook/
│   ├── gain_matrix/
│   ├── greedy/
│   ├── omp/
│   └── qaoa/
│
└── outputs/
    ├── greedy/
    ├── omp/
    └── qaoa/
```

### Folder Description

| Folder            | Purpose                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `README.md`       | Complete project explanation and implementation status           |
| `references/`     | Research papers, Review 1 document, and other reference material |
| `codes/`          | Python implementation of the complete simulation framework       |
| `outputs/greedy/` | Results generated using the Greedy algorithm                     |
| `outputs/omp/`    | Results that will be generated using OMP                         |
| `outputs/qaoa/`   | Results that will be generated using QAOA                        |

> **Current Status:** Only the `greedy/` implementation and its corresponding outputs have been completed. The `omp/` and `qaoa/` folders are reserved for the next stages of the project.

---

# 🎯 Project Objective

In a massive MIMO system, the base station may contain a large number of antennas but only a limited number of RF chains.

For this project:

```text
Transmit Antennas (Nt) = 16
Users (K)              = 20
RF Chains (NRF)        = 8
Candidate Beams (M)    = 16
Spatial Hotspots       = 3
Channel Seeds          = 15
```

Since there are only **8 RF chains**, the beam-selection stage selects exactly **8 feasible user-beam pairs**.

The objective is to select user-beam combinations that provide strong channel-beam alignment while satisfying:

* One user → at most one beam
* One beam → at most one user
* Total selected pairs = number of RF chains

## These system parameters are defined in the Review 1 implementation.

# 🏗️ Overall System Flow

The current and planned project workflow is:

```text
                 ┌──────────────────────┐
                 │   System Parameters  │
                 │ Nt=16, K=20, NRF=8  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Channel Model     │
                 │  Multi-user mmWave   │
                 │  Clustered Channel   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Beam Codebook      │
                 │   DFT-based Beams    │
                 │   M = 16 Beams       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Gain Matrix G      │
                 │       20 × 16        │
                 └──────────┬───────────┘
                            │
                            ▼
          ┌────────────────────────────────────┐
          │       Beam Selection Algorithms    │
          │                                    │
          │  ┌────────┐  ┌──────┐  ┌────────┐ │
          │  │ Greedy │  │ OMP  │  │ QAOA   │ │
          │  │   ✅   │  │  🔄  │  │   🔄   │ │
          │  └────────┘  └──────┘  └────────┘ │
          └────────────────┬───────────────────┘
                           │
                           ▼
                 ┌──────────────────────┐
                 │ Selected User-Beam   │
                 │      Pairs (8)       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Performance Analysis │
                 │                      │
                 │ • SINR               │
                 │ • Spectral Efficiency│
                 │ • Energy Efficiency  │
                 └──────────────────────┘
```

The currently implemented stages are:

**Channel Generation → Beam Codebook → Gain Matrix → Greedy Selection → Performance Evaluation**

OMP and QAOA will use the same overall problem formulation and will be compared against the Greedy baseline.

---

# 1. 📡 Channel Model

A **clustered geometric mmWave channel model** is used to represent the propagation between the base station and users.

The channel of user `k` is represented as:

```text
              Clusters
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     Rays      Rays       Rays
       │         │         │
       └─────────┼─────────┘
                 │
                 ▼
            User Channel
```

The channel is modeled as:

$$
\mathbf{h}_k =
\sqrt{\frac{N_t}{N_{cl}N_{ray}}}
\sum_{c=1}^{N_{cl}}
\sum_{\ell=1}^{N_{ray}}
\alpha_{k,c,\ell}
\mathbf{a}_t(\phi_{k,c,\ell})
$$

where:

* `Nt` = number of transmit antennas
* `Ncl` = number of scattering clusters
* `Nray` = number of rays per cluster
* `α` = complex path gain
* `φ` = angle of departure
* `aₜ(φ)` = transmit steering vector

The complete multi-user channel matrix is:

$$
\mathbf{H}\in\mathbb{C}^{20\times16}
$$

Therefore, the current system has **20 users and 16 transmit antennas**.

---

# 2. 📶 Transmit Antenna Array

The base station uses a:

**Uniform Linear Array (ULA)**

with:

```text
Number of antennas = 16
Antenna spacing    = λ/2
```

The steering vector for an angle `φ` is:

$$
\mathbf{a}_t(\phi)
=
\frac{1}{\sqrt{N_t}}
\begin{bmatrix}
1 &
e^{j\pi\sin(\phi)} &
e^{j2\pi\sin(\phi)} &
\cdots &
e^{j(N_t-1)\pi\sin(\phi)}
\end{bmatrix}^{T}
$$

This steering-vector formulation is used to generate the candidate beams.

---

# 3. 🔦 DFT Beam Codebook

A **DFT-based beam codebook** is constructed from the steering vectors.

The number of candidate beams is:

```text
M = Nt = 16
```

Therefore:

```text
16 antennas
      ↓
16 candidate beams
```

Each candidate beam is represented as:

$$
\mathbf{f}_m = \mathbf{a}_t(\theta_m)
$$

for:

$$
m=1,2,\ldots,16
$$

Thus, the beam-selection problem starts with:

```text
20 Users × 16 Candidate Beams
```

which produces a total of:

```text
20 × 16 = 320
```

possible user-beam combinations before applying the assignment constraints.

---

# 4. 📊 User-Beam Gain Matrix

The compatibility between a user and a beam is measured using the channel-beam alignment:

$$
G_{k,m}
=
\left|
\mathbf{h}_k^H\mathbf{f}_m
\right|
$$

Here:

* `k` represents a user
* `m` represents a candidate beam
* `hₖ` represents the user's channel
* `fₘ` represents the candidate beam

A larger value of `Gₖ,ₘ` means that the beam is better aligned with the user's channel.

The resulting gain matrix is:

$$
\mathbf{G}\in\mathbb{R}^{20\times16}
$$

Therefore:

```text
                 16 Candidate Beams
              B1 B2 B3 ... B16
             ┌───────────────────┐
User 1       │                   │
User 2       │                   │
User 3       │       G           │
  .          │      20×16        │
  .          │                   │
User 20      │                   │
             └───────────────────┘
```

This gain matrix becomes the main input to the beam-selection algorithms.

---

# 5. 🔗 Beam-to-RF-Chain Assignment Problem

The beam-selection stage can be formulated as a binary optimization problem.

Define:

$$
x_{k,m}\in\{0,1\}
$$

where:

```text
x[k,m] = 1 → Beam m is assigned to User k
x[k,m] = 0 → Beam m is not assigned to User k
```

The objective is to maximize the total user-beam gain:

$$
\max_{\mathbf{X}}
\sum_{k=1}^{K}
\sum_{m=1}^{M}
G_{k,m}x_{k,m}
$$

Subject to three main constraints.

### Constraint 1 — One beam per user

$$
\sum_{m=1}^{M}x_{k,m}\leq1
$$

A user cannot receive multiple selected beams during this assignment stage.

### Constraint 2 — One user per beam

$$
\sum_{k=1}^{K}x_{k,m}\leq1
$$

A beam cannot be assigned to multiple users.

### Constraint 3 — RF-chain limitation

$$
\sum_{k=1}^{K}\sum_{m=1}^{M}x_{k,m}
=
N_{RF}
$$

Since:

```text
NRF = 8
```

exactly **8 user-beam pairs** are selected.

These constraints define the common optimization problem that Greedy, OMP and QAOA will address.

---

# 6. 🟢 Greedy Beam Selection — IMPLEMENTED

The first implemented algorithm is the **Greedy beam-selection algorithm**.

### Basic Idea

At every iteration:

1. Look at all currently available user-beam combinations.
2. Find the pair with the highest gain.
3. Select that user-beam pair.
4. Remove the selected user.
5. Remove the selected beam.
6. Repeat until all 8 RF chains are assigned.

In simple terms:

```text
Gain Matrix
     │
     ▼
Find highest available gain
     │
     ▼
Select (User, Beam)
     │
     ├── Remove User
     │
     └── Remove Beam
     │
     ▼
Repeat
     │
     ▼
8 Selected User-Beam Pairs
```

This guarantees that:

```text
One User → Maximum 1 Beam
One Beam → Maximum 1 User
Total Pairs → 8
```

The direct implementation has computational complexity:

$$
O(N_{RF}KM)
$$

For the current system:

```text
O(8 × 20 × 16)
```

The complete Greedy procedure and complexity are defined in the Review 1 document.

---

# 7. 📈 Performance Evaluation

After selecting the 8 user-beam pairs, the selected beams are evaluated using three metrics.

## 7.1 SINR

The Signal-to-Interference-plus-Noise Ratio measures the desired signal strength relative to:

* Inter-user interference
* Noise

For user `k`:

$$
\mathrm{SINR}_k =
\frac{
P_k|\mathbf{h}_k^H\mathbf{f}_{m_k}|^2
}{
\sum_{j\neq k}
P_j|\mathbf{h}_k^H\mathbf{f}_{m_j}|^2
+\sigma^2
}
$$

An important point is that the Greedy algorithm currently selects beams using **individual channel-beam gain**, not directly using the SINR objective.

Therefore:

```text
High individual gain
        ≠
Always high SINR
```

because simultaneously selected beams can interfere with each other.

---

# 7.2 📡 Spectral Efficiency

Spectral efficiency is calculated from SINR:

$$
R_k=\log_2(1+\mathrm{SINR}_k)
$$

The total spectral efficiency is:

$$
R_{sum}
=
\sum_{k\in\mathcal{S}}
\log_2(1+\mathrm{SINR}_k)
$$

Unit:

```text
bits/s/Hz
```

Spectral efficiency indicates how much data rate can be achieved per unit bandwidth.

---

# 7.3 ⚡ Energy Efficiency

Energy efficiency measures the achieved spectral efficiency per unit of total consumed power:

$$
EE=
\frac{R_{sum}}{P_{total}}
$$

Unit:

```text
bits/s/Hz/W
```

The total power model includes:

$$
P_{total}
=
P_{tx}
+
P_{RF}
+
P_{PS}
+
P_{BB}
$$

where:

* `Ptx` = transmit power
* `PRF` = RF-chain power
* `PPS` = phase-shifter power
* `PBB` = baseband processing power

The same power assumptions will be maintained across the different algorithms to provide a fair comparison.

---

# 8. 🧪 Current Simulation Configuration

The current simulation uses:

| Parameter         |     Value |
| ----------------- | --------: |
| Transmit antennas |        16 |
| Users             |        20 |
| RF chains         |         8 |
| Candidate beams   |        16 |
| Spatial hotspots  |         3 |
| Channel seeds     |        15 |
| Array type        |       ULA |
| Antenna spacing   |       λ/2 |
| Beam codebook     | DFT-based |
| Current algorithm |    Greedy |

The results are averaged over **15 independent channel seeds**.

---

# 9. 📊 Current Greedy Results

The current Greedy implementation produces the following baseline:

| Metric              | Greedy Result | Unit        |
| ------------------- | ------------: | ----------- |
| Average SINR        |    **0.3089** | dB          |
| Spectral Efficiency |   **11.3559** | bits/s/Hz   |
| Energy Efficiency   |    **1.2904** | bits/s/Hz/W |

These results are obtained for:

```text
20 Users
16 Antennas
8 RF Chains
16 Candidate Beams
3 Hotspots
15 Channel Seeds
```

The results establish the **classical Greedy baseline** for the next stages of the project.

---

# 10. 🔵 OMP — NEXT STAGE

The second beam-selection approach planned for implementation is:

**Orthogonal Matching Pursuit (OMP)**

OMP will be implemented after establishing the Greedy baseline.

The purpose of the OMP stage is to provide another classical beam-selection approach that can be compared against Greedy.

The OMP implementation will use the same:

* Channel model
* Antenna configuration
* Candidate beam codebook
* Number of users
* Number of RF chains
* Performance metrics
* Channel realizations

This allows a fair comparison between the algorithms.

```text
Current
  │
  ▼
Greedy ✅
  │
  ▼
OMP 🔄
  │
  ▼
Performance Comparison
```

OMP is explicitly identified as a subsequent beam-selection method in the Review 1 framework.

---

# 11. 🟣 QAOA — FUTURE STAGE

The final planned optimization approach is:

**QAOA — Quantum Approximate Optimization Algorithm**

The beam-selection problem will be mapped into an optimization formulation suitable for quantum optimization.

The existing binary assignment variables:

$$
x_{k,m}\in\{0,1\}
$$

provide the foundation for the future optimization formulation.

The planned flow is:

```text
User-Beam Gain Matrix
          │
          ▼
Binary Assignment Problem
          │
          ▼
QUBO Formulation
          │
          ▼
QAOA
          │
          ▼
Optimized Beam Selection
```

QAOA will ultimately be compared with the classical Greedy and OMP approaches.

The Review 1 document identifies **QUBO/QAOA-based beam selection** as a future stage of the project.

---

# 12. 🔬 Final Planned Comparison

Once all three algorithms are implemented, the project will compare:

```text
                 Beam Selection
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
    Greedy            OMP             QAOA
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
              Performance Analysis
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      SINR             SE             EE
```

### Comparison Metrics

The final comparison will focus on:

1. **SINR**
2. **Spectral Efficiency**
3. **Energy Efficiency**

All algorithms will use identical:

* Channel conditions
* Number of antennas
* Number of users
* RF chains
* Candidate beams
* Power assumptions

This ensures a fair comparison between Greedy, OMP and QAOA.

---

# 13. 📌 Current Project Status

| Component                      | Status          |
| ------------------------------ | --------------- |
| System model                   | ✅ Completed     |
| Multi-user channel model       | ✅ Completed     |
| ULA antenna model              | ✅ Completed     |
| DFT beam codebook              | ✅ Completed     |
| User-beam gain matrix          | ✅ Completed     |
| Beam assignment formulation    | ✅ Completed     |
| Greedy algorithm               | ✅ Completed     |
| SINR evaluation                | ✅ Completed     |
| Spectral efficiency evaluation | ✅ Completed     |
| Energy efficiency evaluation   | ✅ Completed     |
| Greedy baseline results        | ✅ Completed     |
| OMP implementation             | 🔄 Next Stage   |
| QUBO formulation               | 🔄 Future Stage |
| QAOA implementation            | 🔄 Future Stage |
| Final algorithm comparison     | 🔄 Future Stage |

---

# 14. 🚀 Roadmap

### Phase 1 — System Modeling ✅

* Define massive MIMO system
* Generate multi-user mmWave channels
* Model 16-antenna ULA
* Generate DFT candidate beams
* Construct user-beam gain matrix

### Phase 2 — Classical Baseline ✅

* Formulate beam-to-RF-chain assignment
* Implement Greedy beam selection
* Select 8 user-beam pairs
* Calculate SINR
* Calculate spectral efficiency
* Calculate energy efficiency
* Generate baseline results

### Phase 3 — OMP 🔄

* Implement OMP
* Generate OMP beam assignments
* Evaluate SINR
* Evaluate spectral efficiency
* Evaluate energy efficiency
* Compare OMP with Greedy

### Phase 4 — QUBO/QAOA 🔄

* Convert beam-selection problem into binary optimization form
* Develop QUBO formulation
* Map the problem to QAOA
* Implement QAOA-based beam selection
* Evaluate performance

### Phase 5 — Final Comparison 🔄

Compare:

```text
Greedy vs OMP vs QAOA
```

using:

```text
SINR
Spectral Efficiency
Energy Efficiency
```

---

# 📚 References

The project is based on research related to:

* mmWave massive MIMO
* Hybrid beamforming
* DFT beam codebooks
* Beam selection
* Greedy optimization
* Future quantum optimization

Important references include:

1. O. El Ayach et al., *Spatially Sparse Precoding in Millimeter Wave MIMO Systems*, IEEE Transactions on Wireless Communications, 2014.

2. C.-X. Wang et al., *6G Wireless Channel Measurements and Models: Trends and Challenges*, IEEE Vehicular Technology Magazine, 2020.

3. A. F. Molisch et al., *Hybrid Beamforming for Massive MIMO: A Survey*, 2016.

4. Y. Huang et al., *DFT Codebook-Based Hybrid Precoding for Multiuser mmWave Massive MIMO Systems*, 2020.

5. D. Yang et al., *DFT-Based Beamforming Weight-Vector Codebook Design*, IEEE ICC, 2010.

The complete reference list is available in the `references/` folder and the Review 1 document.

---

# 👥 Project Team

### Bharath Jeyakkumar S

Computer Technology
Reg. No. 2024503013

### Bilu Besto H

Computer Technology
Reg. No. 2024503569

### Kamalesh T

Computer Technology
Reg. No. 2024503539

### Abdul Wahith M

Computer Technology
Reg. No. 2024503559

---

# 📌 Summary

The project currently has a complete **classical beam-selection baseline**.

The implemented pipeline is:

```text
Multi-user Channel Generation
            ↓
16-Antenna ULA
            ↓
DFT Beam Codebook
            ↓
20 × 16 User-Beam Gain Matrix
            ↓
Greedy Beam Selection
            ↓
8 Selected User-Beam Pairs
            ↓
SINR
            ↓
Spectral Efficiency
            ↓
Energy Efficiency
```

The current Greedy implementation provides the baseline against which the next optimization approaches will be evaluated.

The next major implementation stages are:

```text
        Greedy ✅
           ↓
         OMP 🔄
           ↓
      QUBO / QAOA 🔄
           ↓
 Final Performance Comparison
```

The ultimate goal is to determine whether optimization-based beam selection, particularly **QAOA**, can provide improvements in **SINR, spectral efficiency, and energy efficiency** compared with classical beam-selection methods.
