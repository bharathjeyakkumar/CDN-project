import numpy as np

# QUBO-only construction for the Quantum-Enhanced Hybrid Precoder project.
# Decision variable: x[k,m] = 1 if user k selects beam m, else 0.
# Final problem: minimize x^T Q x.

Nt = 8
K = 3
M = 3
N_RF = 2

noise_power = 1.0
user_power = 1.0

lam_I = 0.5
lam_F = 0.2
lam_A = 6.0

np.random.seed(42)

# Channel matrix H: K users x Nt antennas
H = (np.random.randn(K, Nt) + 1j * np.random.randn(K, Nt)) / np.sqrt(2)

print("=" * 60)
print("CHANNEL MATRIX H")
print("=" * 60)
print(H)


# Candidate beam codebook
def steering_vector(Nt, angle):
    n = np.arange(Nt)
    return np.exp(1j * np.pi * n * np.sin(angle)) / np.sqrt(Nt)


angles = np.linspace(-np.pi / 3, np.pi / 3, M)

# F: Nt x M, each column is a candidate beam
F = np.column_stack([steering_vector(Nt, angle) for angle in angles])

print("\n" + "=" * 60)
print("BEAM CODEBOOK F")
print("=" * 60)
print(F)


# User-beam gain:
# G[k,m] = |h_k^H f_m|
G = np.abs(H @ F)
G2 = G ** 2

print("\n" + "=" * 60)
print("USER-BEAM GAIN MATRIX G")
print("=" * 60)
print(G)

print("\n" + "=" * 60)
print("USER-BEAM POWER GAIN MATRIX G^2")
print("=" * 60)
print(G2)


# Flatten x[k,m] into one binary variable index.
N_VARS = K * M


def idx(k, m):
    return k * M + m


print("\n" + "=" * 60)
print("BINARY VARIABLES")
print("=" * 60)

for k in range(K):
    for m in range(M):
        print(f"x[{k},{m}] -> variable {idx(k, m)}")


# QUBO matrix for min x^T Q x
Q = np.zeros((N_VARS, N_VARS))


# 1. Signal / beam-gain reward
# Maximize sum G^2*x -> minimize -sum G^2*x
for k in range(K):
    for m in range(M):
        i = idx(k, m)
        Q[i, i] += -G2[k, m]


# 2. Interference penalty
# For two different users selecting two different beams:
# leakage = G[k,n]^2 + G[j,m]^2
for k in range(K):
    for j in range(k + 1, K):
        for m in range(M):
            for n in range(m + 1, M):
                i = idx(k, m)
                l = idx(j, n)

                leak = G2[k, n] + G2[j, m]
                coefficient = lam_I * leak

                # Split pair coefficient across symmetric Q entries.
                Q[i, l] += coefficient / 2
                Q[l, i] += coefficient / 2


# 3. Fairness proxy
mean_gain = G2.mean()

for k in range(K):
    for m in range(M):
        i = idx(k, m)
        fairness_cost = lam_F * (G2[k, m] - mean_gain) ** 2 * 0.01
        Q[i, i] += fairness_cost


# 4. Constraint: each user selects at most one beam
# Penalize selecting two beams for the same user.
for k in range(K):
    for m in range(M):
        for n in range(m + 1, M):
            i = idx(k, m)
            j = idx(k, n)

            Q[i, j] += lam_A / 2
            Q[j, i] += lam_A / 2


# 5. Constraint: each beam is assigned to at most one user
for m in range(M):
    for k in range(K):
        for j in range(k + 1, K):
            i = idx(k, m)
            l = idx(j, m)

            Q[i, l] += lam_A / 2
            Q[l, i] += lam_A / 2


# 6. Constraint: exactly N_RF assignments
# Penalty = lam_A * (sum_i x_i - N_RF)^2
# Constant term is omitted because it does not affect optimization.

for i in range(N_VARS):
    Q[i, i] += lam_A * (1 - 2 * N_RF)

for i in range(N_VARS):
    for j in range(i + 1, N_VARS):
        coefficient = 2 * lam_A
        Q[i, j] += coefficient / 2
        Q[j, i] += coefficient / 2


# Final QUBO matrix
np.set_printoptions(precision=4, suppress=True)

print("\n" + "=" * 60)
print("FINAL QUBO MATRIX Q")
print("=" * 60)
print(Q)

print("\n" + "=" * 60)
print("QUBO INFORMATION")
print("=" * 60)
print(f"Number of users       : {K}")
print(f"Number of antennas    : {Nt}")
print(f"Candidate beams       : {M}")
print(f"RF chains             : {N_RF}")
print(f"Number of variables   : {N_VARS}")

print("\nQUBO objective:")
print("    minimize x^T Q x")

print("\nDecision variable:")
print("    x[k,m] = 1 -> user k selects beam m")
print("    x[k,m] = 0 -> otherwise")

print("\nObjectives included:")
print("    1. Signal / beam gain")
print("    2. Interference penalty")
print("    3. Fairness penalty")

print("\nConstraints included:")
print("    1. Each user selects at most one beam")
print("    2. Each beam is assigned to at most one user")
print("    3. Exactly N_RF assignments are selected")

print("\n" + "=" * 60)
print("QUBO CONSTRUCTION COMPLETE")
print("=" * 60)
