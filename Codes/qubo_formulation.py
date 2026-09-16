import numpy as np


# ============================================================
# BUILD QUBO MATRIX
# ============================================================

def build_qubo_matrix(
    G,
    n_rf_chains,
    user_penalty=None,
    beam_penalty=None,
    chain_penalty=None
):
    """
    Build QUBO matrix for user-beam assignment.

    Decision variable:

        x[k,m] = 1
            User k is assigned Beam m

        x[k,m] = 0
            Otherwise

    Objective:

        Maximize:
            sum G[k,m] * x[k,m]

        Since QUBO is minimized:

            Minimize:
                -sum G[k,m] * x[k,m]

    Constraints:

        1. One user can receive at most one beam.
        2. One beam can be assigned to at most one user.
        3. Exactly n_rf_chains assignments must be selected.

    Parameters
    ----------
    G : np.ndarray
        Gain matrix of shape (K, M).

    n_rf_chains : int
        Number of RF chains / assignments.

    user_penalty : float
        Penalty for assigning multiple beams to the same user.

    beam_penalty : float
        Penalty for assigning the same beam to multiple users.

    chain_penalty : float
        Penalty for selecting a number of assignments different
        from n_rf_chains.

    Returns
    -------
    Q : np.ndarray
        Symmetric QUBO matrix of shape (K*M, K*M).

    variable_map : dict
        Maps variable index -> (user, beam).

    reverse_map : dict
        Maps (user, beam) -> variable index.
    """

    # --------------------------------------------------------
    # Convert G to numpy array
    # --------------------------------------------------------

    G = np.asarray(G, dtype=float)

    if G.ndim != 2:
        raise ValueError("G must be a 2-dimensional matrix.")

    K, M = G.shape

    n_variables = K * M

    # --------------------------------------------------------
    # Validate number of RF chains
    # --------------------------------------------------------

    if n_rf_chains <= 0:
        raise ValueError("n_rf_chains must be greater than 0.")

    if n_rf_chains > min(K, M):
        raise ValueError(
            "n_rf_chains cannot be greater than "
            "the number of users or beams."
        )

    # --------------------------------------------------------
    # Penalty values
    # --------------------------------------------------------

    max_gain = np.max(G)

    if max_gain <= 0:
        max_gain = 1.0

    if user_penalty is None:
        user_penalty = 2.0 * max_gain

    if beam_penalty is None:
        beam_penalty = 2.0 * max_gain

    if chain_penalty is None:
        chain_penalty = 2.0 * max_gain

    # --------------------------------------------------------
    # Create QUBO matrix
    #
    # Energy:
    #
    #       E = x^T Q x
    #
    # Q is symmetric.
    #
    # Therefore, if a quadratic term has coefficient C:
    #
    #       C * xi * xj
    #
    # we put:
    #
    #       Q[i,j] += C/2
    #       Q[j,i] += C/2
    #
    # because x^T Q x counts both.
    # --------------------------------------------------------

    Q = np.zeros(
        (n_variables, n_variables),
        dtype=float
    )

    # --------------------------------------------------------
    # Variable mapping
    # --------------------------------------------------------

    variable_map = {}
    reverse_map = {}

    index = 0

    for k in range(K):

        for m in range(M):

            variable_map[index] = (k, m)

            reverse_map[(k, m)] = index

            index += 1

    # ========================================================
    # 1. OBJECTIVE FUNCTION
    # ========================================================

    # Maximize:
    #
    #       sum G[k,m] x[k,m]
    #
    # Convert to minimization:
    #
    #       -sum G[k,m] x[k,m]
    #
    # Therefore diagonal:
    #
    #       Q[i,i] += -G[k,m]
    # ========================================================

    for k in range(K):

        for m in range(M):

            i = reverse_map[(k, m)]

            Q[i, i] += -G[k, m]

    # ========================================================
    # 2. USER CONSTRAINT
    # ========================================================

    # At most one beam per user:
    #
    #       sum_m x[k,m] <= 1
    #
    # Penalty:
    #
    #       Pu * xi * xj
    #
    # for every pair of beams belonging to the same user.
    #
    # Because Q is symmetric:
    #
    #       Q[i,j] += Pu/2
    #       Q[j,i] += Pu/2
    # ========================================================

    for k in range(K):

        for m1 in range(M):

            for m2 in range(m1 + 1, M):

                i = reverse_map[(k, m1)]
                j = reverse_map[(k, m2)]

                Q[i, j] += user_penalty / 2.0
                Q[j, i] += user_penalty / 2.0

    # ========================================================
    # 3. BEAM CONSTRAINT
    # ========================================================

    # At most one user per beam:
    #
    #       sum_k x[k,m] <= 1
    #
    # Penalty:
    #
    #       Pb * xi * xj
    #
    # for every pair of users using the same beam.
    # ========================================================

    for m in range(M):

        for k1 in range(K):

            for k2 in range(k1 + 1, K):

                i = reverse_map[(k1, m)]
                j = reverse_map[(k2, m)]

                Q[i, j] += beam_penalty / 2.0
                Q[j, i] += beam_penalty / 2.0

    # ========================================================
    # 4. EXACTLY N_RF ASSIGNMENTS
    # ========================================================

    # Constraint:
    #
    #       sum_i xi = N_RF
    #
    # Penalty:
    #
    #       Pc (sum_i xi - N_RF)^2
    #
    # Expand:
    #
    #       Pc [
    #           sum_i xi
    #           + 2 sum_(i<j) xi*xj
    #           - 2*N_RF*sum_i xi
    #           + N_RF^2
    #       ]
    #
    # Since:
    #
    #       xi^2 = xi
    #
    # Linear coefficient:
    #
    #       Pc(1 - 2*N_RF)
    #
    # Quadratic coefficient:
    #
    #       2*Pc
    #
    # Since Q is symmetric, each side receives:
    #
    #       (2*Pc)/2 = Pc
    #
    # ========================================================

    # Linear terms
    for i in range(n_variables):

        Q[i, i] += (
            chain_penalty
            * (1 - 2 * n_rf_chains)
        )

    # Quadratic terms
    for i in range(n_variables):

        for j in range(i + 1, n_variables):

            Q[i, j] += chain_penalty
            Q[j, i] += chain_penalty

    # --------------------------------------------------------
    # Return QUBO
    # --------------------------------------------------------

    return Q, variable_map, reverse_map


# ============================================================
# CONVERT BITSTRING TO USER-BEAM ASSIGNMENT
# ============================================================

def bitstring_to_assignment(
    bitstring,
    variable_map
):
    """
    Convert a binary solution into:

        [(user, beam), ...]

    Example:

        [1,0,0,0,1,0]

    becomes:

        [(0,0), (1,1)]
    """

    assignment = []

    for i, bit in enumerate(bitstring):

        if int(bit) == 1:

            user, beam = variable_map[i]

            assignment.append(
                (user, beam)
            )

    return assignment


# ============================================================
# CALCULATE QUBO ENERGY
# ============================================================

def calculate_qubo_energy(
    Q,
    bitstring
):
    """
    Calculate:

        E = x^T Q x
    """

    x = np.asarray(
        bitstring,
        dtype=float
    )

    if len(x) != Q.shape[0]:
        raise ValueError(
            "Bitstring length must match QUBO dimension."
        )

    return float(
        x @ Q @ x
    )


# ============================================================
# CHECK ASSIGNMENT VALIDITY
# ============================================================

def check_assignment(
    assignment,
    n_users,
    n_beams,
    n_rf_chains
):
    """
    Check whether an assignment satisfies:

        1. One beam per user
        2. One user per beam
        3. Exactly n_rf_chains assignments
    """

    users = [user for user, beam in assignment]

    beams = [beam for user, beam in assignment]

    # Number of assignments
    correct_count = (
        len(assignment) == n_rf_chains
    )

    # No duplicate users
    unique_users = (
        len(users) == len(set(users))
    )

    # No duplicate beams
    unique_beams = (
        len(beams) == len(set(beams))
    )

    valid = (
        correct_count
        and unique_users
        and unique_beams
    )

    return valid


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # PROJECT PARAMETERS
    # --------------------------------------------------------

    K = 20              # Number of users
    M = 16              # Number of candidate beams
    N_RF = 8            # Number of RF chains

    # --------------------------------------------------------
    # Generate example gain matrix
    #
    # IMPORTANT:
    # This is only for testing.
    #
    # In your actual project, replace this with your
    # real gain matrix generated from the channel model.
    # --------------------------------------------------------

    np.random.seed(42)

    G = np.random.rand(K, M)

    # --------------------------------------------------------
    # Print gain matrix
    # --------------------------------------------------------

    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=200
    )

    print("=" * 70)
    print("GAIN MATRIX")
    print("=" * 70)

    print("Shape:", G.shape)

    print(G)

    # --------------------------------------------------------
    # Build QUBO
    # --------------------------------------------------------

    Q, variable_map, reverse_map = build_qubo_matrix(
        G,
        n_rf_chains=N_RF
    )

    # --------------------------------------------------------
    # Print QUBO information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUBO INFORMATION")
    print("=" * 70)

    print("Number of users:", K)
    print("Number of beams:", M)
    print("Number of RF chains:", N_RF)

    print(
        "Number of binary variables:",
        K * M
    )

    print(
        "QUBO matrix shape:",
        Q.shape
    )

    # --------------------------------------------------------
    # Print complete QUBO matrix
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLETE QUBO MATRIX")
    print("=" * 70)

    print(Q)

    # --------------------------------------------------------
    # Save QUBO matrix
    # --------------------------------------------------------

    np.savetxt(
        "qubo_matrix.csv",
        Q,
        delimiter=",",
        fmt="%.6f"
    )

    print("\nQUBO matrix saved as:")
    print("qubo_matrix.csv")

    # --------------------------------------------------------
    # Variable mapping
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VARIABLE MAPPING")
    print("=" * 70)

    print("\nFirst 20 variables:")

    for i in range(min(20, len(variable_map))):

        user, beam = variable_map[i]

        print(
            f"x[{i}] -> "
            f"User {user}, Beam {beam}"
        )

    # --------------------------------------------------------
    # Example bitstring
    # --------------------------------------------------------

    example_bitstring = np.zeros(
        K * M,
        dtype=int
    )

    # Select 8 valid user-beam pairs
    #
    # User 0 -> Beam 0
    # User 1 -> Beam 1
    # ...
    # User 7 -> Beam 7

    for k in range(N_RF):

        index = reverse_map[(k, k)]

        example_bitstring[index] = 1

    # --------------------------------------------------------
    # Convert bitstring to assignment
    # --------------------------------------------------------

    assignment = bitstring_to_assignment(
        example_bitstring,
        variable_map
    )

    print("\n" + "=" * 70)
    print("EXAMPLE ASSIGNMENT")
    print("=" * 70)

    print("Bitstring:")
    print(example_bitstring)

    print("\nAssignment:")

    for user, beam in assignment:

        print(
            f"User {user} -> Beam {beam}"
        )

    # --------------------------------------------------------
    # Calculate energy
    # --------------------------------------------------------

    energy = calculate_qubo_energy(
        Q,
        example_bitstring
    )

    print("\nQUBO Energy:")

    print(energy)

    # --------------------------------------------------------
    # Check validity
    # --------------------------------------------------------

    valid = check_assignment(
        assignment,
        K,
        M,
        N_RF
    )

    print("\nAssignment valid:", valid)

    print("\n" + "=" * 70)
    print("QUBO GENERATION COMPLETED")
    print("=" * 70)