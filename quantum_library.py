"""
Quantum Library - QAOA-based Implementation
Uses Quantum Approximate Optimization Algorithm (QAOA) to find optimal moves.
"""

import random
from typing import List, TYPE_CHECKING

import numpy as np
from itertools import combinations

# ----------------------------
# SciPy optimizer (real optimizer)
# ----------------------------
from scipy.optimize import minimize

# ----------------------------
# Qrisp + simulation
# ----------------------------
from qrisp import QuantumVariable, h
from qrisp.qaoa import RX_mixer
from qrisp.operators.qubit import Z
from qiskit.quantum_info import Statevector

if TYPE_CHECKING:
    from battlefield import Soldier

def quantum_random_walk(current_position):
    """
    Classic random walk implementation.
    """
    x, y = current_position
    dx = random.randint(-1, 1)
    dy = random.randint(-1, 1)
    return (x + dx, y + dy)

# Dictionary to store available algorithms
ALGORITHMS = {
    'random_walk': quantum_random_walk,
}

def quantum_best_move(soldier: "Soldier", enemies: List["Soldier"], algorithm: str = "quantum_step", teammates: List["Soldier"] = None) -> tuple:
    """
    Determine the best move for a soldier using quantum algorithms.

    Builds the omega dictionary from H gradient values and uses quantum_step
    to find the optimal move based on offensive/defensive positioning.

    Args:
        soldier: The soldier making the move.
        enemies: List of enemy soldiers on the battlefield.
        algorithm: Algorithm to use (default: "quantum_step").
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Tuple (new_x, new_y) with the new position.
    """
    if algorithm == "quantum_step":
        # Calculate H gradients for all 4 cardinal directions
        h_left = calculate_h_left(soldier, enemies, teammates)   # I (Izquierda)
        h_right = calculate_h_right(soldier, enemies, teammates)  # D (Derecha)
        h_up = calculate_h_up(soldier, enemies, teammates)        # + (Arriba)
        h_down = calculate_h_down(soldier, enemies, teammates)    # - (Abajo)

        # Build omega dictionary
        # Negative gradient = more offensive direction (preferred)
        # Positive gradient = more vulnerable direction (avoided)
        omega = {
            "I": int(h_left),   # Left (x-1)
            "D": int(h_right),  # Right (x+1)
            "+": int(h_up),     # Up (y+1)
            "-": int(h_down)    # Down (y-1)
        }

        # Call quantum_step with current position and omega
        new_x, new_y = quantum_step(soldier.x, soldier.y, omega)
        return (new_x, new_y)

    elif algorithm in ALGORITHMS:
        # Fallback to legacy algorithms
        current_position = (soldier.x, soldier.y)
        algorithm_func = ALGORITHMS[algorithm]
        return algorithm_func(current_position)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys()) + ['quantum_step']}")

# Calcular ofensiva y vulnerabilidad
def calculate_offensive_power(soldier: "Soldier", enemies: List["Soldier"]) -> int:
    """
    Calculate the offensive power of a soldier towards enemies in range.

    Args:
        soldier: The soldier whose offensive power is being calculated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Total offensive power of the soldier.
    """
    offensive_power = -1

    weakest = soldier.find_weakest_to_attack(enemies)

    # si encuentra weakest, compara si el ataque del soldado es suficiente para matar al mas debil, en ese caso devuelve 1, si no -1
    if weakest:
        if soldier.strength >= weakest.health:
            offensive_power = 1  # Can kill the weakest enemy
        # else: offensive_power stays -1 (cannot kill)

    return offensive_power

def calculate_vulnerability(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the vulnerability of a soldier based on enemies' offensive power.

    Args:
        soldier: The soldier whose vulnerability is being calculated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Total vulnerability of the soldier.
    """
    vulnerability = -1

    # si la suma de las fuerzas de los enemigos es mayor que 2 veces tu vida, devuelve 1, si no devuelve -1
    total_enemy_strength = 0
    for enemy in enemies:
        if enemy.can_fight(soldier):
            total_enemy_strength += enemy.strength

    if total_enemy_strength > 2 * soldier.health:
        vulnerability = 1  # Very vulnerable (enemies can overwhelm)
    # else: vulnerability stays -1 (not critically vulnerable)

    return vulnerability

# Calcular las H
def calculate_h_value(soldier: "Soldier", enemies: List["Soldier"], teammates: List["Soldier"] = None) -> float:
    """
    Calculate the h heuristic value for a soldier based on vulnerability vs offensive power.

    Higher h value indicates a more vulnerable/defensive position.
    Lower h value indicates a more offensive/aggressive position.

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Heuristic value h (higher = more vulnerable, lower = more offensive).
    """
    offensive = calculate_offensive_power(soldier, enemies)
    vulnerability = calculate_vulnerability(soldier, enemies)

    # si somos menos soldados defensivo, retornamos offensive - (unidades de diferencia) * vulnerability
    # si mas soldados ofensivo, retornamos (unidades de diferencia) * offensive - vulnerability

    if teammates is not None:
        num_allies = len(teammates)
        num_enemies = len(enemies)
        unit_difference = num_allies - num_enemies

        if unit_difference < 0:
            # Menos soldados (defensivo)
            return offensive - abs(unit_difference) * vulnerability
        else:
            # Más soldados o igual (ofensivo)
            return unit_difference * offensive - vulnerability
    else:
        # Fallback a la lógica anterior si no hay información de teammates
        return 2 * offensive - vulnerability

# Calcular h arriba, que es la diferencia de calcular el valor h desplazando el valor de la variable y de nuestro soldado una casilla hacia arriba menos el valor de h sin desplazar ninguna variable.
def calculate_h_up(soldier: "Soldier", enemies: List["Soldier"], teammates: List["Soldier"] = None) -> float:
    """
    Calculate the h gradient when moving up (y+1).

    Returns the difference: h(y+1) - h(y)
    Positive value means h increases when moving up (becoming more vulnerable).
    Negative value means h decreases when moving up (becoming more offensive).

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Gradient of h in the upward direction.
    """
    from battlefield import Soldier

    # Calculate h at current position
    h_current = calculate_h_value(soldier, enemies, teammates)

    # Create a temporary soldier with y shifted up by 1
    soldier_shifted = Soldier(
        x=soldier.x,
        y=soldier.y + 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    # Calculate h at shifted position
    h_shifted = calculate_h_value(soldier_shifted, enemies, teammates)

    # Return the difference
    return h_shifted - h_current

def calculate_h_down(soldier: "Soldier", enemies: List["Soldier"], teammates: List["Soldier"] = None) -> float:
    """
    Calculate the h gradient when moving down (y-1).

    Returns the difference: h(y-1) - h(y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Gradient of h in the downward direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies, teammates)

    soldier_shifted = Soldier(
        x=soldier.x,
        y=soldier.y - 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies, teammates)
    return h_shifted - h_current

def calculate_h_left(soldier: "Soldier", enemies: List["Soldier"], teammates: List["Soldier"] = None) -> float:
    """
    Calculate the h gradient when moving left (x-1).

    Returns the difference: h(x-1) - h(x)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Gradient of h in the left direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies, teammates)

    soldier_shifted = Soldier(
        x=soldier.x - 1,
        y=soldier.y,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies, teammates)
    return h_shifted - h_current

def calculate_h_right(soldier: "Soldier", enemies: List["Soldier"], teammates: List["Soldier"] = None) -> float:
    """
    Calculate the h gradient when moving right (x+1).

    Returns the difference: h(x+1) - h(x)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.
        teammates: List of ally soldiers on the battlefield (optional).

    Returns:
        Gradient of h in the right direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies, teammates)

    soldier_shifted = Soldier(
        x=soldier.x + 1,
        y=soldier.y,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies, teammates)
    return h_shifted - h_current

def calculate_h_right_up(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the h gradient when moving diagonally right-up (x+1, y+1).

    Returns the difference: h(x+1, y+1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the right-up diagonal direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies)

    soldier_shifted = Soldier(
        x=soldier.x + 1,
        y=soldier.y + 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_right_down(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the h gradient when moving diagonally right-down (x+1, y-1).

    Returns the difference: h(x+1, y-1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the right-down diagonal direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies)

    soldier_shifted = Soldier(
        x=soldier.x + 1,
        y=soldier.y - 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_left_up(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the h gradient when moving diagonally left-up (x-1, y+1).

    Returns the difference: h(x-1, y+1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the left-up diagonal direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies)

    soldier_shifted = Soldier(
        x=soldier.x - 1,
        y=soldier.y + 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_left_down(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the h gradient when moving diagonally left-down (x-1, y-1).

    Returns the difference: h(x-1, y-1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the left-down diagonal direction.
    """
    from battlefield import Soldier

    h_current = calculate_h_value(soldier, enemies)

    soldier_shifted = Soldier(
        x=soldier.x - 1,
        y=soldier.y - 1,
        strength=soldier.strength,
        health=soldier.health,
        unit_type=soldier.unit_type,
        team=soldier.team,
        range_dist=soldier.range_dist,
        max_health=soldier.max_health
    )

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

# =========================================================
# QAOA-based Quantum Step Functions
# =========================================================

# =========================================================
# CONFIGURATION: Choose quantum algorithm
# =========================================================
# Set this to "qaoa" or "exact" to switch between methods
QUANTUM_METHOD = "qaoa"  # Options: "qaoa" or "exact"

# QAOA parameters (only used if QUANTUM_METHOD = "qaoa")
QAOA_P = 1              # QAOA depth (layers)
QAOA_N_RESTARTS = 2     # Number of optimizer restarts

# Global cache for QAOA results to avoid recomputation
_qaoa_cache = {}

def clear_qaoa_cache():
    """Clear the QAOA cache. Useful when changing parameters."""
    global _qaoa_cache
    _qaoa_cache = {}

def set_quantum_method(method: str):
    """
    Change the quantum method used for decision-making.

    Args:
        method: "qaoa" for QAOA optimization, "exact" for exact diagonalization
    """
    global QUANTUM_METHOD
    if method not in ["qaoa", "exact"]:
        raise ValueError(f"Method must be 'qaoa' or 'exact', got '{method}'")
    QUANTUM_METHOD = method
    print(f"Quantum method set to: {method}")
    if method == "exact":
        clear_qaoa_cache()  # Clear cache when switching to exact

def build_qrisp_H_operator_no_xx(omega, K_ID=1000, K_pm=1000):
    """
    Build Hamiltonian (Qrisp operator): only Z fields + |11><11| penalties.
    H = sum_a omega[a] Z_a + K_ID |11><11|_{I,D} + K_pm |11><11|_{+,-}
    No XX interaction terms.
    """
    labels = ["I", "D", "+", "-"]
    idx = {lab: i for i, lab in enumerate(labels)}

    H = 0
    # local Z fields
    for a in labels:
        H += omega[a] * Z(idx[a])

    # |11><11|_{ij} = (1/4)(I - Zi - Zj + ZiZj)
    def P11(i, j):
        return 0.25 * (1 - Z(i) - Z(j) + Z(i) * Z(j))

    H += K_ID * P11(idx["I"], idx["D"])
    H += K_pm * P11(idx["+"], idx["-"])

    return H, idx


def classical_exact_ground_state(omega, K_ID=1000, K_pm=1000):
    """
    Enumerate all 16 computational basis states and find
    the exact classical ground state(s).
    """
    Hmat, idx = build_numpy_H_no_xx(omega, K_ID=K_ID, K_pm=K_pm)
    diagE = np.real(np.diag(Hmat))

    Emin = diagE.min()
    ground_states = []

    for k, E in enumerate(diagE):
        if np.isclose(E, Emin):
            ground_states.append(format(k, "04b"))

    return Emin, ground_states, diagE


def build_numpy_H_no_xx(omega, K_ID=1000, K_pm=1000):
    """
    Build full 16x16 matrix for H with only Z terms + penalties.
    This is diagonal in computational basis.
    """
    labels = ["I", "D", "+", "-"]
    idx = {lab: i for i, lab in enumerate(labels)}
    n = 4
    dim = 2**n

    # Precompute Z eigenvalues for each basis state: Z|0>=+|0>, Z|1>=-|1>
    Hdiag = np.zeros(dim, dtype=float)

    def z_eig(bit):
        return +1.0 if bit == 0 else -1.0

    for k in range(dim):
        bits = [(k >> (n - 1 - q)) & 1 for q in range(n)]  # [I,D,+,-] bits

        # local fields: omega[a] * Z_a
        E = 0.0
        for a in labels:
            q = idx[a]
            E += omega[a] * z_eig(bits[q])

        # penalties: add K if (I,D) are both 1 and/or (+,-) both 1
        Ibit, Dbit, pbit, mbit = bits[idx["I"]], bits[idx["D"]], bits[idx["+"]], bits[idx["-"]]
        if Ibit == 1 and Dbit == 1:
            E += K_ID
        if pbit == 1 and mbit == 1:
            E += K_pm

        Hdiag[k] = E

    # Return as diagonal matrix for expectation <psi|H|psi>
    H = np.diag(Hdiag.astype(complex))
    return H, idx


def qaoa_statevector(omega, gammas, betas, K_ID=1000, K_pm=1000):
    """
    Build p-layer QAOA circuit:
      |+>^n
      for l=1..p: exp(-i gamma_l H) then RX mixer with beta_l
    Returns statevector (numpy array).
    """
    p = len(gammas)
    if len(betas) != p:
        raise ValueError("gammas and betas must have same length (depth p).")

    H_op, _ = build_qrisp_H_operator_no_xx(omega, K_ID=K_ID, K_pm=K_pm)

    # H is commuting/diagonal -> trotterization is exact with 1 step in practice
    U_cost = H_op.trotterization(order=1, method="commuting_qw", forward_evolution=True)

    qv = QuantumVariable(4)
    h(qv)  # |+>^4

    for l in range(p):
        U_cost(qv, t=float(gammas[l]), steps=1)
        RX_mixer(qv, float(betas[l]))

    qc = qv.qs.compile().to_qiskit()
    sv = Statevector.from_instruction(qc).data
    return sv


def energy_expectation(Hmat, sv):
    """Calculate energy expectation <psi|H|psi>."""
    return float(np.real(np.vdot(sv, Hmat @ sv)))


def most_likely_bitstring(sv, n_qubits=4):
    """Get the most likely bitstring from statevector."""
    probs = np.abs(sv) ** 2
    k = int(np.argmax(probs))
    return format(k, f"0{n_qubits}b"), float(probs[k])


def qaoa_optimize_no_xx(
    omega,
    p=2,
    K_ID=1000,
    K_pm=1000,
    n_restarts=10,
    seed=0,
):
    """
    Minimize <H> over QAOA angles using scipy.optimize.minimize with bounds.
    Returns best solution dict.
    """
    rng = np.random.default_rng(seed)
    Hmat, idx = build_numpy_H_no_xx(omega, K_ID=K_ID, K_pm=K_pm)

    # parameter vector theta = [gammas(0..p-1), betas(0..p-1)]
    # bounds: gammas in [0, 2pi], betas in [0, pi]
    bounds = [(0.0, 2*np.pi)] * p + [(0.0, np.pi)] * p

    def objective(theta):
        gammas = theta[:p]
        betas = theta[p:]
        sv = qaoa_statevector(omega, gammas, betas, K_ID=K_ID, K_pm=K_pm)
        return energy_expectation(Hmat, sv)

    best = {"E": np.inf, "theta": None, "bit": None, "pbit": None}

    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(0, 2*np.pi) for _ in range(p)] +
                      [rng.uniform(0, np.pi)   for _ in range(p)], dtype=float)

        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)

        if res.fun < best["E"]:
            gammas = res.x[:p]
            betas = res.x[p:]
            sv = qaoa_statevector(omega, gammas, betas, K_ID=K_ID, K_pm=K_pm)
            bit, pbit = most_likely_bitstring(sv, n_qubits=4)
            best.update({"E": float(res.fun), "theta": res.x, "bit": bit, "pbit": pbit})

    return best, idx


# ----------------------------
# Legacy Pauli matrices (kept for compatibility)
# ----------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],
               [1, 0]], dtype=complex)
Z_mat  = np.array([[1,  0],
               [0, -1]], dtype=complex)

def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def one_body(n, i, op):
    ops = [I2] * n
    ops[i] = op
    return kron_n(ops)

def two_body(n, i, j, op_i, op_j):
    ops = [I2] * n
    ops[i] = op_i
    ops[j] = op_j
    return kron_n(ops)

def projector_11(n, i, j):
    """
    Projector onto |11> on qubits (i,j):
    P_11 = (1/4)(I - Z_i - Z_j + Z_i Z_j)
    Penalizes ONLY the |11> state in the computational basis.
    """
    dim = 2**n
    I_full = np.eye(dim, dtype=complex)
    Zi = one_body(n, i, Z_mat)
    Zj = one_body(n, j, Z_mat)
    ZiZj = two_body(n, i, j, Z_mat, Z_mat)
    return 0.25 * (I_full - Zi - Zj + ZiZj)

# =========================================================
# QUANTUM STEP IMPLEMENTATIONS
# =========================================================

def quantum_step_exact(x, y, omega, seed=42, K_ID=1000, K_pm=1000):
    """
    Perform one quantum decision step using EXACT diagonalization.
    This is the original fast method that finds the exact ground state.

    Inputs:
      x, y   : current position (ints)
      omega  : dict { "I","D","+","-" : int }
      seed   : RNG seed (not used in exact method, kept for compatibility)
      K_ID   : penalty for I=D=1
      K_pm   : penalty for +=-=1

    Returns:
      new_x, new_y
    """
    labels = ["I", "D", "+", "-"]
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    # Build Hamiltonian (without XX interactions for speed)
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    # Z fields only
    for a in labels:
        H += omega[a] * one_body(n, idx[a], Z_mat)

    # Constraint penalties
    H += K_ID * projector_11(n, idx["I"], idx["D"])
    H += K_pm * projector_11(n, idx["+"], idx["-"])

    # Exact diagonalization
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]  # Ground state

    # Most likely bitstring
    probs = np.abs(psi0)**2
    k_map = int(np.argmax(probs))
    bit_map = format(k_map, "04b")

    # Interpret bitstring as move
    I_bit, D_bit, plus_bit, minus_bit = map(int, bit_map)

    new_x, new_y = x, y
    if I_bit == 1:
        new_x -= 1
    if D_bit == 1:
        new_x += 1
    if plus_bit == 1:
        new_y += 1
    if minus_bit == 1:
        new_y -= 1

    return new_x, new_y


def quantum_step_qaoa(x, y, omega, seed=42, K_ID=1000, K_pm=1000):
    """
    Perform one quantum decision step using QAOA optimization.
    This uses variational quantum algorithms (more realistic for real quantum hardware).

    Inputs:
      x, y   : current position (ints)
      omega  : dict { "I","D","+","-" : int }
      seed   : RNG seed for optimizer initialization
      K_ID   : penalty for I=D=1
      K_pm   : penalty for +=-=1

    Returns:
      new_x, new_y

    Note: Uses global QAOA_P and QAOA_N_RESTARTS parameters
    """
    # Create a cache key from omega values
    omega_tuple = tuple(sorted(omega.items()))
    cache_key = (omega_tuple, K_ID, K_pm, QAOA_P, QAOA_N_RESTARTS)

    # Check cache first
    global _qaoa_cache
    if cache_key in _qaoa_cache:
        best = _qaoa_cache[cache_key]
    else:
        # Run QAOA optimization to find optimal bitstring
        best, idx = qaoa_optimize_no_xx(
            omega=omega,
            p=QAOA_P,
            K_ID=K_ID,
            K_pm=K_pm,
            n_restarts=QAOA_N_RESTARTS,
            seed=seed
        )
        # Store in cache (limit cache size to avoid memory issues)
        if len(_qaoa_cache) < 1000:
            _qaoa_cache[cache_key] = best

    # Get the most likely bitstring from QAOA
    bit_map = best["bit"]

    # Interpret bitstring as move
    I_bit, D_bit, plus_bit, minus_bit = map(int, bit_map)

    new_x, new_y = x, y
    if I_bit == 1:
        new_x -= 1
    if D_bit == 1:
        new_x += 1
    if plus_bit == 1:
        new_y += 1
    if minus_bit == 1:
        new_y -= 1

    return new_x, new_y


def quantum_step(x, y, omega, seed=42, K_ID=1000, K_pm=1000):
    """
    Main quantum step function - dispatches to either exact or QAOA method.

    The method used is determined by the global QUANTUM_METHOD variable.
    Change it at the top of this file or use set_quantum_method().

    Inputs:
      x, y   : current position (ints)
      omega  : dict { "I","D","+","-" : int }
      seed   : RNG seed
      K_ID   : penalty for I=D=1
      K_pm   : penalty for +=-=1

    Returns:
      new_x, new_y
    """
    if QUANTUM_METHOD == "qaoa":
        return quantum_step_qaoa(x, y, omega, seed, K_ID, K_pm)
    elif QUANTUM_METHOD == "exact":
        return quantum_step_exact(x, y, omega, seed, K_ID, K_pm)
    else:
        raise ValueError(f"Unknown QUANTUM_METHOD: {QUANTUM_METHOD}")