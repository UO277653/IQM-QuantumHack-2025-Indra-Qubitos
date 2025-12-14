"""
Quantum Library - Mock Implementation
This is a placeholder/mock implementation of the quantum move calculation.
Replace this with actual quantum algorithm implementation.
"""

import random
from typing import List, TYPE_CHECKING

import numpy as np
from itertools import combinations

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
        1 if the soldier can kill the weakest enemy in range, -1 otherwise.
    """
    offensive_power = -1

    weakest = soldier.find_weakest_to_attack(enemies)

    # Check if the soldier can kill the weakest enemy
    if weakest:
        if soldier.strength >= weakest.health:
            offensive_power = 1  # Can kill the weakest enemy

    return offensive_power

def calculate_vulnerability(soldier: "Soldier", enemies: List["Soldier"]) -> float:
    """
    Calculate the vulnerability of a soldier based on enemies' offensive power.

    Args:
        soldier: The soldier whose vulnerability is being calculated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        1 if the total enemy strength is greater than twice the soldier's health, -1 otherwise.
    """
    vulnerability = -1

    # Calculate the total strength of enemies that can attack the soldier
    total_enemy_strength = 0
    for enemy in enemies:
        if enemy.can_fight(soldier):
            total_enemy_strength += enemy.strength

    if total_enemy_strength > 2 * soldier.health:
        vulnerability = 1  # Very vulnerable (enemies can overwhelm)

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

# ----------------------------
# Pauli matrices
# ----------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],
               [1, 0]], dtype=complex)
Z  = np.array([[1,  0],
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
    Zi = one_body(n, i, Z)
    Zj = one_body(n, j, Z)
    ZiZj = two_body(n, i, j, Z, Z)
    return 0.25 * (I_full - Zi - Zj + ZiZj)

# =========================================================
# MAIN FUNCTION YOU ASKED FOR
# =========================================================
def quantum_step(x, y, omega, seed=42, K_ID=1000, K_pm=1000):
    """
    Perform one quantum decision step.

    Inputs:
      x, y   : current position (ints)
      omega  : dict { "I","D","+","-" : int }
      seed   : RNG seed for J couplings
      K_ID   : penalty for I=D=1
      K_pm   : penalty for +=-=1

    Returns:
      new_x, new_y
    """

    # ----------------------------
    # Spin labels and indexing
    # ----------------------------
    labels = ["I", "D", "+", "-"]
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    # ----------------------------
    # Random XX couplings
    # ----------------------------
    rng = np.random.default_rng(seed)
    J = {}
    for a, b in combinations(labels, 2):
        J[(a, b)] = 0

    # ----------------------------
    # Build Hamiltonian
    # ----------------------------
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    # Z fields
    for a in labels:
        H += omega[a] * one_body(n, idx[a], Z)

    # XX interactions
    for a, b in combinations(labels, 2):
        ia, ib = idx[a], idx[b]
        H += J[(a, b)] * two_body(n, ia, ib, X, X)

    # Constraint penalties
    H += K_ID * projector_11(n, idx["I"], idx["D"])
    H += K_pm * projector_11(n, idx["+"], idx["-"])

    # ----------------------------
    # Diagonalize
    # ----------------------------
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]

    # ----------------------------
    # MOST LIKELY BITSTRING (MAP)
    # ----------------------------
    probs = np.abs(psi0)**2
    k_map = int(np.argmax(probs))
    bit_map = format(k_map, "04b")  # |I D + ->
    p_map = probs[k_map]

    # ----------------------------
    # INTERPRET BITSTRING AS (x, y) MOVE
    # ----------------------------
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

    # Optional debug output
    #print("\nQuantum step:")
    #print(f"  MAP bitstring = |{bit_map}>  (p = {p_map:.6f})")
    #print(f"  (x, y) : ({x}, {y}) → ({new_x}, {new_y})")

    return new_x, new_y