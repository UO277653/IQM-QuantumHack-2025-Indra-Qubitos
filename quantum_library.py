"""
Quantum Library - Mock Implementation
This is a placeholder/mock implementation of the quantum move calculation.
Replace this with actual quantum algorithm implementation.
"""

import random
from typing import List

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

def quantum_best_move(current_position, quantum_algorithm):
    """
    Dynamically selects and applies a specified quantum algorithm to determine the best move.
    """
    algorithm_func = ALGORITHMS.get(quantum_algorithm)

    if algorithm_func:
        return algorithm_func(current_position)
    else:
        raise ValueError(f"Unknown algorithm: {quantum_algorithm}. Available algorithms are: {list(ALGORITHMS.keys())}")

# Calcular ofensiva y vulnerabilidad
def calculate_offensive_power(soldier: Soldier, enemies: List[Soldier]) -> int:
    """
    Calculate the offensive power of a soldier towards enemies in range.

    Args:
        soldier: The soldier whose offensive power is being calculated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Total offensive power of the soldier.
    """
    offensive_power = 0
    for enemy in enemies:
        if soldier.can_fight(enemy):
            offensive_power += soldier.strength
    return offensive_power

def calculate_vulnerability(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the vulnerability of a soldier based on enemies' offensive power.

    Args:
        soldier: The soldier whose vulnerability is being calculated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Total vulnerability of the soldier.
    """
    vulnerability = 0
    for enemy in enemies:
        if enemy.can_fight(soldier):
            vulnerability += enemy.strength
    return vulnerability

# Calcular las H
def calculate_h_value(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h heuristic value for a soldier based on vulnerability vs offensive power.

    Higher h value indicates a more vulnerable/defensive position.
    Lower h value indicates a more offensive/aggressive position.

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Heuristic value h (higher = more vulnerable, lower = more offensive).
    """
    offensive = calculate_offensive_power(soldier, enemies)
    vulnerability = calculate_vulnerability(soldier, enemies)

    # H = vulnerability - offensive_power
    # Higher vulnerability and lower offensive power -> higher h
    # Lower vulnerability and higher offensive power -> lower h
    return vulnerability - offensive

# Calcular h arriba, que es la diferencia de calcular el valor h desplazando el valor de la variable y de nuestro soldado una casilla hacia arriba menos el valor de h sin desplazar ninguna variable.
def calculate_h_up(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving up (y+1).

    Returns the difference: h(y+1) - h(y)
    Positive value means h increases when moving up (becoming more vulnerable).
    Negative value means h decreases when moving up (becoming more offensive).

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the upward direction.
    """
    # Calculate h at current position
    h_current = calculate_h_value(soldier, enemies)

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
    h_shifted = calculate_h_value(soldier_shifted, enemies)

    # Return the difference
    return h_shifted - h_current

def calculate_h_down(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving down (y-1).

    Returns the difference: h(y-1) - h(y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the downward direction.
    """
    h_current = calculate_h_value(soldier, enemies)

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

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_left(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving left (x-1).

    Returns the difference: h(x-1) - h(x)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the left direction.
    """
    h_current = calculate_h_value(soldier, enemies)

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

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_right(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving right (x+1).

    Returns the difference: h(x+1) - h(x)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the right direction.
    """
    h_current = calculate_h_value(soldier, enemies)

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

    h_shifted = calculate_h_value(soldier_shifted, enemies)
    return h_shifted - h_current

def calculate_h_right_up(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving diagonally right-up (x+1, y+1).

    Returns the difference: h(x+1, y+1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the right-up diagonal direction.
    """
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

def calculate_h_right_down(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving diagonally right-down (x+1, y-1).

    Returns the difference: h(x+1, y-1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the right-down diagonal direction.
    """
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

def calculate_h_left_up(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving diagonally left-up (x-1, y+1).

    Returns the difference: h(x-1, y+1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the left-up diagonal direction.
    """
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

def calculate_h_left_down(soldier: Soldier, enemies: List[Soldier]) -> float:
    """
    Calculate the h gradient when moving diagonally left-down (x-1, y-1).

    Returns the difference: h(x-1, y-1) - h(x, y)

    Args:
        soldier: The soldier being evaluated.
        enemies: List of enemy soldiers on the battlefield.

    Returns:
        Gradient of h in the left-down diagonal direction.
    """
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