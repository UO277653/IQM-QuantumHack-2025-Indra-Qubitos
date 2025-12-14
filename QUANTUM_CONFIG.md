# Quantum Algorithm Configuration Guide

## How to Switch Between Quantum Methods

The system supports two quantum methods for decision-making:

### 1. **EXACT** (Default) - Fast & Deterministic
- Uses exact diagonalization of the Hamiltonian
- Finds the exact ground state
- **Speed**: ~1 second per battle
- **Accuracy**: Exact solution
- **Best for**: Quick testing, many battles

### 2. **QAOA** - Realistic Quantum Algorithm
- Uses Quantum Approximate Optimization Algorithm
- More realistic for actual quantum hardware
- **Speed**: ~10-180 seconds per battle (depending on parameters)
- **Accuracy**: Approximation (quality depends on p and n_restarts)
- **Best for**: Research, realistic quantum simulations

## Method 1: Edit quantum_library.py Directly

Open `quantum_library.py` and find lines 442-446:

```python
# =========================================================
# CONFIGURATION: Choose quantum algorithm
# =========================================================
# Set this to "qaoa" or "exact" to switch between methods
QUANTUM_METHOD = "exact"  # Options: "qaoa" or "exact"

# QAOA parameters (only used if QUANTUM_METHOD = "qaoa")
QAOA_P = 2              # QAOA depth (layers)
QAOA_N_RESTARTS = 3     # Number of optimizer restarts
```

**To use EXACT (fast)**:
```python
QUANTUM_METHOD = "exact"
```

**To use QAOA**:
```python
QUANTUM_METHOD = "qaoa"
QAOA_P = 2              # Higher = better quality, slower (try 1-3)
QAOA_N_RESTARTS = 3     # Higher = more likely to find optimum (try 2-10)
```

## Method 2: Change Programmatically

In your Python code:

```python
import quantum_library as qlib

# Switch to exact method
qlib.set_quantum_method("exact")

# Or switch to QAOA
qlib.set_quantum_method("qaoa")

# Adjust QAOA parameters
qlib.QAOA_P = 2
qlib.QAOA_N_RESTARTS = 3
```

## QAOA Parameter Guide

### QAOA_P (Circuit Depth)
- **p=1**: Fast, basic approximation (~10-30 sec/battle)
- **p=2**: Good balance (~30-90 sec/battle) ⭐ **Recommended**
- **p=3**: High quality (~90-180 sec/battle)
- **p≥4**: Diminishing returns, very slow

### QAOA_N_RESTARTS (Optimizer Attempts)
- **n=2**: Fast, may miss optimum (~1x speed)
- **n=3-4**: Good balance ⭐ **Recommended**
- **n=5-10**: More thorough (~2-5x slower)
- **n≥10**: Overkill for most cases

## Performance Comparison (100 battles)

| Method | Config | Time | Quantum Win Rate |
|--------|--------|------|------------------|
| EXACT  | - | ~2 min | ~70-73% |
| QAOA   | p=1, n=2 | ~30 min | ~68-70% |
| QAOA   | p=2, n=3 | ~2-3 hrs | ~72-75% (estimated) |
| QAOA   | p=2, n=4 | ~4-5 hrs | ~73-76% (estimated) |

## Recommended Configurations

### For Quick Testing
```python
QUANTUM_METHOD = "exact"
```

### For Research/Publication
```python
QUANTUM_METHOD = "qaoa"
QAOA_P = 2
QAOA_N_RESTARTS = 3
```

### For Maximum Quality (overnight runs)
```python
QUANTUM_METHOD = "qaoa"
QAOA_P = 3
QAOA_N_RESTARTS = 5
```

## Troubleshooting

**QAOA is too slow**:
- Reduce `QAOA_P` to 1
- Reduce `QAOA_N_RESTARTS` to 2
- Use fewer battles for testing

**Want better QAOA results**:
- Increase `QAOA_P` to 3
- Increase `QAOA_N_RESTARTS` to 5-10
- Let it run longer or overnight

**Cache issues**:
```python
import quantum_library as qlib
qlib.clear_qaoa_cache()  # Clear QAOA cache
```
