# Quantum Library – Hackathon 2025

This repository contains the implementation of a **simulated quantum decision-making library** that combines ideas from quantum mechanics with classical heuristics to compute optimal movements for units on a battlefield. The goal is not to provide a physically accurate quantum algorithm, but rather a **conceptually inspired, interpretable, and extensible framework** that leverages quantum-like optimization principles.

---

## **1. Introduction**

The purpose of this library is to provide a strategic decision system for battlefield units. Each unit evaluates its environment and decides how to move in order to:

* Maximize offensive opportunities.
* Minimize vulnerability to enemy attacks.
* Respect hard movement constraints (e.g., mutually exclusive directions).

Two complementary approaches are combined:

* **Classical heuristics**, which encode domain knowledge such as offensive power and vulnerability.
* **Quantum-inspired simulation**, where a Hamiltonian encodes these heuristics and constraints, and the ground state determines the most likely movement.

This hybrid approach offers a balance between **interpretability**, **flexibility**, and **global optimization**.

---

## **2. Classical Heuristics**

Classical heuristics provide an interpretable layer that grounds the quantum model in intuitive battlefield logic. They are also useful on their own as a baseline or fallback strategy.

### **2.1 Offensive Power**

#### **`calculate_offensive_power`**

This function evaluates whether a unit can eliminate the weakest enemy within its attack range.

$$
P_{\text{offensive}} = \begin{cases}
1 & \text{if the unit can kill the weakest enemy in range} \
-1 & \text{otherwise}
\end{cases}
$$

**Interpretation**:

* `1` indicates a clear offensive opportunity.
* `-1` indicates that attacking is unlikely to succeed.

This binary formulation keeps the heuristic simple and robust, avoiding overfitting to fine-grained combat statistics.

---

### **2.2 Vulnerability**

#### **`calculate_vulnerability`**

This function measures how exposed a unit is by summing the strength of all enemies capable of attacking it.

$$
V_{\text{unit}} = \begin{cases}
1 & \text{if } \sum \text{enemy strength} > 2 \cdot \text{unit health} \
-1 & \text{otherwise}
\end{cases}
$$

**Interpretation**:

* `1` means the unit is in a highly dangerous position.
* `-1` means the unit is relatively safe.

The factor of `2` acts as a conservative safety margin, discouraging suicidal positioning.

---

## **2.3 Heuristic Energy Function (h-value)**

### **`calculate_h_value`**

The core heuristic is the scalar value ( h ), which balances offense and defense:

* Lower ( h ) → vulnearable positions.
* Higher ( h ) → offensive positions.

When information about allies is available, the heuristic adapts dynamically:

* **Numerical disadvantage** → defensive bias.
* **Numerical advantage** → offensive bias.

This adaptive behavior allows the same unit logic to work across very different tactical situations.

---

## **2.4 Spatial Gradients**

For each possible movement direction (left, right, up, down), a **finite-difference gradient** is computed:

$$
\Delta h = h(\text{shifted position}) - h(\text{current position})
$$

**Interpretation**:

* ( \Delta h > 0 ): movement improves the position (preferred).
* ( \Delta h < 0 ): movement worsens the position (penalized).

These gradients form the input parameters (`omega`) for the quantum-inspired decision model.

---

## **3. Quantum Decision Model**

### **3.1 Encoding Movements as Qubits**

Each possible movement direction is mapped to a binary variable:

* `I` → move left
* `D` → move right
* `+` → move up
* `-` → move down

Each variable is treated as a qubit, resulting in a 4-qubit system.

---

### **3.2 Hamiltonian Construction**

The Hamiltonian is composed of three terms:

3.2.1. **Local Z-fields** (heuristic bias):

$$
   H_Z = \sum_a h_a Z_a
$$

3.2.2. **Interaction terms** (currently disabled or extensible):

$$
   H_{XX} = \sum_{a < b} J_{ab} X_{a} X_{b}
$$

3.2.3. **Hard Constraint Hamiltonian**

Hard movement constraints are enforced by adding **energy penalty terms** to the Hamiltonian. These terms penalize forbidden action combinations by projecting onto invalid quantum states.

For two qubits \( i, j \), the projector onto the forbidden state \(|11\rangle\) in the computational basis is given by:

$$
P^{(i,j)}_{11} = |11\rangle\langle 11| = \frac{1}{4}\left(\mathbb{I}- Z_i- Z_j+ Z_i Z_j\right)
$$

where \( Z_i \) and \( Z_j \) are Pauli-\(Z\) operators acting on qubits \( i \) and \( j \), and \( \mathbb{I} \) is the identity operator.

In this model, two hard constraints are imposed:

* Left (`I`) and right (`D`) movements cannot be selected simultaneously.
* Up (`+`) and down (`-`) movements cannot be selected simultaneously.

The corresponding constraint Hamiltonian is:

$$
H_{\text{constraints}} = K_{ID}\, P^{(I,D)}_{11} + K_{+-}\, P^{(+,-)}_{11}
$$

or, written explicitly,

$$
H_{\text{constraints}} = \frac{K_{ID}}{4}\left(\mathbb{I} - Z_I- Z_D+ Z_I Z_D\right)+\frac{K_{+-}}{4}\left(\mathbb{I}- Z_+- Z_-+ Z_+ Z_-\right)
$$

Here, \( K_{ID} \) and \( K_{+-} \) are large positive penalty coefficients. In the regime \( K \gg |\omega| \), forbidden configurations are energetically suppressed and do not appear in the ground state, ensuring logical consistency of the selected action.

---

### **3.3 Ground State Selection**

The Hamiltonian is diagonalized, and the **ground state** is interpreted as a probability distribution over all movement combinations.

The algorithm selects the **Maximum A Posteriori (MAP)** bitstring, which corresponds to the most likely valid move.

This mimics a quantum annealing or QUBO-style optimization process. In future extensions, the same formulation allows the interaction terms to be solved explicitly within a QUBO framework, enabling more complex coupled decision-making to be handled by dedicated QUBO solvers or quantum annealing hardware.

---

## **4. Scalability and Extensibility**

### **Scalability**

The scalability of this approach does **not** rely on increasing the number of qubits per agent. Instead, the core design principle is that **each unit is described by a fixed-size quantum system** (currently 4 qubits), independent of the global battlefield size.

Key scalability properties:

* **Constant qubit count per unit**: each agent always solves the same small quantum optimization problem, making the per-decision cost bounded and predictable.
* **Linear scaling in the number of units**: total computational cost scales with the number of agents, not with the size of the quantum state.
* **Local decision-making**: each Hamiltonian is constructed from local information (nearby enemies and allies), avoiding global state explosion.
* **Classical optimisation**: Potential optimization of classical heuristic functions using standard classical optimization techniques to improve decision accuracy and performance.

This makes the approach suitable for large-scale simulations, where many agents act simultaneously, each solving a small and tractable optimization problem.

---

### **Extensibility**

Possible extensions include:

* Non-zero interaction terms ( J_{ab} ) to encode synergies or conflicts.
* Continuous-valued heuristics instead of binary ( \pm 1 ).
* Learning ( \omega ) weights from data.
* Multi-agent coupled Hamiltonians for coordinated strategies.
* Explicit modeling of **team synergies**, allowing units from the same team to influence each other’s decision-making through shared interaction terms.

---

## **5. Summary**

This library demonstrates how **quantum-inspired optimization** can be combined with **interpretable classical heuristics** to produce flexible, explainable, and scalable decision-making systems for games or simulations.

It is designed as a conceptual and experimental framework, suitable for hackathons, research prototypes, and future quantum-enhanced AI systems.

---

## **6. Project File Structure**

The project is organized to clearly separate the main library, configuration, version history, and simulation tools:

```text
quantum_library.py        # Main library implementing quantum-inspired decision making (explained in the accompanying notebook)
QuantumConfig.md          # Configuration guide for future extensions to model team interactions and obtain ground states
versions_battlefield/     # Directory containing historical versions of the project as it evolved
battlefield_tester.py     # Simulation pipeline that runs battlefield scenarios and generates a dashboard to visualize results
```

**File and folder descriptions:**

* **quantum_library.py**: Core library implementing heuristics, gradient calculations, and quantum-inspired movement decisions.
* **QuantumConfig.md**: Guide for configuring the Hamiltonian and QUBO setup for potential future extensions to handle inter-unit interactions.
* **versions_battlefield/**: Contains previous project versions to track changes, improvements, and evolution over time.
* **battlefield_tester.py**: Provides the simulation pipeline to execute battlefield scenarios, generating an interactive dashboard to analyze results.

This structure ensures **clarity, maintainability, and extensibility**, allowing future integration of new features such as **team synergy interactions**.
