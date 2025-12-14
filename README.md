# Quantum Library - Hackathon 2025

Este archivo contiene la implementación de una biblioteca cuántica simulada que utiliza principios de mecánica cuántica y heurísticas clásicas para calcular los movimientos óptimos de las piezas en un campo de batalla. A continuación, se describe el propósito, las funciones principales y las posibles mejoras del código.

---

## **1. Introducción**

El objetivo de esta biblioteca es proporcionar un sistema de decisión estratégico para las piezas en el campo de batalla. Utilizamos dos enfoques principales:

- **Heurísticas clásicas**: Basadas en la evaluación de vulnerabilidad y poder ofensivo.
- **Simulación cuántica**: Utilizando un modelo de Hamiltoniano para calcular probabilidades de movimiento.

El sistema permite a las piezas tomar decisiones basadas en su entorno, maximizando las oportunidades de ataque y minimizando su vulnerabilidad.

---

## **2. Funciones Principales**

### **2.1. Cálculo de Poder Ofensivo y Vulnerabilidad**

A través de estas heurísticas clásicas corroboramos la interpretabilidad de los parámetros de interacción de nuestro modelo. Estas funciones evalúan la capacidad de una pieza para atacar y su exposición a ataques enemigos:

#### **`calculate_offensive_power`**

Calcula el poder ofensivo de una pieza sumando la fuerza de ataque hacia los enemigos en su rango.

- **Fórmula**:  
  \[
  P_{\text{ofensivo}} = \sum_{\text{enemigos en rango}} \text{fuerza}_{\text{soldado}}
  \]

#### **`calculate_vulnerability`**

Calcula la vulnerabilidad de una pieza sumando la fuerza de los enemigos que pueden atacarla.

- **Fórmula**:  
  \[
  V_{\text{soldado}} = \sum_{\text{enemigos en rango}} \text{fuerza}_{\text{enemigo}}
  \]

---

### **2.2. Gradientes Heurísticos (H)**

#### **`calculate_h_value`**

Calcula el valor heurístico \(H\) como la diferencia entre vulnerabilidad y poder ofensivo:

- **Fórmula**:  
  \[
  H = V_{\text{soldado}} - P_{\text{ofensivo}}
  \]

- **Interpretación**:
  - Un \(H\) alto indica una posición vulnerable.
  - Un \(H\) bajo indica una posición ofensiva.

#### **Gradientes Direccionales**

Cada gradiente (\(H_{\text{up}}, H_{\text{down}}, H_{\text{left}}, H_{\text{right}}\)) evalúa cómo cambia \(H\) al moverse en una dirección específica.

Ejemplo: `calculate_h_up` (moverse hacia arriba):

- **Fórmula**:  
  \[
  H_{\text{up}} = H(y+1) - H(y)
  \]

- **Interpretación**:
  - \(H_{\text{up}} > 0\): Moverse hacia arriba aumenta la vulnerabilidad.
  - \(H_{\text{up}} < 0\): Moverse hacia arriba reduce la vulnerabilidad.

---

### **2.3. Simulación Cuántica**

#### **`quantum_step`**

Realiza un paso de decisión cuántica utilizando un modelo de Hamiltoniano. Los pasos principales son:

1. Construir el Hamiltoniano \(H\) utilizando los valores de \(H_{\text{gradiente}}\).
2. Diagonalizar \(H\) para obtener los estados propios.
3. Seleccionar el estado más probable y traducirlo a un movimiento.

- **Fórmula del Hamiltoniano**:
  \[
  H = \sum_{a} \omega_a Z_a + \sum_{a,b} J_{ab} X_a X_b + K_{\text{penalización}}
  \]

- **Interpretación**:
  - El estado más probable (\(k_{\text{map}}\)) determina el movimiento óptimo.
  - Penalizamos configuraciones no válidas para garantizar movimientos estratégicos.

#### **Matrices de Pauli y Proyectores**

Utilizamos matrices de Pauli (\(X, Z\)) y proyectores para construir el Hamiltoniano. Estas herramientas permiten modelar interacciones entre estados cuánticos.

- **Proyector \(P_{11}\)**:
  Penaliza configuraciones no válidas (como movimientos simultáneos en direcciones opuestas):
  \[
  P_{11} = \frac{1}{4}(I - Z_i - Z_j + Z_i Z_j)
  \]

---

## **3. Justificación del Diseño**

1. **Interpretabilidad**:
   - Las heurísticas (\(H\)) permiten entender cómo las decisiones afectan la vulnerabilidad y el poder ofensivo.
   - Los gradientes direccionales explican por qué una dirección es preferida sobre otra.

2. **Escalabilidad**:
   - El modelo cuántico puede ampliarse para incluir más direcciones o restricciones.
   - La estructura modular permite integrar un backend cuántico real (como Qiskit o Cirq).

3. **Justificación Cuántica**:
   - Aunque es una simulación, el uso de Hamiltonianos y proyectores refleja principios reales de la mecánica cuántica.

---

## **4. Futuras Mejoras**

1. **Integración Cuántica Real**:
   - Sustituir la simulación por un backend cuántico real para ejecutar los cálculos.

2. **Optimización de Heurísticas**:
   - Ajustar los valores de \(H\) para escenarios más complejos.

3. **Ampliación del Modelo**:
   - Incluir más tipos de piezas y reglas de combate.
   - Incorporar interacciones más complejas en el Hamiltoniano.

---

## **5. Conclusión**

Este proyecto demuestra cómo los principios de la mecánica cuántica y las heurísticas clásicas pueden combinarse para tomar decisiones estratégicas en un entorno competitivo. Aunque el modelo es una aproximación, proporciona una base sólida para explorar estrategias cuánticas en simulaciones y juegos.
