# Quantum-Enhanced Support Vector Machines (QSVMs) For Cryptogaphic Classification

## Project Overview

The project explores **Generalised Quantum Support Vector Machines (QSVMs)** an adaptation of the traditional machine learning approach that leverages the high-dimensionality of quantum states that aims to tackle **complex classification tasks**. Specifically, we focus on cryptographically motivated data derived from the **Discrete Logarithm Problem (DLP)**, illustrating how quantum-enhanced kernels can capture intricate structure that classical methods often fail to detect.

> **Read the Full Report**  
> For an in-depth exploration of the theoretical background, experimental methods, and results, please see the [Full Project Report](./path/to/project_report.pdf) (PDF).


---

## Why Quantum SVMs?

Classical Support Vector Machines are a staple of modern data science, offering reliable performance and well-understood mathematical properties. However, real-world applications like **cryptography**, **genomics**, and **advanced financial modeling** generate datasets with **deeply non-linear** or **highly entangled** relationships. Classical SVM kernels (e.g., polynomial or RBF) can struggle to capture these subtle dependencies.

By contrast, **Quantum SVMs** encode data into multi-qubit states, taking advantage of:

- **Superposition and interference** to explore exponentially large feature spaces  
- **Entanglement** to represent correlations beyond the reach of conventional kernels  
- **Fidelity-based quantum kernels** that measure state overlap directly via the Born Rule  

These features enable QSVMs to learn decision boundaries in a feature space that may be exponentially larger than classical methods can realistically handle.

---

## Project Highlights

### 1. Discrete Logarithm Problem Dataset

- We construct a binary classification task using a prime modulus $p$, generator $g$, and exponentiation

  $x = g^y \bmod p.$

- Labels reflect different residue class properties, creating a non-trivial and highly non-linear decision boundary—an ideal stress test for comparing quantum and classical kernels.

### 2. Quantum Feature Maps

- Classical inputs $\mathbf{x}$ are embedded into quantum circuits $U_{\phi}(\mathbf{x})$, producing states $\lvert \phi(\mathbf{x}) \rangle$.
- The **quantum kernel** $K_{Q}(\mathbf{x}, \mathbf{y})$ is computed via fidelity (overlap) measurements of these states.
- Circuit depth and entangling gates are systematically varied to gauge how they affect classification accuracy and runtime.

### 3. Classical vs. Quantum Benchmarking

- We implement three classical SVM kernels (linear, polynomial, and RBF) for baseline comparisons.
- Statistical evaluations show the QSVM consistently achieves higher accuracy on the cryptographic dataset, demonstrating **quantum advantage** at the 5% confidence level.
- Deeper circuits can capture richer structure but risk longer simulation times and potential overfitting—underscoring the practical trade-off in **NISQ-era** hardware.

### 4. Scalability and Future Directions

- While this project focuses on DLP-based data as a proof of concept, QSVMs hold promise for a variety of **complex classification tasks** ranging from genomic analysis to secure communications.
- Ongoing work involves **optimizing circuit designs** to mitigate quantum noise and enhance real-device performance, essential steps for scaling to genuine quantum hardware.


## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Main Pipeline (`main.py`)](#1-main-pipeline-mainpy)
  - [2. Classical Accuracies Comparison (`classical_accuraciespy`)](#2-classical-accuracies-comparison-classical_accuraciespy)
  - [3. Depth Optimization (`depth_optimisationpy`)](#3-depth-optimization-depth_optimisationpy)
  - [4. Qiskit Circuit Key Illustration (`gates_keypy`)](#4-qiskit-circuit-key-illustration-gates_keypy)


---

## Overview
1. **Dataset Generation**  
   Uses a prime `p`, finds a generator `g`, and generates `(x, y)` data via discrete exponentiation `x = g^y mod p`.  
   A secret interval defines binary labels.

2. **Feature Engineering**  
   Projects `x` onto a unit circle by computing `sin(2πx/p)` and `cos(2πx/p)`.

3. **Classical vs Quantum SVM**  
   Compares classical SVM kernels (linear, RBF, polynomial) with a quantum kernel SVM built via Qiskit’s `PauliFeatureMap`.

4. **Additional Scripts**  
   - `classical_accuracies.py` runs multiple primes over multiple runs to gather statistics for classical SVM kernels.

   - `depth_optimisation.py` searches different quantum feature map depths and stops if test accuracies consistently drop.

   - `gates_key.py` generates a reference figure demonstrating several Qiskit gates.


---

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── classical_accuracies.py
│   ├── depth_optimisation.py
│   ├── gates_key.py
│   ├── dlp_utils.py
│   ├── classical_utils.py
│   ├── quantum_utils.py
│   ├── plot_utils.py
│   └── results_utils.py
└── (Output folders such as "results/" or "depth_optimisation/" get created automatically)
```

- **`dlp_utils.py`**: Utilities for discrete log dataset generation.  

- **`classical_utils.py`**: Functions for training and evaluating classical SVMs.  

- **`quantum_utils.py`**: Functions for computing quantum kernels and training quantum SVMs.  

- **`plot_utils.py`**: Plotting functions (e.g., kernel matrices, decision boundaries, confusion matrices).

- **`results_utils.py`**: Manages creation of results directories.  

- **`main.py`**: Main pipeline for generating data, training SVMs, and evaluating.  

- **`classical_accuracies.py`**: Repeated classical SVM experiments over a set of primes.  

- **`depth_optimisation.py`**: Optimizes the quantum feature map depth for certain primes.

- **`gates_key.py`**: Outputs a small reference figure of various quantum gates.


---


## Installation
1. Clone this repository:

   ```bash
   git clone https://github.com/james46/Quantum-Support-Vector-Machines-Research.git
   cd YourRepoName

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv\Scripts\activate

5. Install required packages:

   ```bash
   pip install -r requirements.txt



## Usage
1. **Main Pipeline (`main.py`)**
   Generates a DLP-based dataset, projects onto a unit circle, trains the classical and Quantum SVMs, and plots the results (Note that this script simulates large quantum systems in Qiskit and therefore will have a very long runtime on most local machines):

   ```bash
   cd src
   python main.py

Outputs go to a `results/` directory containing: datasets, models, kernels, plots, circuits.

2. **Classical Accuracies Comparison (`classical_accuracies.py`)**
   Evaluates linear, RBF, and polynomial SVM accuracies across multiple primes with 100 runs:

   ```bash
   python classical_accuracies.py

Outputs include `svm_evaluation_results.csv` and `svm_accuracies_across_primes.csv`.

3. **Depth Optimisation (`depth_optimisation.py`)**
   Vary quantum feature map depth for chosen primes and halt when test accuracy declines consecutively:

   ```bash
   python depth_optimsation.py

Outputs include a `depth_optimsation/` folder with subfolders per prime storing results as well as a `quantum_results_summary.csv` for each prime.

4. **Qiskit Circuit Key Illustration (`gates_key.py`)**
   Creates `qiskit_circuit_key.png` descirbing each quantum gate symbol:
   ```bash
   python gates_key.py
