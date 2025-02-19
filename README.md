# Quantum-Enhanced SVM with Discrete Logarithm Problem

This repository demonstrates how to generate a dataset using the Discrete Logarithm Problem (DLP), then train both classical and quantum-kernel-based Support Vector Machines (SVM) to classify the generated data. The code compares multiple SVM variants (linear, RBF, polynomial, and quantum kernel) and also includes scripts for deeper analysis of classical accuracies across various primes and for depth optimization of quantum feature maps.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Main Pipeline (`main.py`)](#1-main-pipeline-mainpy)
  - [2. Classical Accuracies Comparison (`classical_accuraciespy`)](#2-classical-accuracies-comparison-classical_accuraciespy)
  - [3. Depth Optimization (`depth_optimisationpy`)](#3-depth-optimization-depth_optimisationpy)
  - [4. Qiskit Circuit Key Illustration (`gates_keypy`)](#4-qiskit-circuit-key-illustration-gates_keypy)
- [Notes](#notes)
- [License](#license)

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
   git clone https://github.com/james46/FILL IN.git
   cd YourRepoName
