# Quantum-Enhanced Support Vector Machines (QSVMs) For Cryptogaphic Classification

## Project Overview

The project explores **Generalised Quantum Support Vector Machines (QSVMs)**—an innovative machine learning approach that leverages the high-dimensionality of quantum states to tackle **complex classification tasks**. Specifically, we focus on cryptographically motivated data derived from the **Discrete Logarithm Problem (DLP)**, illustrating how quantum-enhanced kernels can capture intricate structure that classical methods often fail to detect.

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
