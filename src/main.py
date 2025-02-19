import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from dlp_utils import generate_dataset
from classical_utils import train_classical_svm, evaluate_model_detailed, save_model
from quantum_utils import compute_quantum_kernel_sequential, train_quantum_svm_fixed, evaluate_quantum_svm, save_quantum_model
from plot_utils import (
    save_circuit, plot_data_on_circle, plot_and_save_kernel_matrix,
    plot_and_save_accuracies_all, plot_and_save_decision_boundary,
    plot_and_save_confusion_matrices_all
)
from qiskit.circuit.library import PauliFeatureMap
from results_utils import setup_results_directory

def main():
    random.seed(42)
    np.random.seed(42)
    p = 31
    num_samples = 1000
    test_size = 0.2
    feature_dim = 2
    depth = 5
    C_val = 0.5

    results_paths = setup_results_directory('results')
    df, g, s = generate_dataset(p, num_samples)
    df = plot_data_on_circle(df, p, results_paths)

    X = df[['x_sin','x_cos']].values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    np.save(os.path.join(results_paths['datasets'], 'X_train.npy'), X_train)
    np.save(os.path.join(results_paths['datasets'], 'X_test.npy'), X_test)
    np.save(os.path.join(results_paths['datasets'], 'y_train.npy'), y_train)
    np.save(os.path.join(results_paths['datasets'], 'y_test.npy'), y_test)

    clf_lin = train_classical_svm(X_train, y_train, kernel='linear', C=C_val)
    acc_lin, y_lin = evaluate_model_detailed(clf_lin, X_test, y_test, "Classical SVM (Linear)")
    save_model(clf_lin, os.path.join(results_paths['models'], 'clf_classical_linear.pkl'))

    clf_rbf = train_classical_svm(X_train, y_train, kernel='rbf', C=C_val)
    acc_rbf, y_rbf = evaluate_model_detailed(clf_rbf, X_test, y_test, "Classical SVM (RBF)")
    save_model(clf_rbf, os.path.join(results_paths['models'], 'clf_classical_rbf.pkl'))

    clf_poly = train_classical_svm(X_train, y_train, kernel='poly', C=C_val)
    acc_poly, y_poly = evaluate_model_detailed(clf_poly, X_test, y_test, "Classical SVM (Polynomial)")
    save_model(clf_poly, os.path.join(results_paths['models'], 'clf_classical_poly.pkl'))

    fm = PauliFeatureMap(feature_dimension=feature_dim, reps=depth, entanglement='full', paulis=['Z','YY'])
    save_circuit(fm, 'pauli_feature_map_decomposed.png', results_paths)
    K_train, K_test = compute_quantum_kernel_sequential(X_train, X_test, fm)
    np.save(os.path.join(results_paths['kernels'], 'K_train.npy'), K_train)
    np.save(os.path.join(results_paths['kernels'], 'K_test.npy'), K_test)
    plot_and_save_kernel_matrix(K_train, "Quantum Kernel Matrix (K_train)", 'quantum_kernel_matrix.png', results_paths)

    clf_q = train_quantum_svm_fixed(K_train, y_train, C=C_val)
    clf_q.feature_map = fm
    acc_q, y_q = evaluate_quantum_svm(clf_q, K_test, y_test, "Quantum Kernel SVM")
    save_quantum_model(clf_q, os.path.join(results_paths['models'], 'clf_quantum.pkl'))

    all_info = [
        ("Classical Linear", acc_lin),
        ("Classical RBF", acc_rbf),
        ("Classical Poly", acc_poly),
        ("Quantum Kernel", acc_q)
    ]
    plot_and_save_accuracies_all(all_info, results_paths)

    plot_and_save_decision_boundary(clf_lin, X_test, y_test, "Decision Boundary (Linear)", 'linear_boundary.png', results_paths)
    plot_and_save_decision_boundary(clf_rbf, X_test, y_test, "Decision Boundary (RBF)", 'rbf_boundary.png', results_paths)
    plot_and_save_decision_boundary(clf_poly, X_test, y_test, "Decision Boundary (Poly)", 'poly_boundary.png', results_paths)
    plot_and_save_decision_boundary(clf_q, X_test, y_test, "Decision Boundary (Quantum)", 'quantum_boundary.png', results_paths, True, X_train)

    model_preds = [
        ("Linear", y_lin),
        ("RBF", y_rbf),
        ("Poly", y_poly),
        ("Quantum", y_q)
    ]
    plot_and_save_confusion_matrices_all(model_preds, y_test, results_paths)

    for n, a in all_info:
        print(f"{n} Accuracy: {a*100:.2f}%")

    with open(os.path.join(results_paths['base'], 'accuracies.txt'), 'w') as f:
        for n, a in all_info:
            f.write(f"{n} Accuracy: {a*100:.2f}%\n")

if __name__ == "__main__":
    main()
