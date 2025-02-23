import numpy as np
from tqdm import tqdm
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

def compute_quantum_kernel_sequential(X_train, X_test, feature_map):
    backend = Aer.get_backend('statevector_simulator')
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K_train = kernel.evaluate(x_vec=X_train, y_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test

def train_quantum_svm_fixed(K_train, y_train, C=):
    clf = SVC(kernel='precomputed', C=C)
    with tqdm(total=1, desc="Training Quantum Kernel SVM", unit="model") as pbar:
        clf.fit(K_train, y_train)
        pbar.update(1)
    return clf

def evaluate_quantum_svm(clf, K_test, y_test, name="Quantum SVM"):
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred

def save_quantum_model(clf, path):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
