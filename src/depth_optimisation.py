import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sympy import isprime
import warnings
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import PauliFeatureMap
from results_utils import setup_results_directory
from dlp_utils import generate_dataset
from plot_utils import plot_data_on_circle, save_circuit
from quantum_utils import compute_quantum_kernel_sequential, train_quantum_svm_fixed
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

def main():
    primes = [101]
    num_samples = 1000
    test_size = 0.2
    C_quantum = 1.0
    depth_search_params = {
        101: {'start': 1, 'max': 10, 'step': 1},
        1009: {'start': 1, 'max': 10, 'step': 1},
        29363: {'start': 1, 'max': 20, 'step': 2}
    }
    res_paths = setup_results_directory('depth_optimisation')
    summary = {}
    for p in primes:
        df, g, s = generate_dataset(p, num_samples, )
        df_path = df.copy()  # Overwrite so we can do circle plot
        df_path = plot_data_on_circle(df_path, p, res_paths[p])
        X = df_path[['x_sin','x_cos']].values
        y = df_path['y'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        np.save(os.path.join(res_paths[p]['datasets'], 'X_train.npy'), X_train)
        np.save(os.path.join(res_paths[p]['datasets'], 'X_test.npy'), X_test)
        np.save(os.path.join(res_paths[p]['datasets'], 'y_train.npy'), y_train)
        np.save(os.path.join(res_paths[p]['datasets'], 'y_test.npy'), y_test)
        dconf = depth_search_params.get(p, {'start':1,'max':10,'step':1})
        ds = list(range(dconf['start'], dconf['max']+1, dconf['step']))
        qresults = []
        cons_falls = 0
        prev_acc = None
        for d in ds:
            fm = PauliFeatureMap(feature_dimension=2, reps=d, entanglement='full', paulis=['Z','YY'])
            f_name = f'pauli_feature_map_depth{d}_decomposed.png'
            save_circuit(fm, f_name, res_paths[p])
            K_train, K_test = compute_quantum_kernel_sequential(X_train, X_test, fm)
            np.save(os.path.join(res_paths[p]['kernels'], f'K_train_depth{d}.npy'), K_train)
            np.save(os.path.join(res_paths[p]['kernels'], f'K_test_depth{d}.npy'), K_test)
            clf_q = train_quantum_svm_fixed(K_train, y_train, C=C_quantum)
            clf_q.feature_map = fm
            test_pred = clf_q.predict(K_test)
            test_acc = accuracy_score(y_test, test_pred)
            train_acc = accuracy_score(y_train, clf_q.predict(K_train))
            qresults.append({
                'depth': d,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'circuit_path': os.path.join(res_paths[p]['circuits'], f_name)
            })
            if prev_acc is not None and test_acc < prev_acc:
                cons_falls += 1
            else:
                cons_falls = 0
            prev_acc = test_acc
            if cons_falls >= 3:
                break
        qdf = pd.DataFrame(qresults)
        qdf.to_csv(os.path.join(res_paths[p]['base'], 'quantum_results_summary.csv'), index=False)
        summary[p] = qdf
        if not qdf.empty:
            best_row = qdf.loc[qdf['test_accuracy'].idxmax()]
            best_d = best_row['depth']
            src_path = best_row['circuit_path']
            dst_path = os.path.join(res_paths[p]['circuits'], f'optimal_circuit_depth{best_d}.png')
            os.rename(src_path, dst_path)

if __name__ == "__main__":
    main()
