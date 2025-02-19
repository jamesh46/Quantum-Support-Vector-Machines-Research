import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit

def save_circuit(circuit, filename, results_paths, title="Quantum Circuit"):
    dc = circuit.decompose()
    fp = os.path.join(results_paths['circuits'], filename)
    dc.draw('mpl', filename=fp, style={'backgroundcolor': 'white'})
    print(f"Saved quantum circuit to '{fp}'.")

def plot_data_on_circle(df, p, results_paths, label_col='y'):
    df['x_sin'] = np.sin(2 * np.pi * df['x'] / p)
    df['x_cos'] = np.cos(2 * np.pi * df['x'] / p)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='x_sin', y='x_cos', hue=label_col, data=df, palette='coolwarm', edgecolor='k', alpha=0.8)
    c = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1.5, linestyle='--')
    plt.gca().add_patch(c)
    plt.gca().set_aspect('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title('Data on Unit Circle')
    path = os.path.join(results_paths['plots'], 'circle_projection.png')
    plt.savefig(path)
    plt.close()
    return df

def plot_and_save_kernel_matrix(K, title, filename, results_paths):
    plt.figure(figsize=(10, 8))
    sns.heatmap(K, cmap='viridis')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Data')
    p = os.path.join(results_paths['plots'], filename)
    plt.savefig(p)
    plt.close()

def plot_and_save_accuracies_all(models_info, results_paths):
    names = [m[0] for m in models_info]
    accs = [m[1] for m in models_info]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accs, color=['skyblue','lightgreen','plum','salmon'])
    plt.ylim(0, 1)
    for b, a in zip(bars, accs):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2.0, h+0.02, f"{a*100:.2f}%", ha='center')
    plt.title('Comparison of SVM Accuracies')
    p = os.path.join(results_paths['plots'], 'accuracy_comparison_all.png')
    plt.savefig(p)
    plt.close()

def plot_and_save_decision_boundary(clf, X, y, title, filename, results_paths, is_quantum=False, X_train=None, h=0.05, sample_fraction=1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    if is_quantum:
        np.random.seed(42)
        sz = int(len(grid_points)*sample_fraction)
        idxs = np.random.choice(len(grid_points), size=sz, replace=False)
        sub_grid = grid_points[idxs]
        kernel = FidelityQuantumKernel(feature_map=clf.feature_map)
        K_grid = kernel.evaluate(x_vec=sub_grid, y_vec=X_train)
        Z_sub = clf.predict(K_grid)
        Z = np.full(xx.shape, -1)
        Z_flat = Z.ravel()
        Z_flat[idxs] = Z_sub
        Z = Z_flat.reshape(xx.shape)
    else:
        Z = clf.predict(grid_points).reshape(xx.shape)
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    sc = plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    p = os.path.join(results_paths['plots'], filename)
    plt.savefig(p)
    plt.close()

def plot_and_save_confusion_matrices_all(model_preds, y_true, results_paths):
    from sklearn.metrics import confusion_matrix
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = len(model_preds)
    r = math.ceil(n/2)
    fig, axes = plt.subplots(r, 2, figsize=(14, 12))
    axes = axes.flatten() if r>1 else [axes]
    for i, (mn, yp) in enumerate(model_preds):
        cm = confusion_matrix(y_true, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{mn} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    p = os.path.join(results_paths['plots'], 'confusion_matrices_all.png')
    plt.savefig(p)
    plt.close()
