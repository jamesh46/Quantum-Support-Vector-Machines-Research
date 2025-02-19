import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from dlp_utils import generate_dataset
from classical_utils import train_classical_svm
from sympy import isprime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def feature_engineering(df, p):
    df['x_sin'] = np.sin(2 * np.pi * df['x'] / p)
    df['x_cos'] = np.cos(2 * np.pi * df['x'] / p)
    return df

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel='linear', C=0.5, degree=3):
    from sklearn.metrics import accuracy_score
    clf = train_classical_svm(X_train, y_train, kernel=kernel, C=C, degree=degree)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def main():
    primes = [3, 13, 31, 59, 101, 151, 223, 373, 1009, 2399, 7919, 29363]
    num_samples = 1000
    test_size = 0.2
    C = 0.5
    degree = 3
    num_runs = 100
    base_seed = 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    results = []

    for p in primes:
        if not isprime(p):
            continue
        lin_acc, rbf_acc, poly_acc = [], [], []
        for run in tqdm(range(num_runs), desc=f"Prime={p}", leave=False):
            seed = base_seed + p + run
            random.seed(seed)
            np.random.seed(seed)
            try:
                df, g, s = generate_dataset(p, num_samples)
            except ValueError:
                continue
            df = feature_engineering(df, p)
            X = df[['x_sin','x_cos']].values
            y = df['y'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
            lin_acc.append(train_and_evaluate_svm(X_train, X_test, y_train, y_test, 'linear', C))
            rbf_acc.append(train_and_evaluate_svm(X_train, X_test, y_train, y_test, 'rbf', C))
            poly_acc.append(train_and_evaluate_svm(X_train, X_test, y_train, y_test, 'poly', C, degree))
        lin_acc, rbf_acc, poly_acc = np.array(lin_acc), np.array(rbf_acc), np.array(poly_acc)
        ml, mr, mp = lin_acc.mean(), rbf_acc.mean(), poly_acc.mean()
        sl, sr, sp = lin_acc.std(ddof=1), rbf_acc.std(ddof=1), poly_acc.std(ddof=1)
        se_l, se_r, se_p = sl/np.sqrt(num_runs), sr/np.sqrt(num_runs), sp/np.sqrt(num_runs)
        results.append({
            'Prime': p,
            'Mean_Linear_SVM_Accuracy (%)': ml*100,
            'SE_Linear_SVM (%)': se_l*100,
            'Mean_RBF_SVM_Accuracy (%)': mr*100,
            'SE_RBF_SVM (%)': se_r*100,
            'Mean_Poly_SVM_Accuracy (%)': mp*100,
            'SE_Poly_SVM (%)': se_p*100,
            'SD_Linear_SVM (%)': sl*100,
            'SD_RBF_SVM (%)': sr*100,
            'SD_Poly_SVM (%)': sp*100,
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv("svm_evaluation_results.csv", index=False)
    plt.figure(figsize=(12,8))
    sns.set(style="whitegrid")
    data_plot = df_res.melt(id_vars='Prime', value_vars=[
        'Mean_Linear_SVM_Accuracy (%)','Mean_RBF_SVM_Accuracy (%)','Mean_Poly_SVM_Accuracy (%)'
    ], var_name='SVM_Model', value_name='Mean_Accuracy')
    mapper = {
        'Mean_Linear_SVM_Accuracy (%)': 'Linear',
        'Mean_RBF_SVM_Accuracy (%)': 'RBF',
        'Mean_Poly_SVM_Accuracy (%)': 'Poly'
    }
    data_plot['SVM_Model'] = data_plot['SVM_Model'].map(mapper)
    sns.pointplot(x='Prime', y='Mean_Accuracy', hue='SVM_Model', data=data_plot,
                  dodge=True, markers=['o','s','D'], capsize=.1, errwidth=1, palette='Set2')
    for idx, row in df_res.iterrows():
        p = row['Prime']
        xp = list(df_res['Prime']).index(p)
        plt.errorbar(x=xp,
                     y=row['Mean_Linear_SVM_Accuracy (%)'],
                     yerr=row['SE_Linear_SVM (%)'],
                     fmt='none', c='C0', capsize=5)
        plt.errorbar(x=xp,
                     y=row['Mean_RBF_SVM_Accuracy (%)'],
                     yerr=row['SE_RBF_SVM (%)'],
                     fmt='none', c='C1', capsize=5)
        plt.errorbar(x=xp,
                     y=row['Mean_Poly_SVM_Accuracy (%)'],
                     yerr=row['SE_Poly_SVM (%)'],
                     fmt='none', c='C2', capsize=5)
    plt.title('SVM Accuracies Across Primes')
    plt.xlabel('Prime p')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='SVM Model')
    plt.savefig("svm_accuracies_across_primes.png")
    plt.close()

if __name__ == "__main__":
    main()
