import numpy as np
import pickle
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_classical_svm(X_train, y_train, kernel='linear', C=0.5, degree=3, gamma='scale'):
    clf = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    with tqdm(total=1, desc=f"Training SVM ({kernel})", unit="model") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)
    return clf

def evaluate_model_detailed(clf, X_test, y_test, model_name="Model"):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Accuracy: {acc*100:.2f}%")
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc, y_pred

def save_model(clf, path):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
