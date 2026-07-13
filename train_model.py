import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "wdbc_named.csv"
MODEL_OUTPUT_PATH = "breast_cancer_model.pkl"
SCALER_OUTPUT_PATH = "scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["ID"])
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})
    return df


def split_data(df):
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n===== {name} =====")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key.capitalize()}: {value:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return metrics


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def select_best_model(results):
    best = max(results, key=lambda r: r["roc_auc"])
    return best


def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    trained_models = train_models(X_train_scaled, y_train)

    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(name, model, X_test_scaled, y_test)
        results.append(metrics)

    best_metrics = select_best_model(results)
    best_model_name = best_metrics["model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model based on ROC-AUC: {best_model_name}")

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    print(f"Saved model to {MODEL_OUTPUT_PATH}")
    print(f"Saved scaler to {SCALER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
