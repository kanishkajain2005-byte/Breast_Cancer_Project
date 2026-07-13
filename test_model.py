import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "wdbc_named.csv"
MODEL_PATH = "breast_cancer_model.pkl"
SCALER_PATH = "scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["ID"])
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})
    return df


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    print("Sample predictions vs actual labels:")
    comparison = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": predictions[:10]
    })
    comparison["Actual"] = comparison["Actual"].map({1: "Malignant", 0: "Benign"})
    comparison["Predicted"] = comparison["Predicted"].map({1: "Malignant", 0: "Benign"})
    print(comparison)

    print("\nOverall Accuracy on test set:", accuracy_score(y_test, predictions))
    print("\nFull Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Benign", "Malignant"]))


if __name__ == "__main__":
    main()
