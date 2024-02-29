import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys

def load_model(path):
    """Load saved model from path."""
    try:
        model = joblib.load(path)
        print(f"Loaded model from: {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def load_test_data():
    """Load testing data from ath."""
    try:
        X_test = pd.read_csv("data/split/test/X_test.csv")
        y_test = pd.read_csv("data/split/test/y_test.csv").squeeze()
        print(f"Loaded test data")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

def evaluate(model, X_test, y_test):
    """Evaluate the model using accuracy, precision, recall, f1-score, and AUROC."""
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')

    probabilities = model.predict_proba(X_test)[:,1]
    auroc = roc_auc_score(y_test, probabilities)

    print("\nModel performance")
    print("----------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    model = load_model(model_path)
    X_test, y_test = load_test_data()

    evaluate(model, X_test, y_test)