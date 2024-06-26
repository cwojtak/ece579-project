import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
import os
import evaluate_model

def load_training_data():
    """Load the training data from the split directory."""
    train_dir = "data/split/train/"
    X_train = pd.read_csv(train_dir + "X_train.csv")
    y_train = pd.read_csv(train_dir + "y_train.csv").squeeze()
    org_indices_train = pd.read_csv(train_dir + "org_indices_train.csv").squeeze()
    return X_train, y_train, org_indices_train

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model (the baseline) using the provided training data."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save the trained model to the specified path using joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

if __name__ == "__main__":
    # Load the training data
    X_train, y_train, org_indices_train = load_training_data()

    # Train the logistic regression model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate the model on training data
    evaluate_model.evaluate(model, X_train, y_train, "TRAIN", org_indices_train)

    # Evaluate the model on testing data
    X_test, y_test, org_indices_test = evaluate_model.load_test_data()
    evaluate_model.evaluate(model, X_test, y_test, "TEST", org_indices_test)

    # Save the trained model
    save_model(model, "saved_models/logistic_regression_baseline.joblib")
    print("Saved model: logistic_regression_baseline.joblib")
