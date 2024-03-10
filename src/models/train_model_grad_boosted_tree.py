# Using reference https://scikit-learn.org/stable/modules/tree.html

from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import evaluate_model
import pandas as pd
import os

def load_training_data():
    """Load the training data from the split directory."""
    train_dir = "data/split/train/"
    X_train = pd.read_csv(train_dir + "X_train.csv")
    y_train = pd.read_csv(train_dir + "y_train.csv").squeeze()
    return X_train, y_train

def train_grad_boosted_tree(X_train, y_train):
    """Train a decision tree using the provided training data."""
    model = GradientBoostingClassifier(verbose=1)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save the trained model to the specified path using joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

if __name__ == "__main__":
    # Load the training data
    X_train, y_train = load_training_data()

    # Train the logistic regression model
    model = train_grad_boosted_tree(X_train, y_train)

    # Evaluate the model on training data
    evaluate_model.evaluate(model, X_train, y_train, "TRAIN")

    # Evaluate the model on testing data
    X_test, y_test = evaluate_model.load_test_data()
    evaluate_model.evaluate(model, X_test, y_test, "TEST")

    # Save the trained model
    save_model(model, "saved_models/grad_boosted_tree.joblib")
    print("Saved model: grad_boosted_tree.joblib")
