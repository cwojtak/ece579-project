# Using reference https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.naive_bayes import GaussianNB
from joblib import dump

import pandas as pd

import os

def load_training_data():
    """Load the training data from the split directory."""
    train_dir = "data/split/train/"
    X_train = pd.read_csv(train_dir + "X_train.csv")
    y_train = pd.read_csv(train_dir + "y_train.csv").squeeze()
    return X_train, y_train

def train_naive_bayes(X_train, y_train):
    """Train a naive bayes model using the provided training data."""
    model = GaussianNB(verbose=1)
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
    model = train_naive_bayes(X_train, y_train)

    # Save the trained model
    save_model(model, "saved_models/naive_bayes.joblib")
