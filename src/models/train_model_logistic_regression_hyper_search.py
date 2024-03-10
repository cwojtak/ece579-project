import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import evaluate_model
from joblib import dump
import os

def load_training_data():
    """Load the training data from the split directory."""
    train_dir = "data/split/train/"
    X_train = pd.read_csv(train_dir + "X_train.csv")
    y_train = pd.read_csv(train_dir + "y_train.csv").squeeze()
    return X_train, y_train

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model (advanced) using the provided training data."""
    # Hyperparameter Search Space
    param_grid = {
        'solver': ["liblinear"],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']  # TODO maybe try defining specific values for this
    }
    
    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Perform Grid Search with Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}\n")

    # Return best performing model
    best_model = grid_search.best_estimator_
    return best_model

def save_model(model, model_path):
    """Save the trained model to the specified path using joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

if __name__ == "__main__":
    # Load the training data
    X_train, y_train = load_training_data()

    # Train the logistic regression model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate the model on training data
    evaluate_model.evaluate(model, X_train, y_train, "TRAIN")

    # Evaluate the model on testing data
    X_test, y_test = evaluate_model.load_test_data()
    evaluate_model.evaluate(model, X_test, y_test, "TEST")

    # Save the trained model
    save_model(model, "saved_models/logistic_regression_hyper_search.joblib")
    print("Saved model: logistic_regression_hyper_search.joblib")
