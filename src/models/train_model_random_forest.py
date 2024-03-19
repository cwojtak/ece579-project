# Using reference
# https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump
import evaluate_model
import pandas as pd
import os

def load_training_data():
    """Load the training data from the split directory."""
    train_dir = "data/split/train/"
    X_train = pd.read_csv(train_dir + "X_train.csv")
    y_train = pd.read_csv(train_dir + "y_train.csv").squeeze()
    org_indices_train = pd.read_csv(train_dir + "org_indices_train.csv").squeeze()
    return X_train, y_train, org_indices_train

def train_random_forest(X_train, y_train):
    """Train a random forest model using the provided training data."""
    # Hyperparameter Search Space
    param_grid = {
        'class_weight': ['balanced', 'balanced_subsample', None],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [50, 100, 200]
    }

    # Initialize the Random Forest model
    model = RandomForestClassifier(verbose=1)

    # Using stratified K fold sampling to address class imbalances as base SVM had low recall
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Performing a grid hyperparameter search because grid search takes a long time
    print("Performing a grid hyperparameter search with Stratified K-Fold cross validation")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}\n")

    return grid_search.best_estimator_

def save_model(model, model_path):
    """Save the trained model to the specified path using joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

if __name__ == "__main__":
    # Load the training data
    X_train, y_train, org_indices_train = load_training_data()

    # Train the logistic regression model
    model = train_random_forest(X_train, y_train)

    # Evaluate the model on training data
    evaluate_model.evaluate(model, X_train, y_train, "TRAIN", org_indices_train)

    # Evaluate the model on testing data
    X_test, y_test, org_indices_test = evaluate_model.load_test_data()
    evaluate_model.evaluate(model, X_test, y_test, "TEST", org_indices_test)

    # Save the trained model
    save_model(model, "saved_models/random_forest.joblib")
    print("Saved model: saved_models/random_forest.joblib")
