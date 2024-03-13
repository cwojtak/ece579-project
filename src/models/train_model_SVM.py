# Using reference https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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

def train_SVM(X_train, y_train):
    """Train an SVM model using the provided training data."""
    # Hyperparameter Search Space
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3]
    }

    # Initialize the SVM
    # Grid search on base SVM found C=1, degree=3, kernel='linear' was best
    model = svm.SVC(C=1, degree=3, kernel='linear', probability=True, verbose=1)

    # Using stratified K fold sampling to address class imbalances as base SVM had low recall
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Performing a randomized grid hyperparameter search because grid search takes a long time
    print("Performing a randomized grid hyperparameter search with Stratified K-Fold cross validation")

    rand_search = RandomizedSearchCV(estimator=model, n_iter=10, param_distributions=param_grid, cv=skf, n_jobs=-1)
    rand_search.fit(X_train, y_train)
    print(f"Best parameters found: {rand_search.best_params_}\n")

    return rand_search.best_estimator_

def save_model(model, model_path):
    """Save the trained model to the specified path using joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

if __name__ == "__main__":
    # Load the training data
    X_train, y_train = load_training_data()

    # Train the logistic regression model
    model = train_SVM(X_train, y_train)

    # Evaluate the model on training data
    evaluate_model.evaluate(model, X_train, y_train, "TRAIN")

    # Evaluate the model on testing data
    X_test, y_test = evaluate_model.load_test_data()
    evaluate_model.evaluate(model, X_test, y_test, "TEST")

    # Save the trained model
    save_model(model, "saved_models/SVM.joblib")
    print("Saved model: saved_models/SVM.joblib")
