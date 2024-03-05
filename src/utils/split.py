import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys


def split_save():
    """
    Split the dataset into training and testing sets. A fixed size and random_state
    ensure that the split is the SAME for every experiment, which is necessary to
    properly compare models.
    """
    data = pd.read_csv("data/processed/preprocessed_data.csv")

    X = data.drop('label', axis=1)
    y = data['label']
    

    # Stratified split to maintain class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Save split data
    X_train.to_csv("data/split/train/" + "X_train.csv", index=False)
    y_train.to_csv("data/split/train/" + "y_train.csv", index=False)
    X_test.to_csv("data/split/test/" + "X_test.csv", index=False)
    y_test.to_csv("data/split/test/" + "y_test.csv", index=False)
    

if __name__ == "__main__":
    split_save()
