#!/bin/bash

# Identify python interpreter
PYTHON3=.venv/Scripts/python.exe

# Make directories to store data in
mkdir data/processed
mkdir data/split
mkdir data/split/test
mkdir data/split/train

# Only do data preprocessing and splitting if it hasn't already been done
if [ ! -f data/split/test/X_test.csv ] ||
   [ ! -f data/split/test/y_test.csv ] ||
   [ ! -f data/split/train/X_train.csv ] ||
   [ ! -f data/split/train/y_train.csv ]; then

# Data preprocessing
echo "Preprocessing data..."
$PYTHON3 src/data/preprocess_binary_BoW.py

# Data splitting
echo "Splitting data..."
$PYTHON3 src/utils/split.py

fi

# Train a model
echo "Training model..."
$PYTHON3 src/models/train_model_random_forest.py

# Evaluate a model
echo "Evaluating model..."
$PYTHON3 src/models/evaluate_model.py saved_models/random_forest.joblib