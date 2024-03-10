#!/bin/bash

# Function to send commands and python output to text file for documentation.
output_file="docs/logistic_regression_hyper.txt"
if [ -f "$output_file" ]; then
    rm "$output_file"
    echo "Deleted existing output file: $output_file"
fi
function run_command {
    command="$1"
    echo "Running command: $command"
    {
        $command 2>&1
    } >> "$output_file"
    echo "" >> "$output_file"
}

# Check if directories exist, and create them if they don't
if [ ! -d "data/processed" ]; then
    mkdir "data/processed"
    echo "Created directory: data/processed"
fi
if [ ! -d "data/split" ]; then
    mkdir "data/split"
    echo "Created directory: data/split"
fi
if [ ! -d "data/split/test" ]; then
    mkdir "data/split/test"
    echo "Created directory: data/split/test"
fi
if [ ! -d "data/split/train" ]; then
    mkdir "data/split/train"
    echo "Created directory: data/split/train"
fi

# Only do data preprocessing and splitting if it hasn't already been done
if [ ! -f data/split/test/X_test.csv ] ||
   [ ! -f data/split/test/y_test.csv ] ||
   [ ! -f data/split/train/X_train.csv ] ||
   [ ! -f data/split/train/y_train.csv ]; then

# Data preprocessing
run_command "python3 src/data/preprocess_binary_BoW.py"

# Data splitting
run_command "python3 src/utils/split.py"

fi

# Train a model. NOTE: This also evaluates on training and tessting data.
run_command "python3 src/models/train_model_logistic_regression_hyper_search.py"

# Cleanup workspace
run_command "bin/cleanup.sh"
