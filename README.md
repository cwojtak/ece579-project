# SMS Spam Detection

## Requirements
This project requires Python (tested on 3.8.10) and the requirements in requirements.txt to be installed.       
They can be installed with ```pip install -r requirements.txt``` in the project root directory.

## Running an Experiment

*NOTE: All example commands should be executed from the root project directory (i.e. /ece579-project)  
*NOTE: Some pre-made configurations are available in ./bin. Running any of the *_default_run.sh scripts in this folder
will automatically execute steps 1-6 below.

1. Data preprocessing
    - Create/Use a preprocessing script from 'src/data'. This may include processes like tokenization, vectorization, text normalization, etc.
    - The script should save the preprocessed data to 'data/processed'.
    - NO DATA PREPROCESSING SCRIPT SHOULD CHANGE THE ORDER OF ROWS OR COLUMNS IN THE DATASET; DOING SO WILL CHANGE THE SPLIT, COMPROMIZING MODEL COMPARISON.
    - NEVER DIRECTLY EDIT THE DATA IN 'data/raw'.

    - e.g. python3 src/data/preprocess_binary_BoW.py

2. Data splitting
    - Run 'src/utils/split.py' to split the preprocessed data into training and testing sets.
    - The script will save the training and testing sets to 'data/split/train' and 'data/split/test' directories, respectively.
    - Using this script ensures that the data split is stratified (maintains class proportions), and is the same across each experiment.

    - e.g. python3 src/utils/split.py

3. Train a model
    - Create/Use a training script from 'src/models' to train a model on the preprocessed and split training data.
    - Each script should load the training data, configure the model, perform the training process, and save the trained model to 'saved_models'.

    - e.g. python3 src/models/train_model_BASELINE.py

4. Evaluate the model
    - Run 'evaluate_model.py' to evaluate the trained model.

    - e.g. python3 src/models/evaluate_model.py saved_models/logistic_regression_baseline.joblib

5. Document findings
    - Document the experiement using a pdf or similar, and save to 'docs'.
    - This should be done automatically by the automated shell scripts but should be done if experimentation is being
performed manually.
    - Ensure all steps are documented for reproducibility.

6. Clean up the workspace
    - Run 'bin/cleanup.sh' or manually delete files in 'data/processed', 'data/split/train', and 'data/split/test'.
    -   Note: you may have to run 'chmod +x bin/cleanup.sh' before executing the shell script.
    - If you are training multiple models with the same preprocessing, you can skip this step until you are done running experiments.

    - e.g. bin/cleanup.sh

7. Push to Git
