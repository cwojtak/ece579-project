import pandas as pd
import re

def load_and_label_data(file_path):
    """
    Load data, map labels (spam:1. ham:0), and return a DataFrame.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                label, message = re.split(r'\s+', line.strip(), maxsplit=1)
                label = 1 if label.lower() == 'spam' else 0
                data.append([label, message])
            except ValueError:
                continue
    return pd.DataFrame(data, columns=['label', 'message'])