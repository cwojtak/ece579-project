import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_and_label_data
import os
import re


def vectorize_messages(messages):
    """Vectorizes the messages using a binary bag-of-words approach but does some feature extraction"""
    # Convert long strings of numbers (i.e., phone numbers) to a unique word to make whether a message has a phone
    # number or not more significant
    messages.apply(lambda message:
                   re.sub("[0-9]{7,100}", "feature235693", message)
                   )
    print("Replaced long strings of numbers with a unique word.")

    vectorizer = CountVectorizer(binary=True, stop_words=["a"])
    features = vectorizer.fit_transform(messages)
    print("Removed stop words: a")
    print("Vectorized data with binary BoW.")
    return pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())


if __name__ == "__main__":
    data_path = "data/raw/SMSSpamCollection"
    output_path = "data/processed/preprocessed_data.csv"

    data = load_and_label_data(data_path)

    # Vectorize message (BoW)
    vectorized_data = vectorize_messages(data['message'])

    # Combine labels with vectorized messages
    preprocessed_data = pd.concat([data["org_indices"], data['label'], vectorized_data], axis=1)

    # Save preprocessed data
    preprocessed_data.to_csv(output_path, index=False)
