import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_and_label_data
import os


def vectorize_messages(messages):
    """Vectorizes the messages using a count bag-of-words approach"""
    vectorizer = CountVectorizer(stop_words='english')
    features = vectorizer.fit_transform(messages)
    print("Removed stopwords: english.")
    print("Vectorized data with binary BoW.")
    return pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())


if __name__ == "__main__":
    data_path = "data/raw/SMSSpamCollection"
    output_path = "data/processed/preprocessed_data.csv"

    data = load_and_label_data(data_path)

    # Vectorize message (BoW)
    vectorized_data = vectorize_messages(data['message'])

    # Combine labels with vectorized messages
    preprocessed_data = pd.concat([data['label'], vectorized_data], axis=1)

    # Save preprocessed data
    preprocessed_data.to_csv(output_path, index=False)
