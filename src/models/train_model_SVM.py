# Using reference https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

import numpy as np

from data import get_datasets

texts, labels, eval_texts, eval_labels = get_datasets()

count_vectorizer = CountVectorizer()
vectorized_samples = count_vectorizer.fit_transform(texts)

support = svm.SVC()
support.fit(vectorized_samples, labels)
pred = np.array(support.predict(count_vectorizer.transform(eval_texts)))
ground_truth = np.array(eval_labels)

tp = np.sum(np.logical_and(pred, ground_truth))
tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(ground_truth)))
fp = np.sum(pred) - tp
fn = np.sum(np.logical_not(pred)) - tn

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Number Correct: %5d/%5d" % (tp + tn, len(ground_truth)))
print("Recall: %2.5f" % recall)
print("Precision: %2.5f" % precision)
print("F1 Score: %2.5f" % (2 * (precision * recall) / (precision + recall)))
print("Specificity: %2.5f" % (tn / (tn + fp)))