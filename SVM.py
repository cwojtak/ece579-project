# Using reference https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np

texts = []
labels = []
eval_texts = []
eval_labels = []
i = 0

with open("data/SMSSpamCollection", "r") as data_file:
    data_file_contents = data_file.readlines()
    for line in data_file_contents:
        divide = line.find("am") + 2
        label = 1 if line[:divide] == "spam" else 0
        text = line[divide + 1:]
        if i % 2 == 0:
            texts.append(text)
            labels.append(label)
        else:
            eval_texts.append(text)
            eval_labels.append(label)
        i += 1


count_vectorizer = CountVectorizer()
vectorized_samples = count_vectorizer.fit_transform(texts)

support = svm.SVC()
support.fit(vectorized_samples, labels)
pred = np.array(support.predict(count_vectorizer.transform(eval_texts)))
ground_truth = np.array(eval_labels)
print("Number Correct: %5d/%5d" % (np.sum(np.logical_and(pred, ground_truth)) +
      np.sum(np.logical_and(np.logical_not(pred), np.logical_not(ground_truth))),
      len(ground_truth)))
