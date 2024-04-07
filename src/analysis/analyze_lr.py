from joblib import load

# Load Model
model = load('saved_models/BEST_logistic_regression_hyper_search.joblib')

# Load Vectorizer
vectorizer = load('data/vectorizers/fe_vectorizer.joblib')

# Get feature importances from model and map to words
feature_importances = model.coef_[0]
feature_names = vectorizer.get_feature_names_out()
word_importances = list(zip(feature_names, feature_importances))
word_importances = sorted(word_importances, key=lambda x: x[1], reverse=True)

# Extract top 10 most important for each class
top_spam = word_importances[:10]
top_ham = word_importances[-10:]

# Print results
print("BEST_logistic_regression_hyper_search Interpretation")
print("Most important features for classifying as SPAM:")
print("----------------------------------------------------")
for word, importance in top_spam:
    print(f"{word}: {importance:.4f}")

print("\nMost important features for classifying as HAM:")
print("----------------------------------------------------")
for word, importance in reversed(top_ham):
    print(f"{word}: {importance:.4f}")