from joblib import load

# Load Model
model = load('saved_models/BEST_random_forest.joblib')

# Load Vectorizer
vectorizer = load('data/vectorizers/fe_vectorizer.joblib')

# Get feature importances from model and map to words
feature_importances = model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
word_importances = zip(feature_names, feature_importances)
word_importances = sorted(word_importances, key=lambda x:x[1], reverse=True)

# Print results
print("BEST_random_forest Interpretation")
print("Most important features for classifying")
print("----------------------------------------------------")
for word, importance in word_importances[:20]:
    print(f"{word}: {importance:.4f}")
