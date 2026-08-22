import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the cleaned dataset (the one with clean_text column from preprocessing)
df = pd.read_csv('dataset/deceptive-opinion-clean.csv')

# Create the TF-IDF "translator" machine
# max_features=5000 means: only keep the 5000 most useful words (keeps things fast and manageable)
vectorizer = TfidfVectorizer(max_features=5000)

# Fit + transform: learn the vocabulary AND convert every review into numbers, in one step
X = vectorizer.fit_transform(df['clean_text'])

# y is our answer key: what we're trying to predict (truthful or deceptive)
y = df['deceptive']

# Sanity checks
print("Shape of X (rows, word-features):", X.shape)
print("First 20 words in vocabulary:", list(vectorizer.vocabulary_.keys())[:20])
print("Sample deceptive labels:", y.head())

# Save the vectorizer itself — we'll need this EXACT same one later
# to convert any new review a user submits, using the same vocabulary
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save X and y too, so Naive Bayes/SVM scripts can just load them directly
import scipy.sparse
scipy.sparse.save_npz('X_tfidf.npz', X)
y.to_csv('y_labels.csv', index=False)

print("\nSaved vectorizer, X (features), and y (labels) to disk.")