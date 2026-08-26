import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse

df = pd.read_csv('dataset_aayush/amazon_reviews_clean.csv')

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

y = df['LABEL']

print("Shape of X (rows, word-features):", X.shape)
print("First 20 words in vocabulary:", list(vectorizer.vocabulary_.keys())[:20])
print("Sample labels:", y.head())

with open('dataset_aayush/vectorizer_aayush.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

scipy.sparse.save_npz('dataset_aayush/X_tfidf_aayush.npz', X)
y.to_csv('dataset_aayush/y_labels_aayush.csv', index=False)

print("\nSaved vectorizer, X (features), and y (labels) to disk.")