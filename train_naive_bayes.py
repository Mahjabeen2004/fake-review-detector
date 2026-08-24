import pickle
import scipy.sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load exactly what tfidf.py saved
X = scipy.sparse.load_npz('X_tfidf.npz')
y = pd.read_csv('y_labels.csv')['deceptive']

# Split: 80% train, 20% test, random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sanity check
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Test it
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nNaive Bayes Accuracy: {accuracy:.2%}")

# Save the trained model
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to naive_bayes_model.pkl")