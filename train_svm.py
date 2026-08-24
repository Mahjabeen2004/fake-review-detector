import pickle
import scipy.sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the same saved features and labels
X = scipy.sparse.load_npz('X_tfidf.npz')
y = pd.read_csv('y_labels.csv')['deceptive']

# Same split, same random_state=42 — so it's a fair comparison against Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test it
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"SVM Accuracy: {accuracy:.2%}")

# Save the trained model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to svm_model.pkl")