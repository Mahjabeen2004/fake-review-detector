import pickle
import scipy.sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X = scipy.sparse.load_npz('dataset_aayush/X_tfidf_aayush.npz')
y = pd.read_csv('dataset_aayush/y_labels_aayush.csv')['LABEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nNaive Bayes Accuracy: {accuracy:.2%}")

with open('dataset_aayush/naive_bayes_model_aayush.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved.")