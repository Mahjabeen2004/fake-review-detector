import pickle
import scipy.sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

X = scipy.sparse.load_npz('X_tfidf.npz')
y = pd.read_csv('y_labels.csv')['deceptive']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# New recommended way to get probability/confidence scores
base_model = SVC(kernel='linear')
model = CalibratedClassifierCV(base_model, ensemble=False)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"SVM Accuracy: {accuracy:.2%}")

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to svm_model.pkl")