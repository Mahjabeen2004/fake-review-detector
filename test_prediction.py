import pickle
from preprocess import preprocess_text

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def test_review(sample_review):
    cleaned = preprocess_text(sample_review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    print("Review:", sample_review)
    print("Non-zero features in vector:", vector.nnz)
    print("Prediction:", prediction)
    print("Confidence scores:", dict(zip(model.classes_, probabilities)))
    print()

# Test 1: product review (out-of-domain)
test_review("Amazing product!! Best purchase ever, life changing, everyone must buy this now!!!")

# Test 2: hotel review (matches training domain)
test_review("This hotel was absolutely amazing, best stay ever, the staff were incredible and I would recommend it to everyone!")