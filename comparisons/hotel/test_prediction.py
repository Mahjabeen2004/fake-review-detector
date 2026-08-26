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
    print("Prediction:", prediction)
    print("Confidence:", {k: round(v, 3) for k, v in zip(model.classes_, probabilities)})
    print()

# Obvious fake-style (superlatives, no detail)
# Obvious fake-style (superlatives, product-focused)
test_review("Amazing product!! Best purchase ever, life changing, everyone must buy this now!!!")
test_review("Absolutely perfect! Five stars! Highly recommend to everyone!!!")
test_review("This is hands down the best product I have ever bought! Everyone needs this in their life!!!")
test_review("Incredible quality, exceeded my expectations completely! Will definitely buy again, five stars!")
test_review("Perfect in every way! Fast shipping, amazing packaging, best purchase of the year!!!")
test_review("I am absolutely obsessed with this product! It changed my daily routine completely, must buy!")
test_review("Outstanding! Works flawlessly, looks amazing, and the price is unbeatable. Highly recommend to all!")
test_review("This product is a game changer! Nothing else compares, I recommend it to everyone I know!")
test_review("Five stars all the way! Excellent quality, excellent service, excellent everything!!!")

# (keep your existing genuine/plain/negative/mixed reviews below unchanged)