import pickle
from preprocess_aayush import preprocess_text

with open('dataset_aayush/naive_bayes_model_aayush.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dataset_aayush/vectorizer_aayush.pkl', 'rb') as f:
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

# Fake-style
test_review("Amazing product!! Best purchase ever, life changing, everyone must buy this now!!!")
test_review("Absolutely perfect! Five stars! Highly recommend to everyone!!!")
test_review("This is hands down the best product I have ever bought! Everyone needs this in their life!!!")
test_review("Incredible quality, exceeded my expectations completely! Will definitely buy again, five stars!")
test_review("Perfect in every way! Fast shipping, amazing packaging, best purchase of the year!!!")
test_review("I am absolutely obsessed with this product! It changed my daily routine completely, must buy!")
test_review("Outstanding! Works flawlessly, looks amazing, and the price is unbeatable. Highly recommend to all!")
test_review("This product is a game changer! Nothing else compares, I recommend it to everyone I know!")
test_review("Five stars all the way! Excellent quality, excellent service, excellent everything!!!")

# Genuine-style
test_review("Good quality but a bit smaller than expected. Works fine for daily use.")
test_review("Arrived two days late and the box was dented, but the product itself works fine.")
test_review("Battery life is shorter than advertised, only lasts about 4 hours instead of 8.")
test_review("Bought this for my kitchen. Works as expected, nothing special but does the job.")
test_review("It's okay. Does what it says. Nothing more to add.")
test_review("Decent product for the price. Would consider buying again.")
test_review("Stopped working after two weeks. Very disappointed, would not recommend.")
test_review("The color was different from the picture and it feels cheaply made.")
test_review("Great sound quality but the battery drains too fast for my liking.")