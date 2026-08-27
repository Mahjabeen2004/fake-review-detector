"""
reviews/views.py
----------------
Day 3: the core working piece of the whole project.

Flow:
  1. User submits a review through a form (product name + review text)
  2. This view loads your trained Naive Bayes model + vectorizer
  3. Classifies the review, saves it to the database
  4. Redirects to a Result page showing the verdict

Model/vectorizer are loaded ONCE, when Django starts -- not on every
request. Loading a .pkl file from disk is relatively slow, so doing it
on every single submission would make the site sluggish.
"""

import os
import pickle

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404

from preprocess_aayush import preprocess_text
from .models import Product, Reviewer, Review


# ---------------------------------------------------------
# Load the trained model + vectorizer ONCE at startup.
# These live in dataset_aayush/, at the project root level.
# ---------------------------------------------------------
MODEL_PATH = os.path.join(settings.BASE_DIR, 'dataset_aayush', 'naive_bayes_model_aayush.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'dataset_aayush', 'vectorizer_aayush.pkl')

with open(MODEL_PATH, 'rb') as f:
    nb_model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Maps the model's real output labels to what we display/store internally.
# NEVER change this mapping during training -- only used here, at display/save time.
LABEL_MAP = {'__label1__': 'fake', '__label2__': 'genuine'}


def get_top_words(vectorized_input, class_index, top_n=5):
    """
    Which words in THIS specific review pushed the model toward its
    predicted class? We only look at words that are actually present
    in this review, then rank them by the model's learned log-probability
    for the predicted class -- higher log-probability = stronger pull
    toward that verdict.
    """
    feature_names = vectorizer.get_feature_names_out()
    log_probs = nb_model.feature_log_prob_[class_index]

    present_indices = vectorized_input.nonzero()[1]
    word_scores = [(feature_names[i], log_probs[i]) for i in present_indices]
    word_scores.sort(key=lambda item: item[1], reverse=True)

    return [word for word, score in word_scores[:top_n]]


def submit_review(request):
    """Shows the submit form (GET) or processes a submitted review (POST)."""
    if request.method == 'POST':
        product_name = request.POST.get('product_name', '').strip()
        reviewer_name = request.POST.get('reviewer_name', '').strip()
        review_text = request.POST.get('review_text', '').strip()

        # --- Preprocess exactly the same way training data was processed ---
        cleaned_text = preprocess_text(review_text)
        vectorized = vectorizer.transform([cleaned_text])

        # --- Predict ---
        predicted_class = nb_model.predict(vectorized)[0]           # e.g. '__label1__'
        probabilities = nb_model.predict_proba(vectorized)[0]
        class_index = list(nb_model.classes_).index(predicted_class)
        confidence = probabilities[class_index]

        final_label = LABEL_MAP[predicted_class]
        top_words_list = get_top_words(vectorized, class_index)

        # --- Find or create the Product / Reviewer rows ---
        product, _ = Product.objects.get_or_create(name=product_name)

        reviewer = None
        if reviewer_name:
            reviewer, _ = Reviewer.objects.get_or_create(name=reviewer_name)

        # --- Save this review ---
        review = Review.objects.create(
            product=product,
            reviewer=reviewer,
            text=review_text,
            predicted_label=final_label,
            confidence=confidence,
            word_count=len(review_text.split()),
            top_words=', '.join(top_words_list),
        )

        return redirect('review_result', review_id=review.id)

    return render(request, 'reviews/submit.html')


def review_result(request, review_id):
    """Shows the verdict for one saved review."""
    review = get_object_or_404(Review, id=review_id)
    context = {
        'review': review,
        'confidence_percent': round(review.confidence * 100),
        'top_words_list': review.top_words.split(', ') if review.top_words else [],
    }
    return render(request, 'reviews/result.html', context)