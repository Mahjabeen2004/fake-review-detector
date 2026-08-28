"""
reviews/views.py
----------------
Day 3: Submit -> Classify -> Result (single review)
Day 4: Bulk Check -> Bulk Results -> Product Summary

Flow (single):
  1. User submits a review through a form (product name + review text)
  2. This view loads your trained Naive Bayes model + vectorizer
  3. Classifies the review, saves it to the database
  4. Redirects to a Result page showing the verdict

Flow (bulk):
  1. User pastes multiple reviews (one per line) for one product
  2. Each line is classified separately, all tagged with the same batch_id
  3. Redirects to Bulk Results, showing just this batch
  4. From there, Product Summary aggregates ALL reviews for that product
     (all-time, ignoring batch) -- that's what batch_id exists to separate.

Model/vectorizer are loaded ONCE, when Django starts -- not on every
request. Loading a .pkl file from disk is relatively slow, so doing it
on every single submission would make the site sluggish.
"""

import os
import pickle
import uuid

import pandas as pd
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from preprocess_aayush import preprocess_text
from reviewer_trust import compute_trust_score
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

# The reverse mapping -- needed because reviewer_trust.py's compute_trust_score()
# was already tested and committed expecting the ORIGINAL '__label1__'/'__label2__'
# format. Rather than touch that already-working file, we translate our internal
# 'fake'/'genuine' values back before handing data to it.
INTERNAL_TO_TRAINING_LABEL = {'fake': '__label1__', 'genuine': '__label2__'}


def refresh_reviewer_trust_score(reviewer):
    """
    Recomputes and saves ONE reviewer's trust score using their REAL review
    history from the database -- reusing the exact same compute_trust_score()
    logic already tested on dummy data (Day 1), completely unchanged.

    Called whenever:
      - that reviewer posts a new single review, or
      - an admin overrides one of their reviews' verdict (Fake <-> Genuine)
    """
    reviews = Review.objects.filter(reviewer=reviewer)
    if not reviews.exists():
        return

    data = {'reviewer_name': [], 'review_text': [], 'predicted_label': [], 'timestamp': []}
    for r in reviews:
        data['reviewer_name'].append(reviewer.name)
        data['review_text'].append(r.text)
        data['predicted_label'].append(INTERNAL_TO_TRAINING_LABEL[r.final_label])
        data['timestamp'].append(r.created_at)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    trust_score, _, _, _ = compute_trust_score(df)
    reviewer.trust_score = trust_score
    reviewer.save()


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


def classify_text(text):
    """
    Runs one piece of review text through the full pipeline:
    clean -> vectorize -> predict -> find top contributing words.
    Used by BOTH single and bulk submission, so the exact same
    logic is never duplicated in two places.

    Returns a dict ready to unpack into Review.objects.create(...).
    """
    cleaned_text = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned_text])

    predicted_class = nb_model.predict(vectorized)[0]           # e.g. '__label1__'
    probabilities = nb_model.predict_proba(vectorized)[0]
    class_index = list(nb_model.classes_).index(predicted_class)
    confidence = probabilities[class_index]

    final_label = LABEL_MAP[predicted_class]
    top_words_list = get_top_words(vectorized, class_index)

    return {
        'text': text,
        'predicted_label': final_label,
        'confidence': confidence,
        'word_count': len(text.split()),
        'top_words': ', '.join(top_words_list),
    }


def submit_review(request):
    """Shows the submit form (GET) or processes a submitted single review (POST)."""
    if request.method == 'POST':
        product_name = request.POST.get('product_name', '').strip()
        reviewer_name = request.POST.get('reviewer_name', '').strip()
        review_text = request.POST.get('review_text', '').strip()

        # Guard against near-empty text (e.g. just spaces slipping past the
        # HTML 'required' attribute). With zero real words, the model has
        # nothing to classify -- it would just return a meaningless ~50%
        # coin-flip result dressed up to look like a real assessment.
        if len(review_text.split()) < 3:
            return render(request, 'reviews/submit.html', {
                'error': 'Please enter a real review (at least a few words) before checking.',
            })

        result = classify_text(review_text)

        product, _ = Product.objects.get_or_create(name=product_name)

        reviewer = None
        if reviewer_name:
            reviewer, _ = Reviewer.objects.get_or_create(name=reviewer_name)

        review = Review.objects.create(
            product=product,
            reviewer=reviewer,
            **result,
        )

        if reviewer:
            refresh_reviewer_trust_score(reviewer)

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


def submit_bulk_review(request):
    """Shows the bulk submit form (GET) or processes pasted reviews (POST)."""
    if request.method == 'POST':
        product_name = request.POST.get('product_name', '').strip()
        raw_text = request.POST.get('bulk_reviews', '')

        # One review per line, ignore blank lines AND lines too short to
        # classify meaningfully (same reasoning as the single-review guard)
        lines = [line.strip() for line in raw_text.split('\n') if len(line.strip().split()) >= 3]

        # If EVERY line got filtered out (too short/blank), stop here rather
        # than creating an empty batch and redirecting to a confusing
        # "0 reviews checked" results page.
        if not lines:
            return render(request, 'reviews/submit_bulk.html', {
                'error': 'No valid reviews found. Each line needs at least a few words.',
            })

        product, _ = Product.objects.get_or_create(name=product_name)

        # Every review in THIS submission shares one batch_id, so the
        # Bulk Results page can show just this batch later.
        batch_id = uuid.uuid4()

        for line in lines:
            result = classify_text(line)
            Review.objects.create(
                product=product,
                reviewer=None,  # bulk mode doesn't track individual reviewers
                batch_id=batch_id,
                **result,
            )

        return redirect('bulk_results', batch_id=batch_id)

    return render(request, 'reviews/submit_bulk.html')


def bulk_results(request, batch_id):
    """Shows only the reviews submitted together in ONE bulk batch."""
    reviews = Review.objects.filter(batch_id=batch_id)
    product = reviews.first().product if reviews.exists() else None
    context = {
        'reviews': reviews,
        'product': product,
        'batch_id': batch_id,
    }
    return render(request, 'reviews/bulk_results.html', context)


def product_summary(request, batch_id):
    """
    Aggregates reviews from ONE specific batch -- matching the mockup's
    "20 reviews pasted for this product" framing. This is a snapshot of
    one checking session, NOT an all-time history for the product name --
    two unrelated people bulk-checking a same-named product on different
    platforms at different times should never be mixed into one score.
    (Contrast with reviewer trust scores, which SHOULD stay all-time --
    that's tracking one person's behavior pattern across sessions.)
    """
    reviews = Review.objects.filter(batch_id=batch_id)
    product = reviews.first().product if reviews.exists() else None

    total = reviews.count()
    fake_count = sum(1 for r in reviews if r.final_label == 'fake')
    genuine_count = total - fake_count

    genuine_pct = round(100 * genuine_count / total) if total else 0
    fake_pct = 100 - genuine_pct if total else 0

    context = {
        'product': product,
        'total': total,
        'fake_count': fake_count,
        'genuine_count': genuine_count,
        'genuine_pct': genuine_pct,
        'fake_pct': fake_pct,
    }
    return render(request, 'reviews/product_summary.html', context)


@staff_member_required
def admin_dashboard(request):
    """
    Custom Admin Dashboard (Screen 3 from the mockup) -- separate from
    Django's built-in /admin/. Lists every review with its verdict,
    reviewer trust score, and an override button.

    @staff_member_required means only logged-in staff/superuser accounts
    can reach this page -- anyone else gets redirected to the admin login.
    """
    reviews = Review.objects.select_related('product', 'reviewer').order_by('-created_at')
    return render(request, 'reviews/admin_dashboard.html', {'reviews': reviews})


@staff_member_required
@require_POST
def override_review(request, review_id):
    """
    Flips one review's verdict (Fake <-> Genuine). The ORIGINAL model
    prediction is never touched -- only override_label and is_overridden
    change, so we always keep both "what the model said" and "what the
    admin corrected it to."

    If the review has a reviewer attached, their trust score is
    recalculated immediately using the corrected label.
    """
    review = get_object_or_404(Review, id=review_id)

    # Figure out the new label FIRST, before changing anything -- final_label
    # depends on is_overridden, so if we flip is_overridden before checking
    # the current verdict, we'd be checking against a value that isn't set yet.
    new_label = 'genuine' if review.final_label == 'fake' else 'fake'

    review.override_label = new_label
    review.is_overridden = True
    review.save()

    if review.reviewer:
        refresh_reviewer_trust_score(review.reviewer)

    return redirect('admin_dashboard')