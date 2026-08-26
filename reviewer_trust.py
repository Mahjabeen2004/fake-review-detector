"""
reviewer_trust.py
------------------
Computes a "trust score" (0-100) for each reviewer based on:
  1. % of their reviews flagged as fake by the ML model
  2. How fast they post reviews back-to-back (bursty posting = suspicious)
  3. How similar their own reviews are to each other (copy-paste = suspicious)

Labels match the real dataset format: __label1__ = fake, __label2__ = genuine.
This runs on DUMMY data first (no database yet). Once confirmed working,
we plug in real reviews from the Django database later.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# STEP 1: Dummy data (already built and confirmed by you)
# ---------------------------------------------------------
dummy_reviews = pd.DataFrame({
    'reviewer_name': ['john_d', 'john_d', 'john_d', 'sara_k', 'sara_k', 'mike_t', 'mike_t', 'mike_t', 'mike_t'],
    'review_text': [
        "Amazing product, best purchase ever!!!",
        "Best product ever, amazing quality!!!",
        "Everyone must buy this, life changing!!!",
        "Good quality, works as expected.",
        "Arrived on time, packaging was fine.",
        "Decent for the price.",
        "Not bad, does the job.",
        "Works fine, nothing special.",
        "Okay product, would consider again."
    ],
    'predicted_label': ['__label1__', '__label1__', '__label1__', '__label2__', '__label2__',
                         '__label2__', '__label2__', '__label2__', '__label2__'],
    'timestamp': pd.to_datetime([
        '2026-08-01 10:00', '2026-08-01 10:05', '2026-08-01 10:08',  # john_d: 3 reviews in 8 minutes
        '2026-08-01 09:00', '2026-08-03 14:00',                      # sara_k: spread out normally
        '2026-08-01 08:00', '2026-08-05 12:00', '2026-08-10 16:00', '2026-08-15 09:00'  # mike_t: spread out
    ])
})


# ---------------------------------------------------------
# STEP 2: Signal 1 - Honesty ratio
# % of a reviewer's reviews NOT flagged fake.
# 0 fake -> 100. All fake -> 0.
# ---------------------------------------------------------
def honesty_ratio(reviewer_df):
    fake_count = (reviewer_df['predicted_label'] == '__label1__').sum()
    total = len(reviewer_df)
    return 100 * (1 - fake_count / total)


# ---------------------------------------------------------
# STEP 3: Signal 2 - Posting speed score
# Look at the smallest gap (in minutes) between any two consecutive posts.
# Under 30 minutes apart = suspicious burst posting.
# Only one review ever -> no burst possible -> full score.
# ---------------------------------------------------------
def posting_speed_score(reviewer_df, burst_threshold_minutes=30):
    if len(reviewer_df) < 2:
        return 100

    times = reviewer_df['timestamp'].sort_values()
    gaps_minutes = times.diff().dropna().dt.total_seconds() / 60
    smallest_gap = gaps_minutes.min()

    if smallest_gap >= burst_threshold_minutes:
        return 100
    else:
        return max(0, 100 * (smallest_gap / burst_threshold_minutes))


# ---------------------------------------------------------
# STEP 4: Signal 3 - Self-similarity score
# Compare a reviewer's own reviews to each other with TF-IDF + cosine similarity.
# High average similarity among their own texts = likely copy-pasted.
# Only one review -> nothing to compare -> full score.
# ---------------------------------------------------------
def self_similarity_score(reviewer_df):
    texts = reviewer_df['review_text'].tolist()
    if len(texts) < 2:
        return 100

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    n = sim_matrix.shape[0]
    total, count = 0, 0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim_matrix[i, j]
                count += 1
    avg_similarity = total / count

    return 100 * (1 - avg_similarity)


# ---------------------------------------------------------
# STEP 5: Combine the three signals (equal weight, 33% each for now)
# ---------------------------------------------------------
def compute_trust_score(reviewer_df):
    h = honesty_ratio(reviewer_df)
    p = posting_speed_score(reviewer_df)
    s = self_similarity_score(reviewer_df)
    final_score = (h + p + s) / 3
    return round(final_score, 1), round(h, 1), round(p, 1), round(s, 1)


# ---------------------------------------------------------
# STEP 6: Run it for every reviewer and print results
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"{'Reviewer':<10} {'Trust':<8} {'Honesty':<10} {'Speed':<8} {'Similarity':<10}")
    print("-" * 50)
    for name, group in dummy_reviews.groupby('reviewer_name'):
        trust, h, p, s = compute_trust_score(group)
        print(f"{name:<10} {trust:<8} {h:<10} {p:<8} {s:<10}")