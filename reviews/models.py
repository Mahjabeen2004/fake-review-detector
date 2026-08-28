"""
reviews/models.py
------------------
Three tables, matching what the UI mockup actually displays:

Product  -> just a name, reviews attach to it (Screens 1, 2b, 4)
Reviewer -> a username + a stored trust score (Screens 1, 3)
Review   -> everything the ML pipeline outputs for one review (Screens 2, 2b, 3)

Design choice: predicted_label (what the model said) is kept separate from
is_overridden / override_label (what the admin changed it to). We never
overwrite the model's original answer -- the admin table needs to show both.
"""

import uuid

from django.db import models


class Product(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Reviewer(models.Model):
    # "reviewer name" on the submit form -- optional there, but if given,
    # we link to (or create) a Reviewer row so we can track their history.
    name = models.CharField(max_length=150, unique=True)

    # Stored, not computed live on every page load -- recalculated by
    # reviewer_trust.py's compute_trust_score() whenever this reviewer
    # posts a new review or gets an admin override.
    trust_score = models.FloatField(default=100.0)

    def __str__(self):
        return self.name


class Review(models.Model):
    LABEL_CHOICES = [
        ("fake", "Fake"),
        ("genuine", "Genuine"),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="reviews")

    # Optional, exactly like the mockup's "Reviewer name (optional)" field.
    reviewer = models.ForeignKey(
        Reviewer, on_delete=models.SET_NULL, null=True, blank=True, related_name="reviews"
    )

    text = models.TextField()

    # --- ML output (Result page / Screen 2) ---
    predicted_label = models.CharField(max_length=10, choices=LABEL_CHOICES)
    confidence = models.FloatField()  # e.g. 0.92 for "92% confidence"

    # --- Feature chips shown on the Result page ---
    sentiment_score = models.FloatField(null=True, blank=True)  # e.g. +0.86
    word_count = models.IntegerField(null=True, blank=True)     # e.g. 14
    # Human-readable similarity flag, e.g. "3rd similar post" or blank if unique
    similarity_flag = models.CharField(max_length=50, blank=True)

    # --- "Why flagged" box ---
    # Comma-separated top contributing words, e.g. "amazing,best ever,life changing"
    top_words = models.CharField(max_length=500, blank=True)
    # One reason per line, e.g. "Excessive exclamation marks and superlatives"
    reasons = models.TextField(blank=True)

    # --- Admin override (Screen 3) ---
    is_overridden = models.BooleanField(default=False)
    override_label = models.CharField(max_length=10, choices=LABEL_CHOICES, null=True, blank=True)

    # Groups reviews submitted together in one go. A single review is its
    # own batch of one; a bulk submission shares one batch_id across all
    # its reviews, generated once in the view and assigned to each row.
    # Lets the Bulk Results page show "just this submission" separately
    # from the Product Summary page's "all-time history for this product".
    batch_id = models.UUIDField(default=uuid.uuid4, editable=False)

    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def final_label(self):
        """What should actually be displayed/counted: the admin's override
        if one exists, otherwise the model's original prediction."""
        return self.override_label if self.is_overridden else self.predicted_label

    @property
    def confidence_percent(self):
        """Confidence stored as a decimal (0.92) -> clean whole percent (92)."""
        return round(self.confidence * 100)

    def __str__(self):
        return f"{self.product.name} — {self.final_label} ({self.confidence:.0%})"