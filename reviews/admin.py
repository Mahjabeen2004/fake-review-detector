"""
reviews/admin.py
-----------------
Registers Product, Reviewer, and Review so you can see/edit them in
Django's built-in admin UI at /admin/.

This ALSO includes a real Django admin action ("Flip verdict") so
overrides are genuinely possible directly through Django's built-in
admin interface -- not just the custom Admin Dashboard page at
/admin-dashboard/. Both exist; the custom dashboard is the primary,
mockup-matching UI, and this action covers direct database-level
moderation through Django's native admin.
"""

from django.contrib import admin
from .models import Product, Reviewer, Review
from .views import refresh_reviewer_trust_score


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(Reviewer)
class ReviewerAdmin(admin.ModelAdmin):
    list_display = ("name", "trust_score")
    search_fields = ("name",)


def flip_verdict(modeladmin, request, queryset):
    """
    Django admin action: select one or more reviews in the built-in
    /admin/ list view, then choose "Flip verdict" from the Actions
    dropdown to override Fake<->Genuine directly here. Reuses the
    exact same trust-score refresh logic as the custom dashboard's
    override button, so both stay consistent.
    """
    reviewers_to_refresh = set()

    for review in queryset:
        new_label = 'genuine' if review.final_label == 'fake' else 'fake'
        review.override_label = new_label
        review.is_overridden = True
        review.save()
        if review.reviewer:
            reviewers_to_refresh.add(review.reviewer)

    for reviewer in reviewers_to_refresh:
        refresh_reviewer_trust_score(reviewer)

    modeladmin.message_user(request, f"Flipped verdict for {queryset.count()} review(s).")


flip_verdict.short_description = "Flip verdict (Fake <-> Genuine)"


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = (
        "product",
        "reviewer",
        "predicted_label",
        "confidence",
        "is_overridden",
        "override_label",
        "created_at",
    )
    list_filter = ("predicted_label", "is_overridden")
    search_fields = ("text",)
    actions = [flip_verdict]