"""
reviews/admin.py
-----------------
Registers Product, Reviewer, and Review so you can see/edit them in
Django's built-in admin UI at /admin/ while testing — before you build
your own custom Admin Dashboard page (that's Day 5).

This is just for YOUR testing convenience right now. Your mockup's
"Screen 3: Admin Dashboard" with override buttons is a separate,
custom-built page you'll make later — this is not that page.
"""

from django.contrib import admin
from .models import Product, Reviewer, Review


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(Reviewer)
class ReviewerAdmin(admin.ModelAdmin):
    list_display = ("name", "trust_score")
    search_fields = ("name",)


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