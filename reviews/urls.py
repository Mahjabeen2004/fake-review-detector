"""
reviews/urls.py
----------------
Maps web addresses to the view functions in views.py.

/submit/                     -> single review submit form
/result/<id>/                -> result for one specific saved review
/submit/bulk/                -> bulk submit form (paste multiple reviews)
/bulk-results/<batch_id>/    -> results for just ONE bulk batch
/product-summary/<batch_id>/ -> aggregate stats for one batch
/admin-dashboard/            -> custom dashboard, all reviews + override buttons
/override/<id>/              -> flips one review's verdict (POST only)
"""

from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.submit_review, name='submit_review'),
    path('result/<int:review_id>/', views.review_result, name='review_result'),
    path('submit/bulk/', views.submit_bulk_review, name='submit_bulk_review'),
    path('bulk-results/<uuid:batch_id>/', views.bulk_results, name='bulk_results'),
    path('product-summary/<uuid:batch_id>/', views.product_summary, name='product_summary'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('override/<int:review_id>/', views.override_review, name='override_review'),
]