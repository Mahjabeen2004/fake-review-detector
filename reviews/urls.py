"""
reviews/urls.py
----------------
Maps web addresses to the view functions in views.py.

/submit/                     -> single review submit form
/result/<id>/                -> result for one specific saved review
/submit/bulk/                -> bulk submit form (paste multiple reviews)
/bulk-results/<batch_id>/    -> results for just ONE bulk batch
/product-summary/<id>/       -> all-time aggregate stats for one product
"""

from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.submit_review, name='submit_review'),
    path('result/<int:review_id>/', views.review_result, name='review_result'),
    path('submit/bulk/', views.submit_bulk_review, name='submit_bulk_review'),
    path('bulk-results/<uuid:batch_id>/', views.bulk_results, name='bulk_results'),
    path('product-summary/<uuid:batch_id>/', views.product_summary, name='product_summary'),
]