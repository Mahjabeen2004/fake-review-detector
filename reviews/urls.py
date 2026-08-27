"""
reviews/urls.py
----------------
Maps web addresses to the view functions in views.py.

/submit/         -> shows the submit form, and processes it when submitted
/result/<id>/    -> shows the result for one specific saved review
"""

from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.submit_review, name='submit_review'),
    path('result/<int:review_id>/', views.review_result, name='review_result'),
]