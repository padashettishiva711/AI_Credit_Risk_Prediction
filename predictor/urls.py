from django.urls import path

from .views import CreditRiskPredictionView


urlpatterns = [
    path(
        "predict/",
        CreditRiskPredictionView.as_view(),
        name="credit-risk-predict"
    ),
]