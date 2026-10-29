from django.contrib import admin

from .models import CreditRiskPrediction


@admin.register(CreditRiskPrediction)
class CreditRiskPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "loan_amnt",
        "annual_inc",
        "fico_score",
        "grade",
        "default_probability",
        "prediction",
        "created_at",
    )

    list_filter = (
        "prediction",
        "grade",
        "home_ownership",
        "verification_status",
        "created_at",
    )

    search_fields = (
        "id",
        "purpose",
    )

    ordering = ("-created_at",)