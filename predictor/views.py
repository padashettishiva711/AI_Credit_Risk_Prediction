from django.shortcuts import render
import json
from pathlib import Path

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import CreditRiskPrediction
from .serializers import CreditRiskInputSerializer
from .ml_service import predict_credit_risk

from django.db.models import Avg, Count

from .ml_service import predict_credit_risk, MODEL_METRICS

class CreditRiskPredictionView(APIView):

    def post(self, request):
        serializer = CreditRiskInputSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        applicant_data = serializer.validated_data

        # Run ML prediction
        result = predict_credit_risk(applicant_data)

        # Save result in database
        record = CreditRiskPrediction.objects.create(
            **applicant_data,
            default_probability=result["default_probability"],
            prediction=result["prediction"]
        )

        return Response(
            {
                "id": record.id,
                "default_probability": result["default_probability"],
                "default_probability_percent": result["default_probability_percent"],
                "prediction": result["prediction"],
                "risk_label": result["risk_label"],
                "threshold": result["threshold"],
                "explanation": result["explanation"],
            },
            status=status.HTTP_201_CREATED
        )


def prediction_form(request):
    return render(
        request,
        "predictor/prediction_form.html"

   )

def prediction_history(request):
    predictions = CreditRiskPrediction.objects.all().order_by(
        "-created_at"
    )

    return render(
        request,
        "predictor/prediction_history.html",
        {
            "predictions": predictions
        }
    )


def model_info(request):
    return render(
        request,
        "predictor/model_info.html"
    )

def dashboard(request):
    predictions = CreditRiskPrediction.objects.all()

    total_predictions = predictions.count()

    higher_risk = predictions.filter(
        prediction=1
    ).count()

    lower_risk = predictions.filter(
        prediction=0
    ).count()

    average_probability = predictions.aggregate(
        average=Avg("default_probability")
    )["average"]

    recent_predictions = predictions.order_by(
        "-created_at"
    )[:8]

    context = {
        "total_predictions": total_predictions,
        "higher_risk": higher_risk,
        "lower_risk": lower_risk,
        "average_probability": (
            round(average_probability * 100, 2)
            if average_probability is not None
            else 0
        ),
        "recent_predictions": recent_predictions,
        "model_metrics": MODEL_METRICS,
    }

    return render(
        request,
        "predictor/dashboard.html",
        context
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_FILE = PROJECT_ROOT / "models" / "ml_analytics.json"


def model_analytics(request):
    analytics = {}

    if ANALYTICS_FILE.exists():
        with ANALYTICS_FILE.open("r", encoding="utf-8") as file:
            analytics = json.load(file)

    return render(
        request,
        "predictor/model_analytics.html",
        {"analytics": analytics}
    )