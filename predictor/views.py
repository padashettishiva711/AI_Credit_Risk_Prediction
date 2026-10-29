from django.shortcuts import render

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import CreditRiskPrediction
from .serializers import CreditRiskInputSerializer
from .ml_service import predict_credit_risk


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
                "default_probability_percent":
                    result["default_probability_percent"],
                "prediction": result["prediction"],
                "risk_label": result["risk_label"],
                "threshold": result["threshold"],
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