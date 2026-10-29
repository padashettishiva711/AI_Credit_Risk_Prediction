from django.test import SimpleTestCase

from .ml_service import (
    THRESHOLD,
    predict_credit_risk,
)

from rest_framework.test import APITestCase
from .models import CreditRiskPrediction


class CreditRiskMLServiceTest(SimpleTestCase):

    def setUp(self):
        self.applicant = {
            "loan_amnt": 15000,
            "term": 36,
            "int_rate": 12.5,
            "installment": 500,
            "emp_length": 5,
            "home_ownership": "RENT",
            "annual_inc": 65000,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 18.5,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 10,
            "pub_rec": 0,
            "revol_bal": 12000,
            "revol_util": 45.0,
            "total_acc": 25,
            "mort_acc": 1,
            "pub_rec_bankruptcies": 0,
            "fico_score": 700,
            "grade": "B",
        }

    def test_credit_risk_prediction(self):
        result = predict_credit_risk(
            self.applicant
        )

        self.assertIn(
            "default_probability",
            result,
        )

        self.assertIn(
            "prediction",
            result,
        )

        self.assertIn(
            "risk_label",
            result,
        )

        self.assertIn(
            "threshold",
            result,
        )

        self.assertGreaterEqual(
            result["default_probability"],
            0.0,
        )

        self.assertLessEqual(
            result["default_probability"],
            1.0,
        )

        self.assertIn(
            result["prediction"],
            [0, 1],
        )

        self.assertEqual(
            result["threshold"],
            THRESHOLD,
        )


class CreditRiskAPITest(APITestCase):

    def setUp(self):
        self.applicant = {
            "loan_amnt": 15000,
            "term": 36,
            "int_rate": 12.5,
            "installment": 500,
            "emp_length": 5,
            "home_ownership": "RENT",
            "annual_inc": 65000,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 18.5,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 10,
            "pub_rec": 0,
            "revol_bal": 12000,
            "revol_util": 45.0,
            "total_acc": 25,
            "mort_acc": 1,
            "pub_rec_bankruptcies": 0,
            "fico_score": 700,
            "grade": "B",
        }

    def test_prediction_api(self):
        response = self.client.post(
            "/api/predict/",
            self.applicant,
            format="json",
        )

        self.assertEqual(
            response.status_code,
            201,
        )

        self.assertIn(
            "default_probability",
            response.data,
        )

        self.assertIn(
            "prediction",
            response.data,
        )

        self.assertIn(
            "risk_label",
            response.data,
        )

        self.assertIn(
            "threshold",
            response.data,
        )

        self.assertEqual(
            CreditRiskPrediction.objects.count(),
            1,
        )

    def test_prediction_api_rejects_invalid_data(self):
            invalid_applicant = self.applicant.copy()

            # FICO score must be between 300 and 850
            invalid_applicant["fico_score"] = 100

            response = self.client.post(
                "/api/predict/",
                invalid_applicant,
                format="json",
            )

            self.assertEqual(
                response.status_code,
                400,
            )

            self.assertIn(
                "fico_score",
                response.data,
            )

            # Invalid requests must not be saved
            self.assertEqual(
                CreditRiskPrediction.objects.count(),
                0,
            )
