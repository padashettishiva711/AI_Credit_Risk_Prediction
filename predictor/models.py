from django.db import models


class CreditRiskPrediction(models.Model):
    # Loan information
    loan_amnt = models.FloatField()
    term = models.IntegerField()
    int_rate = models.FloatField()
    installment = models.FloatField()

    # Borrower information
    emp_length = models.FloatField(null=True, blank=True)
    home_ownership = models.CharField(max_length=30)
    annual_inc = models.FloatField()
    verification_status = models.CharField(max_length=30)
    purpose = models.CharField(max_length=50)

    # Credit information
    dti = models.FloatField(null=True, blank=True)
    delinq_2yrs = models.FloatField()
    inq_last_6mths = models.FloatField(null=True, blank=True)
    open_acc = models.FloatField()
    pub_rec = models.FloatField()
    revol_bal = models.FloatField()
    revol_util = models.FloatField(null=True, blank=True)
    total_acc = models.FloatField()
    mort_acc = models.FloatField(null=True, blank=True)
    pub_rec_bankruptcies = models.FloatField(null=True, blank=True)
    fico_score = models.FloatField()

    # LendingClub grade
    grade = models.CharField(max_length=5)

    # Prediction results
    default_probability = models.FloatField()
    prediction = models.IntegerField()

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def default_probability_percent(self):
           return self.default_probability * 100


    def __str__(self):
        risk = "Higher Risk" if self.prediction == 1 else "Lower Risk"

        return (
            f"Prediction #{self.pk} - "
            f"{risk} - "
            f"{self.default_probability:.2%}"
        )
