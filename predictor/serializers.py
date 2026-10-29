from rest_framework import serializers


class CreditRiskInputSerializer(serializers.Serializer):
    loan_amnt = serializers.FloatField(min_value=0)
    term = serializers.IntegerField()
    int_rate = serializers.FloatField(min_value=0)
    installment = serializers.FloatField(min_value=0)

    emp_length = serializers.FloatField(
        required=False,
        allow_null=True
    )

    home_ownership = serializers.CharField()
    annual_inc = serializers.FloatField(min_value=0)
    verification_status = serializers.CharField()
    purpose = serializers.CharField()

    dti = serializers.FloatField(
        required=False,
        allow_null=True
    )

    delinq_2yrs = serializers.FloatField(min_value=0)

    inq_last_6mths = serializers.FloatField(
        required=False,
        allow_null=True
    )

    open_acc = serializers.FloatField(min_value=0)
    pub_rec = serializers.FloatField(min_value=0)
    revol_bal = serializers.FloatField(min_value=0)

    revol_util = serializers.FloatField(
        required=False,
        allow_null=True
    )

    total_acc = serializers.FloatField(min_value=0)

    mort_acc = serializers.FloatField(
        required=False,
        allow_null=True
    )

    pub_rec_bankruptcies = serializers.FloatField(
        required=False,
        allow_null=True
    )

    fico_score = serializers.FloatField(
        min_value=300,
        max_value=850
    )

    grade = serializers.ChoiceField(
        choices=["A", "B", "C", "D", "E", "F", "G"]
    )

    def validate_term(self, value):
        if value not in [36, 60]:
            raise serializers.ValidationError(
                "Term must be either 36 or 60 months."
            )

        return value