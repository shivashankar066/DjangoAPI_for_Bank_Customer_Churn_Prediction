from configparser import ConfigParser
import pandas as pd
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder
import pickle
from .apps import ChurnPredictionAppConfig
import warnings

warnings.filterwarnings("ignore")
import logging

config = ConfigParser()
config.read("churn_prediction_app/config/config.ini")

# Load the DataFrame
try:
    df = pd.read_csv(config["path"]["data"])
except FileNotFoundError:
    df = None


class churn_prediction(APIView):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        extra = {
            "cls_name": self.__class__.__name__,
        }
        self.logger = logging.LoggerAdapter(self.logger, extra)

    def post(self, request):
        start = datetime.now()
        request_data = request.data
        feature_names = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]
        X = pd.DataFrame([request_data], columns=feature_names)
        X['Balance'] = float(X['Balance'])
        X["EstimatedSalary"] = float(X['EstimatedSalary'])
        if X.shape[0] == 0:
            response = {"No records found in df"}
            return Response(response)

        for column in ChurnPredictionAppConfig.categorical_columns:
            le = LabelEncoder()
            le.fit(X[column])
            X[column] = le.transform(X[column])
        self.logger.info("First few rows of X:\n%s", X.head())
        self.logger.info("Data types of columns in X:\n%s", X.dtypes)
        predict_result = ChurnPredictionAppConfig.DT_model.predict(X)
        predicted_probabilities = ChurnPredictionAppConfig.DT_model.predict_proba(X)
        self.logger.info("Predicted Result: %s", predict_result)
        self.logger.info("Predicted Probabilities: %s", predicted_probabilities)

        if predict_result == 1:
            prediction_mapped = 'Yes'
        else:
            prediction_mapped = 'No'

        end = datetime.now()
        response = {
            "message": "Bank Churn Prediction Engine Service completed successfully.",
            "status": "Success",
            "statusCode": 200,
            "respTime": (end - start).total_seconds(),
            "Probability Score": str(predicted_probabilities),
            "(Customer Churn prediction) customer closed account ? ": str(prediction_mapped)

        }

        return Response(response)
