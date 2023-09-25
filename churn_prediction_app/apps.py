from django.apps import AppConfig
from configparser import ConfigParser
import json
import pickle
import os
import warnings
import logging
warnings.filterwarnings("ignore")
config = ConfigParser()
config.read(os.path.join("churn_prediction_app", "config", "config.ini"))

# Load all files into app
class ChurnPredictionAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "churn_prediction_app"
    logger = logging.getLogger(__name__)
    cat_cos_list = config['path']['categorical_cols_list_path']
    with open(cat_cos_list, 'r') as f:
        categorical_columns = json.load(f)

    label_encoders = {}
    for col in categorical_columns:
        file_name = os.path.join('churn_prediction_app', 'config', 'label_encoder_{}.pkl'.format(col))

        with open(file_name, 'rb') as f:
            encoder = pickle.load(f)
            label_encoders[col] = encoder
    categorical_columns = categorical_columns
    label_encoders = label_encoders

    #  Load Model
    DT_model_path = config['path']['model_path']
    with open(DT_model_path, 'rb') as f:
        DT_model = pickle.load(f)

