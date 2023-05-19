import os
import sys

from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import get_best_params
from config.config import model_parameter


class ModelTrainer:
    def __init__(self) -> None:
        pass
        
    def initiate_model_trainer(self, X_train, y_train):

        logging.info("Starting model training")

        try:
            trained_models = []

            models = {
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Entering the train process")

            for i in range(len(models)):

                model = list(models.values())[i]

                parameter_grid = model_parameter[list(models.keys())[i]]

                best_parameter = get_best_params(X_train, y_train, model, parameter_grid)

                model.set_params(**best_parameter)

                model.fit(X_train, y_train)

                trained_models.append(model)

            logging.info("Model training completed!")
            return trained_models

        except Exception as e:
            CustomException(e, sys)
