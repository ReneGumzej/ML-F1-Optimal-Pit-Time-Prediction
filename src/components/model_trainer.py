import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging

from src.utils import get_best_params
from config.config import model_parameter

@dataclass
class ModelTrainerConfiguration:
    model_object_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_config = ModelTrainerConfiguration()
        self.parameter = model_parameter
        
        
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        
        logging.info("Starting model training")
        
        try:
            trained_model = []
            reached_score = []

            models = {
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "KNearest Regressor": KNeighborsRegressor()
            }
            
            logging.info("Entering the train process")
            
            for i in range(len(models)):
                
                model = list(models.values())[i]

                parameter_grid = self.parameter[list(models.keys())[i]]
                
                
                logging.info(f"preparing Model: '{models.keys()[i]}' with Parameter {parameter_grid} for training")
                
                best_parameter = get_best_params(X_train, y_train, model, parameter_grid)
                
                logging.info("Estimated best parameters")
                
                model.set_params(**best_parameter)
                logging.info("passing the parameters to models")
                
                logging.info("Start to train the Model")
                model.fit(X_train, y_train)
                
                print(best_parameter)
                
                logging.info(f"Finished training for Model: {models.keys()[i]} ")
                score = model.score(X_test, y_test)
                
                logging.info("Finished computation of model score")
                
                trained_model.append(model)
                reached_score.append(score)
            
            model_performance = list(zip(trained_model,reached_score))
            
            logging.info("Completed model training loop")
            return model_performance

            
        except Exception as e:
            CustomException(e,sys)