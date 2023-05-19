import os
import sys

import numpy as np 
import pandas as pd

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path: str, object):
    try:
        dir_path: str = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)
                  
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(y_true: pd.Series, y_predicted: pd.Series):
    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_predicted)
    return mae, rmse, r2

        
def get_best_params(X_train, y_train ,model, parameter):
    grid_clf = GridSearchCV(estimator=model, param_grid=parameter, cv=3)

    grid_clf.fit(X_train, y_train)
    
    return grid_clf.best_params_

def load_objet(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    