import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformation object")

            numerical_columns = [
                "RoundNumber",
                "eventYear",
                "Stint",
                "bestPreRaceTime",
                "meanAirTemp",
                "meanTrackTemp",
                "meanHumid",
                "Rainfall",
                "GridPosition",
                "Position",
                "raceStintsNums",
                "TyreAge",
                "lapNumberAtBeginingOfStint",
                "CircuitLength",
                "designedLaps",
                "fuel_slope",
                "fuel_bias",
                "deg_slope",
                "deg_bias",
                "lag_slope_mean",
                "lag_bias_mean",
            ] #The target feature must be dropped
            categorical_columns = [
                "EventName",
                "Team",
                "Compound",
                "Driver",
                "bestLapTimeIsFrom",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info(
                "Initialised numerical pipeline with 'Simpleimputer' and 'Standardscaler'"
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info(
                "Initialised categorial pipeline with 'Simpleimputer', 'OneHotEncoder' and 'Standardscaler'"
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            logging.info(
                "Initialised ColumnTransformer with 'numerical pipeline' and 'categorial pipeline'"
            )
            logging.info("Preprocessor object is created!")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read in train and test data")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()

            label = "StintLen"
            drop_column = "Unnamed: 0"

            y_train = np.array(train_df[label])
            y_test = np.array(test_df[label])

            X_train = train_df.drop(
                columns=[label, drop_column], axis=1
            )
            X_test = test_df.drop(
                columns=[label, drop_column], axis=1
            )
            

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_array = preprocessing_obj.fit_transform(
                X_train
            )
            X_test_array = preprocessing_obj.transform(X_test)

            
            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessing_obj,
            )

            logging.info("Data transformation completed!")
            return (
                X_train_array,
                X_test_array,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
