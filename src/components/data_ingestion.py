import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initialisation of data ingestion started")

        try:
            logging.info("Reading the dataset as DataFrame")
            df = pd.read_csv(
                r".\notebook\data\data.csv"
            )  # here we can read from any datascource
            logging.info("Reading the dataset completed")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(
                "Created 'artifacts' folder and saved the raw data as 'data.csv'"
            )

            logging.info("Train-Test-Split initiated")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            logging.info(
                "Train dataset created and saved as 'train.csv' in the 'artifacts' folder"
            )
            test_data.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(
                "Test dataset created and saved as 'test.csv' in the 'artifacts' folder"
            )

            logging.info("Data Ingestion is Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
