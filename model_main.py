from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformer = DataTransformation()
    X_train, X_test, y_train, y_test, _ = data_transformer.initiate_data_transformation(
        train_data, test_data)

    model_trainer = ModelTrainer()
    model_performance = model_trainer.initiate_model_trainer(
        X_train, y_train, X_test, y_test)

    print(model_performance)
