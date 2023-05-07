import os
import sys

from src.utils import save_object
from dataclasses import dataclass
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelEvaluationConfig:
    model_object_path = os.path.join("artifacts", "model.pkl")


class ModelEvaluator:
    def __init__(self) -> None:
        self.model_eval_config = ModelEvaluationConfig()

    def get_best_model(self, models, X_test, y_test) -> None:
        logging.info("initializing the evaluation process")

        logging.info("Start to evaluate the models")
        try:
            best_models = {}

            for i in range(len(models)):

                model = models[i]

                y_test_predicted = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_predicted)

                best_models[model] = test_model_score

            logging.info("Best model found")
            best_r2_score = max(sorted(best_models.values()))

            logging.info("Check if the model score is above 60%")

            if best_r2_score < 0.6:
                print("No model reached the 60% mark!")

            best = sorted(best_models, key=best_models.get, reverse=True)
            logging.info(
                f"Best model found: Model: {best[0]}, R2-Score: {best_r2_score}")

            logging.info("Saving model object")

            save_object(
                file_path=self.model_eval_config.model_object_path,
                object=best[0]
            )

            logging.info("Model evalutaion completed!")
            return best_r2_score, best[0]

        except Exception as e:
            raise CustomException(e, sys)
