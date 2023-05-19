import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_objet
import os

from dataclasses import dataclass


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_objet(file_path=model_path)
            preprocessor = load_objet(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class CustomData:
    event_name: str
    event_year: str
    team: str
    compound: str
    driver: str
    best_lap_time_from: str

    round_number: int
    stint: float
    best_pre_race_time: float
    mean_air_temp: float
    mean_track_temp: float
    mean_humid: float
    rainfall: int
    grid_position: int
    position: int
    race_stint_nums: int
    tyre_age: float
    lap_number_begin_stint: float
    circiut_lenght: float
    designed_laps: int
    fuel_slope: float
    fuel_bias: float
    deg_slope: float
    deg_bias: float
    lag_slope_mean: float
    lag_bias_mean: float

    def get_data_as_dataframe(self):
        try:
            custom_data = {
                "event_name": [self.event_name],
                "event_year": [self.event_year],
                "team": [self.team],
                "compound": [self.compound],
                "driver": [self.driver],
                "best_lap_time_from": [self.best_lap_time_from],
                "round_number": [self.round_number],
                "stint":[ self.stint],
                "best_pre_race_time":[ self.best_pre_race_time],
                "mean_air_temp": [self.mean_air_temp],
                "mean_track_temp": [self.mean_track_temp],
                "mean_humid": [self.mean_humid],
                "rainfall": [self.rainfall],
                "grid_position": [self.grid_position],
                "position": [self.position],
                "race_stint_nums": [self.race_stint_nums],
                "tyre_age": [self.tyre_age],
                "lap_number_begin_stint": [self.lap_number_begin_stint],
                "circiut_lenght": [self.circiut_lenght],
                "designed_laps": [self.designed_laps],
                "fuel_slope": [self.fuel_slope],
                "fuel_bias": [self.fuel_bias],
                "deg_slope": [self.deg_slope],
                "deg_bias": [self.deg_bias],
                "lag_slope_mean": [self.lag_slope_mean],
                "lag_bias_mean": [self.lag_bias_mean],
            }

            return pd.DataFrame(custom_data)

        except Exception as e:
            raise CustomException(e, sys)
