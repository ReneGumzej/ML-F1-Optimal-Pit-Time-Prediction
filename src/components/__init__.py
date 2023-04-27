import os
from definitions import ROOT_PATH

data_path = "data.csv"

ABS_DATA_PATH = os.path.join(ROOT_PATH, data_path)

print(ABS_DATA_PATH)

