# Main file to run the training and evaluation of the models.
# Then we run the evalution of the strategies.
import os

import pandas as pd

from wflib.evaluator import WaterFuturesEvaluator as WFE


print("Running the evaluation of the models")

wfe = WFE(data_dir= os.path.join('paper', 'data'),
          forecasting_horizon=24*7, # 1 week
          testing_starts_after_n_horizons_with_n=52, # We use at least 1 year of data for training the very first model that we test
          ignore_previously_saved_results=False
            )

# Add the parent directory to the path to find the models used in the battle
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.benchmark import RollingAverageWeek, AutoRollingAverageWeek
from preprocessing.impute_and_fill import FillZero, FillAvgWeek

wfe.add_model_configuration({
    'name': 'PrevWeek',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'target': [FillZero()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'PrevWeek_v2',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'AutoRollingAverage',
    'model': AutoRollingAverageWeek(),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': False,
    'n_eval_runs': 10
})
