from eval.strategies.base import Strategy

import numpy as np
import pandas as pd

class BestOnLastNW(Strategy):

    def __init__(self, n_weeks) -> None:
        self.best_models = None
        self.n_weeks = n_weeks

    def find_best_models(self, testresults: dict) -> list:
        # Create numpy array with model names to be able to list-index them
        model_names = list(testresults.keys())

        # Extract what are the number of the weeks in the results so that you can extract the last n_weeks
        weeks = np.unique(testresults[model_names[0]].index.get_level_values('Test week'))
        sel_weeks = weeks[-self.n_weeks:]

        for model in model_names:
            testresults[model] = pd.concat([testresults[model].xs(week, level='Test week') for week in sel_weeks])

        # Calculate the average of the performance indicators
        average_pis = np.stack([testresults[model].groupby('DMA').mean().to_numpy() for model in model_names])
        average_pi12 = np.mean(average_pis[:,:,:2], axis=2)
        average_pi3 = average_pis[:,:,2]

        # Find best models
        best_model_pi12 = [model_names[i] for i in np.argmin(average_pi12, axis=0)]
        best_model_pi3 = [model_names[i] for i in np.argmin(average_pi3, axis=0)]
        self.best_models = np.concatenate([[best_model_pi12, best_model_pi3]])

        return list(set(self.best_models.flatten()))

    def combine_forecasts(self, forecasts) -> np.ndarray:
        ensemble_forecast = np.zeros((24*7,10))

        for dma in range(10):
            ensemble_forecast[:168,dma] = forecasts[self.best_models[0][dma]].groupby('Date').mean().iloc[:168,dma].to_numpy('float')
            ensemble_forecast[168:,dma] = forecasts[self.best_models[1][dma]].groupby('Date').mean().iloc[168:,dma].to_numpy('float')

        return ensemble_forecast

class BestOnTest(BestOnLastNW):

    def __init__(self) -> None:
        super().__init__(4)