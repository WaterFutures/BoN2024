from eval.strategies.base import Strategy

import numpy as np

class BestOnTest(Strategy):

    def find_best_models(self, testresults: dict) -> list:
        # Create numpy array with model names to be able to list-index them
        model_names = list(testresults.keys())

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