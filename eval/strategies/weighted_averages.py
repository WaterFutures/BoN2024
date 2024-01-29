from eval.strategies.base import Strategy

import numpy as np
import pandas as pd

from eval.data_loading_helpers import DMAS_NAMES
WEEK_LEN = 24*7

class WeightedAverage(Strategy):

    def __init__(self, best_models, weights) -> None:
        # if you pass a list of models and a list of weights, it means you are using the same for all dmas
        self.rule = {}
        if isinstance(best_models, list):
            for dma in DMAS_NAMES:
                self.rule[dma] = {}
                self.rule[dma]['bm'] = best_models
        else:
            # I expect a dictionary with the best models for each dma
            for dma in DMAS_NAMES:
                self.rule[dma] = {}
                self.rule[dma]['bm'] = best_models[dma]
            
        if isinstance(weights, list):
            weights = np.array(weights)

        if isinstance(weights, np.ndarray):
            # check that the weights sum to 1
            assert np.isclose(weights.sum(), 1), "The weights must sum to 1"
            for dma in DMAS_NAMES:
                self.rule[dma]['w'] = weights
        else:
            # I expect a dictionary with the weights for each dma
            for dma in DMAS_NAMES:
                assert np.isclose(weights[dma].sum(), 1), "The weights must sum to 1"
                self.rule[dma]['w'] = weights[dma]

    def find_best_models(self, testresults: dict) -> list:
        all_models = []
        for dma in DMAS_NAMES:
            all_models += self.rule[dma]['bm']

        return list(set(all_models))
    
    def combine_forecasts(self, forecasts) -> np.ndarray:
        ensemble_forecast = np.zeros((WEEK_LEN,len(DMAS_NAMES)))

        for i, dma in enumerate(DMAS_NAMES):
            dma_fcst = np.stack([forecasts[model].groupby('Date').mean().loc[:,dma].to_numpy('float') for model in self.rule[dma]['bm']]).reshape(-1, WEEK_LEN).T
            ensemble_forecast[:,i] = np.sum(dma_fcst * self.rule[dma]['w'], axis=1)

        return ensemble_forecast