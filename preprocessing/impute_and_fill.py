from preprocessing.base import Preprocessing

import numpy as np
import pandas as pd

from models.benchmark import RollingAverageWeek

WEEK_LEN = 7*24

# With inpute I mean the functions that fill nans in the original training data!
# With fill I mean functions that fill nans in predictions/forecasts (e.g., for previous week)

class FillZero(Preprocessing):
    def inverse_transform(self, X):
        return X.fillna(0)

class FillStandard(Preprocessing):
    def inverse_transform(self, X):
        return X.interpolate(method='linear').ffill().bfill()
    
class FillMean(Preprocessing):
    def fit(self, X):
        self.mean = X.mean(axis=0)

    def inverse_transform(self, X):
        return X.fillna(self.mean)

class FillAvgWeek(Preprocessing):
    def fit(self, X):
        self.avg_week_model = RollingAverageWeek(window_size=X.shape[0]//WEEK_LEN).fit(X, None)

    def inverse_transform(self, X):
        for dma in X.columns:
            mask = X[dma].isna()
            X.loc[mask, dma] = self.avg_week_model.dmas_models[dma][mask]
        return X