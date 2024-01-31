import models
from models.base import Model
import pandas as pd
import numpy as np
    
class ExpWeightedRollingWeek(Model):

    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, demands, weather):
        demands = demands.to_numpy()
        demands = demands.reshape(demands.shape[0] // 168, 168, demands.shape[1])

        cur_window = demands[-self.window_size:]
        cur_nan_mask = ~np.isnan(cur_window)
        cur_weights = cur_nan_mask * np.exp(np.arange(self.window_size))[:,None,None]
        cur_normalization_factors = np.sum(cur_weights, axis=0) / np.sum(cur_nan_mask, axis=0)
        self.avg_week = np.nanmean(cur_window * cur_weights, axis=0) / cur_normalization_factors


    def forecast(self, demand_test, weather_test):
        return np.nan_to_num(self.avg_week, nan=0)

