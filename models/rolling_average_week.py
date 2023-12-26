import models
from models.base import Model
import pandas as pd
import numpy as np
    
class RollingAverageWeek(Model):

    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, demands, weather):
        demands = demands.to_numpy()
        self.avg_week = np.nanmean(demands.reshape((demands.shape[0] // 168, 168, demands.shape[1]))[-self.window_size:], axis=0)

    def forecast(self, weather):
        return np.nan_to_num(self.avg_week, nan=0)

def rolling_average_week(x):
    return {
        'name': f'RollingAvgWeek{x}',
        'model': RollingAverageWeek(x),
        'preprocessing': {
            'demand': [],
            'weather': []
        }
    }