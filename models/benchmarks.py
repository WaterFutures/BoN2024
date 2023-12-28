import models
from models.base import Model
import pandas as pd
import numpy as np

class PreviousWeek(Model):

    def fit(self, demands, weather):
        self.prev_week = demands.iloc[-168:].fillna(0).to_numpy()

    def forecast(self, demand_test, weather_test):
        return self.prev_week

previous_week = {
    'name': 'PrevWeek',
    'model': PreviousWeek(),
    'preprocessing': {
        'demand': [],
        'weather': []
    }
}

    
class AverageWeek(Model):

    def fit(self, demands, weather):
        demands = demands.to_numpy()
        self.avg_week = np.nanmean(demands.reshape((demands.shape[0] // 168, 168, demands.shape[1])), axis=0)

    def forecast(self, demand_test, weather_test):
        return self.avg_week

average_week = {
    'name': 'AvgWeek',
    'model': AverageWeek(),
    'preprocessing': {
        'demand': [],
        'weather': []
    }
}
