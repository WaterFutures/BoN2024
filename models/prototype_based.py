from models.base import Model
import pandas as pd
import numpy as np
from preprocessing.advanced_transforms import LGBM_weather_features
from preprocessing.simple_transforms import LinearInterpolation
import matplotlib.pyplot as plt


WEEK_LEN = 24 * 7

class Nearest_Weather(Model):
    def __init__(self, nneighbors=1, weather_features='all'):
        self.nneighbors = nneighbors
        self.weather_features = weather_features

    def fit(self, demands, weather):


        if type(self.weather_features) == list:
            weather = weather[self.weather_features]
        weather = weather.to_numpy()

        self.data = {}
        self.weather = {}
        self.clf = {}

        self.dmas = demands.columns
        for dma in self.dmas:

            data = demands[dma].to_numpy()
            data_samples = []
            weather_samples = []
            # plt.figure()
            for i in range(0, len(data), WEEK_LEN):
                # if np.sum(np.isnan(data[i:i+WEEK_LEN])) == 0:
                if not np.any(np.isnan(data[i:i+WEEK_LEN])):
                    
                    data_samples.append(data[i:i+WEEK_LEN])
                    weather_samples.append(weather[i:i+WEEK_LEN])
                    # plt.plot(data[i:i+WEEK_LEN])
            self.weather[dma] = weather_samples #np.hstack(weather_samples)
            self.data[dma] = data_samples #np.hstack(data_samples)
            # plt.show()
            # plt.close()
            

        
        

    def forecast(self, demand_test, weather_test):
        if type(self.weather_features) == list:
            weather_test = weather_test[self.weather_features]
        weather_test = weather_test.to_numpy()
        
        prediction = np.zeros((WEEK_LEN, len(self.dmas)))
        for i,dma in enumerate(self.dmas):
            weather = self.weather[dma]
            
            distances = [np.linalg.norm(weather_test-w) for w in weather]
            closest = np.argsort(distances)
            prediction[:,i] = np.mean([self.data[dma][i] for i in closest[:min(self.nneighbors,len(closest))]], axis=0)

        return prediction


prototype_5 = {
    'name': 'Prototype_simple5',
    'model': Nearest_Weather(),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LinearInterpolation(interpolation_range=5)]
    }
}

prototype_3 = {
    'name': 'Prototype_simple3',
    'model': Nearest_Weather(nneighbors=3),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LinearInterpolation(interpolation_range=5)]
    }
}

prototype_7 = {
    'name': 'Prototype_simple7',
    'model': Nearest_Weather(nneighbors=7),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LinearInterpolation(interpolation_range=5)]
    }
}

prototype_fe = {
    'name': 'Prototype_features',
    'model': Nearest_Weather(),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LGBM_weather_features(), LinearInterpolation(interpolation_range=5)]#
    }
}

prototype_fe_subset = {
    'name': 'Prototype_selected_features',
    'model': Nearest_Weather(weather_features=['real_feel','heat_index', 'days_since_rain']),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LGBM_weather_features(), LinearInterpolation(interpolation_range=5)]#
    }
}
