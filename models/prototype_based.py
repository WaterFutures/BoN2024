from models.base import Model
import pandas as pd
import numpy as np
from preprocessing.advanced_transforms import LGBM_weather_features
from preprocessing.simple_transforms import LinearInterpolation
from preprocessing.weather_feature_engineering import RealFeel, AmountOfRainDecayed, RemoveBasicFeatures
import matplotlib.pyplot as plt


WEEK_LEN = 24 * 7
DAY_LEN = 24


# standardize weather features
class Nearest_Weather(Model):
    def __init__(self, nneighbors=1, weather_features='all'):
        self.nneighbors = nneighbors
        self.weather_features = weather_features
        

    def fit(self, demands, weather):


        if type(self.weather_features) == list:
            weather = weather[self.weather_features]
        weather = weather.to_numpy()

        self.weather_mean = np.mean(weather, axis=0)
        self.weather_std = np.std(weather, axis=0)
        weather = (weather-self.weather_mean) / self.weather_std

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
                if not np.any(np.isnan(data[i:i+WEEK_LEN])):
                    
                    data_samples.append(data[i:i+WEEK_LEN])
                    weather_samples.append(weather[i:i+WEEK_LEN])
                    print(weather[i:i+WEEK_LEN].shape)
            self.weather[dma] = weather_samples 
            self.data[dma] = data_samples 
                   

    def forecast(self, demand_test, weather_test):
        if type(self.weather_features) == list:
            weather_test = weather_test[self.weather_features]
        weather_test = weather_test.to_numpy()

        weather_test = (weather_test-self.weather_mean) / self.weather_std 
        
        prediction = np.zeros((WEEK_LEN, len(self.dmas)))
        for i,dma in enumerate(self.dmas):
            weather = self.weather[dma]
        
            distances = [np.linalg.norm(weather_test-w) for w in weather]
            closest = np.argsort(distances)
            prediction[:,i] = np.mean([self.data[dma][i] for i in closest[:min(self.nneighbors,len(closest))]], axis=0)

        return prediction
    
class Nearest_Weather_Daily(Model):
    def __init__(self, nneighbors=1, weather_features='all'):
        self.nneighbors = nneighbors
        self.weather_features = weather_features


    def fit(self, demands, weather):
        if type(self.weather_features) == list:
            weather = weather[self.weather_features]
        weather = weather.to_numpy()

        self.weather_mean = np.mean(weather, axis=0)
        self.weather_std = np.std(weather, axis=0)
        weather = (weather-self.weather_mean) / self.weather_std

        self.data = {}
        self.weather = {}
        self.clf = {}

        self.dmas = demands.columns
        for dma in self.dmas:

            data = demands[dma].to_numpy()
            data_samples = []
            weather_samples = []
            
            for i in range(0, len(data), DAY_LEN):
                if not np.any(np.isnan(data[i:i+DAY_LEN])):
                    
                    data_samples.append(data[i:i+DAY_LEN])
                    weather_samples.append(weather[i:i+DAY_LEN])
            self.weather[dma] = weather_samples 
            self.data[dma] = data_samples 
                   

    def forecast(self, demand_test, weather_test):
        if type(self.weather_features) == list:
            weather_test = weather_test[self.weather_features]
        weather_test = weather_test.to_numpy()
        weather_test = (weather_test-self.weather_mean) / self.weather_std 

        prediction = np.zeros((WEEK_LEN, len(self.dmas)))
        for day in range(7):
            weather_test_day = weather_test[day:day+24, :]
            for i,dma in enumerate(self.dmas):
                weather = self.weather[dma]
                
                distances = [np.linalg.norm(weather_test_day-w) for w in weather]
                closest = np.argsort(distances)
                prediction[day*24:(day+1)*24,i] = np.mean([self.data[dma][i] for i in closest[:min(self.nneighbors,len(closest))]], axis=0)

        return prediction


proto_daily_5 = {
    'name': 'Prototype_daily_5',
    'model': Nearest_Weather_Daily(),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LinearInterpolation(interpolation_range=5),RealFeel(),AmountOfRainDecayed(),RemoveBasicFeatures()]
    }
}


proto_weekly_5 = {
    'name': 'Prototype_weekly_5',
    'model': Nearest_Weather(),
    'preprocessing': {
        'demand': [LinearInterpolation(interpolation_range=5)],
        'weather': [LinearInterpolation(interpolation_range=5),RealFeel(),AmountOfRainDecayed(),RemoveBasicFeatures()]
    }
}
