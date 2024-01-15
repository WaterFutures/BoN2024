from models.base import Model
import numpy as np
from sklearn.linear_model import LinearRegression

class PatternRegression(Model):

    def __init__(self, model=LinearRegression, model_args={}):
        self.model_mean = model(**model_args)
        self.model_std = model(**model_args)
        self.cur_pattern = np.zeros((24*7, 10))

    def fit(self, demands, weather):
        demands = demands.to_numpy().reshape((int(demands.shape[0] / 168), 168, 10))
        weather = weather.to_numpy().reshape((int(weather.shape[0] / 168), 168, weather.shape[1]))

        # Train regressors
        weather_week_means = np.nanmean(weather, axis=1)

        means = np.nanmean(demands, axis=1)
        stds = np.nanstd(demands, axis=1)

        # Filter nans
        nan_mask = np.where(~np.any(np.isnan(means), axis=1))
        weather_week_means = weather_week_means[nan_mask]
        means = means[nan_mask]
        stds = stds[nan_mask]

        # Predict
        self.model_mean.fit(weather_week_means, means)
        self.model_std.fit(weather_week_means, stds)

        # For Pattern
        demand_pattern = np.nanmean(demands, axis=0)
        self.cur_pattern = (demand_pattern - np.mean(demand_pattern, axis=0)) / np.std(demand_pattern, axis=0)

                
    def forecast(self, demand_test, weather_test):
        mean_weather = np.nanmean(weather_test.to_numpy(), axis=0)

        pred_mean = self.model_mean.predict(mean_weather[None,:])[0]
        pred_std = self.model_std.predict(mean_weather[None,:])[0]

        pred = (self.cur_pattern * pred_std) + pred_mean
        return pred
        
from preprocessing.simple_transforms import Logarithm
from preprocessing.weather_feature_engineering import RealFeel, DewPoint, WindChill

pattern_regression = {
    'name': f'PatternRegression',
    'model': PatternRegression(),
    'preprocessing': {
        'demand': [Logarithm()],
        'weather': [RealFeel(), DewPoint(), WindChill()]
    }
}