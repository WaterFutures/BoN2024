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

from models.base import Model
import numpy as np
from sklearn.linear_model import LinearRegression

class PatternRegressionDaily(Model):

    def __init__(self, model=LinearRegression, model_args={}):
        self.model_mean_weekday = model(**model_args)
        self.model_std_weekday = model(**model_args)
        self.cur_pattern_weekday = np.zeros((24, 10))
        self.model_mean_weekend = model(**model_args)
        self.model_std_weekend = model(**model_args)
        self.cur_pattern_weekend = np.zeros((24, 10))

    def fit(self, demands, weather):
        demands = demands.to_numpy().reshape((int(demands.shape[0] / 168), 7, 24, 10))
        weather = weather.to_numpy().reshape((int(weather.shape[0] / 168), 7, 24, weather.shape[1]))

        demands_weekdays = demands[:, [0,1,2,3,4],:, :]
        demands_weekend = demands[:, [5,6],:, :]

        weather_weekdays = weather[:, [0,1,2,3,4],:, :]
        weather_weekend = weather[:, [5,6],:, :]

        # Train regressors
        weather_means_weekdays = np.nanmean(weather_weekdays, axis=2).reshape((weather_weekdays.shape[0]*5, weather_weekdays.shape[3]))
        weather_means_weekend = np.nanmean(weather_weekend, axis=2).reshape((weather_weekend.shape[0]*2, weather_weekend.shape[3]))

        means_weekdays = np.nanmean(demands_weekdays, axis=2).reshape((demands_weekdays.shape[0] * 5, 10))
        stds_weekdays = np.nanstd(demands_weekdays, axis=2).reshape((demands_weekdays.shape[0] * 5, 10))

        means_weekend = np.nanmean(demands_weekend, axis=2).reshape((demands_weekend.shape[0] * 2, 10))
        stds_weekend = np.nanstd(demands_weekend, axis=2).reshape((demands_weekend.shape[0] * 2, 10))

        # Filter nans
        nan_mask_weekdays = np.where(~np.any(np.isnan(means_weekdays), axis=1))
        weather_means_weekdays = weather_means_weekdays[nan_mask_weekdays]
        means_weekdays = means_weekdays[nan_mask_weekdays]
        stds_weekdays = stds_weekdays[nan_mask_weekdays]

        nan_mask_weekend = np.where(~np.any(np.isnan(means_weekend), axis=1))
        weather_means_weekend = weather_means_weekend[nan_mask_weekend]
        means_weekend = means_weekend[nan_mask_weekend]
        stds_weekend = stds_weekend[nan_mask_weekend]

        # Fit models
        self.model_mean_weekday.fit(weather_means_weekdays, means_weekdays)
        self.model_std_weekday.fit(weather_means_weekdays, stds_weekdays)

        self.model_mean_weekend.fit(weather_means_weekend, means_weekend)
        self.model_std_weekend.fit(weather_means_weekend, stds_weekend)

        # For Pattern
        demand_pattern_weekday = np.nanmean(demands_weekdays, axis=(0,1))
        self.cur_pattern_weekday = (demand_pattern_weekday - np.mean(demand_pattern_weekday, axis=0)) / np.std(demand_pattern_weekday, axis=0)

        demand_pattern_weekend = np.nanmean(demands_weekend, axis=(0,1))
        self.cur_pattern_weekend = (demand_pattern_weekend - np.mean(demand_pattern_weekend, axis=0)) / np.std(demand_pattern_weekend, axis=0)

                
    def forecast(self, demand_test, weather_test):
        mean_weather = np.nanmean(weather_test.to_numpy().reshape((7,24,weather_test.shape[1])), axis=1)

        pred = []
        for i in range(7):
            if i < 5:
                # Weekday
                pred_mean = self.model_mean_weekday.predict(mean_weather[i][None,:])[0]
                pred_std = self.model_std_weekday.predict(mean_weather[i][None,:])[0]

                pred.append((self.cur_pattern_weekday * pred_std) + pred_mean)
            else:
                #Weekdend
                pred_mean = self.model_mean_weekend.predict(mean_weather[i][None,:])[0]
                pred_std = self.model_std_weekend.predict(mean_weather[i][None,:])[0]

                pred.append((self.cur_pattern_weekend * pred_std) + pred_mean)
        return np.concatenate(pred)
        
from preprocessing.simple_transforms import Logarithm
from preprocessing.weather_feature_engineering import RealFeel, DewPoint, WindChill

pattern_regression_daily = {
    'name': f'PatternRegressionDaily',
    'model': PatternRegressionDaily(),
    'preprocessing': {
        'demand': [Logarithm()],
        'weather': [RealFeel(), DewPoint(), WindChill()]
    }
}