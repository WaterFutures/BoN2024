from model import Model
from data_loader import DMAS_NAMES
import numpy as np
from sklearn.linear_model import LinearRegression

class PatternRegression(Model):

    def __init__(self, model=LinearRegression, model_args={}):
        self.model_mean = model(**model_args)
        self.model_std = model(**model_args)
        self.cur_pattern = np.zeros((24*7, 10))
        self.model_name = model.__name__

    def fit(self, X_train):
        # TODO: Interpolate better?
        X_train = X_train.interpolate(limit_direction='both').to_numpy()
        
        dma_weeks = X_train[:,:len(DMAS_NAMES)].reshape((int(X_train.shape[0] / 168), 168, 10))
        weather_weeks = X_train[:,len(DMAS_NAMES):].reshape((int(X_train.shape[0] / 168), 168, 4))

        # Train regressors
        weather_week_means = np.mean(weather_weeks, axis=1)

        means = np.mean(dma_weeks, axis=1)
        stds = np.std(dma_weeks, axis=1)

        self.model_mean.fit(weather_week_means, means)
        self.model_std.fit(weather_week_means, stds)

        # For Pattern
        dma_week_averages = np.mean(dma_weeks, axis=0)
        self.cur_pattern = (dma_week_averages - np.mean(dma_week_averages, axis=0)) / np.std(dma_week_averages, axis=0)

                

    def forecast(self, X_test):
        weather = X_test.interpolate(limit_direction='both').to_numpy()[-24*7:, len(DMAS_NAMES):]
        mean_weather = np.mean(weather, axis=0)

        pred_mean = self.model_mean.predict(mean_weather[None,:])[0]
        pred_std = self.model_std.predict(mean_weather[None,:])[0]

        pred = (self.cur_pattern * pred_std) + pred_mean
        return pred
        

    def name (self):
        return f'PatternRegression_{self.model_name}'
    
    def preprocess_data(self, train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h):
        return train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h
    
    def forecasted_dmas(self):
        return DMAS_NAMES
    
    def forecasted_dmas_idx(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]