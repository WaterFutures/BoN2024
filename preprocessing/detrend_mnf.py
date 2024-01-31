from preprocessing.base import Preprocessing
import numpy as np
import pandas as pd

WEEK_LEN = 7*24

class DetrendMNF(Preprocessing):

    def __init__(self, n_weeks=4):
        super().__init__()
        self.trend = None
        self.n_weeks = n_weeks

    def fit(self, X):
        dmas_night_q = X[X.index.hour.isin([1, 2, 3, 4, 5])]
        dmas_mnf = dmas_night_q.groupby(dmas_night_q.index.date).min()

        # Now I have the MNF for each day in the dataset
        dmas_mnf_trend = dmas_mnf.rolling(self.n_weeks*7, min_periods=1).median() # median not mean to avoid outliers
        dmas_mnf_trend.index = pd.to_datetime(dmas_mnf_trend.index)
        # I add already the next week, with the idea that it will be the same as this last week
        dmas_mnf_trend.loc[dmas_mnf_trend.index[-1] + pd.Timedelta(hours=23, days=7)] = dmas_mnf_trend.iloc[-1,:]
        
        # Now I resample to hourly and interpolate
        self.trend = dmas_mnf_trend.resample('H').interpolate(method='linear').ffill().bfill()
        

    def transform(self, X):
        return X/self.trend.loc[X.index,:] # so it goes from 1 to n times the MNF

    def inverse_transform(self, X):
        # When I convert back into the original scale, I need to add the trend but 
        # only for the new week. Since for the new week I don't have a value of the trend
        # I use the last value of the trend
        return X*self.trend.iloc[-WEEK_LEN:, :].mean(axis=0)
