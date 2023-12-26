from models.model import Model
import pandas as pd
import numpy as np

WEEK_LEN = 24 * 7

from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pmd

class AutoRegressive(Model):
    def __init__(self, lags=WEEK_LEN):
        self.dmas_models = {}
        self.lags = lags

    def fit(self, demands, weather):
        self.dmas = demands.columns
        for dma in self.dmas:
            model = AutoReg(demands[dma].to_numpy(), 
                              lags=self.lags, 
                              missing="drop")
            self.dmas_models[dma] = model.fit()
        

    def forecast(self, weather):
        pred = np.array([self.dmas_models[dma].forecast(steps=WEEK_LEN) for dma in self.dmas])
        return pred.T
    

autoreg_no_preprocess = {
    'name': 'AutoReg',
    'model': AutoRegressive(),
    'preprocessing': {
        'demand': [],
        'weather': []
    }
}


class Arima(Model):
    def __init__(self):
        self.dmas_models = {}

    def fit(self, demands, weather):
        self.dmas = demands.columns
        for dma in self.dmas:
            model = pmd.auto_arima(y=demands[dma].to_numpy(), 
                                   X=weather.to_numpy(),
                                   start_p=0,d = 1,start_q=0,
                                   test="adf", supress_warnings = True,
                                   trace=True)
            self.dmas_models[dma] = model

    def forecast(self, weather):
        pred = np.array([self.dmas_models[dma].predict(n_periods=WEEK_LEN) for dma in self.dmas])
        return pred.T