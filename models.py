from model import Model
from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pmd
import numpy as np

class Model:
    # No attributes for now 

    def __init__(self):
        pass
    
    def fit(self, X_train):
        pass

    def forecast(self, X_test):
        pass


class autoregressive_model(Model):
    def __init__(self, lags=24*7) -> None:
        super().__init__()
        self.model_fit = None
        self.lags = lags

    def fit(self, X_train):
        model = AutoReg(X_train, lags=self.lags)
        self.model_fit = model.fit()

    def forecast(self, X_test):
        return self.model_fit.forecast(steps=24*7)
    

class arima(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.lags = lags

    def fit(self, X_train):
        self.model = pmd.auto_arima(X_train, 
                              start_p=0,d = 1,start_q=0,
                              test="adf", supress_warnings = True,
                              trace=True)

    def forecast(self, X_test):
        return self.model.predict(n_periods=24*7)
    

class dummy_prev_week(Model):
    def __init__(self) -> None:
        super().__init__()
        self.last_week = None

    def fit(self, X_train):
        self.last_week = X_train[-24*7:]

    def forecast(self, X_test):
        return self.last_week
    

class dummy_mean_week(Model):
    def __init__(self) -> None:
        super().__init__()
        self.mean_week = None

    def fit(self, X_train):
        self.mean_week = np.mean(np.array(X_train).reshape(-1, 24*7), axis=0)

    def forecast(self, X_test):
        return self.mean_week
    

