from model import Model
from data_loader import DMAS_NAMES
from constants import WEEK_LEN
import pandas as pd
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pmd

class AutoRegressive(Model):
    def __init__(self, lags=WEEK_LEN) -> None:
        super().__init__()
        self.dmas_models = {}
        self.lags = lags

    def fit(self, X_train: pd.DataFrame) -> None:
        for dma in DMAS_NAMES:
            _model_ = AutoReg(X_train[dma].to_numpy(),
                              lags=self.lags,
                              missing="drop")
            self.dmas_models[dma] = _model_.fit()


    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        pred = np.array([self.dmas_models[dma].forecast(steps=WEEK_LEN) for dma in DMAS_NAMES])
        return pred.T

    def name(self) -> str:
        return f"AutoRegressive-lag_{self.lags}h"

    def forecasted_dmas(self) -> list:
        return DMAS_NAMES

    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame,
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
        # We don't need the exogenous data, discard them.
        # We don't fill the data, it is unnecessary.
        return (train__dmas_h_q, pd.DataFrame())

def impute_nan_(df):

    df['hour'] = df.index.hour
    df[DMAS_NAMES] = df[DMAS_NAMES].apply(lambda x:x.interpolate(limit=1))
    df[DMAS_NAMES] = df.groupby(["hour"])[DMAS_NAMES].transform(lambda x: x.fillna(x.rolling(4).mean()))
    df[DMAS_NAMES] = df.groupby(["hour"])[DMAS_NAMES].transform(lambda x: x.fillna(x.mean()))

    return df[DMAS_NAMES]

class Arima(Model):
    def __init__(self) -> None:
        super().__init__()
        self.dmas_models = {}

    def fit(self, X_train: pd.DataFrame) -> None:
        for dma in DMAS_NAMES:
            _model_ = pmd.auto_arima(X_train[dma].to_numpy(),
                              start_p=0,d=1,start_q=0,
                              test="adf", supress_warnings = True,
                              trace=True)
            self.dmas_models[dma] = _model_

    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        pred = np.array([self.dmas_models[dma].predict(n_periods=WEEK_LEN) for dma in DMAS_NAMES])
        return pred.T

    def name(self) -> str:
        return f"pmd Auto Arima"

    def forecasted_dmas(self) -> list:
        return DMAS_NAMES

    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame,
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
        # We don't need the exogenous data, discard them.
        # We don't fill the data, it is unnecessary.
        return (impute_nan_(train__dmas_h_q), pd.DataFrame())
