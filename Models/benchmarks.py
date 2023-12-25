from model import Model
from data_loader import DMAS_NAMES
from constants import WEEK_LEN
import pandas as pd
import numpy as np

class PreviousWeek(Model):
    def __init__(self) -> None:
        super().__init__()
        self.last_week = None

    def fit(self, X_train: pd.DataFrame) -> None:
        self.last_week = X_train.iloc[-24*7:, self.forecasted_dmas_idx()].to_numpy()

    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(self.last_week) #fill with 0s because I can't return a solution with NaNs

    def name(self) -> str:
        return "Previous Week"

    def forecasted_dmas(self) -> list:
        return DMAS_NAMES

    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame,
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
        # We don't need the exogenous data, discard them.
        # We don't fill the data, it is unnecessary.
        return (train__dmas_h_q, pd.DataFrame())


class AverageWeek(Model):
    def __init__(self) -> None:
        super().__init__()
        self.average_week = None

    def fit(self, X_train: pd.DataFrame) -> None:
        all_values = X_train.iloc[:, self.forecasted_dmas_idx()].to_numpy()
        # average at each hour, for every day of the week
        self.average_week = np.nanmean(all_values.reshape((-1,WEEK_LEN,all_values.shape[1])),
                                       axis=0)

    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.average_week

    def name(self) -> str:
        return "Average Week"

    def forecasted_dmas(self) -> list:
        return DMAS_NAMES

    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame,
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
        # We don't need the exogenous data, discard them.
        # We don't fill the data, it is unnecessary.
        return (train__dmas_h_q, pd.DataFrame())
