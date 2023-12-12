import pandas as pd
import numpy as np
from data_loader import DMAS_NAMES

class Model:
    """Base class for all models.
    
    We expect you to derive this class and override all the following methods:
    - fit
    - forecast
    - name
    - preprocess_data
    - forecasted_dmas

    """

    def __init__(self) -> None:
        pass

    def fit(self, X_train: pd.DataFrame) -> None:
        pass

    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        pass

    def name (self) -> str:
        return "Model"
    
    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame, 
                        test__dmas_h_q: pd.DataFrame, 
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame,
                        eval__exin_h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h
    
    def forecasted_dmas(self) -> list[str]:
        pass

    def forecasted_dmas_idx(self) -> list[int]:
        forecasted_dmas = self.forecasted_dmas()
        dmas_names = DMAS_NAMES
        forecasted_dmas_idx = [dmas_names.index(dma) for dma in forecasted_dmas]
        return forecasted_dmas_idx
        