from model import Model
from data_loader import DMAS_NAMES

class PreviousWeek(Model):
    def __init__(self) -> None:
        super().__init__()
        self.last_week = None

    def fit(self, X_train):
        self.last_week = X_train.iloc[-24*7:, 0:10].to_numpy()

    def forecast(self, X_test):
        return self.last_week
    
    def name(self):
        return "Previous Week"
    
    def forecasted_dmas(self):
        return DMAS_NAMES
    
    def forecasted_dmas_idx(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def preprocess_data(self, 
                        train__dmas_h_q, test__dmas_h_q,
                        train__exin_h, test__exin_h, 
                        eval__exin_h):
        return (train__dmas_h_q.fillna(0), test__dmas_h_q.fillna(0), 
                train__exin_h.fillna(0), test__exin_h.fillna(0), 
                eval__exin_h.fillna(0))