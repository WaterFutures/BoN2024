from preprocessing.base import Preprocessing
import numpy as np
from Utils.data_loader import load_calendar


class DayOfTheWeek(Preprocessing):
    def transform(self, X):
        X['DayOfTheWeek'] = X.index.weekday.to_numpy()
        return X

class HourOfTheDay(Preprocessing):
    def transform(self, X):
        X['HourOfTheDay'] = X.index.hour.to_numpy()
        print(X.head())
        return X

class DaysOff(Preprocessing):
    def transform(self, X):
        cal = load_calendar()
        weekend = cal['Weekend'].to_numpy() == 1
        public_holidays = cal['Holiday'].to_numpy() == 1
        non_business_days = np.array(np.logical_or(weekend, public_holidays), dtype=int)
        X['DaysOff'] = np.repeat(non_business_days[0:int(len(X)/24)], 24)
        return X
