from models.base import Model
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import Prophet as Pf
import holidays

WEEK_LEN = 24 * 7

class Fbprophet(Model):
    def __init__(self):
        self.fb_prophet = {}
        

    def fit(self, demands, weather):


        self.dmas = demands.columns
        for dma in self.dmas:
            model =  Pf(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True,country_holidays="IT",verbose=False)
            data = demands[dma].rename_axis('ds').reset_index(name='y')
            series = TimeSeries.from_dataframe(data, 'ds', 'y')

            self.fb_prophet[dma] = model.fit(series)
        

    def forecast(self, demand_test, weather_test):
        pred= np.array([self.fb_prophet[dma].predict(n=WEEK_LEN).values().T.tolist()[0] for dma in self.dmas ])
        return pred.T
