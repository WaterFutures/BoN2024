"""
This is an evolution of the benchmarks.py.

We realised that a rolling average performs better than the previous week benchmark,
and that previous week, and average week are special cases of the rolling average 
with window size 1 and the whole dataset size respectively.

So, the idea is to have a rolling average that selects automatically the best
window size for the dataset and for each DMA. At the end of the day all other models,
can tune a specific model for each DMA.

To select the best one, we should do this testing all possible window sizes.
However, we know that at the beginning we could expect an improvement, because we are 
removing the stochasticity of the specific week, but as we add more weeks to the
window, at some point it should start perform more or less the same or even worse.
This behaviour is similar to the bias-variance tradeoff, where we are adding more
bias to reduce the variance.
So, if we you use the idea of out-of-sample error to select the best window size, 
as soon as the metrics on the oos data start worsening again after improving, we
can stop the search and select the best window size as the previous one. 

As a metric to test we are using the MAE, so that incorporates both PI1 and PI3 
of the challenge. 

As strategy to validate the models on the metric, we use something that remotely 
resembles cross-validation and leave-one-out:
We will test n random weeks of the dataset, and we will compute the MAE on them. 
There is no problem here to use the same week both in train and testing.

Moreover, the filling of the rolling average was terrible because when nans were
present, we were setting them to 0. Even putting the average consumption of the 
DMA is better. The average of the DMA at that hour of the week is even better.
This filler functions have been written in the preprocessing folder.
"""

import models
from models.base import Model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

WEEK_LEN = 24 * 7

class RollingAverageWeek(Model):

    def __init__(self, window_size=4):
        self.window_size = window_size
        self.dmas_models = {}

    def fit(self, demands, weather):
        n_weeks = demands.shape[0] // WEEK_LEN
        n_dmas = demands.shape[1]
        
        ws = {}
        if self.window_size is None:
            for dma in demands.columns:
                ws[dma] = n_weeks
        elif isinstance(self.window_size, int):
            for dma in demands.columns:
                ws[dma] = self.window_size
        elif isinstance(self.window_size, list) and len(self.window_size) == n_dmas:
            for i,dma in enumerate(demands.columns):
                ws[dma] = self.window_size[i]
        elif isinstance(self.window_size, dict):
            ws = self.window_size
        else:
            raise ValueError("window_size must be ...")

        for dma in demands.columns:
            ws__ = ws[dma]
            dma_h_q = demands[dma].to_numpy().reshape((n_weeks, WEEK_LEN))
            avg_week = np.nanmean(dma_h_q[-ws__:], axis=0)
            
            self.dmas_models[dma] = avg_week
            
        return self

    def forecast(self, demand_test, weather_test):
        return np.concatenate(
            [self.dmas_models[dma] for dma in self.dmas_models.keys()],
            axis=0).reshape((len(self.dmas_models.keys()), WEEK_LEN)).T
    
# The Auto Rolling Average week, given a max number of weeks, trys with a kind of
# cross-validation/leave one-out to find the best window size for the rolling average.
    
def auto_rolling_average_week(demands, max_n_weeks=10, tolerance=0.01, sample_size=10, random_state=None):
    n_weeks = demands.shape[0] // WEEK_LEN
    n_dmas = demands.shape[1]
    
    dmas_h_q = demands.to_numpy().reshape((n_weeks, WEEK_LEN, n_dmas))

    # I will return a dictorionary with a value for each dma
    best_ws = {}

    # I need to tackle each dma separately as they may want different window sizes
    for i,dma in enumerate(demands.columns):
        # on y I have the number of the week I want to forecast
        # on x I have the number of the weeks available to forecast y with this window size
        # for each y the x available is all the previous weeks from y-1 to 0

        # select sample_size weeks at random, I go from max_n_weeks to n_weeks
        # so that independent of the window size I test all the models on the same weeks
        if random_state is not None:
            np.random.seed(random_state)
        y_test = np.random.choice(range(max_n_weeks, n_weeks, 1), size=min(sample_size,n_weeks-max_n_weeks-1), replace=False)
        
        curr_best_mae = np.inf

        for ws in range(1,max_n_weeks+1):
        
            mae = np.zeros((sample_size,))
            for j,tw in enumerate(y_test):
                # I create a rolling average with the current window size
                tw_true = dmas_h_q[tw,:,i]
                tw_pred = RollingAverageWeek(ws).fit(pd.DataFrame(dmas_h_q[np.arange(tw),:,i].reshape((-1,)), columns=[dma]), None).forecast(None, None).reshape((-1,))

                mae[j] = np.nanmean(np.abs(tw_true - tw_pred), axis=0) 

            if np.nanmean(mae) < curr_best_mae-tolerance:
                curr_best_mae = np.nanmean(mae)
                best_ws[dma] = ws
            else:
                break # break the ws loop
        
    return best_ws 

class AutoRollingAverageWeek(Model):
    
        def __init__(self, max_n_weeks=10, tolerance=0.01, sample_size=30, random_state=42):
            self.max_n_weeks = max_n_weeks
            self.tolerance = tolerance
            self.sample_size = sample_size
            self.random_state = random_state
            self.best_wss = {}
            self.RollAW = None
    
        def fit(self, demands, weather):
            self.best_wss = auto_rolling_average_week(demands, self.max_n_weeks, self.tolerance, self.sample_size, self.random_state)
            self.RollAW = RollingAverageWeek(self.best_wss).fit(demands, weather)
            return self
    
        def forecast(self, demand_test, weather_test):
            return self.RollAW.forecast(demand_test, weather_test)