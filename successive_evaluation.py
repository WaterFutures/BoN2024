"""
Successive evaluation following the test-then-train idea frequently used in online learning in the presence of distributional changes (concept drift).
Using the last n weeks for the evaluation of all models will result in severe overfitting on this data.
Thus, I think we should put apart the last n(=4) weeks for testing and deciding on the final model for submission.
For designing and experimenting with different models, I would go for this option training on the data until week t-1, and testing on week t until we reach the last n weeks which we keep aside.
Thereby, we have enough data for development but keep an independent test set aside.
"""

from eval_framework import performance_indicators
import data_loader as dl
import pandas as pd
import numpy as np
from models import autoregressive_model, dummy_prev_week, dummy_mean_week, arima

day_len = 24
week_len = 7*day_len

def evaluate_test_then_train_scheme(model, f_out, first_test_week=4):
    """
    Test a model in the test-then-train scheme, i.e. training on weeks 0 to first_test_week-1 testing on first_test_week in the first run, and then including more data for each run, 
    i.e. training on weeks 0 to first_test_week-1+i testing on first_test_week+i

    Parameters
    ----------
    model : model implementing Model described in model.py
        Initialized model to be evaluated, needs to implement fit(self, X_train) function for fitting the model and forecast(self, X_test) function returnin the forecast for the test week.
    f_out : String
        File for the results of each individual test run to be stored (test week id will be appended as index).
    first_test_week : int, optional
        First week to be considered for testing. The default is 4 - I start counting by 0.

    """

    train__dmas_h_q, test__dmas_h_q, train__wea_h, test__wea_h = dl.load_splitted_data(split_strategy="final_weeks", split_size_w=10, week_selection=0, start_first_monday=True)

    # run over all train test splits
    for test_idx in range(first_test_week,int(train__dmas_h_q.shape[0]/week_len)):
        forecasts = []
        # loop over all demands
        for dma_id in range(10):
            # aggregate trainin data
            X_train = train__dmas_h_q.iloc[:test_idx*week_len,dma_id].fillna(0)

            # fit model
            model.fit(X_train)

            # obtain forecast
            forecast = model.forecast(X_train)
            forecasts.append(forecast)
        
        # accumulate forecasts for all DMAs as dataframe to be compatible with eval_framework
        forecasts = pd.DataFrame(np.vstack(forecasts).T)
        res = performance_indicators(train__dmas_h_q.iloc[test_idx*week_len:(test_idx+1)*week_len,:].fillna(0), forecasts)
        res.to_pickle('{}_{}.pkl'.format(f_out, test_idx)) 





evaluate_test_then_train_scheme(dummy_mean_week(), 'results/dummy_mean_week', first_test_week=4)
evaluate_test_then_train_scheme(dummy_prev_week(), 'results/dummy_prev_week', first_test_week=4)
evaluate_test_then_train_scheme(arima(), 'results/arima', first_test_week=4)
evaluate_test_then_train_scheme(autoregressive_model(), 'results/autoregresssive', first_test_week=4)
evaluate_test_then_train_scheme(autoregressive_model(lags=24*14), 'results/autoregresssive-14', first_test_week=8)
