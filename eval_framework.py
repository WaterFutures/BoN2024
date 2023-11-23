"""
In this file, we will implement the evaluation framework for the Battle of Water
Demand Forecasting project.

3 metrics are used:
- PI1_w^d MAE: Mean Absolute Error in the first 24 hours of week w for DMA d
- PI2_w^d MaxAE: Maximum Absolute Error in the first 24 hours of week w for DMA d
- PI3_w^d MAE: Mean Absolute Error after the first 24 hours till the end of the 
    week w for DMA d

In the evaluation framework, for each model we will have a matrix of values for 
metric and DMA. 

Simply measuring the metrics is not enough, we need to see if they are better 
than a benchmark model (e.g. ARIMA, 
previous week, same week year before)

Moreover, we can think of testing our models on several weeks (e.g., the final 
ones, random weeks, same week year before). 

I would suggest to produce a matrix of results for each model with 
Px_tw^d as the score x of test week tw for DMA d (see above)

This will give us a 3 dimensional matrix (metric, test week, DMA) for each model
 and for each benchmark.

Then we do a comparison between the benchmarks models, choose a benchmark for each 
test week, DMA and score for example removing the not meaningful combinations 
(e.g., same week year before to forecast the test week)

Finally, we can compare the models with the benchmark and see if they are better
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, max_error
import data_loader
import pandas as pd

def performance_indicator_1(dmas_h_q_true, dmas_h_q_pred):
    """
    PI1^d MAE: Mean Absolute Error in the first 24 hours of the week for DMA d.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA
    
    return mean_absolute_error(dmas_h_q_true.head(24), dmas_h_q_pred.head(24), multioutput='raw_values')

def performance_indicator_2(dmas_h_q_true, dmas_h_q_pred):
    """
    PI2^d MaxAE: Max Absolute Error in the first 24 hours of the week for DMA d.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA

    pi2 = np.zeros(dmas_h_q_true.shape[1])
    for i in range(dmas_h_q_true.shape[1]):
        pi2[i] = max_error(dmas_h_q_true.iloc[0:24,i], dmas_h_q_pred.iloc[0:24,i] )
    
    return pi2

def performance_indicator_3(dmas_h_q_true, dmas_h_q_pred):
    """
    PI3^d MAE: Mean Absolute Error after the first 24 hours to the end of the week for DMA d.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA

    return mean_absolute_error(dmas_h_q_true.iloc[24:24*7,:], dmas_h_q_pred.iloc[24:24*7,:], multioutput='raw_values')

def performance_indicators(dmas_h_q_true, dmas_h_q_pred):
    """
    Test the model on a single week.

    :param dmas_h_q_pred: The forecasted data for the week to test.
    :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] == 10     # 10 DMAs

    pi1 = performance_indicator_1(dmas_h_q_true, dmas_h_q_pred)
    pi2 = performance_indicator_2(dmas_h_q_true, dmas_h_q_pred)
    pi3 = performance_indicator_3(dmas_h_q_true, dmas_h_q_pred)

    return pd.DataFrame({'PI1':pi1, 'PI2':pi2, 'PI3':pi3}, index=dmas_h_q_true.columns)

def test_model(dmas_h_q_true, dmas_h_q_pred):
    """
    Test the model on all the weeks in the test set and store the results.

    :param dmas_h_q_pred: The forecasted data for the week to test.
    :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[1] == 10     # 10 DMAs

    # Get the week numbers 
    n_test_weeks = int(len(dmas_h_q_true.index)/24/7)
    test_weeks_idxs = np.zeros(n_test_weeks, dtype=int)
    for i in range(0, n_test_weeks):
        test_weeks_idxs[i] = data_loader.dataset_week_number(dmas_h_q_true.index[i*24*7])

    result = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(tw,dma) for tw in test_weeks_idxs for dma in dmas_h_q_true.columns], names=['Test week', 'DMA']),
        columns=['PI1', 'PI2', 'PI3']
    )
    
    for tw in range(0, n_test_weeks):
        tw_dmas_h_q_true = dmas_h_q_true.iloc[tw*24*7:(tw+1)*24*7,:]
        tw_dmas_h_q_pred = dmas_h_q_pred.loc[tw_dmas_h_q_true.index,:] # Just in case they don't have the same order
        result.loc[test_weeks_idxs[tw],:] = performance_indicators(tw_dmas_h_q_true, tw_dmas_h_q_pred).values

    return result

class EvaluationLogger:
    # No attributes for now 

    def __init__(self, a_test_dmas_h_q):
        self.m_test_dmas_h_q = a_test_dmas_h_q
        self.m_results = {}
    
    def add_model_test(self, model_name, dmas_h_q_pred):
        """
        Test the model on all the weeks in the test set and store the results.

        :param dmas_h_q_pred: The forecasted data for the week to test.
        :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
        """
        self.m_results[model_name] = test_model(self.m_test_dmas_h_q, dmas_h_q_pred)
