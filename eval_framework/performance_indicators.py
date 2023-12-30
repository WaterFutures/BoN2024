"""
In this file, we will implement the metrics for the evaluation of the models.

3 metrics are used:
- PI1_w^d MAE: Mean Absolute Error in the first 24 hours of week w for DMA d
- PI2_w^d MaxAE: Maximum Absolute Error in the first 24 hours of week w for DMA d
- PI3_w^d MAE: Mean Absolute Error after the first 24 hours till the end of the 
    week w for DMA d
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, max_error
# TODO: move everything to numpy

def performance_indicator_1(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI1^d MAE: Mean Absolute Error in the first 24 hours of the week for DMA d.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA
    
    return mean_absolute_error(dmas_h_q_true[:24], dmas_h_q_pred[:24], multioutput='raw_values')

performance_indicator_1_short_name = 'PI1'
performance_indicator_1_long_name = 'Mean Absolute Error in the first 24 hours of the week for a given DMA.'
performance_indicator_1_label = 'MAE [L/s]'

def performance_indicator_2(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI2^d MaxAE: Max Absolute Error in the first 24 hours of the week for DMA d.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA
    
    return np.amax(np.abs(dmas_h_q_true[:24] - dmas_h_q_pred[:24]), axis=0)

performance_indicator_2_short_name = 'PI2'
performance_indicator_2_long_name = 'Max Absolute Error in the first 24 hours of the week for a given DMA.'
performance_indicator_2_label = 'MaxAE [L/s]'

def performance_indicator_3(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI3^d MAE: Mean Absolute Error after the first 24 hours to the end of the week for DMA d.
    
    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA

    return mean_absolute_error(dmas_h_q_true[24:24*7], dmas_h_q_pred[24:24*7], multioutput='raw_values')

performance_indicator_3_short_name = 'PI3'
performance_indicator_3_long_name = 'Mean Absolute Error after the first 24 hours to the end of the week for a given DMA.'
performance_indicator_3_label = 'MAE [L/s]'

def performance_indicators(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> dict[str, np.ndarray] :
    """
    Test the model on a single week.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0      # at least one DMA

    # We decided to not include the nans in the counts 

    # Find the nans in the true data, then put thme to 0 both in the true and 
    # predicted data
    nans_true = np.isnan(dmas_h_q_true)
    dmas_h_q_true[nans_true] = 0
    dmas_h_q_pred[nans_true] = 0

    pi1 = performance_indicator_1(dmas_h_q_true, dmas_h_q_pred)
    pi2 = performance_indicator_2(dmas_h_q_true, dmas_h_q_pred)
    pi3 = performance_indicator_3(dmas_h_q_true, dmas_h_q_pred)

    return {'PI1':pi1, 'PI2':pi2, 'PI3':pi3,
            'n_nans_1d': nans_true[:24].sum(axis=0),
            'n_nans_w': nans_true.sum(axis=0)}

performance_indicators_short_names = [performance_indicator_1_short_name,
                                        performance_indicator_2_short_name,
                                        performance_indicator_3_short_name,
                                        'n_nans_1d', 'n_nans_w']
PIs = performance_indicators_short_names
performance_indicators_long_names = { performance_indicators_short_names[0]: performance_indicator_1_long_name,
                                        performance_indicators_short_names[1]: performance_indicator_2_long_name,
                                        performance_indicators_short_names[2]: performance_indicator_3_long_name,
                                        performance_indicators_short_names[3]: 'Number of nans in the first 24 hours for a given DMA.',
                                        performance_indicators_short_names[4]: 'Number of nans in the week for a given DMA.'
                                        }
performance_indicators_labels = { performance_indicators_short_names[0]: performance_indicator_1_label,
                                        performance_indicators_short_names[1]: performance_indicator_2_label,
                                        performance_indicators_short_names[2]: performance_indicator_3_label,
                                        performance_indicators_short_names[3]: 'Number of nans [-]',
                                        performance_indicators_short_names[4]: 'Number of nans [-]'
                                        }