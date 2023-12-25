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
import scipy.stats as ss
import pandas as pd
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
    pis = (pi1 + pi2 + pi3) / 3

    return {'PI1':pi1, 'PI2':pi2, 'PI3':pi3, 'avg_PIs':pis,
            'n_nans_1d': nans_true[:24].sum(axis=0),
            'n_nans_w': nans_true.sum(axis=0)}

performance_indicators_short_names = [performance_indicator_1_short_name,
                                        performance_indicator_2_short_name,
                                        performance_indicator_3_short_name,
                                        'avg_PIs',
                                        'n_nans_1d', 'n_nans_w']
PIs = performance_indicators_short_names
performance_indicators_long_names = { performance_indicators_short_names[0]: performance_indicator_1_long_name,
                                        performance_indicators_short_names[1]: performance_indicator_2_long_name,
                                        performance_indicators_short_names[2]: performance_indicator_3_long_name,
                                        performance_indicators_short_names[3]: 'Average of PIs',
                                        performance_indicators_short_names[4]: 'Number of nans in the first 24 hours for a given DMA.',
                                        performance_indicators_short_names[5]: 'Number of nans in the week for a given DMA.'
                                        }
performance_indicators_labels = { performance_indicators_short_names[0]: performance_indicator_1_label,
                                        performance_indicators_short_names[1]: performance_indicator_2_label,
                                        performance_indicators_short_names[2]: performance_indicator_3_label,
                                        performance_indicators_short_names[3]: 'MAE [L/s]',
                                        performance_indicators_short_names[4]: 'Number of nans [-]',
                                        performance_indicators_short_names[5]: 'Number of nans [-]'
                                        }

def create_report(results, model_names, test_weeks, dma_names, pi):

    ranks = []
    ranks_p1 = []
    ranks_p2 = []
    ranks_p3 = []
    ranks_dma_a = []
    ranks_dma_b = []
    ranks_dma_c = []
    ranks_dma_d = []
    ranks_dma_e = []
    ranks_dma_f = []
    ranks_dma_g = []
    ranks_dma_h = []
    ranks_dma_i = []
    ranks_dma_j = []
    for w in test_weeks:
        for dma in dma_names:
            for p in ['PI1','PI2','PI3']:
                temp = []
                temp_p1 = []
                temp_p2 = []
                temp_p3 = []
                temp_dma_a = []
                temp_dma_b = []
                temp_dma_c = []
                temp_dma_d = []
                temp_dma_e = []
                temp_dma_f = []
                temp_dma_g = []
                temp_dma_h = []
                temp_dma_i = []
                temp_dma_j = []
                for i,m in enumerate(model_names):
                    temp.append(results[m]['validation'].loc[(w, dma), p]) # Model m, week w, column j and metric p
                    if p == 'PI1':
                        temp_p1.append(results[m]['validation'].loc[(w, dma), p])
                    elif p == 'PI2':
                        temp_p2.append(results[m]['validation'].loc[(w, dma), p])
                    else:
                        temp_p3.append(results[m]['validation'].loc[(w, dma), p])

                    if dma == 'DMA_A':
                        temp_dma_a.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_B':
                        temp_dma_b.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_C':
                        temp_dma_c.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_D':
                        temp_dma_d.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_E':
                        temp_dma_e.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_F':
                        temp_dma_f.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_G':
                        temp_dma_g.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_H':
                        temp_dma_h.append(results[m]['validation'].loc[(w, dma), p])
                    elif dma == 'DMA_I':
                        temp_dma_i.append(results[m]['validation'].loc[(w, dma), p])
                    else:
                        temp_dma_j.append(results[m]['validation'].loc[(w, dma), p])

                ranks.append(ss.rankdata(temp))
                ranks_p1.append(ss.rankdata(temp_p1))
                ranks_p2.append(ss.rankdata(temp_p2))
                ranks_p3.append(ss.rankdata(temp_p3))
                ranks_dma_a.append(ss.rankdata(temp_dma_a))
                ranks_dma_b.append(ss.rankdata(temp_dma_b))
                ranks_dma_c.append(ss.rankdata(temp_dma_c))
                ranks_dma_d.append(ss.rankdata(temp_dma_d))
                ranks_dma_e.append(ss.rankdata(temp_dma_e))
                ranks_dma_f.append(ss.rankdata(temp_dma_f))
                ranks_dma_g.append(ss.rankdata(temp_dma_g))
                ranks_dma_h.append(ss.rankdata(temp_dma_h))
                ranks_dma_i.append(ss.rankdata(temp_dma_i))
                ranks_dma_j.append(ss.rankdata(temp_dma_j))

    ranks = [arr for arr in ranks if arr.size > 0]
    ranks_p1 = [arr for arr in ranks_p1 if arr.size > 0]
    ranks_p2 = [arr for arr in ranks_p2 if arr.size > 0]
    ranks_p3 = [arr for arr in ranks_p3 if arr.size > 0]
    ranks_dma_a = [arr for arr in ranks_dma_a if arr.size > 0]
    ranks_dma_b = [arr for arr in ranks_dma_b if arr.size > 0]
    ranks_dma_c = [arr for arr in ranks_dma_c if arr.size > 0]
    ranks_dma_d = [arr for arr in ranks_dma_d if arr.size > 0]
    ranks_dma_e = [arr for arr in ranks_dma_e if arr.size > 0]
    ranks_dma_f = [arr for arr in ranks_dma_f if arr.size > 0]
    ranks_dma_g = [arr for arr in ranks_dma_g if arr.size > 0]
    ranks_dma_h = [arr for arr in ranks_dma_h if arr.size > 0]
    ranks_dma_i = [arr for arr in ranks_dma_i if arr.size > 0]
    ranks_dma_j = [arr for arr in ranks_dma_j if arr.size > 0]

    all_ranks = np.vstack((model_names,np.round(np.array(ranks_p1).mean(axis=0), 2),
                          np.round(np.array(ranks_p2).mean(axis=0), 2), np.round(np.array(ranks_p3).mean(axis=0), 2),
                          np.round(np.array(ranks_dma_a).mean(axis=0), 2),np.round(np.array(ranks_dma_b).mean(axis=0), 2),
                          np.round(np.array(ranks_dma_c).mean(axis=0), 2),np.round(np.array(ranks_dma_d).mean(axis=0), 2),
                          np.round(np.array(ranks_dma_e).mean(axis=0), 2),np.round(np.array(ranks_dma_f).mean(axis=0), 2),
                          np.round(np.array(ranks_dma_g).mean(axis=0), 2),np.round(np.array(ranks_dma_h).mean(axis=0), 2),
                          np.round(np.array(ranks_dma_i).mean(axis=0), 2),np.round(np.array(ranks_dma_j).mean(axis=0), 2),
                          np.round(np.array(ranks).mean(axis=0), 2)))

    avg_statistics = pd.DataFrame(index=model_names, columns=dma_names)
    std_statistics = pd.DataFrame(index=model_names, columns=dma_names)
    pct_statistics = pd.DataFrame(index=model_names, columns=dma_names)

    for m in model_names:
        g = results[m]['validation'].index.get_level_values('DMA')

        avg_statistics.loc[m,:] = results[m]['validation'].groupby(g).mean()[pi].T
        std_statistics.loc[m,:] = results[m]['validation'].groupby(g).std()[pi].T

    for dma in dma_names:
        res_for_pct = pd.DataFrame(index=test_weeks, columns=model_names)
        for m in model_names:
            for w in test_weeks:
                res_for_pct.loc[w,m] = results[m]['validation'].loc[(w, dma), pi]
        t = res_for_pct.apply(lambda x: x.eq(x.min()), axis=1).sum(axis=0)
        pct_statistics.loc[:,dma] = t / len(test_weeks)
    '''for m in model_names:
        t = pd.DataFrame(results).apply(lambda x: x.eq(x.min()), axis=1).sum()
        pct_statistics.loc[m,dma] = round(100 * t[m] / len(test_weeks), 2)'''

    avg_statistics['Average'] = avg_statistics.iloc[:,:-1].mean(axis=1)
    std_statistics['Average'] = std_statistics.iloc[:,:-1].mean(axis=1)
    pct_statistics['Average'] = pct_statistics.iloc[:,:-1].mean(axis=1)

    return (avg_statistics.reset_index().rename(columns={'index':'Models'}),
            std_statistics.reset_index().rename(columns={'index':'Models'}),
            pct_statistics.reset_index().rename(columns={'index':'Models'}),
            pd.DataFrame(index=model_names, data=np.transpose(all_ranks)).rename(columns={0:'Models',
                                                                                    1:'Rank_P1',
                                                                                    2:'Rank_P2',
                                                                                    3:'Rank_P3',
                                                                                    4:'Rank_A',
                                                                                    5:'Rank_B',
                                                                                    6:'Rank_C',
                                                                                    7:'Rank_D',
                                                                                    8:'Rank_E',
                                                                                    9:'Rank_F',
                                                                                    10:'Rank_G',
                                                                                    11:'Rank_H',
                                                                                    12:'Rank_I',
                                                                                    13:'Rank_J',
                                                                                    14:'Average'})
                                                                                    )
