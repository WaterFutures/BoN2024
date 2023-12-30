import pandas as pd
from .performance_indicators import performance_indicators_short_names as PIs

def ModelResults() -> dict:
    """
    Return a dictionary with all the necessary information to describe the whole 
    process to evaluate a model.

    Processed data used for training
        train and test, for each type of variable (dma consumption and exogenous), 
        eval, for exogenous variables always known at time t (e.g., weather)
    Model
    Validation results pandas dataframe with multiindex (test week, dma) and columns
        the performance indicators
    Test results pandas dataframe with multiindex (test week, dma) and columns
        the performance indicators
    BWDF forecast results pandas dataframe with index the week we need to forecast 
        and columns the DMAs
    
    :return: A dictionary with the structure of the results of a model.
    """
    return {
        "processed_data": {
            "train__dmas_h_q": None, 
            "test__dmas_h_q": None,
            "train__exin_h": None,
            "test__exin_h": None,
            "eval__exin_h": None
        },
        "model": None,
        "validation": None,
        "test": None,
        "bwdf_forecast": None
    }

def ProcessResults(eval_weeks: range, dmas: list[str]) -> pd.DataFrame:
    """
    Return a dataframe with the structure of the results of the evaluation over
    the trainig/validation or test set.
    It differs from ModelResults as this only keeps the scores for each week 
    and dma

    :param eval_weeks: The weeks to evaluate.
    :param dmas: The DMAs to evaluate.
    :return: A Pandas dataframe with a two level index (test week level 0; DMA
     level 1) and columns the perfromances indicators.
    """

    return pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(ew,dma) for ew in eval_weeks for dma in dmas], names=['Test week', 'DMA']),
            columns=PIs
        )