import numpy as np
import pandas as pd

from eval.data_loading_helpers import DMAS_NAMES
WEEK_LEN = 24*7

def stack_seeds(dict_of_seeds, sublevel):
    
    df_list = []
    for seed in dict_of_seeds:
        df_list.append(dict_of_seeds[seed][sublevel])

    return pd.concat(df_list, 
                     keys=range(len(df_list)),
                     names=['Seed']
                    )

def extract_from(result, iter_n=1, phase='train', sublevel='forecasts'):

    return stack_seeds(result['iter_'+str(iter_n)][phase], sublevel)