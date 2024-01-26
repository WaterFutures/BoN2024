import numpy as np
import pandas as pd
import scipy as sp
import os
import pathlib
import pickle
import tqdm

from eval.data_loading_helpers import load_data, DMAS_NAMES, WEEK_LEN

class WaterFuturesEvaluator:

    def __init__(self):
        self.n_iter=4   # Number of iterations of the whole competition
        self.curr_it=0  # Current iteration
        self.demand=None    # Current iteration demand dataframe
        self.weather=None   # Current iteration weather dataframe
        self.n_weeks=0      # Number of weeks in the current iteration demand dataframe
        self.eval_week=self.n_weeks+1 # Current evaluation week (present only in the weather dataframe)
        
        self.n_train_weeks=52 # Parameter to decide how many weeks to use for training (52 so that we don't bias ourselves on any particolar week)
        self.n_test_weeks=4   # Parameter to decide how many weeks to use for testing (4 is basically the month of the evaluation week)
        self.n_train_seeds=5  # Number of seeds to use for models during training
        self.n_test_seeds=10    # Number of seeds to use for models during testing
        self.week_start=0      # week to start the training = n_weeks-n_train_weeks-n_test_weeks
        self.train_weeks=None  # range of weeks to use for training (range(week_start, week_start+n_train_weeks))
        self.test_weeks=None   # range of weeks to use for testing (range(n_weeks-n_test_weeks, n_weeks))
        
        self.curr_phase = None  # Phase of the competition (train or test)

        self.results_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'data', 'results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        """
        # Load data, omitting the last 4 weeks
        demand, weather = load_data()
        self.demand = demand.iloc[:-WEEK_LEN*4]
        self.weather = weather.iloc[:-WEEK_LEN*4]

        self.week_start = 12 #12
        self.total_weeks = self.demand.shape[0] // (WEEK_LEN)

        self.results = {}

        self.results_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'wfe_results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.load_saved_results()
        """

    def next_iter(self):
        self.curr_it = min(self.curr_it+1, self.n_iter) 

        self.demand, self.weather = load_data(self.curr_it)
        self.n_weeks = self.demand.shape[0] // (WEEK_LEN)
        self.eval_week = self.n_weeks+1

        self.week_start = self.n_weeks-self.n_train_weeks-self.n_test_weeks
        self.train_weeks = range(self.week_start, self.week_start+self.n_train_weeks)
        self.test_weeks = range(self.n_weeks-self.n_test_weeks, self.n_weeks)


    def load_saved_results(self):
        files = os.listdir(self.results_folder)
        for cur_file in files:
            cur_file_path = os.path.join(self.results_folder, cur_file)
            cur_model_name = '.'.join(cur_file.split('.')[:-1])
            with open(cur_file_path, 'rb') as f:
                self.results[cur_model_name] = pd.compat.pickle_compat.load(f)

    def add_model(self, config, force=False):
        # Check force condition and skip computation if desired
        if (not force) and (config['name'] in self.results.keys()):
            return

        # Evaluate model
        performance_indicators, forecast = self.eval_model(config)
        self.results[config['name']] = {
            'performance_indicators': performance_indicators,
            'forecast': forecast
        }

        # Save results to disk
        cur_file_path = os.path.join(self.results_folder, f'{config["name"]}.pkl')
        with open(cur_file_path, 'wb') as f:
            pickle.dump(self.results[config["name"]], f)


    def eval_model(self, config):
        test_week_idcs = range(self.week_start, self.total_weeks)

        results = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(week,dma) for week in test_week_idcs for dma in DMAS_NAMES], names=['Test week', 'DMA']),
            columns=['PI1', 'PI2', 'PI3'],
            dtype=float
        )

        forecast = self.demand.copy()
        forecast.iloc[:] = pd.NA

        for test_week_idx in tqdm.tqdm(test_week_idcs):
            # Load current train and test data
            demand_train = self.demand.iloc[:WEEK_LEN*test_week_idx]
            weather_train = self.weather.iloc[:WEEK_LEN*test_week_idx]
            ground_truth = self.demand.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)]
            weather_test = self.weather.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)]

            # If applicable, prepare test dataframes
            if 'prepare_test_dfs' in config['preprocessing']:
                for preprocessing_step in config['preprocessing']['prepare_test_dfs']:
                    demand_test, weather_test = preprocessing_step.transform(demand_train, weather_train, weather_test)

            # Apply preprocessing for demands
            for preprocessing_step in config['preprocessing']['demand']:
                demand_train = preprocessing_step.fit_transform(demand_train)

            # Apply preprocessing for weather
            for preprocessing_step in config['preprocessing']['weather']:
                weather_train = preprocessing_step.fit_transform(weather_train)

            # Train model
            config['model'].fit(demand_train, weather_train)

            # If applicable, prepare test dataframes
            if 'prepare_test_dfs' in config['preprocessing']:
                for preprocessing_step in config['preprocessing']['demand']:
                    demand_test = preprocessing_step.transform(demand_test)

                for preprocessing_step in config['preprocessing']['weather']:
                    weather_test = preprocessing_step.transform(weather_test)

                # demand_test = demand_test.iloc[-WEEK_LEN:,:]
                # weather_test = weather_test.iloc[-WEEK_LEN:,:]
            else:
                # Prepare test weather anyways
                for preprocessing_step in config['preprocessing']['weather']:
                    weather_test = preprocessing_step.transform(weather_test)

                demand_test = None

            # Forecast next week
            demand_forecast = config['model'].forecast(demand_test, weather_test)
            demand_forecast = pd.DataFrame(demand_forecast, index=ground_truth.index, columns=ground_truth.columns)

            # Transform forecast back into original unit
            for preprocessing_step in reversed(config['preprocessing']['demand']):
                demand_forecast = preprocessing_step.inverse_transform(demand_forecast)


            # Save forecast and calculate Performance indicators
            forecast.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)] = demand_forecast
            results.loc[test_week_idx] = performance_indicators(demand_forecast, ground_truth)

        return results, forecast
    
    def ranks_report(self, model_names):
        # The .loc[:13] makes this compatible with the ensembled files
        df_shape = self.results[model_names[0]]['performance_indicators'].loc[13:]
        shape = (*df_shape.index.levshape, df_shape.shape[1])
        # Collect all performance indicators in shape (model, week, dma, PI)
        performance_indicators = np.array([self.results[model_name]['performance_indicators'].loc[13:].to_numpy().reshape(shape) for model_name in model_names])

        # Calculate ranks
        ranks = sp.stats.rankdata(performance_indicators, axis=0)
        ranks_pis = np.nanmean(ranks, axis=(1, 2))
        ranks_dmas = np.nanmean(ranks, axis=(1, 3))
        ranks_average = np.nanmean(ranks, axis=(1, 2, 3))[:,np.newaxis]

        # Return rank report dataframe
        return pd.DataFrame(np.concatenate((ranks_pis, ranks_dmas, ranks_average), axis=1), 
                            index=model_names, 
                            columns=['PI1', 'PI2', 'PI3', *[f'Rank_{x}' for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']], 'Average'])




### Performance Indicators
def performance_indicators(y_pred, y_true):
    assert not np.any(np.isnan(y_pred)), 'Model forecasted NaN values'
    return np.vstack((pi1(y_pred, y_true), pi2(y_pred, y_true), pi3(y_pred, y_true))).T

def pi1(y_pred, y_true):
    return np.nanmean(np.abs(y_pred[:24] - y_true[:24]), axis=0)

def pi2(y_pred, y_true):
    return np.nanmax(np.abs(y_pred[:24] - y_true[:24]), axis=0)

def pi3(y_pred, y_true):
    return np.nanmean(np.abs(y_pred[24:] - y_true[24:]), axis=0)
