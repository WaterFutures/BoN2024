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
        self.eval_week=self.n_weeks # Current evaluation week (present only in the weather dataframe) # No plus 1 as we start already couinting from 0
        
        self.n_train_weeks=52 # Parameter to decide how many weeks to use for training (52 so that we don't bias ourselves on any particolar week)
        self.n_test_weeks=4   # Parameter to decide how many weeks to use for testing (4 is basically the month of the evaluation week)
        self.n_train_seeds=5  # Number of seeds to use for models during training
        self.n_test_seeds=10    # Number of seeds to use for models during testing
        self.week_start=0      # week to start the training = n_weeks-n_train_weeks-n_test_weeks
        self.train_weeks=None  # range of weeks to use for training (range(week_start, week_start+n_train_weeks))
        self.test_weeks=None   # range of weeks to use for testing (range(n_weeks-n_test_weeks, n_weeks))
        
        self.curr_phase = None  # Phase of the competition (train, test or eval)

        self.configs = {} # Dictionary of configs for each model

        data_folder = os.getenv('BON2024_DATA_FOLDER')
        if data_folder is None:
            data_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'data')
        self.results_folder = os.path.join(data_folder, 'results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
       
        self.results = {}
        self.load_saved_results()

        self.strategies = {}
        self.resstrategies = {}
        self.load_saved_strategies()

        self.selected_models = []
        self.selected_strategy = None

    def next_iter(self):
        self.curr_it = min(self.curr_it+1, self.n_iter) 

        self.demand, self.weather = load_data(self.curr_it)
        self.n_weeks = self.demand.shape[0] // (WEEK_LEN)
        self.eval_week = self.n_weeks # this is used as index so it starts from 0

        self.week_start = self.n_weeks-self.n_train_weeks-self.n_test_weeks
        self.train_weeks = range(self.week_start, self.week_start+self.n_train_weeks)
        self.test_weeks = range(self.n_weeks-self.n_test_weeks, self.n_weeks)


    def load_saved_results(self):
        if not os.path.exists(os.path.join(self.results_folder,'models')):
            return
        models = os.listdir(os.path.join(self.results_folder,'models'))
        models = [m for m in models if os.path.isdir(os.path.join(self.results_folder, 'models', m)) ]
        for mode_dir in models:
        
            iters = os.listdir(os.path.join(self.results_folder,'models',mode_dir))
            iters = [m for m in iters if os.path.isdir(os.path.join(self.results_folder, 'models', mode_dir, m)) ]
            for iter_dir in iters:

                phases = os.listdir(os.path.join(self.results_folder,'models',mode_dir,iter_dir))
                phases = [m for m in phases if os.path.isdir(os.path.join(self.results_folder, 'models', mode_dir, iter_dir, m)) ]
                for phase_dir in phases:

                    files = os.listdir(os.path.join(self.results_folder,'models',mode_dir,iter_dir,phase_dir))
                    files = [fi for fi in files if os.path.isfile(os.path.join(self.results_folder, 'models', mode_dir, iter_dir, phase_dir, fi)) and fi.split('.')[-1] == 'pkl']
                    for cur_file in files:

                        cur_file_path = os.path.join(self.results_folder,'models',mode_dir,iter_dir,phase_dir,cur_file)
                        cur_file_name = cur_file.split('.')[0]

                        cur_model_name = cur_file_name.split('__')[0]
                        iter = cur_file_name.split('__')[1]
                        phase = cur_file_name.split('__')[2]
                        seed = cur_file_name.split('__')[3]

                        if cur_model_name not in self.results.keys():
                            self.results[cur_model_name] = {}

                        if iter not in self.results[cur_model_name].keys():
                            self.results[cur_model_name][iter] = {}

                        if phase not in self.results[cur_model_name][iter].keys():
                            self.results[cur_model_name][iter][phase] = {}

                        with open(cur_file_path, 'rb') as f:
                            self.results[cur_model_name][iter][phase][seed] = pd.compat.pickle_compat.load(f)
                            self.results[cur_model_name][iter][phase][seed]['forecast'] = self.results[cur_model_name][iter][phase][seed]['forecast'].replace({pd.NA: np.nan})
                            self.results[cur_model_name][iter][phase][seed]['forecast'] = self.results[cur_model_name][iter][phase][seed]['forecast'].astype('float64')
                            if 'performance_indicators' in self.results[cur_model_name][iter][phase][seed].keys():
                                self.results[cur_model_name][iter][phase][seed]['performance_indicators'] = self.results[cur_model_name][iter][phase][seed]['performance_indicators'].replace({pd.NA: np.nan})
                                self.results[cur_model_name][iter][phase][seed]['performance_indicators'] = self.results[cur_model_name][iter][phase][seed]['performance_indicators'].astype('float64')
                            
    def load_saved_strategies(self):
        if not os.path.exists(os.path.join(self.results_folder,'strategies')):
            return

        strategies = os.listdir(os.path.join(self.results_folder,'strategies'))
        strategies = [m for m in strategies if os.path.isdir(os.path.join(self.results_folder, 'strategies', m)) ]
        for strategy_dir in strategies:
                       
                iters = os.listdir(os.path.join(self.results_folder,'strategies',strategy_dir))
                iters = [m for m in iters if os.path.isdir(os.path.join(self.results_folder, 'strategies', strategy_dir, m)) ]
                for iter_dir in iters:

                    files = os.listdir(os.path.join(self.results_folder,'strategies',strategy_dir,iter_dir))
                    files = [fi for fi in files if os.path.isfile(os.path.join(self.results_folder, 'strategies', strategy_dir, iter_dir, fi)) and fi.split('.')[-1] == 'pkl']
                    for cur_file in files:

                        cur_file_path = os.path.join(self.results_folder,'strategies',strategy_dir,iter_dir,cur_file)
                        cur_file_name = cur_file.split('.')[0]

                        cur_strategy_name = cur_file_name.split('__')[0]
                        iter = cur_file_name.split('__')[1]
                        phase = cur_file_name.split('__')[2]

                        if cur_strategy_name not in self.resstrategies.keys():
                            self.resstrategies[cur_strategy_name] = {}

                        if iter not in self.resstrategies[cur_strategy_name].keys():
                            self.resstrategies[cur_strategy_name][iter] = {}

                        with open(cur_file_path, 'rb') as f:
                            self.resstrategies[cur_strategy_name][iter][phase] = pd.compat.pickle_compat.load(f)     

    def add_model(self, config, force=False):
        # if config is not in configs yet add it, or overwrite
        self.configs[config['name']] = config

        # Check the folder exists
        iter = 'iter_'+str(self.curr_it)
        res_dir = os.path.join(self.results_folder, 
                               'models',
                               config['name'], 
                               iter,
                               self.curr_phase)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Check force condition and skip computation if desired
        if (not force) and (config['name'] in self.results.keys()) and (iter in self.results[config['name']].keys()) and (self.curr_phase in self.results[config['name']][iter].keys()):
            return

        seed_range = [0]
        if self.curr_phase == 'train':
            if not config['deterministic']:  
                seed_range = range(self.n_train_seeds)
            week_range = self.train_weeks
        elif self.curr_phase == 'test':
            if not config['deterministic']:
                seed_range = range(self.n_test_seeds)
            week_range = self.test_weeks
        else:
            raise ValueError(r'You can\'t add new models in this phase')

        # Evaluate model
        for seed in seed_range:
            l__seed = 'seed_'+str(seed)
            print(f'Evaluating {config["name"]} with seed {seed} in {self.curr_phase} phase')
            performance_indicators, forecast = self.eval_model(config, week_range, seed)
            if config['name'] not in self.results.keys():
                self.results[config['name']] = {}

            if iter not in self.results[config['name']].keys():
                self.results[config['name']][iter] = {}
            
            if self.curr_phase not in self.results[config['name']][iter].keys():
                self.results[config['name']][iter][self.curr_phase] = {}

            self.results[config['name']][iter][self.curr_phase][l__seed] = {
                    'performance_indicators': performance_indicators,
                    'forecast': forecast
                }
            # Save results to disk
            cur_file_path = os.path.join(res_dir,
                                        f'{config["name"]}__{iter}__{self.curr_phase}__{l__seed}__.pkl')
            with open(cur_file_path, 'wb') as f:
                pickle.dump(self.results[config["name"]][iter][self.curr_phase][l__seed], f)


    def eval_model(self, config, test_week_idcs, seed=0):

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

            # Set the seed for the model
            config['model'].set_seed(seed)

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

    def add_strategy(self, name, strategy, force=False):
        # only available in training phase of the first iteration
        assert self.curr_phase == 'train', 'Strategies can only be added during training phase'
        assert self.curr_it == 1, 'Strategies can only be added during the first iteration'

        # if strategy is not in strategies yet add it, or overwrite
        self.strategies[name] = strategy

        # Check the folder exists
        iter = 'iter_'+str(self.curr_it)
        res_dir = os.path.join(self.results_folder, 
                                'strategies',
                                name, 
                                iter) 
        # unlike models there is no phase or seed for strategies
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Check force condition and skip computation if desired
        if (not force) and (name in self.resstrategies.keys()) and (iter in self.resstrategies[name].keys()):
            return
        
        # Evaluate strategy
        print(f'Evaluating strategy: {name}')
        performance_indicators, forecast = self.eval_strategy(strategy)
        if name not in self.resstrategies.keys():
            self.resstrategies[name] = {}

        self.resstrategies[name][iter] = {
                'performance_indicators': performance_indicators,
                'forecast': forecast
            }
        # Save results to disk
        cur_file_path = os.path.join(res_dir,
                                    f'{name}__{iter}__{self.curr_phase}__.pkl')
        with open(cur_file_path, 'wb') as f:
            pickle.dump(self.resstrategies[name][iter], f)

    def eval_strategy(self, strategy):
        test_week_idcs = range(self.week_start+self.n_test_weeks, self.week_start+self.n_train_weeks) 
        results = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(week,dma) for week in test_week_idcs for dma in DMAS_NAMES], names=['Test week', 'DMA']),
            columns=['PI1', 'PI2', 'PI3'],
            dtype=float
        )

        forecast = self.demand.copy()
        forecast.iloc[:] = pd.NA

        for test_week_idx in tqdm.tqdm(test_week_idcs):
            # Load the truth for this week
            ground_truth = self.demand.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)]
            
            # Load how the selected models perfromed for the test_weeks before this week
            testresults = extract_results({k: self.results[k] for k in self.selected_models if k in self.results}, 
                                        self.curr_it,
                                        self.curr_phase,
                                        range(test_week_idx-self.n_test_weeks, test_week_idx)
                                        )

            # Select the best model(s) for each DMA
            best_models = strategy.find_best_models(testresults)

            # Take the forecast of the best model(s) for each DMA that forecasts the ground truth week
            forecasts = extract_forecasts({k: self.results[k] for k in best_models if k in self.results}, 
                                            self.curr_it,
                                            self.curr_phase,
                                            pd.date_range(start=ground_truth.index[0], 
                                                          periods=WEEK_LEN, freq='H')
                                        )   

            # Combine the forecasts
            demand_forecast = strategy.combine_forecasts(forecasts)

            # Save forecast and calculate Performance indicators
            forecast.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)] = demand_forecast
            results.loc[test_week_idx] = performance_indicators(demand_forecast, ground_truth)

        return results, forecast

    def forecast_next(self):
        assert self.selected_strategy is not None, 'No strategy selected'
        assert len(self.selected_models)>0, 'No models selected'

        self.curr_phase = 'test'

        for model_name in self.selected_models:
            self.add_model(self.configs[model_name]) # add model will send the correct database to the eval model function

        # etract the results of the selected models accordingly
        testresults = extract_results({k: self.results[k] for k in self.selected_models if k in self.results}, 
                                        self.curr_it,
                                        self.curr_phase,
                                        self.test_weeks
                                    )

        self.curr_phase = 'eval'

        best_models = self.strategies[self.selected_strategy].find_best_models(testresults)

        # create dataframes of the forecasts with the selected models
        forecasts = self.get_forecasts_an_all(best_models)

        demand_forecast = self.strategies[self.selected_strategy].combine_forecasts(forecasts)

        # demand forecast to dataframe
        demand_forecast = pd.DataFrame(demand_forecast, 
                                       index=self.weather.index[-WEEK_LEN:], 
                                        columns=self.demand.columns)

        l__iter = 'iter_'+str(self.curr_it)
        # save the forecast of the strategy 
        res_dir = os.path.join(self.results_folder, 
                                'strategies',
                                self.selected_strategy, 
                                l__iter)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        cur_file_path = os.path.join(res_dir,
                                    f'{self.selected_strategy}__{l__iter}__{self.curr_phase}__.pkl')
        with open(cur_file_path, 'wb') as f:
            pickle.dump(demand_forecast, f)

        # save also the dataframe as an excel file
        demand_forecast.to_excel(os.path.join(res_dir,
                                    f'{self.selected_strategy}__{l__iter}__{self.curr_phase}__.xlsx'))


    def get_forecasts_an_all(self, best_models) -> dict:
        forecasts = {} # Returns a dictionary of dataframes with the forecasts of the selected models for each DMA and multiple seeds (index)
        l__iter = 'iter_'+str(self.curr_it)

        for model_name in best_models:

            # Check the folder where I save the models results exists  
            res_dir = os.path.join(self.results_folder, 
                                    'models',
                                    model_name, 
                                    l__iter,
                                    self.curr_phase)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            df_list = [] # df for multiple seeds in the same model
            for seed in range(self.n_test_seeds):
                l__seed = 'seed_'+str(seed)

                # if it exists already it was loaded in the constructor, so use that because it means the previous evaluation went wrong 
                if self.curr_phase in self.results[model_name][l__iter].keys() and l__seed in self.results[model_name][l__iter][self.curr_phase].keys() and 'forecast' in self.results[model_name][l__iter][self.curr_phase][l__seed].keys():
                    df_list.append(self.results[model_name][l__iter][self.curr_phase][l__seed]['forecast']) 
                    # I append a dataframe with Date as index and DMA as columns

                # otherwise evaluate it
                else:
                    df_list.append(self.get_forecast_on_all(self.configs[model_name], seed)) # from here I get the same dataframe with Date as index and DMA as columns
                    
                    # save it in results in case one of the other models breaks the run
                    if self.curr_phase not in self.results[model_name][l__iter].keys():
                        self.results[model_name][l__iter][self.curr_phase] = {}
                    self.results[model_name][l__iter][self.curr_phase][l__seed] = dict(forecast=df_list[-1])

                    cur_file_path = os.path.join(res_dir,
                                            f'{model_name}__{l__iter}__{self.curr_phase}__{l__seed}__.pkl')
                    with open(cur_file_path, 'wb') as f:
                        pickle.dump(self.results[model_name][l__iter][self.curr_phase][l__seed], f)

            forecasts[model_name] = pd.concat(df_list,
                                        keys=range(len(df_list)),
                                        names=['Seed', 'Date'])
            
            # Return the promised dictionary of double index (seed and date) dataframes 
        return forecasts
    
    def get_forecast_on_all(self, config, seed: int) -> pd.DataFrame:
        print(f'Evaluating {config["name"]} with seed {seed} in {self.curr_phase} phase')

        demand_train = self.demand.iloc[:WEEK_LEN*self.eval_week]
        weather_train = self.weather.iloc[:WEEK_LEN*self.eval_week]
        weather = self.weather.iloc[WEEK_LEN*self.eval_week: WEEK_LEN*(self.eval_week+1)]
        
        # If applicable, prepare test dataframes
        if 'prepare_test_dfs' in config['preprocessing']:
            for preprocessing_step in config['preprocessing']['prepare_test_dfs']:
                demand_test, weather = preprocessing_step.transform(demand_train, weather_train, weather)

        # Apply preprocessing for demands
        for preprocessing_step in config['preprocessing']['demand']:
            demand_train = preprocessing_step.fit_transform(demand_train)

        # Apply preprocessing for weather
        for preprocessing_step in config['preprocessing']['weather']:
            weather_train = preprocessing_step.fit_transform(weather_train)

        # Set the seed for the model
        config['model'].set_seed(seed)

        # Train model
        config['model'].fit(demand_train, weather_train)

        # If applicable, prepare test dataframes
        if 'prepare_test_dfs' in config['preprocessing']:
            for preprocessing_step in config['preprocessing']['demand']:
                demand_test = preprocessing_step.transform(demand_test)

            for preprocessing_step in config['preprocessing']['weather']:
                weather = preprocessing_step.transform(weather)

            # demand_test = demand_test.iloc[-WEEK_LEN:,:]
            # weather_test = weather_test.iloc[-WEEK_LEN:,:]
        else:
            # Prepare test weather anyways
            for preprocessing_step in config['preprocessing']['weather']:
                weather = preprocessing_step.transform(weather)

            demand_test = None

        # Forecast next week
        demand_forecast = config['model'].forecast(demand_test, weather)
        demand_forecast = pd.DataFrame(demand_forecast, 
                                       index=self.weather.iloc[WEEK_LEN*self.eval_week: WEEK_LEN*(self.eval_week+1)].index, 
                                       columns=self.demand.columns)

        # Transform forecast back into original unit
        for preprocessing_step in reversed(config['preprocessing']['demand']):
            demand_forecast = preprocessing_step.inverse_transform(demand_forecast)

        return demand_forecast


def extract_results(results, iter_n, phase, weeks):
    # Extract the results 
    iter = 'iter_'+str(iter_n)
    testres = {}
    for model_name in results.keys():
        testres[model_name] = {}
        df_list = []
        for seed in results[model_name][iter][phase].keys():
            df_list.append(results[model_name][iter][phase][seed]['performance_indicators'].loc[weeks])

        testres[model_name] = pd.concat(df_list,
                                        keys=range(len(df_list)),
                                        names=['Seed'])

    return testres

def extract_forecasts(results, iter_n, phase, hours):
    # Extract the forecasts
    iter = 'iter_'+str(iter_n)
    testres = {}
    for model_name in results.keys():
        testres[model_name] = {}
        df_list = []
        for seed in results[model_name][iter][phase].keys():
            df_list.append(results[model_name][iter][phase][seed]['forecast'].loc[hours])

        testres[model_name] = pd.concat(df_list,
                                        keys=range(len(df_list)),
                                        names=['Seed', 'Date'])

    return testres

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
