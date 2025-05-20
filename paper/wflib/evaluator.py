import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

DATE__l='date'

class WaterFuturesEvaluator():
    """
    This is a more generic evaluator that can be used with custom approaches.
    """

    def __init__(self):
        """
        
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.results_dir = os.path.join(data_dir, 'output')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        target_observations, exogenous_observations = self.load_bwdf_final_data(data_dir)
        
        self.target_df= target_observations
        self.exog_df= exogenous_observations
        
        self.forecasting_horizon = 24*7 # 1 week
        self.n_horizon_test_starts = 52 # We start after a year of data
        
        self.models = {}
        self.load_saved_models()

        self.strategies = {}
        self.load_saved_strategies()

    @property
    def demand(self):
        return self.target_df

    def load_bwdf_final_data(self, data_dir):
        
        demands = pd.read_excel(os.path.join(data_dir, 'input', 'Inflows.xlsx'))
        weather = pd.read_excel(os.path.join(data_dir, 'input', 'Weather.xlsx'))

        # basic preprocessing to set the first column as the index with name date
        def preprocess_date_columns(df):
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], format='%d/%m/%Y %H:%M')
            df = df.rename(columns={df.columns[0]: DATE__l})
            df = df.set_index(DATE__l)
            return df

        demands = preprocess_date_columns(demands)
        weather = preprocess_date_columns(weather)

        # Set the units of the columns for the weather data
        demands.columns = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']
        weather.columns = ['Rain', 'Temperature', 'Humidity', 'Windspeed']
        weather.attrs['units'] = {'Rain':'mm', 'Temperature':'°C', 'Humidity':'%', 'Windspeed':'km/h'}

        # Adjust the dataset to account for the missing hour in summer time and average the duplicates
        def adjust_summer_time(df):
            days_missing_hour = ['2021-03-28', '2022-03-27', '2023-03-26']

            # Copy 1AM and 3AM data to 2AM for days missing 2AM
            for day in days_missing_hour:
                df = pd.concat([df, df.loc[f'{day} 01:00:00':f'{day} 03:00:00']
                            .reset_index()
                            .assign(date=pd.to_datetime(f'{day} 02:00:00'))
                            .set_index(DATE__l)]) \
                                .sort_index()

            # Average 2AM values for days with duplicates
            return df.groupby(DATE__l).mean().sort_index()

        demands = adjust_summer_time(demands)
        weather = adjust_summer_time(weather)

        # Adjust the dataset to start on the first Monday
        def adjust_first_monday(df):
            return df.loc['2021-01-04':]

        demands = adjust_first_monday(demands)
        weather = adjust_first_monday(weather)

        return demands, weather
    
    def load_saved_models(self):
        """
        Upload the saved results from the disk.
        """
        if not os.path.exists(os.path.join(self.results_dir,'models')):
            os.makedirs(os.path.join(self.results_dir,'models'))
            return
        
        # Check which models are saved and if they are actually directories
        models_dirs = os.listdir(os.path.join(self.results_dir,'models'))
        models_dirs = [d for d in models_dirs if os.path.isdir(os.path.join(self.results_dir,'models',d))]

        # Upload the models
        for model_dir in models_dirs:
            model_fcs = pickle.load(open(os.path.join(self.results_dir,'models',model_dir, f'{model_dir}__forecasts.pkl'), 'rb'))
            self.models[model_dir] = dict(forecasts= model_fcs.infer_objects(), config= {})
            
    def load_saved_strategies(self):
        """
        Upload the saved results from the disk.
        """
        if not os.path.exists(os.path.join(self.results_dir,'strategies')):
            os.makedirs(os.path.join(self.results_dir,'strategies'))
            return
        
        # Check which models are saved and if they are actually directories
        strategies_dirs = os.listdir(os.path.join(self.results_dir,'strategies'))
        strategies_dirs = [d for d in strategies_dirs if os.path.isdir(os.path.join(self.results_dir,'strategies',d))]

        # Upload the models
        for strategy_dir in strategies_dirs:
            strategy_fcs = pickle.load(open(os.path.join(self.results_dir,'strategies',strategy_dir, f'{strategy_dir}__forecasts.pkl'), 'rb'))
            self.strategies[strategy_dir] = dict(forecasts= strategy_fcs)

    def add_model_configuration(self, configuration: dict, force= False):

        model_name = configuration['name']
        # New model or force set a clean status: clean the dict and the directory where the results are saved
        if force or model_name not in self.models:
            print(f'Adding (or resetting) model {model_name} to the evaluator.')

            self.models[model_name] = dict(forecasts=[])

            # if the directory exists already we will overwrite it, otherwise we will create it
            if os.path.exists(os.path.join(self.results_dir,'models',model_name)):
                os.rmdir(os.path.join(self.results_dir,'models',model_name))
            
            os.makedirs(os.path.join(self.results_dir,'models',model_name))

        self.models[model_name]['config'] = configuration

        self._run_model_configuration(model_name)

        # Save the results on the disk
        pickle.dump(self.models[model_name]['forecasts'], open(os.path.join(self.results_dir,'models',model_name, f'{model_name}__forecasts.pkl'), 'wb'))
    
    def _run_model_configuration(self, model_name: str):        
        """
        This function runs a model configuration and saves the results in the dictionary and on the disk.
        If some results are already present, it will append the new ones, until the number of runs is reached.
        This is the case for example if some stochastic models or optimization algorithms are used.
        """
        config = self.models[model_name]['config']

        # If model is deterministic, we will have to run it only once with a fake seed 0
        # otherwise, take the param n_eval_runs from the config.
        # The number of training runs that have been done already is implicitly defined in the index of the forecast DataFrame.
        # This contains a (seed, date) index and a value for each variable that is forecasted.
        n_eval_runs_requested = 1 if config['deterministic'] else config['n_eval_runs']
        n_eval_runs_done = 0 if len(self.models[model_name]['forecasts'])==0 else self.models[model_name]['forecasts'].index.get_level_values('seed').nunique()
        seeds = np.random.Generator(np.random.PCG64()).integers(0, 2**32-1, n_eval_runs_requested - n_eval_runs_done)
        n_eval_runs_todo = len(seeds)

        # If seeds is empty, nothing to be done
        if n_eval_runs_todo == 0:
            print(f'No new seeds to run for model {model_name}.')
            return
        print(f'Running {n_eval_runs_todo} new seeds for model {model_name}.')

        # Only the training of the model is stochastic. The preparation of the data is assumed to be deterministic.
        # So we will prepare the data only once and use it for all the seeds.
        
        # Create a empty (nans) DataFrame with the same shape of the observations and the multiple seeds to do.
        # This will be filled with the forecasts for each seed.
        fcst_df = pd.DataFrame(index=pd.MultiIndex.from_product([seeds, self.target_df.index], names=['seed', DATE__l]),
                            columns=self.target_df.columns)

        # The model will be evaluated on the testing set, starting from the n-th horizon until the end,
        # with a streaming approach with a step that is the horizon of the model.

        for testing_current_idx in tqdm(
            range(
                self.n_horizon_test_starts*self.forecasting_horizon,
                self.target_df.shape[0],
                self.forecasting_horizon)
            ):
            # The model will be trained on all the data until the current testing horizon starts.
            # The forecast will be made on the next forecasting_horizon steps.            
            target_df__4train = self.target_df.iloc[:testing_current_idx]
            exog_df__4train = self.exog_df.iloc[:testing_current_idx]

            # Our assumption is that to create a forecast on the next horizon,
            # you will not get anything from the target data, but exogenous data can be used as perfect forecasts.
            target_df__4fcst = None
            exog_df__4fcst = self.exog_df.iloc[testing_current_idx:testing_current_idx+self.forecasting_horizon]
            idxs2fcst = exog_df__4fcst.index

            # Autoregressive models may require data from previous steps to make the forecast.
            # So we add the training data to the auxiliary dataframes to make the prediction.
            if 'prediction_requires_extended_dfs' in self.models[model_name]['config']['preprocessing'] and self.models[model_name]['config']['preprocessing']['prediction_requires_extended_dfs']:
                
                target_df__4fcst = pd.concat([
                    self.target_df.iloc[:testing_current_idx],
                    pd.DataFrame(index=idxs2fcst, columns=self.target_df.columns)
                ], axis=0)
                
                exog_df__4fcst = self.exog_df.iloc[:testing_current_idx+self.forecasting_horizon]
            
            # Apply pre-processing to the data
            for preprocessor in self.models[model_name]['config']['preprocessing']['target']:
                target_df__4train = preprocessor.transform(target_df__4train)
                if target_df__4fcst is not None:
                    target_df__4fcst = preprocessor.transform(target_df__4fcst)

            for preprocessor in self.models[model_name]['config']['preprocessing']['exogenous']:
                exog_df__4train = preprocessor.transform(exog_df__4train)
                exog_df__4fcst = preprocessor.transform(exog_df__4fcst)

            # Now that we prepared the data, we can train the model for every seed and make the forecast.
            # The assumptions are that (i) during training, it is up to the model to split internally the data in training 
            # and validation, and (ii) during forecasting, the model will use the last 'forecasting_horizon' steps to make the forecast.
            for seed in seeds:
                seed = int(seed)

                self.models[model_name]['config']['model'].set_seed(seed)

                self.models[model_name]['config']['model'].fit(target_df__4train, exog_df__4train)

                target_prediction = self.models[model_name]['config']['model'].forecast(target_df__4fcst, exog_df__4fcst)

                # We expect the prediction to be a numpy array of shape (forecasting_horizon, n_variables)
                assert target_prediction.shape == (self.forecasting_horizon, self.target_df.shape[1])
                # We will store the forecast in the DataFrame, with the indexes to forecast and the seed used.
                fcst_df.loc[(seed, idxs2fcst), :] = target_prediction

        # Append the new forecasts to the existing ones
        if self.models[model_name]['forecasts'] == []:
            self.models[model_name]['forecasts'] = fcst_df
        else:
            self.models[model_name]['forecasts'] = pd.concat([self.models[model_name]['forecasts'], fcst_df], axis=0)

        

        