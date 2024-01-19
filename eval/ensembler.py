import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import os
import pathlib
import pickle

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']
WEEK_LEN = 24 * 7

class WaterFuturesEnsembler:

    # TODO: REMOVE
    # Constructor to inject own data
    def __init__(self, demand, weather):
        # Load data
        self.demand, self.weather = demand, weather
        self.weather_train, self.weather_test = self.weather.iloc[:-WEEK_LEN], self.weather.iloc[-WEEK_LEN:]
        self.total_weeks = self.demand.shape[0] // WEEK_LEN

        # Initialize collections
        self.model_configs = []
        self.model_performances = {}
        self.forecasts = {}
    '''
    def __init__(self):
        # Load data
        self.demand, self.weather = load_data()
        self.weather_train, self.weather_test = self.weather.iloc[:-WEEK_LEN], self.weather.iloc[-WEEK_LEN:]
        self.total_weeks = self.demand.shape[0] // WEEK_LEN

        # Initialize collections
        self.model_configs = []
        self.model_performances = {}
        self.forecasts = {}
    '''
        
    def add_model(self, config):
        self.model_configs.append(config)

    def run(self):
        # Determine weeks to consider during ensembling
        self.get_ensembling_weeks()

        # Evaluate candidate models on those weeks
        self.eval_models()
        
        # Calculate best model of test weeks for first 24h / the rest
        self.find_best_models()

        # Make forecast for selected best models
        self.generate_forecasts()

        # Combine forecasts to use best models in each situation
        self.combine_forecasts()

        # Save eneseble forecast to disk
        # TODO

    def get_ensembling_weeks(self):
        #TODO: Do more advanced techiques I guess
        last_n = 1
        self.evaluation_weeks = np.array([self.total_weeks - 1 - n for n in range(last_n)])

    def eval_models(self):
        # Evaluate all added model configs
        for config in self.model_configs:
            cur_performances = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(week,dma) for week in self.evaluation_weeks for dma in self.demand.columns], names=['Week', 'DMA']),
                columns=['PI1', 'PI2', 'PI3'],
                dtype=float
            )

            # on all choose evaluation weeks
            for test_week_idx in self.evaluation_weeks:
                # Load current train and test data
                demand_train = self.demand.iloc[:WEEK_LEN*test_week_idx]
                weather_train = self.weather.iloc[:WEEK_LEN*test_week_idx]
                ground_truth = self.demand.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)]
                weather_test = self.weather.iloc[WEEK_LEN*test_week_idx: WEEK_LEN*(test_week_idx+1)]

                # Make forecast
                demand_forecast = get_forecast(config, demand_train, weather_train, weather_test)

                # Add performances to dataframe
                # TODO: Remove astype statements...
                cur_performances.loc[test_week_idx] = performance_indicators(demand_forecast.astype('float64'), ground_truth.astype('float64'))

            # Save performances
            self.model_performances[config['name']] = cur_performances

    def find_best_models(self):
        # Create numpy array with model names to be able to list-index them
        model_names = np.array([config['name'] for config in self.model_configs])

        # Calculate the average of the performance indicators
        average_pis = np.stack([self.model_performances[model].groupby('DMA').mean().to_numpy() for model in model_names])
        average_pi12 = np.mean(average_pis[:,:,2:], axis=2)
        average_pi3 = average_pis[:,:,2]
        
        # Find best models
        best_model_pi12 = model_names[np.argmin(average_pi12, axis=0)]
        best_model_pi3 = model_names[np.argmin(average_pi3, axis=0)]

        self.best_models = np.concatenate([[best_model_pi12, best_model_pi3]])

    def generate_forecasts(self):
        # Generate forecasts for all models that are choosen among the best models
        for model_name in np.unique(self.best_models):
            model_config = [config for config in self.model_configs if config['name'] == model_name][0]
            self.forecasts[model_name] = get_forecast(model_config, self.demand, self.weather_train, self.weather_test)

    def combine_forecasts(self):
        # Combine the forecasts according to the ensembling strategy
        # TODO: Try out different ensembling strategies
        self.ensemble_forecast = pd.DataFrame(index=self.weather.iloc[-168:].index, columns=self.demand.columns)
        for dma_idx, dma in enumerate(self.ensemble_forecast.columns):
            self.ensemble_forecast[dma].iloc[:168] = self.forecasts[self.best_models[0,dma_idx]][dma].iloc[:168]
            self.ensemble_forecast[dma].iloc[168:] = self.forecasts[self.best_models[1,dma_idx]][dma].iloc[168:]


# TODO: REMOVE
def get_forecast(config, demand_train, weather_train, weather):
    # Temporary get forecast method that loads preexisting forecasts
    results_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'wfe_results')
    cur_file = f'{config["name"]}.pkl'
    cur_file_path = os.path.join(results_folder, cur_file)
        
    with open(cur_file_path, 'rb') as f:
        cur_res = pd.compat.pickle_compat.load(f)

    pis = cur_res['forecast']
    return pis.loc[weather.index]

# TODO: Uncomment
'''
def get_forecast(config, demand_train, weather_train, weather):
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
    demand_forecast = pd.DataFrame(demand_forecast, index=weather.index, columns=demand_train.columns)

    # Transform forecast back into original unit
    for preprocessing_step in reversed(config['preprocessing']['demand']):
        demand_forecast = preprocessing_step.inverse_transform(demand_forecast)

    return demand_forecast
'''
    
def load_data():
    data_folder = os.getenv('BON2024_DATA_FOLDER')
    if data_folder is None:
        data_folder = 'data'

    # Load the excel file
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'InflowData_1.xlsx') )

    # Make the first column to datetime format
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M')
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})

    demand = rawdata
    demand.set_index('Date', inplace=True) # Make the Date column the index of the dataframe

    # Rename the columns from DMA_A to DMA_J
    demand.columns = DMAS_NAMES

    # Load weather data
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'WeatherData_1.xlsx') )

    #Same stuff for weather data
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M')
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})
    weather = rawdata

    weather.set_index('Date', inplace=True)
    weather.columns = ['Rain', 'Temperature', 'Humidity', 'Windspeed']

    # Adjust for Summer/Winter time
    demand = adjust_summer_time(demand)
    weather = adjust_summer_time(weather)

    # Make Data start at first monday
    demand = demand.loc['2021-01-04':]
    weather = weather.loc['2021-01-04':]

    return demand, weather

    
### Data loading helpers
def adjust_summer_time(df):
    days_missing_hour = ['2021-03-28', '2022-03-27', '2023-03-26']

    # Copy 1AM and 3AM data to 2AM for days missing 2AM
    for day in days_missing_hour:
        df = pd.concat([df, df.loc[f'{day} 01:00:00':f'{day} 03:00:00']
                      .reset_index()
                      .assign(Date=pd.to_datetime(f'{day} 02:00:00'))
                      .set_index('Date')]) \
                      .sort_index()

    # Average 2AM values for days with duplicates
    return df.groupby('Date').mean().sort_index()

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
