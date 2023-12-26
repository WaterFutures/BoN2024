import numpy as np
import pandas as pd
import os
import pathlib
import pickle

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']

class WaterFuturesEvaluator:

    def __init__(self):
        # Load data, omitting the last 4 weeks
        demand, weather = load_data()
        self.demand = demand.iloc[:-24*7*4]
        self.weather = weather.iloc[:-24*7*4]

        self.week_start = 12
        self.total_weeks = self.demand.shape[0] // (24*7)

        self.results = {}

        self.results_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'wfe_results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.load_saved_results()

    def load_saved_results(self):
        files = os.listdir(self.results_folder)
        for cur_file in files:
            cur_file_path = os.path.join(self.results_folder, cur_file)
            cur_model_name = '.'.join(cur_file.split('.')[:-1])
            with open(cur_file_path, 'rb') as f:
                self.results[cur_model_name] = pickle.load(f)

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
            columns=['PI1', 'PI2', 'PI3']
        )

        forecast = self.demand.copy()
        forecast.iloc[:] = pd.NA

        for test_week_idx in test_week_idcs:
            # Load current train and test data
            demand_train = self.demand.iloc[:24*7*test_week_idx]
            weather_train = self.weather.iloc[:24*7*test_week_idx]
            demand_test = self.demand.iloc[24*7*test_week_idx: 24*7*(test_week_idx+1)]
            weather_test = self.weather.iloc[24*7*test_week_idx: 24*7*(test_week_idx+1)]

            # Apply preprocessing for demands
            for preprocessing_step in config['preprocessing']['demand']:
                demand_train = preprocessing_step.fit_transform(demand_train)

            # Apply preprocessing for weather
            for preprocessing_step in config['preprocessing']['weather']:
                weather_train = preprocessing_step.fit_transform(weather_train)

            # Train model
            config['model'].fit(demand_train, weather_train)


            # Apply preprocessing for test weather
            for preprocessing_step in config['preprocessing']['weather']:
                weather_train = preprocessing_step.transform(weather_train)

            # Forecast next week
            demand_forecast = config['model'].forecast(weather_test)
            demand_forecast = pd.DataFrame(demand_forecast, index=demand_test.index, columns=demand_test.columns)

            # Transform forecast back into original unit
            for preprocessing_step in reversed(config['preprocessing']['demand']):
                demand_forecast = preprocessing_step.inverse_transform(demand_forecast)

            
            # Save forecast and calculate Performance indicators 
            forecast.iloc[24*7*test_week_idx: 24*7*(test_week_idx+1)] = demand_forecast
            results.loc[test_week_idx] = performance_indicators(demand_forecast, demand_test)

        return results, forecast

    
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

### Performance Indicators
def performance_indicators(y_pred, y_true):
    assert not np.any(np.isnan(y_pred)), 'Model forecasted NaN values'
    return np.vstack((pi1(y_pred, y_true), pi2(y_pred, y_true), pi3(y_pred, y_true))).T

def pi1(y_pred, y_true):
    return np.nanmean(np.abs(y_pred[:24] - y_true[:24]), axis=0)

def pi2(y_pred, y_true):
    return np.max(np.abs(y_pred[:24] - y_true[:24]), axis=0)

def pi3(y_pred, y_true):
    return np.nanmean(np.abs(y_pred[24:] - y_true[24:]), axis=0)