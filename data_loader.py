import pandas as pd
import os 
from sklearn.model_selection import train_test_split

def load_calendar():
    # Calendar, with holidays and weekends
    dates = pd.date_range(start='2021-01-01', end='2023-03-31', freq='D')
    calendar = pd.DataFrame(index=dates)
    calendar['Holiday'] = 0
    calendar['Weekend'] = 0
    calendar['SummerTime'] = 0
    holidays = ['2021-01-01', '2021-01-06','2021-04-04','2021-04-05'] #TODO: add all holidays
    calendar.loc[holidays, 'Holiday'] = 1
    calendar.loc[:, 'Weekend'] = calendar.index.dayofweek.isin([5, 6]).astype(int)
    calendar.loc['2021-03-28', 'SummerTime'] = 1
    calendar.loc['2021-10-31', 'SummerTime'] = -1
    calendar.loc['2022-03-27', 'SummerTime'] = 1
    calendar.loc['2022-10-30', 'SummerTime'] = -1
    calendar.loc['2023-03-26', 'SummerTime'] = 1

    return calendar

def load_characteristics():
    dma_characts_json = {
        'name_short':['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 
        'name_long': ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J'],
        'description' : ['Hospital district', 'Residential district in the countryside', 'Residential district in the countryside',
                        'Suburban residential/commercial district', 'Residential/commercial district close to the city centre',
                        'Suburban district including sport facilities and office buildings', 'Residential district close to the city centre',
                        'City centre district', 'Commercial/industrial district close to the port', 'Commercial/industrial district close to the port'],
        'desc_short' : ['Hosp', 'Res cside', 'Res cside', 
                        'Suburb res/com', 'Res/com close',
                        'Suburb sport/off', 'Res close',
                        'City', 'Port', 'Port'],
        'population' : [162, 531, 607, 2094, 7955, 1135, 3180, 2901, 425, 776],
        'h_mean' : [8.4, 9.6, 4.3, 32.9, 78.3, 8.1, 25.1, 20.8, 20.6, 26.4] # L/s/per hour
    }
    dmas_characteristics = pd.DataFrame(dma_characts_json)
    dmas_characteristics.set_index('name_long', inplace=True)

    return dmas_characteristics

def load_original_data():
    # This function is used to load the data from the excel files and have a common format for the names and the structure of the work.
    # The original data are saved in the same folder of the code under BoN2024/data/original. 
    # The path to the folder BoN2024 can be changed and specidied in the environment variable BON2024_DATA_FOLDER, if this doesn't exist the data are considered to be in the code directory 'data'.
    # 
    # The created variables are the following:
    # - raw_dmas_h_cons: dataframe with the consumption/inflow data, the columns are the names of the DMAs and the index is the date
    # - raw_weather_h:     dataframe with the weather data, the columns are the names of the variables and the index is the date
    # - calendar:        dataframe with the dates, the columns are 'Holiday', 'Weekend', 'Summertime' and the index is the date
    #                           (Summertime is +1 when the summer time starts and you skip 02:00, -1 when you end it and you will have 02:00 twice)
    # - dmas_characteristics: dataframe with the characteristics of the DMAs, the columns are the names of the variables and the index is the name of the DMAs

    ## Data  defined by the problem (see pdf in data/instructions)
    calendar = load_calendar()
    dmas_characteristics = load_characteristics()

    ## Data  need to be uploaded from the excel files
    # Get the folder where the data are saved from the environment variable.
    # If the variable is not set, use the user's home directory.
    data_folder = os.getenv('BON2024_DATA_FOLDER')
    if data_folder is None:
        data_folder = 'data'

    # Print the path to the data folder
    #print(data_folder)

    # Load the excel file
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'InflowData_1.xlsx') )

    # Make the first column to datetime format
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M') 
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})

    raw_dmas_h_cons = rawdata
    raw_dmas_h_cons.set_index('Date', inplace=True) # Make the Date column the index of the dataframe

    # Rename the columns from DMA_A to DMA_J
    raw_dmas_h_cons.columns = dmas_characteristics.index
    raw_dmas_h_cons.attrs['units'] = {'DMA_A':'L/s', 'DMA_B':'L/s', 'DMA_C':'L/s', 'DMA_D':'L/s', 'DMA_E':'L/s', 'DMA_F':'L/s', 'DMA_G':'L/s', 'DMA_H':'L/s', 'DMA_I':'L/s', 'DMA_J':'L/s'}

    # Print the first 5 rows of the dataframe
    #print(raw_dmas_h_cons.head())

    # Load weather data
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'WeatherData_1.xlsx') )

    #Same stuff for weather data
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M') 
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})
    raw_weather_h = rawdata

    raw_weather_h.set_index('Date', inplace=True)
    raw_weather_h.columns = ['Rain', 'Temperature', 'Humidity', 'Windspeed']
    raw_weather_h.attrs['units'] = {'Rain':'mm', 'Temperature':'°C', 'Humidity':'%', 'Windspeed':'km/h'}

    return raw_dmas_h_cons, raw_weather_h, calendar, dmas_characteristics

def load_splitted_data(split_strategy="final_weeks", split_size_w=4, week_selection=0, start_first_monday=True):
    """
    Loads and splits the data according to the specified strategy and parameters.
    
    
    :param split_strategy: The strategy for splitting the data ('final_weeks', 'random', etc.).
    :param split_size_w: The number of weeks to include in the test set (default 4).
    :param week_selection: The specific week to use as the test set (from 1 to split_size_w).
                            If 0, all weeks in split_size_w are used for testing (default 0).
    :return: A tuple of (training_data, testing_data).

    The input dataset is loaded as a Pandas DataFrame with load_original_data().
    """
    raw_dmas_h_cons, raw_weather_h, _, _ = load_original_data()

    # Remove the extra data from the end of the weather dataset
    max_date = raw_dmas_h_cons.index.max()
    raw_weather_h = raw_weather_h.loc[raw_weather_h.index <= max_date,:]

    index_week = pd.Series(0, index=raw_dmas_h_cons.index)
    # assign to each measurement the week number
    week_n_from_beginning = 0
    for i in range(1, len(raw_dmas_h_cons.index), 1):
        if raw_dmas_h_cons.index[i].dayofweek == 0 and raw_dmas_h_cons.index[i-1].dayofweek == 6:
            week_n_from_beginning += 1

        index_week[i] = week_n_from_beginning

    weeks_range = range(1, index_week.max()) # from 1 to n_weeks, I don't want to use week 0 as test set

    if start_first_monday:
        # Remove the extra data from the beginning of the dataset to start on a Monday
        raw_dmas_h_cons = raw_dmas_h_cons.loc[raw_dmas_h_cons.index >= '2021-01-04',:]
        raw_weather_h = raw_weather_h.loc[raw_weather_h.index >= '2021-01-04',:]

    # Split the data
    if split_strategy == "final_weeks":
        test_weeks = range(index_week.max()-split_size_w+1, index_week.max()+1)
        
    elif split_strategy == "random_weeks":
        # Use scikit learn train_test_split
        _, test_weeks = train_test_split(weeks_range, test_size=split_size_w, shuffle=True, random_state=42)
        
    elif split_strategy == "custom1":
        # Here I want to use some specific weeks for testing to "overfit" what we will forecast
        # E.g., the previous week and the same week of the previous year
        raise ValueError("Not implemented yet")
         # TODO: Decide which weeks to use and put it ins test_weeks 
         # then simply do test_indices = index_week.isin(test_weeks)

    else:
        raise ValueError("Unknown split strategy: " + split_strategy)
    
    # Split the data
    if week_selection > 0:
        # should check index but let's use it to throw errors 
        test_weeks = test_weeks[week_selection-1] 
        test_indices = index_week == test_weeks
    else:
        test_indices = index_week.isin(test_weeks)
    
    train_dmas_h_cons = raw_dmas_h_cons.loc[~test_indices,:]
    test_dmas_h_cons = raw_dmas_h_cons.loc[test_indices,:]
    train_weather_h = raw_weather_h.loc[~test_indices,:]
    test_weather_h = raw_weather_h.loc[test_indices,:]

    return train_dmas_h_cons, test_dmas_h_cons, train_weather_h, test_weather_h
