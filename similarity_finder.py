import data_loader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def data_sim(Data_1: pd.DataFrame, Data_2: pd.DataFrame, coef=(1, 1, 1, 1)) -> float:
    """
    This method compares the two datasets with the euclidian distance

    Parameters
    ----------
    Data_1 : DataFrame
        The First Dataset that we want to compare.
    Data_2 : DataFrame
        The Second Dataset that we want to compare.
    coef : SET, optional
        The weights for each weather parameter. The default is (1,1,1,1).

    Returns
    -------
    dist : FLOAT
        The euclidian distance.

    """
    #TODO Maybe try different simillarity function

    #Note this method is still debatable propable future expantion is to add weights to each parameter (some parameters are more importan than others)
    dist = (np.dot(np.sum((Data_1.values-Data_2.values)**2,axis=0),coef)/sum(coef))**0.5
    
    return dist
    

def Simlar_w_finder(num_comp : int, weight=(1,1,1,1), reform_fact: str = "D") -> (list, list):
    """
    This method compares the two datasets with the euclidian distance

    Parameters
    ----------
    num_comp : INT
        The given week that we want to compare with the rest of the timeseries.
    weight : LIST, optional
        The weights for each weather parameter. The default is (1,1,1,1).
    reform_fact : STR, optional
        The Resampling factor e.x. D --> daily. The default is "D".

    Returns
    -------
    most_sim : LIST
        A List from the most simillar to less.
    DIC
        A Dic for with all the values of each week.

    """
    #Data Read
    weather_data= data_loader.load_original_data()[1]
    
    
    #Dataresampling
    Rain_r=weather_data["Rain"].resample(reform_fact).sum()                     # I dont take the average of the total rainfall because i will lose information
    Temp_r=weather_data["Temperature"].resample(reform_fact).mean()
    Humidity_r=weather_data["Humidity"].resample(reform_fact).mean()
    Windspeed_r=weather_data["Windspeed"].resample(reform_fact).mean()
    
    #Connect the dat
    Reformed_data = pd.concat([Rain_r, Temp_r, Humidity_r, Windspeed_r], axis=1)
    
    #dump the first 3 days so we can start from monday  ##If this is get impemented in the data_loader please delete the code below
    
    n = 3
    Reformed_data = Reformed_data.iloc[n:]
    
    ###Data normalazation  ####
    #I normalize each factor by it own in
    
    Reformed_data["Rain_sc"] = MinMaxScaler().fit_transform(Reformed_data[["Rain"]])
    Reformed_data["Temperature_sc"] = MinMaxScaler().fit_transform(Reformed_data[["Temperature"]])
    Reformed_data["Humidity_sc"] = MinMaxScaler().fit_transform(Reformed_data[["Humidity"]])
    Reformed_data["Windspeed_sc"] = MinMaxScaler().fit_transform(Reformed_data[["Windspeed"]])
    
    
    
    scaled_data = Reformed_data[["Rain_sc", "Temperature_sc", "Humidity_sc", "Windspeed_sc"]]
     
    
    
    # Group by week
    weekly_groups = scaled_data.groupby(pd.Grouper(freq='W'))
    
    #Carefull start with 1 for the 1st week
    weekly_list = {i+1: group for i, (_, group) in enumerate(weekly_groups)}
    
    
    dist_list = {}
    
    for i,j in weekly_list.items():
        dist_list[i] = data_sim(weekly_list[num_comp], j, coef=weight  )
    
    most_sim = [key for key, value in sorted(dist_list.items(), key=lambda item: item[1])]
    most_sim.pop(0) #Remove the same week 
    
    return most_sim, weekly_list    #I dont think that weekly_list is a must output but can be helpfull

    

if __name__=="__main__":
    test = Simlar_w_finder(1,weight=(0,0,0,0))