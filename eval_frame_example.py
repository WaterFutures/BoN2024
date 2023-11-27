import data_loader as dl
import pandas as pd
import numpy as np

import eval_framework as ef

"""Explanation on how to use the evaluation framework and the data loader."""

# Extract the data using the data loader with the parameters that you want:
# split_strategy how to get them, split size number of weeks to use for testing,
# week selection if greater than 0 it will override the split size and use the specific
# week coming from the split strategy, start_first_monday if True it will remove the 
# first 3 days of the dataset to start on Monday 4th January 2021
train__dmas_h_q, test__dmas_h_q, train__wea_h, test__wea_h = dl.load_splitted_data(split_strategy="final_weeks", split_size_w=4, week_selection=0, start_first_monday=False)

# example
print("Firs training day is "+str(train__dmas_h_q.index[0]))
print("This day is part of the "+str(dl.dataset_week_number(train__dmas_h_q.index[0]))+"th week of the dataset.")
print("First test day (at noon) is "+str(test__dmas_h_q.index[11]))
print("This day is part of the "+str(dl.dataset_week_number(test__dmas_h_q.index[11]))+"th week of the dataset.")
print("The first monday of week "+str(4)+" is "+str(dl.monday_of_week_number(4)))
print("So the week goes from "+str(dl.monday_of_week_number(4))+" to "+str(dl.monday_of_week_number(5)))
print("You could also use it as "+str(dl.monday_of_week_number(4)+pd.Timedelta(days=7) ) )

# you can also load other data like the calendar and dmas descrp
cal = dl.load_calendar()
print(cal)
dmas_chars = dl.load_characteristics()
print(dmas_chars)
print(dmas_chars.loc[:,"description"])

# You can use the evaluation framework to test your models

# You want the perfromance indicators? 
# Let's see at the perfromance of filling the test weeks with a number (42) to the fake ground truth (0)
# First one is ground truth (like in sckit learn)
print(ef.performance_indicator_1(test__dmas_h_q.iloc[0:24*7,:].fillna(0), test__dmas_h_q.iloc[0:24*7,:].fillna(42)))
print("You can also choose a specific DMA")
#print(ef.performance_indicator_1(test__dmas_h_q.iloc[0:24*7,3].fillna(0), test__dmas_h_q.iloc[0:24*7,3].fillna(42)))

# Want to test on all the PIs?
print(ef.performance_indicators(test__dmas_h_q.iloc[0:24*7,:].fillna(0), test__dmas_h_q.iloc[0:24*7,:].fillna(42)))
#This function also checks for correct sizes [24*7 X 10]!!!
# invalid: print(ef.performance_indicators(test__dmas_h_q.iloc[0:24*7*2,:].fillna(0), test__dmas_h_q.iloc[0:24*7*2,:].fillna(42)))

"""Now let's see how to use the evaluation framework to test your models"""
# To keep track of all the models and their results we use the EvaluationLogger
train__dmas_h_q, test__dmas_h_q, train__wea_h, test__wea_h = dl.load_splitted_data(split_strategy="final_weeks", split_size_w=4, week_selection=0, start_first_monday=False)
el = ef.EvaluationLogger(test__dmas_h_q.fillna(0))

## here youe train your model! 
# training(train__dmas_h_q, train__wea_h)
# test__dmas_h_q__predicted = predict(test__wea_h, train__dmas_h_q, train__wea_h)  
test__dmas_h_q__predicted = test__dmas_h_q.fillna(42)

# To test your model simply do
el.add_model_test("fill42", test__dmas_h_q__predicted)

# Print to screen but we can also save it somehow (e.g., in a file or excel)
print(el.m_results)