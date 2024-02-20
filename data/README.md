# Structure of the Data folder 

- **instructions**
This folder contains the original instructions given for the competition and available on the conference website [WDSA/CCWI2024](https://wdsa-ccwi2024.it/battle-of-water-networks/). 

- **original** 
Place in this folder the Excel files containg the input data. The *x* code in the name indicates the iteration of the competition (maximum 4) and it is used to upload the correct input file available at that time. 
	- InflowData\_*x*.xlsx 
	An Excel file with the date in the (DD/MM/YYYY HH:mm) format on the first column and the hourly net inflow for each DMA in the other columns.
	- WeatherData\_*x*.xlsx
	An Excel file with the date in the (DD/MM/YYYY HH:mm) format on the first column and the hourly measurements for the four weather variables in the following columns (in the order: Date - Rainfall depth - Air temperature - Air humidity - Windspeed ). It is expected to be one week longer (168 values) longer than the inflow file as the last week is used as perfect forecast for the forecasted week. 
	- Calendar.pdf 
	Just a pdf shared during the competition that is reporting the holidays in the town producing the data.

- **results**
	- **models**
		- **AModel**
			- **iter_i**
				- **phase**
					-`AModel__iter_i__phase__seed_s__.pkl`
	- **strategies**
		- **AStrategy**
			- **iter_i**
				- `AStrategy__iter_i__phase__.pkl`

The **results/models** folder contains the results of all the models we developed during the competition. Each model has its correspondent directory (**AModel**). Inside, there is a directory that aggregates the results for each iteration (in the string format `iter_i`). For each iteration, there are 3 possible phases: (i) train (only in first iteration, all models), where we used 52 weeks to compare *ALL* the developed models and decide which would progress to the testing phase, (ii) test (all iterations, only a handful of models that WE selected) where the *selected_models* are tested on the last 4 weeks of the dataset, and (iii) eval, that is the result of the model when trained on the complete dataset (available only for the models chosen by the strategy, i.e., *best_models*). Each pickle file is the result of a training in that iteration, for that phase and with the specific seed value written in the name. Once you upload the `.pkl` file, you will see a dictionary with two entries: (a) `'performance_indicators'`, only for train and test phase: a Pandas Dataframe with the 3 performance indicators on the columns and the tuple (week number, DMA) as index; (b) `'forecast'`, the hourly forecast of that model where, for each week, the model was trained on the whole dataset up to the previous week: a Pandas Dataframe with the hourly Date as index for the 10 DMAs in the columns.
The **results/strategies** folder contains the result of all the strategies to reconcile the ensemble of forecasts that we tried during the competition. Every strategy has its correspondent directory (**AStrategy**). Inside, as for the models, you will find a folder for each iteration, and for each phase there is `.pkl` file with the same dictionary object of the models (i.e., PIs and forecasts of each week). Differently from the models, all the strategies have been tried only in the train phase, and only the selected one (`avg_top5`) has been applied. Hence, only avg_top5 has an eval file containing the final forecast for the evaluation week i (the `.xlsx` contains the same forecast in a different format).

- **solutions**
This folder contains our solutions for the original competition in the templated Excel files. Feel free to use them as your benchmark!