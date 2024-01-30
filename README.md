# BoN2024
This repository contains the solution of the Water Futures team for the Battle of the Network Demand Forecasting, held in Ferarra between 4th and 7th of July 2024.
More information about the competition can be found on the conference website (WDSA/CCWI 2024)[].
Details of the Water-Futures project are available on the website (Water-Futures)[https://waterfutures.eu]. 

## The competition


## The team
The Water-Futures project aims to develop a theoretical basis for designing smart water systems, which can provide a framework for the allocation and development decisions on drinking water infrastructure systems. 
The Water-Futures team builds on synergies from four research groups, transcending methodologies from water science, systems and control theory, economics and decision science, and machine learning and is led by our four Principal Investigators: Marios Policarpou, Barbara Hammer, Phoebe Koundori, and Dragan Savic. 

Colleagues from all the four research groups came together to form a task force to compete in the challenge under a unique Water-Futures team.

## Concept

To leverage the different areas of expertise of the member of the group, we approach the forecasting problem with an ensemble of models. Every partner has the task of developing a number of models based on their knowledge, expertise and time. Then, we evaluate all these models under a common framework for training, validation and testing. A selection of the more performing models is calibrated during the final stage with the whole dataset available for the competition to forecast the evaluation week of the competition. 
The final result is an average of the 5 best performing models. 

## Structure of the repository
The developed models are classes that extend a common `Model` class and implement the `fit` and `forecast` methods. The models takes in input the training dataset of demand and weather and forecast the first week after the training dataset using the weather information of that week as a perfect forecast. All of the developed models are available in the *models* directory. 
The *eval* directory contains the `WaterFuturesEvaluator` class (*evaluator.py* file) and some helper functions for visualization and loading of the data and results (*dashboard.py* and *data_loading_helpers.py*). This class takes care of providing the correct dataset to the models during training, validation, testing and evaluation phases. Using a common class for all the models ensures the most fair comparison between all the models.
Inside you can find the *strategies* directory which contains the classes explaining the different reconciliation strategies between the models. A strategy is defined by having two methods: `find_best_models` and `combine_forecasts`. Only one startegy is then selected in the final step.
The folder *original_inspection* provides some jupyter notebooks to inspect the data and provide some information about that. 
The folder *preprocessing* contains a collection of functions that can be applied to the data before training is performed. 
The folder *utils* contains useful scripts and functions used in other modules.


## Installation 
See the file Installation.txt

## Execution
It is possible to run the execution with `water_futures.py` or with `water_futures.ipynb`. In both cases, select the correct environment as explained in *Installation.md* and just press run all! If the results are already available in the *data/results* folder the training and validation will not run again. 

## Visualisation 
Run water_futures_dash.py to see the results in a nice interactive dashboard using Plotly Dash.

## Acknowledgements
This project has received funding from the European Research Council (ERC) under the ERC Synergy Grant Water-Futures (Grant agreement No. 951424)

We thank the project partners University of Cyprus and the KIOS Reserarch Center of Excellence, Bielefeld University, Athens Univ. of Economics and Business, KWR Water Research Institute, and University of Exeter. 
Many thanks also to the academic supervisor and the universities collaborating with KWR and their PhDs, Politecnico di Milano, National Tehcnical University of Athens, and TU Delft. 