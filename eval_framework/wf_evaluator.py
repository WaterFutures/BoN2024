"""
In this file, we will implement the evaluation framework for the Battle of Water
Demand Forecasting project.

3 metrics are used:
- PI1_w^d MAE: Mean Absolute Error in the first 24 hours of week w for DMA d
- PI2_w^d MaxAE: Maximum Absolute Error in the first 24 hours of week w for DMA d
- PI3_w^d MAE: Mean Absolute Error after the first 24 hours till the end of the 
    week w for DMA d

In the evaluation framework, for each model we will have a matrix of values for 
metric and DMA. 

Simply measuring the metrics is not enough, we need to see if they are better 
than a benchmark model (e.g. ARIMA, 
previous week, same week year before)

Moreover, we can think of testing our models on several weeks (e.g., the final 
ones, random weeks, same week year before). 

I would suggest to produce a matrix of results for each model with 
Px_tw^d as the score x of test week tw for DMA d (see above)

This will give us a 3 dimensional matrix (metric, test week, DMA) for each model
 and for each benchmark.

Then we do a comparison between the benchmarks models, choose a benchmark for each 
test week, DMA and score for example removing the not meaningful combinations 
(e.g., same week year before to forecast the test week)

Finally, we can compare the models with the benchmark and see if they are better
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import pickle
import os
import pathlib

from constants import DAY_LEN, WEEK_LEN
import data_loader
from .performance_indicators import performance_indicators, performance_indicators_long_names, performance_indicators_labels
from .result_structures import ModelResults, ProcessResults
from .dashb_funcs import wf_dashboard_layout, figure_perf_ind_trajectory, figure_dma_q_trajectory

class WaterFuturesEvaluator:

    def __init__(self) -> None:
        self.__first_split_week = 12 # I don't think it makes sense starting before this week 
        # as 2 DMAS are full of nans until the 6th week. Relative number from the beginning of the train object!
        self.__n_test_weeks = 4 # Relative number!

        (self.__train__dmas_h_q, self.__test__dmas_h_q, 
        self.__train__exin_h, self.__test__exin_h, self.__eval__exin_h) = data_loader.load_splitted_data(
            split_strategy="final_weeks", split_size_w=self.__n_test_weeks, start_first_monday=True)

        self.__models_results = {} 
        self.app = None
        self.__results_folder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'wfe_results')
        if not os.path.exists(self.__results_folder):
            os.makedirs(self.__results_folder)
        self.load_saved_results()


    def load_saved_results(self):
        """
        Load results already saved in results folder.
        """
        files = os.listdir(self.__results_folder)
        for cur_file in files:
            cur_file_path = os.path.join(self.__results_folder, cur_file)
            cur_model_name = '.'.join(cur_file.split('.')[:-1])
            with open(cur_file_path, 'rb') as f:
                self.__models_results[cur_model_name] = pickle.load(f)
   
    def add_model(self, model, force=False) -> None:
        """
        Add a model to the evaluator.

        :param model: The model to add.
        :param force: Force re-compute the model even if it's results already exist.
        """

        # Check force condition and skip computation if desired
        if (not force) and (model.name() in self.__models_results.keys()):
            return
        
        self.__models_results[model.name()] = ModelResults()

        (train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h) = model.preprocess_data(
            self.__train__dmas_h_q, self.__test__dmas_h_q, self.__train__exin_h, self.__test__exin_h, self.__eval__exin_h)
        
        self.__models_results[model.name()]["processed_data"]["train__dmas_h_q"] = train__dmas_h_q
        self.__models_results[model.name()]["processed_data"]["test__dmas_h_q"] = test__dmas_h_q
        self.__models_results[model.name()]["processed_data"]["train__exin_h"] = train__exin_h
        self.__models_results[model.name()]["processed_data"]["test__exin_h"] = test__exin_h
        self.__models_results[model.name()]["processed_data"]["eval__exin_h"] = eval__exin_h
        self.__models_results[model.name()]["processed_data"]["fcst__dmas_h_q"] = None

        self.__models_results[model.name()]["model"] = model

        training_results, fcst__dmas_h_q = self.train(model)
        self.__models_results[model.name()]["validation"] = training_results
        self.__models_results[model.name()]["processed_data"]["fcst__dmas_h_q"] = fcst__dmas_h_q

        test_results, fcst__dmas_h_q = self.evaluate(model)
        self.__models_results[model.name()]["test"] = test_results
        self.__models_results[model.name()]["processed_data"]["fcst__dmas_h_q"] = fcst__dmas_h_q

        self.__models_results[model.name()]["bwdf_forecast"] = self.bwdf_forecast(model)

        cur_file_path = os.path.join(self.__results_folder, f'{model.name()}.pkl')
        with open(cur_file_path, 'wb') as f:
            pickle.dump(self.__models_results[model.name()], f)


    def train(self, model) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train the model on all the weeks in the training set.

        :param model: The model to train.
        :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
        """
        train__dmas_h_q = self.__models_results[model.name()]["processed_data"]["train__dmas_h_q"]
        train__exin_h = self.__models_results[model.name()]["processed_data"]["train__exin_h"]
        fcst__dmas_h_q = self.__models_results[model.name()]["processed_data"]["fcst__dmas_h_q"]

        abs_test_weeks = range(self.first_split_absweek(), 
                                self.last_train_absweek()+1)
        results = ProcessResults(abs_test_weeks, model.forecasted_dmas())
        
        absolute_week_shift = 1 # We said the week 0 (1st Jan 2021 to 4th jan 2021) is not used
        vali_weeks = range(self.first_split_week(),  # this is a relative number 
                           train__dmas_h_q.shape[0]//WEEK_LEN)
        
        for vali_week in vali_weeks:
            vali_absweek = vali_week + absolute_week_shift

            train__df = pd.concat(
                [train__dmas_h_q.iloc[:vali_week*WEEK_LEN,:], train__exin_h.iloc[:vali_week*WEEK_LEN,:]],
                axis=1)
            
            model.fit(train__df)

            y_pred = model.forecast(train__df, train__exin_h.iloc[vali_week*WEEK_LEN:(vali_week+1)*WEEK_LEN,:])
            assert y_pred.shape[0] == WEEK_LEN
            assert y_pred.shape[1] == len(model.forecasted_dmas())
            assert not np.isnan(y_pred).any()
            
            dmas_h_q_true = self.__train__dmas_h_q.iloc[vali_week*WEEK_LEN:(vali_week+1)*WEEK_LEN, model.forecasted_dmas_idx()]
            y_true = dmas_h_q_true.to_numpy()

            # Store the results (first create a copy of the forecasted week 
            # before being modified in the performance indicator function)
            fcst__dmas_h_q = pd.concat([fcst__dmas_h_q,
                                        pd.DataFrame(y_pred, 
                                                     index=dmas_h_q_true.index, 
                                                     columns=model.forecasted_dmas())
                                        ], axis=0)
            result = performance_indicators(y_true, y_pred)
            for pi in results.columns:
                assert pi in result.keys()
                results.loc[(vali_absweek,model.forecasted_dmas()), pi] = result[pi]

        return results, fcst__dmas_h_q


    def evaluate(self, model) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the model on all the weeks in the test set.

        :param model: The model to evaluate.
        :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
        """
        train__dmas_h_q = self.__models_results[model.name()]["processed_data"]["train__dmas_h_q"]
        train__exin_h = self.__models_results[model.name()]["processed_data"]["train__exin_h"]
        test__dmas_h_q = self.__models_results[model.name()]["processed_data"]["test__dmas_h_q"]
        test__exin_h = self.__models_results[model.name()]["processed_data"]["test__exin_h"]
        fcst__dmas_h_q = self.__models_results[model.name()]["processed_data"]["fcst__dmas_h_q"]
        
        abs_test_weeks = range(self.first_test_absweek(),
                           self.last_test_absweek()+1)
        results = ProcessResults(abs_test_weeks, model.forecasted_dmas())

        absolute_week_shift = data_loader.dataset_week_number(self.__test__dmas_h_q.index[0])
        test_weeks = range(0,
                           self.__test__dmas_h_q.shape[0]//WEEK_LEN)
        
        for test_week in test_weeks:
            test_absweek = test_week + absolute_week_shift

            test__df = pd.concat([
                pd.concat([train__dmas_h_q, test__dmas_h_q.iloc[:test_week*WEEK_LEN, :] ], axis=0),
                pd.concat([train__exin_h, test__exin_h.iloc[:test_week*WEEK_LEN,:] ], axis=0)
                ], axis=1)
            
            y_pred = model.forecast(test__df, test__exin_h.iloc[test_week*WEEK_LEN:(test_week+1)*WEEK_LEN,:] )
            assert y_pred.shape[0] == WEEK_LEN
            assert y_pred.shape[1] == len(model.forecasted_dmas())
            assert not np.isnan(y_pred).any()
            
            dmas_h_q_true = self.__test__dmas_h_q.iloc[test_week*WEEK_LEN:(test_week+1)*WEEK_LEN, model.forecasted_dmas_idx()]
            y_true = dmas_h_q_true.to_numpy()

            # Store the results (first create a copy of the forecasted week 
            # before being modified in the performance indicator function)
            fcst__dmas_h_q = pd.concat([fcst__dmas_h_q,
                                        pd.DataFrame(y_pred, 
                                                     index=dmas_h_q_true.index, 
                                                     columns=model.forecasted_dmas())
                                        ], axis=0)
            result = performance_indicators(y_true, y_pred)
            for pi in results.columns:
                assert pi in result.keys()
                results.loc[(test_absweek,model.forecasted_dmas()), pi] = result[pi]
            
        return results, fcst__dmas_h_q

    def bwdf_forecast(self, model) -> pd.DataFrame:
        """
        Forecast the model on all the weeks in the BWDF dataset.

        :param model: The model to forecast.
        :return: A Pandas dataframe with index the DMAs and columns the forecasted dmas.
        """
        train__dmas_h_q = self.__models_results[model.name()]["processed_data"]["train__dmas_h_q"]
        train__exin_h = self.__models_results[model.name()]["processed_data"]["train__exin_h"]
        test__dmas_h_q = self.__models_results[model.name()]["processed_data"]["test__dmas_h_q"]
        test__exin_h = self.__models_results[model.name()]["processed_data"]["test__exin_h"]
        eval__exin_h = self.__models_results[model.name()]["processed_data"]["eval__exin_h"]

        complete__df = pd.concat([
            pd.concat([train__dmas_h_q, test__dmas_h_q], axis=0),
            pd.concat([train__exin_h, test__exin_h], axis=0)
        ], axis=1)

        y_pred = model.forecast(complete__df, eval__exin_h)
        assert y_pred.shape[0] == WEEK_LEN
        assert y_pred.shape[1] == len(model.forecasted_dmas())
        
        return pd.DataFrame(y_pred, index=self.__eval__exin_h.index, columns=model.forecasted_dmas())


    def results(self) -> dict:
        """
        Return the results of the evaluation.

        :return: A dictionary with the results of the evaluation.
        """
        return self.__models_results
    
    def result(self, model_name: str) -> dict:
        """
        Return the results of the evaluation for a specific model.

        :return: A dictionary with the results of the evaluation for the model.
        """
        return self.__models_results[model_name]
    
    def models_names(self) -> list[str]:
        """
        Return the names of the models evaluated.

        :return: A list with the names of the models evaluated.
        """
        return list(self.__models_results.keys())
    
    def first_split_week(self) -> int:
        return self.__first_split_week
    
    def first_split_absweek(self) -> int:
        return self.__first_split_week+1
    
    def last_train_absweek(self) -> int:
        return data_loader.dataset_week_number(self.__train__dmas_h_q.index[-1])
        
    def first_test_absweek(self) -> int:
        return data_loader.dataset_week_number(self.__test__dmas_h_q.index[0])
    
    def last_test_absweek(self) -> int:
        return data_loader.dataset_week_number(self.__test__dmas_h_q.index[-1])
        
    def n_test_weeks(self) -> int:
        return self.__n_test_weeks
    
    def fcst_absweek(self) -> int:
        return data_loader.dataset_week_number(self.__eval__exin_h.index[0])

    def run_dashboard(self) -> None:
        """
        Run the dashboard to visualize the results.
        """
        self.app = Dash(__name__)

        self.app.layout = wf_dashboard_layout(self)

        @callback(
            Output('graph-content', 'figure'),
            Output('dma-description', 'children'),
            Output('pi-description', 'children'),
            Output('errors-description', 'children'),
            Output('trajectory-content', 'figure'),
            Input('dma-dropdown', 'value'),
            Input('pi-dropdown', 'value'),
            Input('model-checklist', 'value'),
            Input('trajectory-slider', 'value')
        )
        def update_dash(dma, pi, model_names, wrange):
            
            # limit the number of models to show to the number of colors that can be used
            message = ""
            if len(model_names) > len(px.colors.qualitative.Plotly):
                model_names = model_names[:len(px.colors.qualitative.Plotly)]
                message = "Too many models selected, only the first {} will be shown.\n".format(len(px.colors.qualitative.Plotly))

            figpi = figure_perf_ind_trajectory(self, dma, pi, model_names)
            
            figtraj = figure_dma_q_trajectory(self, dma, pi, model_names, wrange)

            return [figpi, 
                    data_loader.load_characteristics().loc[dma,'description'], 
                    performance_indicators_long_names[pi],
                    message,
                    figtraj]

        self.app.run(debug=True)