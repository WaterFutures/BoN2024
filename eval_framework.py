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
from sklearn.metrics import mean_absolute_error, max_error
import data_loader
import pandas as pd
from constants import DAY_LEN, WEEK_LEN
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import pickle
import os
import pathlib

def performance_indicator_1(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI1^d MAE: Mean Absolute Error in the first 24 hours of the week for DMA d.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA
    
    return mean_absolute_error(dmas_h_q_true[:24], dmas_h_q_pred[:24], multioutput='raw_values')

def performance_indicator_2(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI2^d MaxAE: Max Absolute Error in the first 24 hours of the week for DMA d.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA
    
    return np.amax(np.abs(dmas_h_q_true[:24] - dmas_h_q_pred[:24]), axis=0)

def performance_indicator_3(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> np.ndarray:
    """
    PI3^d MAE: Mean Absolute Error after the first 24 hours to the end of the week for DMA d.
    
    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0       # at least one DMA

    return mean_absolute_error(dmas_h_q_true[24:24*7], dmas_h_q_pred[24:24*7], multioutput='raw_values')

def performance_indicators(dmas_h_q_true: np.ndarray, dmas_h_q_pred: np.ndarray) -> dict[str, np.ndarray] :
    """
    Test the model on a single week.

    :param dmas_h_q_true: The true data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :param dmas_h_q_pred: The forecasted data for the week to test as a numpy array, with shape (24*7, n_dmas).
    :return: The result for each DMA as a numpy array with size n_dmas.
    """
    # Check that the forecast and actual data have the same shape and index
    assert dmas_h_q_true.shape == dmas_h_q_pred.shape
    assert dmas_h_q_true.shape[0] == 24*7   # 24 hours per day, 7 days per week
    assert dmas_h_q_true.shape[1] > 0      # at least one DMA

    # We decided to not include the nans in the counts 

    # Find the nans in the true data, then put thme to 0 both in the true and 
    # predicted data
    nans_true = np.isnan(dmas_h_q_true)
    dmas_h_q_true[nans_true] = 0
    dmas_h_q_pred[nans_true] = 0

    pi1 = performance_indicator_1(dmas_h_q_true, dmas_h_q_pred)
    pi2 = performance_indicator_2(dmas_h_q_true, dmas_h_q_pred)
    pi3 = performance_indicator_3(dmas_h_q_true, dmas_h_q_pred)

    return {'PI1':pi1, 'PI2':pi2, 'PI3':pi3,
            'n_nans_1d': nans_true[:24].sum(axis=0),
            'n_nans_w': nans_true.sum(axis=0)}

PIs = ['PI1', 'PI2', 'PI3', 'n_nans_1d', 'n_nans_w']

def ModelResults() -> dict:
    """
    Return a dictionary with all the necessary information to describe the whole 
    process to evaluate a model.

    Processed data used for training
        train and test, for each type of variable (dma consumption and exogenous), 
        eval, for exogenous variables always known at time t (e.g., weather)
    Model
    Validation results pandas dataframe with multiindex (test week, dma) and columns
        the performance indicators
    Test results pandas dataframe with multiindex (test week, dma) and columns
        the performance indicators
    BWDF forecast results pandas dataframe with index the week we need to forecast 
        and columns the DMAs
    
    :return: A dictionary with the structure of the results of a model.
    """
    return {
        "processed_data": {
            "train__dmas_h_q": None, 
            "test__dmas_h_q": None,
            "train__exin_h": None,
            "test__exin_h": None,
            "eval__exin_h": None
        },
        "model": None,
        "validation": None,
        "test": None,
        "bwdf_forecast": None
    }

def ProcessResults(eval_weeks: range, dmas: list[str]) -> pd.DataFrame:
    """
    Return a dataframe with the structure of the results of the evaluation over
    the trainig/validation or test set.
    It differs from ModelResults as this only keeps the scores for each week 
    and dma

    :param eval_weeks: The weeks to evaluate.
    :param dmas: The DMAs to evaluate.
    :return: A Pandas dataframe with a two level index (test week level 0; DMA
     level 1) and columns the perfromances indicators.
    """

    return pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(ew,dma) for ew in eval_weeks for dma in dmas], names=['Test week', 'DMA']),
            columns=PIs
        )

class WaterFuturesEvaluator:

    def __init__(self) -> None:
        (self.__train__dmas_h_q, self.__test__dmas_h_q, 
        self.__train__exin_h, self.__test__exin_h, self.__eval__exin_h) = data_loader.load_splitted_data(
            split_strategy="final_weeks", split_size_w=4, start_first_monday=True)
        self.__models_results = {} 
        self.app = None
        self.__results_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), 'wfe_results')
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

        first_split_week = 12 # I don't think it makes sense starting before this week
        # as 2 DMAS are full of nans until the 6th week

        absolute_week_shift = 1 # We said the week 0 (1st Jan 2021 to 4th jan 2021) is not used
        abs_test_weeks = range(first_split_week+absolute_week_shift, 
                           train__dmas_h_q.shape[0]//WEEK_LEN+absolute_week_shift)
        test_weeks = range(first_split_week, 
                           train__dmas_h_q.shape[0]//WEEK_LEN)
        
        results = ProcessResults(abs_test_weeks, model.forecasted_dmas())
        
        for split_week in test_weeks:
        
            train__df = pd.concat(
                [train__dmas_h_q.iloc[:split_week*WEEK_LEN,:], train__exin_h.iloc[:split_week*WEEK_LEN,:]],
                axis=1)
            
            model.fit(train__df)

            y_pred = model.forecast(train__df, train__exin_h.iloc[split_week*WEEK_LEN:(split_week+1)*WEEK_LEN,:])
            assert y_pred.shape[0] == 24*7
            assert y_pred.shape[1] == len(model.forecasted_dmas())
            assert not np.isnan(y_pred).any()
            
            dmas_h_q_true = self.__train__dmas_h_q.iloc[split_week*WEEK_LEN:(split_week+1)*WEEK_LEN, model.forecasted_dmas_idx()]
            y_true = dmas_h_q_true.to_numpy()

            # Store the results
            result = performance_indicators(y_true, y_pred)
            for pi in results.columns:
                assert pi in result.keys()
                results.loc[(split_week+absolute_week_shift,model.forecasted_dmas()), pi] = result[pi]
        
            fcst__dmas_h_q = pd.concat([fcst__dmas_h_q,
                                        pd.DataFrame(y_pred, 
                                                     index=dmas_h_q_true.index, 
                                                     columns=model.forecasted_dmas())
                                        ], axis=0)
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
        
        absolute_week_shift = data_loader.dataset_week_number(self.__test__dmas_h_q.index[0])
        abs_test_weeks = range(0+absolute_week_shift,
                           self.__test__dmas_h_q.shape[0]//WEEK_LEN+absolute_week_shift)
        test_weeks = range(0,
                           self.__test__dmas_h_q.shape[0]//WEEK_LEN)
        
        results = ProcessResults(abs_test_weeks, model.forecasted_dmas())

        for test_week in test_weeks:
            test__df = pd.concat([
                pd.concat([train__dmas_h_q, test__dmas_h_q.iloc[:test_week*WEEK_LEN, :] ], axis=0),
                pd.concat([train__exin_h, test__exin_h.iloc[:test_week*WEEK_LEN,:] ], axis=0)
                ], axis=1)
            
            y_pred = model.forecast(test__df, test__exin_h.iloc[test_week*WEEK_LEN:(test_week+1)*WEEK_LEN,:] )
            assert y_pred.shape[0] == 24*7
            assert y_pred.shape[1] == len(model.forecasted_dmas())
            
            dmas_h_q_true = self.__test__dmas_h_q.iloc[test_week*WEEK_LEN:(test_week+1)*WEEK_LEN, model.forecasted_dmas_idx()]
            y_true = dmas_h_q_true.to_numpy()

            # Store the results
            result = performance_indicators(y_true, y_pred)
            for pi in results.columns:
                assert pi in result.keys()
                results.loc[(test_week+absolute_week_shift,model.forecasted_dmas()), pi] = result[pi]
            
            fcst__dmas_h_q = pd.concat([fcst__dmas_h_q,
                                        pd.DataFrame(y_pred, 
                                                     index=dmas_h_q_true.index, 
                                                     columns=model.forecasted_dmas())
                                        ], axis=0)
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
        assert y_pred.shape[0] == 24*7
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
    
    def run_dashboard(self) -> None:
        """
        Run the dashboard to visualize the results.
        """
        self.app = Dash(__name__)

        self.app.layout = html.Div([
            html.H1(children='Water Futures Evaluator Dashboard', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Div(children='Select the performance indicator to visualize:'),
                    dcc.Dropdown(['PI1', 'PI2', 'PI3'], 'PI1', id='pi-dropdown'),
                    html.Div(children={}, id='pi-description')
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(children='Select the DMA to visualize:'),
                    dcc.Dropdown(data_loader.DMAS_NAMES, data_loader.DMAS_NAMES[0], id='dma-dropdown'),
                    html.Div(children={}, id='dma-description')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            dcc.Graph(id='graph-content'),
            dcc.Checklist(
                id="model-checklist",
                options=self.models_names(),
                value=[self.models_names()[0]],
                inline=True
            )
        ])

        @callback(
            Output('graph-content', 'figure'),
            Output('dma-description', 'children'),
            Output('pi-description', 'children'),
            Input('dma-dropdown', 'value'),
            Input('pi-dropdown', 'value'),
            Input('model-checklist', 'value')
        )
        def update_graph(dma, pi, model_names):
            fig = go.Figure()
            
            for model_name in model_names:
                vali_df = self.__models_results[model_name]["validation"]
                vali__weeks = vali_df.index.get_level_values(0).unique()
                vali__dma_pi = vali_df.xs(dma, level=1).loc[:,pi]
                fig.add_trace(go.Scatter(x=vali__weeks, y=vali__dma_pi, name=model_name, mode='lines'))
            
            fig.update_layout(title=dma, xaxis_title='Week', yaxis_title='MAE [L/s]')
            return fig, "DMA: {}".format(dma), "Performance Indicator: {}".format(pi)

        self.app.run(debug=True)