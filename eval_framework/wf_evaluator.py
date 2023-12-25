
import numpy as np
from sklearn.metrics import mean_absolute_error, max_error
import data_loader
import pandas as pd
from constants import DAY_LEN, WEEK_LEN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input, dash_table
from dash.dash_table.Format import Format, Scheme
import pickle
import os
import pathlib
from .performance_indicators import performance_indicators, performance_indicators_long_names, performance_indicators_labels, create_report
from .result_structures import ModelResults, ProcessResults
import tqdm


class WaterFuturesEvaluator:

    def __init__(self) -> None:
        (self.__raw__dmas_h_q, self.__raw__exin_h, self.__eval_wea_h, self.test_weeks) = data_loader.load_splitted_data(
            split_strategy="all_weeks", split_size_w=10, start_first_monday=True)
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
        evaluation_results = ProcessResults(self.test_weeks, model.forecasted_dmas())
        for split_week in tqdm.tqdm(self.test_weeks):

            __train__dmas_h_q = self.__raw__dmas_h_q[self.__raw__dmas_h_q.no_week < split_week]
            __train__exin_h = self.__raw__exin_h[self.__raw__exin_h.no_week < split_week]
            __test__exin_h = self.__raw__exin_h[self.__raw__exin_h.no_week == split_week]

            (df_train, df_test) = model.preprocess_data(
                __train__dmas_h_q, __train__exin_h, __test__exin_h)

            self.__models_results[model.name()]["processed_data"]["df_train"] = df_train
            self.__models_results[model.name()]["processed_data"]["df_test"] = df_test

            self.__models_results[model.name()]["model"] = model

            result, _ = self.train(model, split_week)
            for pi in evaluation_results.columns:
                assert pi in result.keys()
                evaluation_results.loc[(split_week,model.forecasted_dmas()), pi] = result[pi]

        self.__models_results[model.name()]["validation"] = evaluation_results

        self.__models_results[model.name()]["bwdf_forecast"] = self.bwdf_forecast(model)

        cur_file_path = os.path.join(self.__results_folder, f'{model.name()}.pkl')
        with open(cur_file_path, 'wb') as f:
            pickle.dump(self.__models_results[model.name()], f)


    def train(self, model, split_week) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train the model on all the weeks in the training set.

        :param model: The model to train.
        :return: A Pandas dataframe with index the DMAs and columns the perfromances indicators.
        """
        df_train = self.__models_results[model.name()]["processed_data"]["df_train"]
        df_test = self.__models_results[model.name()]["processed_data"]["df_test"]

        model.fit(df_train)

        y_pred = model.forecast(df_test)
        assert y_pred.shape[0] == WEEK_LEN
        assert y_pred.shape[1] == len(model.forecasted_dmas())
        assert not np.isnan(y_pred).any()

        dmas_h_q_true = self.__raw__dmas_h_q.loc[self.__raw__dmas_h_q.no_week == split_week, model.forecasted_dmas()]
        y_true = dmas_h_q_true.to_numpy()

        # Store the results
        result = performance_indicators(y_true, y_pred)
        y_pred = pd.DataFrame(y_pred,
                     index=dmas_h_q_true.index,
                     columns=model.forecasted_dmas())

        return result, y_pred


    def bwdf_forecast(self, model) -> pd.DataFrame:
        """
        Forecast the model on all the weeks in the BWDF dataset.

        :param model: The model to forecast.
        :return: A Pandas dataframe with index the DMAs and columns the forecasted dmas.
        """
        __train__dmas_h_q = self.__raw__dmas_h_q
        __train__exin_h = self.__raw__exin_h
        __test__exin_h = self.__eval_wea_h

        (df_train, df_test) = model.preprocess_data(
            __train__dmas_h_q, __train__exin_h, __test__exin_h)

        y_pred = model.forecast(df_test)
        assert y_pred.shape[0] == WEEK_LEN
        assert y_pred.shape[1] == len(model.forecasted_dmas())

        return pd.DataFrame(y_pred, index=self.__eval_wea_h.index, columns=model.forecasted_dmas())


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
                    dcc.Dropdown(['PI1', 'PI2', 'PI3','avg_PIs'], 'PI1', id='pi-dropdown'),
                    html.Div(children={}, id='pi-description')
                ], style={'width': '30%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(children='Select the DMA to visualize:'),
                    dcc.Dropdown(data_loader.DMAS_NAMES + ['avg_DMAs'], data_loader.DMAS_NAMES[0], id='dma-dropdown'),
                    html.Div(children={}, id='dma-description')
                ], style={'width': '30%', 'float': 'left', 'display': 'inline-block'}),
                html.Div(children={}, id='errors-description',
                         style={'width': '40%', 'float': 'right', 'display': 'inline-block'})
            ]),
            html.Div([
                html.Div([
                    html.Div(children='Select the models to visualize:'),
                    dcc.Checklist(
                        id="model-checklist",
                        options=self.models_names(),
                        value=[self.models_names()[0]],
                        inline=True
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='graph-content',style={'textAlign': 'center','font_size': '26px',}),
            html.H2(style={"margin-left": "15px"}),
            html.H4('Rankings of Models', style={'font-weight': 'bold','textAlign': 'center'}),
            dash_table.DataTable(
                data=[],
                id='ranks-table',
                sort_action='native',
                columns=[
                    {'name': 'Models', 'id': 'Models', 'type': 'text'},
                    {'name': 'Rank by P1', 'id': 'Rank_P1', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by P2', 'id': 'Rank_P2', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by P3', 'id': 'Rank_P3', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA A', 'id': 'Rank_A', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA B', 'id': 'Rank_B', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA C', 'id': 'Rank_C', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA D', 'id': 'Rank_D', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA E', 'id': 'Rank_E', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA F', 'id': 'Rank_F', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA G', 'id': 'Rank_G', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA H', 'id': 'Rank_H', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA I', 'id': 'Rank_I', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Rank by DMA J', 'id': 'Rank_J', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Average', 'id': 'Average', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'width': 'auto',
                    'lineHeight': '15px'
                },
                style_cell={
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden'
                },
                style_data_conditional=[]),
            html.H2(style={"margin-left": "15px"}),
            html.H4('Average Scores across DMAs', style={'font-weight': 'bold','textAlign': 'center'}),
            dash_table.DataTable(
                data=[],
                id='avg-table',
                sort_action='native',
                columns=[
                    {'name': 'Models', 'id': 'Models', 'type': 'text'},
                    {'name': 'DMA_A', 'id': 'DMA_A', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_B', 'id': 'DMA_B', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_C', 'id': 'DMA_C', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_D', 'id': 'DMA_D', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_E', 'id': 'DMA_E', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_F', 'id': 'DMA_F', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_G', 'id': 'DMA_G', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_H', 'id': 'DMA_H', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_I', 'id': 'DMA_I', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_J', 'id': 'DMA_J', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Average', 'id': 'Average', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'width': 'auto',
                    'lineHeight': '15px'
                },
                style_cell={
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden'
                },
                style_data_conditional=[]),
            html.H2(style={"margin-left": "15px"}),
            html.H4('Standard deviation across DMAs', style={'font-weight': 'bold','textAlign': 'center'}),
            dash_table.DataTable(
                data=[],
                id='std-table',
                sort_action='native',
                columns=[
                    {'name': 'Models', 'id': 'Models', 'type': 'text'},
                    {'name': 'DMA_A', 'id': 'DMA_A', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_B', 'id': 'DMA_B', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_C', 'id': 'DMA_C', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_D', 'id': 'DMA_D', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_E', 'id': 'DMA_E', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_F', 'id': 'DMA_F', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_G', 'id': 'DMA_G', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_H', 'id': 'DMA_H', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_I', 'id': 'DMA_I', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'DMA_J', 'id': 'DMA_J', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                    {'name': 'Average', 'id': 'Average', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.fixed)},
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'width': 'auto',
                    'lineHeight': '15px'
                },
                style_cell={
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden'
                },
                style_data_conditional=[]),
            html.H2(style={"margin-left": "15px"}),
            html.H4('% of times the model had the best performance', style={'font-weight': 'bold','textAlign': 'center'}),
            dash_table.DataTable(
                data=[],
                id='pct-table',
                sort_action='native',
                columns=[
                    {'name': 'Models', 'id': 'Models', 'type': 'text'},
                    {'name': 'DMA_A', 'id': 'DMA_A', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_B', 'id': 'DMA_B', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_C', 'id': 'DMA_C', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_D', 'id': 'DMA_D', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_E', 'id': 'DMA_E', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_F', 'id': 'DMA_F', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_G', 'id': 'DMA_G', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_H', 'id': 'DMA_H', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_I', 'id': 'DMA_I', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'DMA_J', 'id': 'DMA_J', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                    {'name': 'Average', 'id': 'Average', 'type': 'numeric', 'format':Format(precision=2, scheme=Scheme.percentage_rounded)},
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'width': 'auto',
                    'lineHeight': '15px'
                },
                style_cell={
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden'
                },
                style_data_conditional=[]),
            html.H2(style={"margin-left": "15px"}),

                html.Div([
                    html.Div(children='Select the Model Description to read:'),
                    dcc.Dropdown(self.models_names(), self.models_names()[0], id='model-description-dropdown'),
                    html.Div(children={}, id='model-description')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),

        ])

        @callback(

            Output('graph-content', 'figure'),
            Output('ranks-table', 'data'),
            Output('ranks-table', 'style_data_conditional'),
            Output('avg-table', 'data'),
            Output('avg-table', 'style_data_conditional'),
            Output('std-table', 'data'),
            Output('std-table', 'style_data_conditional'),
            Output('pct-table', 'data'),
            Output('pct-table', 'style_data_conditional'),
            Output('dma-description', 'children'),
            Output('pi-description', 'children'),
            Output('errors-description', 'children'),

            Input('dma-dropdown', 'value'),
            Input('pi-dropdown', 'value'),
            Input('model-checklist', 'value'),

        )
        def update_dash(dma, pi, model_names):
            fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Performance for {dma} and {pi} for the selected models", f"{dma} BWD forecast for the selected models"))

            i = 0
            # limit the number of models to show to the number of colors that can be used
            message = ""
            if len(model_names) > len(px.colors.qualitative.Plotly):
                model_names = model_names[:len(px.colors.qualitative.Plotly)]
                message = "Too many models selected, only the first {} will be shown.\n".format(len(px.colors.qualitative.Plotly))

            avg_statistics, std_statistics, pct_statistics, table = create_report(self.__models_results, model_names, self.test_weeks, data_loader.DMAS_NAMES, pi)

            for model_name in model_names:
                vali__df = self.__models_results[model_name]["validation"]
                vali__weeks = vali__df.index.get_level_values(0).unique().to_numpy()
                if dma == 'avg_DMAs':
                    g = vali__df.index.get_level_values('Test week')
                    vali__dma_pi = vali__df.groupby(g).mean()[pi].to_numpy()
                else:
                    vali__dma_pi = vali__df.xs(dma, level=1).loc[:,pi].to_numpy()


                # add the line to the plot and the 4 points indepentently
                fig.add_trace(go.Scatter(x=vali__weeks, y=vali__dma_pi, name=model_name,
                                         mode='lines', line=dict(color=px.colors.qualitative.Plotly[i])), row=1, col=1)

                if dma != 'avg_DMAs':
                    fig.add_trace(go.Scatter(x=self.__models_results[model_name]["bwdf_forecast"][dma].index, y=self.__models_results[model_name]["bwdf_forecast"][dma].values, name=model_name,
                                         mode='lines', line=dict(color=px.colors.qualitative.Plotly[i]), showlegend=False), row=2, col=1)

                fig.update_layout(
                    legend=dict(
                        x=0.7,
                        y=1.11,
                        orientation="h",
                        traceorder="normal",
                        font=dict(
                            family="sans-serif",
                            size=12,
                            color="black"
                        ),
                    ),
                    autosize=True,
                    height=700,
                )
                i += 1

            fig.update_layout(title="Performance and BWD forecast for the selected models",
                              xaxis_title='Week', yaxis_title=performance_indicators_labels[pi])


            rank_style_formating = [
                {
                    'if': {
                        'column_type': 'text'  # 'text' | 'any' | 'datetime' | 'numeric'
                    },
                    'textAlign': 'left'
                }] + [
                {
                    'if': {
                        'column_id': col,

                        # since using .format, escape { with {{
                        'filter_query': '{{{}}} = "{}"'.format(col, table[col].min())
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                } for col in table.columns[1:].tolist()
                ]
            avg_style_formating = [
                {
                    'if': {
                        'column_type': 'text'  # 'text' | 'any' | 'datetime' | 'numeric'
                    },
                    'textAlign': 'left'
                }] + [
                {
                    'if': {
                        'column_id': col,

                        # since using .format, escape { with {{
                        'filter_query': '{{{}}} = "{}"'.format(col, avg_statistics[col].min())
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                } for col in avg_statistics.columns[1:].tolist()
                ]
            std_style_formating = [
                {
                    'if': {
                        'column_type': 'text'  # 'text' | 'any' | 'datetime' | 'numeric'
                    },
                    'textAlign': 'left'
                }] + [
                {
                    'if': {
                        'column_id': col,

                        # since using .format, escape { with {{
                        'filter_query': '{{{}}} = "{}"'.format(col, std_statistics[col].min())
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                } for col in std_statistics.columns[1:].tolist()
                ]
            pct_style_formating = [
                {
                    'if': {
                        'column_type': 'text'  # 'text' | 'any' | 'datetime' | 'numeric'
                    },
                    'textAlign': 'left'
                }] + [
                {
                    'if': {
                        'column_id': col,

                        # since using .format, escape { with {{
                        'filter_query': '{{{}}} = "{}"'.format(col, pct_statistics[col].max())
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                } for col in pct_statistics.columns[1:].tolist()
                ]

            return [fig,
                    table.to_dict('records'),
                    rank_style_formating,
                    avg_statistics.to_dict('records'),
                    avg_style_formating,
                    std_statistics.to_dict('records'),
                    std_style_formating,
                    pct_statistics.to_dict('records'),
                    pct_style_formating,
                    "DMA: {}".format(dma),
                    performance_indicators_long_names[pi],
                    message]

        self.app.run(debug=False)
