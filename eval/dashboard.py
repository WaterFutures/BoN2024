from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as suboplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Constants
DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']

DMA_DESCRIPTIONS = {
    'DMA_A': 'Hospital district', 
    'DMA_B': 'Residential district in the countryside', 
    'DMA_C': 'Residential district in the countryside',
    'DMA_D': 'Suburban residential/commercial district', 
    'DMA_E': 'Residential/commercial district close to the city centre',
    'DMA_F': 'Suburban district including sport facilities and office buildings', 
    'DMA_G': 'Residential district close to the city centre',
    'DMA_H': 'City centre district', 
    'DMA_I': 'Commercial/industrial district close to the port', 
    'DMA_J': 'Commercial/industrial district close to the port'
}

PI_DESCRIPTIONS = {
    'PI1': 'Mean Absolute Error of first day', 
    'PI2': 'Max Absolute error of first day', 
    'PI3': 'Mean Absolute Error of days 2-7'
}

PI_UNITS = {
    'PI1': 'MAE [L/s]', 
    'PI2': 'MaxAE [L/s]', 
    'PI3': 'MAE [L/s]'
}

def run_dashboard(wfe):
    app = Dash('WaterFutures Dashboard')
    app.layout = layout(wfe)

    @callback(
        Output('graph-content', 'figure'),
        Input('dma-dropdown', 'value'),
        Input('pi-dropdown', 'value'),
        Input('model-checklist', 'value'),
        Input('strategy-checklist', 'value')
    )
    def figures(dma, pi, model_names, strat_names):
        fig = suboplots.make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                      subplot_titles=(f"Performance for {dma} and {pi} of the selected models",
                                                      f"{dma} error on inflow trajectory for the selected models", 
                                                      f"{dma} inflow trajectory for the selected models",
                                                      f"Weather"))

        #fig_pi = go.Figure()
        for model_idx, model_name in enumerate(model_names):
            # Read current performance indicators

            # The structure is self.results[model_name][iter][phase][seed]['performance_indicators']-> pd.Dataframe(DMA, PI)
            # self.results[model_name][iter][phase][seed]['forecast'] -> pd.Dataframe(Date, DMA)

            # Start with the train phase
            df_list = []
            for seed in wfe.results[model_name]['iter_1']['train'].keys():
                df_list.append(wfe.results[model_name]['iter_1']['train'][seed]['performance_indicators'])
                
            # Concatenate all the seeds
            testres = pd.concat(df_list,
                           keys=range(len(df_list)), 
                            names=['Seed', 'Test week', 'DMA'])
            
            # Select DMA and PI
            testres = testres.loc[(slice(None), slice(None), dma), pi]
            # Here we have a result for each week so let's track the corresponding Monday
            testres_x = testres.index.get_level_values('Test week').unique().map(lambda x: wfe.weather.index[0]+pd.Timedelta(days=x*7))

            fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').median(),
                                     name='Median-PI', mode='lines', line=dict(color=px.colors.qualitative.Plotly[model_idx]),
                                     legendgroup=model_name, legendgrouptitle=dict(text=model_name)
                                     ), row=1, col=1)

            # For each iteration, only test and eval may appear... 

        for i, strategy in enumerate(strat_names):
            # Read current performance indicators

            # The structure is self.resstrategies[strategy][iter][phase]['performance_indicators']-> pd.Dataframe(DMA, PI)
            # self.resstrategies[strategy][iter][phase]['forecast'] -> pd.Dataframe(Date, DMA)
            testres = wfe.resstrategies[strategy]['iter_1']['train']['performance_indicators']
            # Select DMA and PI
            testres = testres.loc[(slice(None), dma), pi]
            # Here we have a result for each week so let's track the corresponding Monday
            testres_x = testres.index.get_level_values('Test week').unique().map(lambda x: wfe.weather.index[0]+pd.Timedelta(days=x*7))

            # Add like it was a forecasts, but use dashed lines and a different family of colors
            fig.add_trace(go.Scatter(x=testres_x, y = testres,
                                     name=strategy+'-PI', mode='lines', 
                                     line=dict(color=px.colors.qualitative.Pastel[i], dash='dash'),
                                     legendgroup=strategy, legendgrouptitle=dict(text=strategy)
                                    ), row=1, col=1)


        # y axis label
        fig.update_yaxes(title_text=PI_UNITS[pi], row=1, col=1)
    
        ### Trajectory error
        # Load and create trace for ground truth data
        demand_gt = wfe.demand[dma]
        weather_gt = wfe.weather

        # Add the demand first as its black and we want it to be on the bottom
        fig.add_trace(go.Scatter(x=demand_gt.index, y=demand_gt,
                                 name='Demand', mode='lines', line=dict(color='black'),
                                    legendgroup='Observations'
                                ), row=3, col=1)
        

        for model_idx, model_name in enumerate(model_names):
            # Load and create trace for forecasted data
            # first get the training data
            df_list = []
            for seed in wfe.results[model_name]['iter_1']['train'].keys():
                df_list.append(wfe.results[model_name]['iter_1']['train'][seed]['forecast'])

            # Concatenate all the seeds
            forecast = pd.concat(df_list,
                                keys=range(len(df_list)), 
                                names=['Seed', 'Date'])
            
            # Select DMA
            forecast = forecast[dma]

            error = forecast.copy()
            for seed in error.index.get_level_values('Seed').unique():
                error.loc[seed] = error.loc[seed].to_numpy() - demand_gt.loc[error.loc[seed].index].to_numpy()

            fig.add_trace(go.Scatter(x=error.index.get_level_values('Date').unique(), y=error.groupby('Date').median(),
                                     name='Median-err', mode='lines', line=dict(color=px.colors.qualitative.Plotly[model_idx]),
                                     legendgroup=model_name
                                    ), row=2, col=1)

            fig.add_trace(go.Scatter(x=forecast.index.get_level_values('Date').unique(), y=forecast.groupby('Date').median(),
                                     name='Median-fcst', mode='lines', line=dict(color=px.colors.qualitative.Plotly[model_idx]),
                                     legendgroup=model_name
                                    ), row=3, col=1)

        
        
        # add the weather
        fig.add_trace(go.Scatter(x=weather_gt.index, y=weather_gt['Temperature'],
                                    name='Temperature', mode='lines', line=dict(color='red'),
                                    legendgroup='Observations', legendgrouptitle=dict(text='Observations')
                                    ), row=4, col=1)
        fig.add_trace(go.Scatter(x=weather_gt.index, y=weather_gt['Rain'],
                                    name='Rain', mode='lines', line=dict(color='blue'),
                                        legendgroup='Observations'
                                    ), row=4, col=1)

        # Title and axes labels
        fig.update_xaxes(title_text="Time", matches='x', row=2, col=1)
        fig.update_yaxes(title_text="Inflow Error [L/s]", row=2, col=1)
        fig.update_xaxes(title_text="Time", matches='x', row=3, col=1)
        fig.update_yaxes(title_text="Inflow [L/s]", row=3, col=1)
        fig.update_xaxes(title_text="Time", matches='x', row=4, col=1)
        fig.update_yaxes(title_text="Temperature [°C]/Rain [mm/hour]", row=4, col=1)

        fig.update_layout(height=1500)
        
        return fig
    
    @callback(
        Output('dma-description', 'children'),
        Input('dma-dropdown', 'value')
    )
    def description_dma(dma):
        return DMA_DESCRIPTIONS[dma]
    
    @callback(
        Output('pi-description', 'children'),
        Input('pi-dropdown', 'value')
    )
    def description_pi(pi):
        return PI_DESCRIPTIONS[pi]

    @app.callback(
        Output("model-checklist", "options"),
        Input("model-checklist", "value"),
    )
    def max_model_selection(value):
        # Limit to as many simultaneous selections as we can visualize
        options = list(wfe.results.keys())
        if len(value) >= len(px.colors.qualitative.Plotly):
            options = [
                {
                    "label": option,
                    "value": option,
                    "disabled": option not in value,
                }
                for option in options
            ]
        return options

    @callback(
        Output('table', 'data'),
        Input('pi-dropdown', 'value'),
        Input('model-checklist', 'value')
    )
    def table(pi, models):
        return [table_row(model, wfe.results[model]['performance_indicators'], pi) for model in models]

    @callback(
        Output('table-scores', 'data'),
        Output('table-scores', 'columns'),
        Input('model-checklist', 'value')
    )
    def table_scores(models):
        # Start at 13 instead of 12 to be compatible with the ensembled files
        scores = calc_scores(models, wfe.results, range(13,77))
        columns = [{'name': model, 'id': model} for model in models]

        return [[scores], columns]
    
    @callback(
        Output('table-ranks', 'data'),
        Input('model-checklist', 'value')
    )
    def table_ranks(models):
        report = wfe.ranks_report(models)
        columns = [{'Model': model, **report.loc[model].to_dict()} for model in models]

        return columns
        
    app.run(debug=True)

def layout(wfe):
    models = list(wfe.results.keys())
    strategies = list(wfe.resstrategies.keys())

    return html.Div([
            html.H1(children='Water Futures Evaluator Dashboard', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Div(children='Select the performance indicator to visualize:'),
                    dcc.Dropdown(['PI1', 'PI2', 'PI3'], 'PI1', id='pi-dropdown'),
                    html.Div(children='', id='pi-description')
                ], style={'width': '30%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(children='Select the DMA to visualize:'),
                    dcc.Dropdown(DMAS_NAMES, DMAS_NAMES[0], id='dma-dropdown'),
                    html.Div(children='', id='dma-description')
                ], style={'width': '30%', 'display': 'inline-block'}),
                html.Div([ 
                    html.Div(children='Select the models to visualize:'),
                    dcc.Checklist(
                        id="model-checklist",
                        options=models,
                        value=[models[0]],
                        inline=True
                    ),
                    html.Div(children='Select the strategies to visualize:'),
                    dcc.Checklist(
                        id="strategy-checklist",
                        options=strategies,
                        value=[strategies[0]],
                        inline=True
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            html.H3(children='PI and Forecasts over Time', style={'textAlign': 'center'}),
            dcc.Graph(id='graph-content'),
            html.H3(children='Overview Table for current PI', style={'textAlign': 'center'}),
            dash_table.DataTable(                                 
                data=[],
                id='table',
                sort_action='native',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'DMA_A', 'id': 'DMA_A'},
                    {'name': 'DMA_B', 'id': 'DMA_B'},
                    {'name': 'DMA_C', 'id': 'DMA_C'},
                    {'name': 'DMA_D', 'id': 'DMA_D'},
                    {'name': 'DMA_E', 'id': 'DMA_E'},
                    {'name': 'DMA_F', 'id': 'DMA_F'},
                    {'name': 'DMA_G', 'id': 'DMA_G'},
                    {'name': 'DMA_H', 'id': 'DMA_H'},
                    {'name': 'DMA_I', 'id': 'DMA_I'},
                    {'name': 'DMA_J', 'id': 'DMA_J'},
                    {'name': 'Average', 'id': 'Average'},
                ]),
            html.H3(children='Who would win? - Current model scores (All weeks)', style={'textAlign': 'center'}),
            dash_table.DataTable(                                 
                data=[],
                id='table-scores',
                sort_action='native',
                columns=[]),
            html.H3(children='Ranks over all DMAs and all PIs', style={'textAlign': 'center'}),
            dash_table.DataTable(                                 
                data=[],
                id='table-ranks',
                sort_action='native',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'PI1', 'id': 'PI1', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'PI2', 'id': 'PI2', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'PI3', 'id': 'PI3', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_A', 'id': 'Rank_A', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_B', 'id': 'Rank_B', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_C', 'id': 'Rank_C', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_D', 'id': 'Rank_D', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_E', 'id': 'Rank_E', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_F', 'id': 'Rank_F', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_G', 'id': 'Rank_G', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_H', 'id': 'Rank_H', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_I', 'id': 'Rank_I', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rank_J', 'id': 'Rank_J', 'type':'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Average', 'id': 'Average', 'type':'numeric', 'format': {'specifier': '.2f'}},
                ]),
        ], style={'margin': '4em', 'margin-bottom': '10em'})

### Helper functions
def table_row(model_name, results, pi):
    means = results[pi].groupby('DMA').mean()
    stds = results[pi].groupby('DMA').std()

    dma_entries = {}
    for dma in means.keys():
        dma_entries[dma] = f'{means[dma]:.3f}+-{stds[dma]:.2f}'

    return {
        'Model': model_name,
        'Average': f'{np.mean(means):.3f}+-{np.mean(stds):.2f}',
        **dma_entries
    }

def calc_scores(models, results, weeks):
    all_indicators = []
    for model in models:
        all_indicators.append(results[model]['performance_indicators'].loc[weeks])

    all_indicators = np.stack(all_indicators)

    score_tuples = zip(*np.unique(all_indicators.argmin(axis=0), return_counts=True))
    scores = {models[model_idx]: model_score for model_idx, model_score in score_tuples}
    return scores
