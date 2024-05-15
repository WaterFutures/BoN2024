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
        # to be transformed in inputs later
        show_train = True
        qt = 0.2

        fig = suboplots.make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                      subplot_titles=(f"Performance for {dma} and {pi} of the selected models",
                                                      f"{dma} error on inflow trajectory for the selected models", 
                                                      f"{dma} inflow trajectory for the selected models",
                                                      f"Weather"),
                                        vertical_spacing=0.05)

        for model_idx, model_name in enumerate(model_names):
            gcolor = px.colors.qualitative.Plotly[model_idx]
            gname = model_name
            # Read current performance indicators

            # The structure is self.results[model_name][iter][phase][seed]['performance_indicators']-> pd.Dataframe(DMA, PI)
            # self.results[model_name][iter][phase][seed]['forecast'] -> pd.Dataframe(Date, DMA)

            # Start with the train phase
            if show_train:
                gname = model_name+' (Train)'
                df_list = []
                for seed in wfe.results[model_name]['iter_1']['train'].keys():
                    df_list.append(wfe.results[model_name]['iter_1']['train'][seed]['performance_indicators'])
                    
                # Concatenate all the seeds
                testres = pd.concat(df_list,
                            keys=range(len(df_list)), 
                                names=['Seed', 'Test week', 'DMA'])
                
                # Select DMA and PI
                testres = testres.loc[(slice(None), slice(None), dma), pi]
                #drop DMA level
                testres = testres.droplevel(2)
                
                # Here we have a result for each week so let's track the corresponding Monday
                testres_x = testres.index.get_level_values('Test week').unique().map(lambda x: wfe.weather.index[0]+pd.Timedelta(days=x*7))
                
                fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').median(),
                                        name='Median-PI', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, legendgrouptitle=dict(text=gname), showlegend=True
                                        ), row=1, col=1)
                
                if testres.index.get_level_values('Seed').unique().shape[0] > 1 and qt > 0:
                    fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').quantile(0.5 + qt),
                                            name='Upper-PI', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname,
                                            fill=None, showlegend=False
                                            ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').quantile(0.5 - qt),
                                            name='Uncertainty', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname, showlegend=True,
                                            #fill='tonexty', fillcolor=gcolor
                                            ), row=1, col=1)
                gname = model_name # reset

            # Now the test and eval phase for all iterations
            df_list = []
            for l__iter in wfe.results[model_name].keys():
                phase = 'test'
                if phase in wfe.results[model_name][l__iter]:
                    seed_list = []
                    for seed in wfe.results[model_name][l__iter][phase].keys():
                        seed_list.append(wfe.results[model_name][l__iter][phase][seed]['performance_indicators'])
                        # Concatenate all the seeds
                    df_list.append(pd.concat(seed_list,
                                        keys=range(len(seed_list)), 
                                        names=['Seed', 'Test week', 'DMA']))
            
            testres = pd.concat(df_list)
            
            # Select DMA and PI
            testres = testres.loc[(slice(None), slice(None), dma), pi]
            #drop DMA level
            testres = testres.droplevel(2)

            # resample the weeks between test where we didn't have data so I can put nan and have a hole in the plot
            # Function to fill in the DataFrame for each group
            def missing_weeks(series):
                missing_idx_list = []

                # Under the assumption that every iteration could have a differen number of seeds...
                for seed in series.index.get_level_values('Seed').unique():
                    min_tw = series.loc[seed].index.get_level_values('Test week').min()
                    max_tw = series.loc[seed].index.get_level_values('Test week').max()

                    missing_tw = set(range(min_tw, max_tw+1)) - set(series.loc[seed].index.get_level_values('Test week').unique())
                
                    for tw in missing_tw:
                        missing_idx_list.append((seed, tw))

                # Create a new Series with NaNs for missing indices
                missing_series = pd.Series(np.nan, index=pd.MultiIndex.from_tuples(missing_idx_list, names=series.index.names))
                
                # Combine the original series with the new one
                # This will automatically fill in NaN for the new indices
                combined_series = pd.concat([series, missing_series]).sort_index()
                
                return combined_series
            testres = missing_weeks(testres)
            
            # Here we have a result for each week so let's track the corresponding Monday
            testres_x = testres.index.get_level_values('Test week').unique().map(lambda x: wfe.weather.index[0]+pd.Timedelta(days=x*7))
            
            fig.add_trace(go.Scatter
                            (x=testres_x, y=testres.groupby('Test week').median(),
                            name='Median-PI', mode='lines', line=dict(color=gcolor),
                            legendgroup=gname, legendgrouptitle=dict(text=gname), showlegend=True
                            ), row=1, col=1)
            
            if testres.index.get_level_values('Seed').unique().shape[0] > 1 and qt > 0:
                fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').quantile(0.5 + qt),
                                        name='Upper-PI', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname,
                                        fill=None, showlegend=False
                                        ), row=1, col=1)

                fig.add_trace(go.Scatter(x=testres_x, y=testres.groupby('Test week').quantile(0.5 - qt),
                                        name='Lower-PI', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=True,
                                        #fill='tonexty', fillcolor=gcolor
                                        ), row=1, col=1)

        if show_train:       
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
                                        legendgroup=strategy, legendgrouptitle=dict(text=strategy),
                                        showlegend=True
                                        ), row=1, col=1)

    
        ### Trajectory error
        # Load and create trace for ground truth data
        demand_gt = wfe.demand[dma]
        weather_gt = wfe.weather

        # Add the demand first as its black and we want it to be on the bottom
        fig.add_trace(go.Scatter(x=demand_gt.index, y=demand_gt,
                                 name='Demand', mode='lines', line=dict(color='black'),
                                 legendgroup='Observations', showlegend=True
                                ), row=3, col=1)
        

        for model_idx, model_name in enumerate(model_names):
            gcolor = px.colors.qualitative.Plotly[model_idx]
            # Load and create trace for forecasted data
            # first get the training data
            if show_train:
                gname = model_name+' (Train)'
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

                edates = error.index.get_level_values('Date').unique()
                fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').median(),
                                        name='Median-err', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=True
                                        ), row=2, col=1)

                fdates = forecast.index.get_level_values('Date').unique()
                fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').median(),
                                        name='Median-fcst', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=True
                                        ), row=3, col=1)

                if forecast.index.get_level_values('Seed').unique().shape[0] > 1 and qt > 0:
                    fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').quantile(0.5 + qt),
                                            name='Upper-err', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname, showlegend=False
                                            ), row=2, col=1)

                    fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').quantile(0.5 - qt),
                                            name='Lower-err', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname, showlegend=False
                                            ), row=2, col=1)

                    fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').quantile(0.5 + qt),
                                            name='Upper-fcst', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname, showlegend=False
                                            ), row=3, col=1)

                    fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').quantile(0.5 - qt),
                                            name='Lower-fcst', mode='lines', line=dict(color=gcolor),
                                            legendgroup=gname, showlegend=False
                                            ), row=3, col=1)
                gname = model_name # reset
                
            # now for the test and eval phase
            df_list = []
            for l__iter in wfe.results[model_name].keys():
                for phase in ['test', 'eval']:
                    if phase in wfe.results[model_name][l__iter]:
                        seed_list = []
                        for seed in wfe.results[model_name][l__iter][phase].keys():
                            seed_list.append(wfe.results[model_name][l__iter][phase][seed]['forecast'])
                        # Concatenate all the seeds
                        df_list.append(pd.concat(seed_list,
                                        keys=range(len(seed_list)), 
                                        names=['Seed', 'Date']))

            # Concatenate all iterations and phases
            forecast = pd.concat(df_list)
            
            # Select DMA
            forecast = forecast[dma]
            
            error = forecast.copy()
            
            # For test and eval phases I have to be a little bit more careful because I don't have the ground truth for the last week
            # Reset the index
            error_reset = error.reset_index()

            # Convert 'Date' to datetime
            error_reset['Date'] = pd.to_datetime(error_reset['Date'])

            # Select rows where 'Date' is before 'demand_gt.index[-1]'
            error_filtered = error_reset[error_reset['Date'] <= demand_gt.index[-1]]

            error = error_filtered
            
            for seed in error['Seed'].unique():
                mask = error['Seed'] == seed
                error.loc[mask, dma] -= demand_gt.loc[error.loc[mask, 'Date']].values
            error.set_index(['Seed', 'Date'], inplace=True)
            error = error[dma] # back to series for plotting
            
            edates = error.index.get_level_values('Date').unique()
            fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').median(),
                            name='Median-err', mode='lines', line=dict(color=gcolor),
                            legendgroup=gname, showlegend=True
                            ), row=2, col=1)
            
            fdates = forecast.index.get_level_values('Date').unique()
            fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').median(),
                            name='Median-fcst', mode='lines', line=dict(color=gcolor),
                            legendgroup=gname, showlegend=True
                            ), row=3, col=1)
            
            if forecast.index.get_level_values('Seed').unique().shape[0] > 1 and qt > 0:
                fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').quantile(0.5 + qt),
                                        name='Upper-err', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=False
                                        ), row=2, col=1)

                fig.add_trace(go.Scatter(x=edates, y=error.groupby('Date').quantile(0.5 - qt),
                                        name='Lower-err', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=False
                                        ), row=2, col=1)

                fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').quantile(0.5 + qt),
                                        name='Upper-fcst', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=False
                                        ), row=3, col=1)

                fig.add_trace(go.Scatter(x=fdates, y=forecast.groupby('Date').quantile(0.5 - qt),
                                        name='Lower-fcst', mode='lines', line=dict(color=gcolor),
                                        legendgroup=gname, showlegend=False
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
        fig.update_yaxes(title_text=PI_UNITS[pi], row=1, col=1)
        fig.update_yaxes(title_text="Inflow Error [L/s]", row=2, col=1)
        fig.update_yaxes(title_text="Inflow [L/s]", row=3, col=1)
        fig.update_xaxes(title_text="Time", matches='x', row=4, col=1)
        fig.update_yaxes(title_text="Temperature [°C]/Rain [mm/hour]", row=4, col=1)

        fig.update_layout(height=1200)
        
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
        Output('table-winner', 'data'),
        Input('model-checklist', 'value')
    )
    def table_winner(models):
        # I want to get what model what be the winner of the competition using the whole training 
        models=[model for model in wfe.results.keys()]
        # The competition ranks the competitors' models accumulating the score, for each DMA, PI and test week
        # We are going to do the same, averaging across all seeds
        test_weeks_m=wfe.results[models[0]]['iter_1']['train']['seed_0']['performance_indicators'].index.get_level_values('Test week').unique()
        
        #let's do the seed concatenation operation only once 
        models_scores={}
        for model_idx, model in enumerate(models): 
            df_list = []
            for seed in wfe.results[model]['iter_1']['train'].keys():
                df_list.append(wfe.results[model]['iter_1']['train'][seed]['performance_indicators'])
        
            # Concatenate all the seeds
            testres = pd.concat(df_list,
                                keys=range(len(df_list)), 
                                names=['Seed', 'Test week', 'DMA'])
            
            models_scores[model]=testres.groupby(['Test week', 'DMA']).mean()
            #print(models_scores[model])

        # do the models only first (ranking is the total competition score)
        total_models_ranking=np.zeros(len(models))
        for dma in DMAS_NAMES:
            for pi in PI_DESCRIPTIONS.keys():
                for tw in test_weeks_m:
                    # Each model has an average performance across its seeds
                    
                    models_score=np.zeros(len(models))
                    for model_idx, model in enumerate(models_scores):    
                        models_score[model_idx]=models_scores[model].loc[(tw,dma), pi]
                        #print(f'Model {model} has {pi}={models_score[model_idx]} on {dma} and week {tw}')
                        
                    # rank them by score and add them to the total competiion score (ranking)
                    
                    # models_ranking=some sort of ranking 

                    #total_models_ranking+=models_ranking
        total_models_ranking=total_models_ranking/(len(DMAS_NAMES)*len(PI_DESCRIPTIONS)*len(test_weeks_m))

        # now let's do the same but adding the strategies in the competition
        test_weeks_s=wfe.resstrategies['avg_top5']['iter_1']['train']['performance_indicators'].index.get_level_values('Test week').unique()
        for strategy in wfe.resstrategies:
            models_scores[strategy]=wfe.resstrategies[strategy]['iter_1']['train']['performance_indicators']
        mers=[model for model in models_scores.keys()]

        total_mers_ranking=np.zeros(len(mers))
        for dma in DMAS_NAMES:
            for pi in PI_DESCRIPTIONS.keys():
                for tw in test_weeks_s:
                    
                    models_score=np.zeros(len(models_scores))
                    for model_idx, model in enumerate(models_scores):
                        models_score[model_idx]=models_scores[model].loc[(tw,dma), pi]
                        #print(f'Model {model} has {pi}={models_score[model_idx]} on {dma} and week {tw}')

                    # rank them by score and add them to the total competiion score (ranking)
                    
                    # models_ranking=some sort of ranking 

                    #total_mers_ranking+=models_ranking    
        total_mers_ranking=total_mers_ranking/(len(DMAS_NAMES)*len(PI_DESCRIPTIONS)*len(test_weeks_s))
        
        # Now return something like '#1 - Model x (1,345)' 
        final_models_ranking=range(len(models)) 
        models_ranked_list=[]
        for idx, model_idx in enumerate(final_models_ranking):
            models_ranked_list.append(f'#{idx+1} - {models[model_idx]} ({total_models_ranking[model_idx]})')
        
        final_mers_ranking=range(len(mers))
        mers_ranked_list=[]
        for idx, mers_idx, in enumerate(final_mers_ranking):
            mers_ranked_list.append(f'#{idx+1} - {mers[mers_idx]} ({total_mers_ranking[mers_idx]})')

        table_data=[]
        for idx in range(len(mers_ranked_list)): #because mers is for sure longer
            if idx < len(models_ranked_list):
                table_data.append({'models': models_ranked_list[idx], 'mers':mers_ranked_list[idx]})
            else:
                table_data.append({'models': '-', 'mers':mers_ranked_list[idx]})
            
        #print(table_data)
        return table_data
    
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
                        value=['LGBMrobust', 'LGBMsimple', 'LGBMsimple_with_last week', 'XGBMsimple', 'WaveNet'],
                        inline=True
                    ),
                    html.Div(children='Select the strategies to visualize:'),
                    dcc.Checklist(
                        id="strategy-checklist",
                        options=strategies,
                        value=['avg_top5'],
                        inline=True
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            html.H3(children='PI and Forecasts over Time', style={'textAlign': 'center'}),
            dcc.Graph(id='graph-content'),
            html.H3(children='Who would win? - All models scores in training', style={'textAlign': 'center'}),
            dash_table.DataTable(                                 
                data=[],
                id='table-winner',
                sort_action='native',
                columns=[
                    {'name': 'Models only', 'id': 'models'},
                    {'name': 'Models and ERS', 'id': 'mers'}
                ],
                #add style data to the left
                )
        ], style={'margin': '4em', 'margin-bottom': '10em'})


"""
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
"""