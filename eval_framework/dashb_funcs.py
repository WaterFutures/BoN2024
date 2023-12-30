import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import pandas as pd

import data_loader
from .performance_indicators import performance_indicators_labels

def wf_dashboard_layout(wfe):
    return html.Div([
            html.H1(children='Water Futures Evaluator Dashboard', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Div(children='Select the performance indicator to visualize:'),
                    dcc.Dropdown(['PI1', 'PI2', 'PI3'], 'PI1', id='pi-dropdown'),
                    html.Div(children={}, id='pi-description')
                ], style={'width': '30%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(children='Select the DMA to visualize:'),
                    dcc.Dropdown(data_loader.DMAS_NAMES, data_loader.DMAS_NAMES[0], id='dma-dropdown'),
                    html.Div(children={}, id='dma-description')
                ], style={'width': '30%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                        html.Div(children='Messages from the dashboard:'),
                        html.Div(children={}, id='errors-description'),
                ], style={'width': '40%', 'float': 'right', 'display': 'inline-block'})
            ]),
            dcc.Graph(id='graph-content'),
            html.Div([ 
                html.Div([ 
                    html.Div(children='Select the models to visualize:'),
                    dcc.Checklist(
                        id="model-checklist",
                        options=wfe.models_names(),
                        value=[wfe.models_names()[0]],
                        inline=True
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(children='Select the Model Description to read:'),
                    dcc.Dropdown(wfe.models_names(), wfe.models_names()[0], id='model-description-dropdown'),
                    html.Div(children={}, id='model-description')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            html.Div([
                dcc.Graph(id='trajectory-content'),
                dcc.RangeSlider(
                    id='trajectory-slider',
                    min=wfe.first_split_absweek(),
                    max=wfe.fcst_absweek(),
                    step=1,
                    value=[wfe.first_test_absweek(),
                           wfe.fcst_absweek()],
                    marks={i: '{}'.format(i) for i in range(wfe.first_split_absweek(),
                                                            wfe.fcst_absweek()+1,
                                                            1)}
                )
            ])
        ])

def figure_perf_ind_trajectory(wfe, dma:str, pi:str, model_names:list[str]) -> go.Figure:
    figpi = go.Figure()
    i = 0
    for model_name in model_names:
        vali__df = wfe.result(model_name)["validation"]
        vali__weeks = vali__df.index.get_level_values(0).unique().to_numpy()
        vali__dma_pi = vali__df.xs(dma, level=1).loc[:,pi].to_numpy()

        test__df = wfe.result(model_name)["test"]
        test__weeks = test__df.index.get_level_values(0).unique().to_numpy()
        test__dma_pi = test__df.xs(dma, level=1).loc[:,pi].to_numpy()

        # A line with the PI time series duringthe validation and the average 
        # of the PI during the test
        plot__weeks = np.append(vali__weeks, test__weeks[[0,-1]])
        plot__dma_pi = np.append(vali__dma_pi, [test__dma_pi.mean(), test__dma_pi.mean()])

        # add the line to the plot and the 4 points indepentently
        figpi.add_trace(go.Scatter(x=plot__weeks, y=plot__dma_pi, name=model_name, 
                                    mode='lines', line=dict(color=px.colors.qualitative.Plotly[i])))
        figpi.add_trace(go.Scatter(x=test__weeks, y=test__dma_pi, name=model_name, 
                                    mode='markers', marker=dict(color=px.colors.qualitative.Plotly[i]), 
                                    showlegend=False))

        week_ticks = week_ticks = np.concatenate(([plot__weeks[0]], 
                                                    np.arange(20, plot__weeks[-1], 10), 
                                                    test__weeks.tolist())).tolist()
        figpi.update_xaxes(range=[vali__weeks[0]-1, test__weeks[-1]+1],
                           tickvals=week_ticks)

        figpi.update_layout(
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
                #bgcolor="LightSteelBlue",
                #bordercolor="Black",
                #borderwidth=2,
                #xanchor='center',  # Anchor the legend at its center
                #yanchor='bottom',  # Anchor the legend at its bottom
            )
        )
        i += 1
    
    figpi.update_layout(title=f"Performance for {dma} and {pi} of the selected models in Validation and Test", 
                        xaxis_title='Week', yaxis_title=performance_indicators_labels[pi])
    return figpi

def figure_dma_q_trajectory(wfe, dma:str, pi:str, model_names:list[str], wrange:list[int]) -> go.Figure:
     # start of the plot with trajectories 
    figtraj = go.Figure()

    # This start from self.__first_split_week
    start_h = pd.to_datetime(data_loader.monday_of_week_number(wrange[0]))
    end_h = pd.to_datetime(data_loader.monday_of_week_number(wrange[1]+1))
    i = 0
    for model_name in model_names:
        pred__dma_h_q = wfe.result(model_name)["processed_data"]["fcst__dmas_h_q"].loc[start_h:end_h,dma]
        pred__dma_h_q = pd.concat([pred__dma_h_q, wfe.result(model_name)["bwdf_forecast"].loc[start_h:end_h,dma]], axis=0)
        
        figtraj.add_trace(go.Scatter(x=pred__dma_h_q.index, y=pred__dma_h_q, name=model_name,
                                        mode='lines', line=dict(color=px.colors.qualitative.Plotly[i])))
        i += 1

    #end of the for plot the truth. 
    true__dma_h_q = pd.concat([
            wfe._WaterFuturesEvaluator__train__dmas_h_q.loc[start_h:end_h, dma],
            wfe._WaterFuturesEvaluator__test__dmas_h_q.loc[start_h:end_h, dma]], axis=0)
    # This starts on the 4th of Jan 2021. So I can simply use the week number
    figtraj.add_trace(go.Scatter(x=true__dma_h_q.index, y=true__dma_h_q, name='Truth',
                                    mode='lines', line=dict(color="#000000")))
    
    figtraj.update_layout(title=f"{dma} inflow trajectory for the selected models"),
    figtraj.update_xaxes(title_text="Time")
    figtraj.update_yaxes(title_text="Inflow [L/s]")

    return figtraj