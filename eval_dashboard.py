from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as suboplots
import pandas as pd

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
        Input('model-checklist', 'value')
    )
    def figures(dma, pi, model_names):
        fig = suboplots.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      subplot_titles=(f"Performance for {dma} and {pi} of the selected models", 
                                                      f"{dma} inflow trajectory for the selected models"))

        #fig_pi = go.Figure()
        for model_idx, model_name in enumerate(model_names):
            # Read current performance indicators
            # This is kinda ugly, sorry but it works
            performance_indicators_np = wfe.results[model_name]['performance_indicators'][wfe.results[model_name]['performance_indicators'].index.get_level_values('DMA') == dma][pi].to_numpy()
            performance_indicators = pd.Series(performance_indicators_np, index=wfe.results[model_name]['forecast'].iloc[-168*performance_indicators_np.shape[0]::168].index)

            # Create trace for performance indicator
            fig.add_trace(go.Scatter(x=performance_indicators.index, y=performance_indicators, name=model_name, 
                                        mode='lines', line=dict(color=px.colors.qualitative.Plotly[model_idx]), showlegend=False),
                                        row=1, col=1)

        # y axis label
        fig.update_yaxes(title_text=PI_UNITS[pi], row=1, col=1)
    
        ### Trajectory
        # Load and create trace for ground truth data
        demand_gt = wfe.demand[dma]
        fig.add_trace(go.Scatter(x=demand_gt.index, y=demand_gt, name='Ground Truth',
                                        mode='lines', line=dict(color="#000000")),
                                        row=2, col=1)
    
        #fig_traj = go.Figure()
        for model_idx, model_name in enumerate(model_names):
            # Load and create trace for forecasted data
            forecast = wfe.results[model_name]['forecast'][dma]
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name=model_name,
                                            mode='lines', line=dict(color=px.colors.qualitative.Plotly[model_idx])),
                                            row=2, col=1)
        
        # Title and axes labels
        fig.update_xaxes(title_text="Time", matches='x', row=2, col=1)
        fig.update_yaxes(title_text="Inflow [L/s]", row=2, col=1)
        

        fig.update_layout(height=700)

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

    app.run(debug=True)

def layout(wfe):
    models = list(wfe.results.keys())

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
                        )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            dcc.Graph(id='graph-content'),
        ])