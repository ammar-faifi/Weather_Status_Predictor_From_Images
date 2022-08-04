"""
    Simple Dash app
"""

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)
# server = app.server # for production

# sample data frame
df = pd.DataFrame({
    'Var1': ['item1', 'item2', 'item3', 'item4', ],
    'Amount': [1, 3, 3, 4],
    'Type': ['T1', 'T2', 'T1', 'T2']
})

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

fig = px.bar(df, x='Var1', y='Amount', color='Type', )
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(
            children='Hello Dash',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div(children='Dash: A web application framework for your data.', style={
            'textAlign': 'center',
            'color': colors['text']
        }),

        dcc.Graph(
            id='example-graph-2',
            figure=fig
        ),
        dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
            dcc.Tab(label='Tab One', value='tab-1-example-graph'),
            dcc.Tab(label='Tab Two', value='tab-2-example-graph'),
        ]),
        html.Div(id='content')
    ]
    )

@app.callback(Output('content', 'children'), 
    Input('tabs-example-graph', 'value'))
def callback(value):
    print(value)


if __name__ == "__main__":
    app.run(debug=True)
