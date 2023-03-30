import os
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from nemd import testutils
from nemd import oplsua
import pandas as pd
from nemd import molview
import plotly.express as px


# Initialize the app
class App(dash.Dash):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.setLayout()

    def setLayout(self):
        self.layout = html.Div([
            html.Div(children='My First App with Data, Graph, and Controls'),
            html.Hr(),
            dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'],
                           value='lifeExp',
                           id='my-final-radio-item-example'),
            dcc.Graph(figure={}, id='traj_fig', style={'height': '80vh'})
        ])

    @callback(Output(component_id='traj_fig', component_property='figure'),
              Input(component_id='my-final-radio-item-example',
                    component_property='value'))
    def update_graph(col_chosen):
        filepath = os.path.join('polym_builder', 'cooh123.data')
        datafile = testutils.test_file(filepath)
        data_reader = oplsua.DataFileReader(datafile)
        data_reader.run()
        frm_vw = molview.FrameView(data_reader)
        frm_vw.setData()
        frm_vw.scatters()
        frm_vw.lines()
        frm_vw.updateLayout()
        return frm_vw.fig


def main():
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
