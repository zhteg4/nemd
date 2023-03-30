import os
import dash
import base64
import dash_bootstrap_components as dbc
from nemd import testutils
from nemd import oplsua
import pandas as pd
from nemd import molview
import plotly.express as px


# Initialize the app
class App(dash.Dash):
    def __init__(self, *arg, **kwarg):
        self.filepath = None
        super().__init__(*arg, **kwarg)
        self.setData()
        self.setLayout()
        self.callback(
            dash.Output(component_id='traj_fig', component_property='figure'),
            dash.Input('select_datafile', 'contents'))(self.updateGraph)

    def setData(self):
        self.filepath = os.path.join('polym_builder', 'cooh123.data')
        datafile = testutils.test_file(self.filepath)
        data_reader = oplsua.DataFileReader(datafile)
        data_reader.run()
        self.frm_vw = molview.FrameView(data_reader)
        self.frm_vw.setData()
        self.frm_vw.updateLayout()

    def setLayout(self):
        self.layout = dash.html.Div([
            dash.html.H1(children='Molecular Trajectory Viewer',
                         style={
                             'textAlign': 'center',
                             'color': '#7FDBFF'
                         }),
            dash.html.Hr(),
            dash.html.Div(children='Datafile:',
                          style={'display': 'inline-block'}),
            dash.html.Div(children='',
                          style={
                              'display': 'inline-block',
                              'margin-left': '10px',
                              'margin-right': '5px'
                          }),
            dash.html.Div(children=[
                dash.dcc.Upload(
                    id='select_datafile',
                    children=dash.html.Div(['click to select']),
                    style={
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    },
                )
            ],
                          style={'display': 'inline-block'}),

            # dash.dcc.RadioItems(options=['datafile', 'trajectory'],
            #                     value='datafile',
            #                     id='data_source',
            #                     labelStyle={'margin-left': '15px'}),
            dash.dcc.Graph(figure={}, id='traj_fig', style={'height': '80vh'})
        ])

    def updateGraph(self, contents):
        if contents:
            content_type, content_string = contents.split(',')
            data_reader = oplsua.DataFileReader(contents=content_string)
            print(type(content_string))
            data_reader.run()
            self.frm_vw = molview.FrameView(data_reader)
            self.frm_vw.setData()
            self.frm_vw.updateLayout()
        self.frm_vw.clearPlot()
        self.frm_vw.plotScatters()
        self.frm_vw.plotLines()
        return self.frm_vw.fig


def main():
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
