import sys
import dash
import dash_bootstrap_components as dbc
from nemd import traj
from nemd import oplsua
from nemd import molview
from nemd import parserutils

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE


class App(dash.Dash):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.frm_vw = molview.FrameView()
        self.setLayout()
        self.callback(
            dash.Output(component_id='traj_fig', component_property='figure'),
            dash.Input('datafile_input', 'contents'),
            dash.Input('traj_input', 'contents'))(self.inputChanged)
        self.callback(
            dash.Output(component_id='datafile_lb',
                        component_property='children'),
            dash.Output(component_id='select_data_lb',
                        component_property='children'),
            dash.Input('datafile_input', 'filename'))(self.updateDataLabel)
        self.callback(
            dash.Output(component_id='traj_lb', component_property='children'),
            dash.Output(component_id='select_traj_lb',
                        component_property='children'),
            dash.Input('traj_input', 'filename'))(self.updateTrajLabel)

    def setLayout(self):
        self.layout = dash.html.Div([
            dash.html.H1(children='Molecular Trajectory Viewer',
                         style={
                             'textAlign': 'center',
                             'color': '#7FDBFF'
                         }),
            dash.html.Hr(),
            dash.html.Div(children='Data File:',
                          style={'display': 'inline-block'}),
            dash.html.Div(children='',
                          id='datafile_lb',
                          style={
                              'display': 'inline-block',
                              'margin-left': '10px',
                              'margin-right': '5px'
                          }),
            dash.html.Div(children=[
                dash.dcc.Upload(
                    id='datafile_input',
                    children=dash.html.Div(children='', id='select_data_lb'),
                    style={
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    },
                )
            ],
                          style={'display': 'inline-block'}),
            # New Line
            dash.html.Div(),
            dash.html.Div(children='Trajectory:',
                          style={'display': 'inline-block'}),
            dash.html.Div(children='',
                          id='traj_lb',
                          style={
                              'display': 'inline-block',
                              'margin-left': '10px',
                              'margin-right': '5px'
                          }),
            dash.html.Div(children=[
                dash.dcc.Upload(
                    id='traj_input',
                    children=dash.html.Div(children='', id='select_traj_lb'),
                    style={
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    },
                )
            ],
                          style={'display': 'inline-block'}),
            dash.dcc.Graph(figure={}, id='traj_fig', style={'height': '80vh'})
        ])

    def inputChanged(self, data_contents, traj_contents):
        if not any([data_contents, traj_contents]):
            return self.cleanPlot()
        # ctx = dash.callback_context
        # input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        self.dataFileChanged(data_contents)
        return self.trajChanged(traj_contents)

    def cleanPlot(self):
        self.frm_vw.clearPlot()
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def dataFileChanged(self, contents):
        if contents is None:
            return self.frm_vw.fig
        self.frm_vw.clearPlot()
        content_type, content_string = contents.split(',')
        data_reader = oplsua.DataFileReader(contents=content_string)
        data_reader.run()
        self.frm_vw.data_reader = data_reader
        self.frm_vw.setData()
        self.frm_vw.setScatters()
        self.frm_vw.setLines()
        self.frm_vw.addTraces()
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def trajChanged(self, contents):
        if contents is None:
            return self.frm_vw.fig
        content_type, content_string = contents.split(',')
        frms = traj.Frame.read(contents=content_string)
        self.frm_vw.setFrames(frms)
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def updateDataLabel(self, filename):
        select_lb = 'X' if filename else 'click to select'
        return filename, select_lb

    def updateTrajLabel(self, filename):
        select_lb = 'X' if filename else 'click to select'
        return filename, select_lb


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument('-' + FlAG_CUSTOM_DUMP,
                        metavar=FlAG_CUSTOM_DUMP.upper(),
                        type=parserutils.type_file,
                        help='Custom dump file to visualize')
    parser.add_argument(FlAG_DATA_FILE,
                        metavar=FlAG_DATA_FILE.upper(),
                        type=parserutils.type_file,
                        help='Data file to get topological information')
    return parser


def main(argv):
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
