import sys
import dash
from types import SimpleNamespace
import dash_bootstrap_components as dbc
from nemd import oplsua
from nemd import traj
from nemd import molview
from nemd import parserutils

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE


class App(dash.Dash):
    def __init__(self, *arg, options=None, **kwarg):
        super().__init__(*arg, **kwarg)
        self.options = options
        if not self.options:
            self.options = SimpleNamespace(data_file=None, custom_dump=None)
        self.setData()
        self.setLayout()
        self.callback(
            dash.Output(component_id='traj_fig', component_property='figure'),
            dash.Input('select_datafile', 'contents'))(self.updateGraph)
        self.callback(
            dash.Output(component_id='datafile_lb',
                        component_property='children'),
            dash.Output(component_id='select_lb',
                        component_property='children'),
            dash.Input('select_datafile', 'filename'))(self.updateDataLabel)

    def setData(self):
        data_reader = None
        if self.options.data_file:
            data_reader = oplsua.DataFileReader(self.options.data_file)
            data_reader.run()
        self.frm_vw = molview.FrameView(data_reader)
        self.frm_vw.setData()

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
                    id='select_datafile',
                    children=dash.html.Div(children='', id='select_lb'),
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

    def updateGraph(self, contents):
        data_reader = None
        if contents:
            content_type, content_string = contents.split(',')
            data_reader = oplsua.DataFileReader(contents=content_string)
            data_reader.run()
        self.frm_vw = molview.FrameView(data_reader)
        self.frm_vw.setData()
        self.frm_vw.clearPlot()
        self.frm_vw.plotScatters()
        self.frm_vw.plotLines()
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def updateDataLabel(self, filename):
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
    options = get_parser().parse_args(argv)
    app = App(__name__,
              options=options,
              external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True, )


if __name__ == '__main__':
    main(sys.argv[1:])
