import sys
import dash
import dash_bootstrap_components as dbc
from nemd import traj
from nemd import ndash
from nemd import oplsua
from nemd import molview

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE


class App(dash.Dash):

    CANCEL_SYMBOL = 'X'
    CLICK_TO_SELECT = 'click to select'
    TRAJ_INPUT = 'traj_input'
    TRAJ_LB = 'traj_lb'
    SELECT_TRAJ_LB = 'select_traj_lb'
    DATAFILE_INPUT = 'datafile_input'
    DATAFILE_LB = 'datafile_lb'
    SELECT_DATA_LB = 'select_data_lb'
    TRAJ_FIG = 'traj_fig'

    BLUE_COLOR_HEX = '#7FDBFF'

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.frm_vw = molview.FrameView()
        self.setLayout()
        self.callback(
            dash.Output(component_id=self.TRAJ_FIG,
                        component_property='figure'),
            dash.Input(self.DATAFILE_INPUT, 'contents'),
            dash.Input(self.TRAJ_INPUT, 'contents'))(self.inputChanged)
        self.callback(
            dash.Output(component_id=self.DATAFILE_LB,
                        component_property='children'),
            dash.Output(component_id=self.SELECT_DATA_LB,
                        component_property='children'),
            dash.Input(self.DATAFILE_INPUT, 'filename'))(self.updateDataLabel)
        self.callback(
            dash.Output(component_id=self.TRAJ_LB,
                        component_property='children'),
            dash.Output(component_id=self.SELECT_TRAJ_LB,
                        component_property='children'),
            dash.Input(self.TRAJ_INPUT, 'filename'))(self.updateTrajLabel)

    def setLayout(self):
        """
        Set the layout of the widget.
        """
        self.layout = dash.html.Div([
            dash.html.H1(children='Molecular Trajectory Viewer',
                         style={
                             'textAlign': 'center',
                             'color': self.BLUE_COLOR_HEX
                         }),
            dash.html.Hr(),
            ndash.LabeledUpload(label='Data File:',
                                status_id=self.DATAFILE_LB,
                                button_id=self.DATAFILE_INPUT,
                                click_id=self.SELECT_DATA_LB),
            ndash.LabeledUpload(label='Trajectory:',
                                status_id=self.TRAJ_LB,
                                button_id=self.TRAJ_INPUT,
                                click_id=self.SELECT_TRAJ_LB),
            dash.dcc.Graph(figure={},
                           id=self.TRAJ_FIG,
                           style={'height': '80vh'})
        ])

    def inputChanged(self, data_contents, traj_contents):
        """
        React to datafile or trajectory change.

        :param data_contents 'str': base64 endecoded str for datafile type and
            contents
        :param traj_contents 'str': base64 endecoded str for trajectory type and
            contents
        :return:
        """
        if not any([data_contents, traj_contents]):
            return self.cleanPlot()

        self.dataFileChanged(data_contents)
        return self.trajChanged(traj_contents)

    def cleanPlot(self):
        """
        Clear data, plot and set style.
        """
        self.frm_vw.clearData()
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def dataFileChanged(self, contents):
        """
        React to datafile change.

        :param contents 'str': base64 endecoded str for datafile type and
            contents
        :return 'plotly.graph_objs._figure.Figure': the figure object
        """
        self.frm_vw.clearData()
        if contents is None:
            return self.frm_vw.fig
        data_reader = oplsua.DataFileReader(contents=contents)
        data_reader.run()
        self.frm_vw.data_reader = data_reader
        self.frm_vw.setData()
        self.frm_vw.setEdges()
        self.frm_vw.setEleSz()
        self.frm_vw.setScatters()
        self.frm_vw.setLines()
        self.frm_vw.addTraces()
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def trajChanged(self, contents):
        """
        React to datafile change.

        :param contents 'str': base64 endecoded str for trajectory type and
            contents
        :return 'plotly.graph_objs._figure.Figure': the figure object
        """
        if contents is None:
            return self.frm_vw.fig
        frms = traj.get_frames(contents=contents)
        self.frm_vw.setFrames(frms)
        self.frm_vw.updateLayout()
        return self.frm_vw.fig

    def updateDataLabel(self, filename):
        """
        React to datafile change.

        :param filename 'str': the datafile filename
        :return str, str: the filename to display and the cancel text for new
            loading.
        """
        select_lb = self.CANCEL_SYMBOL if filename else self.CLICK_TO_SELECT
        return filename, select_lb

    def updateTrajLabel(self, filename):
        """
        React to trajectory change.

        :param filename 'str': the trajectory filename
        :return str, str: the filename to display and the cancel text for new
            loading.
        """
        select_lb = self.CANCEL_SYMBOL if filename else self.CLICK_TO_SELECT
        return filename, select_lb


def main(argv):
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
