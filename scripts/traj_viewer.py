import sys
import dash
import collections
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
from nemd import traj
from nemd import ndash
from nemd import oplsua
from nemd import molview
from nemd import geometry

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE

POINT = collections.namedtuple('POINT',
                               ['idx', 'ele', 'x', 'y', 'z', 'cn', 'pn'])


class App(dash.Dash):

    CANCEL_SYMBOL = 'X'
    CLICK_TO_SELECT = 'click to select'
    TRAJ_INPUT = 'traj_input'
    TRAJ_LB = 'traj_lb'
    SELECT_TRAJ_LB = 'select_traj_lb'
    DATAFILE_INPUT = 'datafile_input'
    DATAFILE_LB = 'datafile_lb'
    SELECT_DATA_LB = 'select_data_lb'
    MEASURE_DD = 'measure_dd'
    MEASURE_INFO_LB = 'measure_info_lb'
    MEASURE_BT = 'measure_bt'
    ALL_FRAME_LB = 'all_frame_lb'
    TRJ_FG = 'TRJ_FG'
    CLICKDATA = 'clickData'
    TRJ_FG_CLICKDATA = f'{TRJ_FG}.{CLICKDATA}'
    POINTS = 'points'
    CUSTOMDATA = 'customdata'
    CURVENUMBER = 'curveNumber'
    POINTNUMBER = 'pointNumber'
    NAME = 'name'
    POSITION = 'Position'
    DISTANCE = 'Distance'
    ANGLE = 'Angle'
    DIHEDRAL = 'Dihedral'
    MEASURE_COUNT = {POSITION: 1, DISTANCE: 2, ANGLE: 3, DIHEDRAL: 4}
    MEASURE_FUNC = {
        POSITION: lambda x: x,
        DISTANCE: geometry.distace,
        ANGLE: geometry.angle,
        DIHEDRAL: geometry.dihedral
    }
    MEASURE_UNIT = {DISTANCE: 'angstrom', ANGLE: 'degree', DIHEDRAL: 'degree'}

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.frm_vw = molview.FrameView()
        self.points = {}
        self.setLayout()
        self.callback(
            Output(component_id=self.TRJ_FG, component_property='figure'),
            Input(self.DATAFILE_INPUT, 'contents'),
            Input(self.TRAJ_INPUT, 'contents'))(self.inputChanged)
        self.callback(
            Output(component_id=self.DATAFILE_LB,
                   component_property='children'),
            Output(component_id=self.SELECT_DATA_LB,
                   component_property='children'),
            Input(self.DATAFILE_INPUT, 'filename'))(self.updateDataLabel)
        self.callback(
            Output(component_id=self.TRAJ_LB, component_property='children'),
            Output(component_id=self.SELECT_TRAJ_LB,
                   component_property='children'),
            Input(self.TRAJ_INPUT, 'filename'))(self.updateTrajLabel)
        self.callback(Output(self.MEASURE_INFO_LB, 'children'),
                      Output(self.TRJ_FG, 'figure', allow_duplicate=True),
                      Input(self.TRJ_FG, 'clickData'),
                      Input(self.MEASURE_DD, 'value'),
                      prevent_initial_call=True)(self.measureData)
        self.callback(Output(self.ALL_FRAME_LB, 'children'),
                      Input(self.MEASURE_BT, 'n_clicks'),
                      Input(self.MEASURE_DD, 'value'))(self.computeFrames)

    def setLayout(self):
        """
        Set the layout of the widget.
        """
        self.layout = html.Div([
            dbc.Row(ndash.H1(children='Molecular Trajectory Viewer')),
            dbc.Row(html.Hr()),
            dbc.Row([
                dbc.Col([
                    ndash.LabeledUpload(label='Data File:',
                                        status_id=self.DATAFILE_LB,
                                        button_id=self.DATAFILE_INPUT,
                                        click_id=self.SELECT_DATA_LB),
                    ndash.LabeledUpload(label='Trajectory:',
                                        status_id=self.TRAJ_LB,
                                        button_id=self.TRAJ_INPUT,
                                        click_id=self.SELECT_TRAJ_LB),
                    ndash.LabeledDropdown(list(self.MEASURE_COUNT.keys()),
                                          label="Measure: ",
                                          value='Position',
                                          id="measure_dd"),
                    html.Pre(id=self.MEASURE_INFO_LB),
                    html.Button('All Frames', id=self.MEASURE_BT),
                    html.Pre(id=self.ALL_FRAME_LB, style={'padding-left': 10})
                ],
                        width=3),
                dbc.Col(dcc.Graph(id=self.TRJ_FG, style=dict(height='80vh')),
                        width=9)
            ])
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
        try:
            data_reader.run()
        except ValueError:
            # Accidentally load xyz into the datafile holder
            return self.frm_vw.fig
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
        try:
            frms = traj.get_frames(contents=contents)
        except ValueError:
            # Empty trajectory file
            return self.frm_vw.fig
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

    def measureData(self, data, mvalue):
        """
        Measure selected atoms.

        :param data dict: newly selected point
        :param mvalue str: measurement type including Position, Distance, Angle,
            and Dihedral
        :return str, `Figure`: measure information to display and figure to plot
        """
        if data is None:
            count = self.MEASURE_COUNT[mvalue]
            return f' Select {count} atoms to measure {mvalue.lower()}', self.frm_vw.fig
        if dash.ctx.triggered[0]['prop_id'] == self.TRJ_FG_CLICKDATA:
            if len(self.points) == self.MEASURE_COUNT[mvalue]:
                # Selection number exceeds and restart as a new collection
                self.points = {}
            pnt = data[self.POINTS][0]
            idx = pnt[self.CUSTOMDATA]
            cn, pn = pnt[self.CURVENUMBER], pnt[self.POINTNUMBER]
            ele = self.frm_vw.fig.data[cn][self.NAME]
            point = POINT(idx=idx,
                          ele=ele,
                          x=pnt['x'],
                          y=pnt['y'],
                          z=pnt['z'],
                          cn=cn,
                          pn=pn)
            self.points[point.idx] = point
        else:
            # Measure type changed
            self.points = {}
        self.markAtoms()
        info = f"Information:\n" + self.measure(mvalue)
        return info, self.frm_vw.fig

    def measure(self, mvalue):
        """
        Measure atom position, distance, angle or dihedral.

        :param mvalue str: measurement type including Position, Distance, Angle,
            and Dihedral
        :return str: the measurement information to display
        """
        points = [x for x in self.points.values()]
        count = self.MEASURE_COUNT[mvalue]
        if len(points) == 0:
            return f' Select {count} atoms to measure {mvalue.lower()}'
        atom_ids = ', '.join(map(str, [x.idx for x in points]))
        if mvalue == self.POSITION:
            point = points[0]
            return f" index={point.idx}, element={point.ele},\n"\
                   f" x={point.x}, y={point.y}, z={point.z}"
        if len(points) < count:
            return f" Atom {atom_ids} been selected.\n" \
                   f" Select more atoms to measure {mvalue.lower()}."
        points = [np.array([x.x, x.y, x.z]) for x in points]
        value = self.MEASURE_FUNC[mvalue](points)
        unit = self.MEASURE_UNIT[mvalue]
        return f' {mvalue} between Atom {atom_ids} is {value:.2f} {unit}'

    def markAtoms(self):
        """
        Mark selected atoms with annotations.
        """

        annotations = [
            dict(showarrow=False,
                 x=pnt.x,
                 y=pnt.y,
                 z=pnt.z,
                 text=f"Atom {i}",
                 xanchor="left",
                 xshift=10,
                 opacity=0.7) for i, pnt in enumerate(self.points.values(), 1)
        ]
        self.frm_vw.fig.update_layout(scene=dict(annotations=annotations),
                                      overwrite=True,
                                      uirevision=True)

    def computeFrames(self, n_clicks, mvalue):
        """
        Compute the point, distance, angle or dihedral measurement for all frames.

        :param n_clicks int: number of button clicks
        :param mvalue str: measurement type including Position, Distance, Angle,
            and Dihedral
        :return str: the measured values for all frames
        """
        if len(self.points
               ) != self.MEASURE_COUNT[mvalue] or not self.frm_vw.fig.frames:
            return 'Select atoms and load trajectory'
        values = []
        for frm in self.frm_vw.fig.frames:
            points = []
            for pnt in self.points.values():
                dat = frm.data[pnt.cn]
                points.append(
                    np.array(
                        [dat['x'][pnt.pn], dat['y'][pnt.pn],
                         dat['z'][pnt.pn]]))
            value = self.MEASURE_FUNC[mvalue](points)
            values.append(value)
        data = pd.DataFrame(values)
        data.index += 1
        return data.to_string(header=False)


def main(argv):
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
