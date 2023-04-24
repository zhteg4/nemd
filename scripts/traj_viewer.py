import sys
import dash
import collections
import numpy as np
import dash_bootstrap_components as dbc
from nemd import traj
from nemd import ndash
from nemd import oplsua
from nemd import molview

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE

POINT = collections.namedtuple('POINT', ['idx', 'ele', 'x', 'y', 'z'])


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
    POINT_SEL = 'point_sel'
    TRAJ_FIG = 'traj_fig'
    BLUE_COLOR_HEX = '#7FDBFF'
    POSITION = 'Position'
    DISTANCE = 'Distance'
    ANGLE = 'Angle'
    DIHEDRAL = 'Dihedral'
    MEASURE_COUNT = {POSITION: 1, DISTANCE: 2, ANGLE: 3, DIHEDRAL: 4}

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.frm_vw = molview.FrameView()
        self.points = {}
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
        self.callback(dash.Output(self.POINT_SEL, 'children'),
                      dash.Output(self.TRAJ_FIG,
                                  'figure',
                                  allow_duplicate=True),
                      dash.Input(self.TRAJ_FIG, 'clickData'),
                      dash.Input(self.MEASURE_DD, 'value'),
                      prevent_initial_call=True)(self.measureData)

    def setLayout(self):
        """
        Set the layout of the widget.
        """
        self.layout = dash.html.Div([
            dbc.Row(
                dash.html.H1(children='Molecular Trajectory Viewer',
                             style={
                                 'textAlign': 'center',
                                 'color': self.BLUE_COLOR_HEX
                             })),
            dbc.Row(dash.html.Hr()),
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
                    dash.html.Div([
                        "Measure: ",
                        dash.dcc.Dropdown(list(self.MEASURE_COUNT.keys()),
                                          value='Position',
                                          id="measure_dd",
                                          style={
                                              'padding-left': 5,
                                              'color': '#000000'
                                          })
                    ]),
                    dash.html.Pre(id=self.POINT_SEL)
                ],
                        width=3),
                dbc.Col(dash.dcc.Graph(figure={},
                                       id=self.TRAJ_FIG,
                                       style={'height': '80vh'}),
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
        if data is None:
            return '', self.frm_vw.fig
        if dash.ctx.triggered[0]['prop_id'] == 'traj_fig.clickData':
            if len(self.points) == self.MEASURE_COUNT[mvalue]:
                self.points = {}
            point = data['points'][0]
            idx = point['customdata']
            ele = self.frm_vw.fig.data[point['curveNumber']]['name']
            pnt = POINT(idx=idx,
                        ele=ele,
                        x=point['x'],
                        y=point['y'],
                        z=point['z'])
            self.points[pnt.idx] = pnt
        else:
            self.points = {}
        self.updateAnnotations()
        info = f"Information:\n" + self.getInfo(mvalue)
        return info, self.frm_vw.fig

    def getInfo(self, mvalue):
        points = [x for x in self.points.values()]
        if mvalue == self.POSITION:
            point = points[0]
            return f" index={point.idx}, element={point.ele},\n"\
                   f" x={point.x}, y={point.y}, z={point.z}"
        if mvalue == self.DISTANCE:
            if len(points) == 0:
                return f' Select two atoms to measure a distance.'
            if len(points) == 1:
                return f' Atom {points[0].idx} has been selected.\n' \
                       f' Select another atom to measure a distance'
            if len(points) == 2:
                xyzs = [np.array([x.x, x.y, x.z]) for x in points]
                dist = np.linalg.norm(xyzs[0] - xyzs[1])
                return f' Distance between Atom {points[0].idx} and ' \
                       f'{points[1].idx}\n' \
                       f' is {dist:.4f} angstrom.'
        if mvalue == self.ANGLE:
            if len(points) == 0:
                return f' Please select three atoms to measure the angle.'
            if len(points) < 3:
                return f" Atom {', '.join([str(x.idx) for x in points])}" \
                       f" has been selected.\n" \
                       f' Select more atoms to measure an angle'
            if len(points) == 3:
                xyzs = [np.array([x.x, x.y, x.z]) for x in points]
                v1 = xyzs[0] - xyzs[1]
                v2 = xyzs[2] - xyzs[1]
                angle = np.arccos(
                    np.dot(v1, v2) / np.linalg.norm(v1) *
                    np.linalg.norm(v2)) / np.pi * 180.
                return f' Angle between Atom {points[0].idx}, {points[1].idx}' \
                       f' and {points[2].idx}\n' \
                       f' is {angle:.2f} degree.'

        return ''

    def updateAnnotations(self):
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
        self.frm_vw.fig.layout.scene.annotations = annotations
        self.frm_vw.fig.update_layout(scene=dict(annotations=annotations))


def main(argv):
    app = App(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.run_server(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
