import mendeleev
import numpy as np
import pandas as pd
import plotly.graph_objects as graph_objects

from nemd import traj


class FrameView:
    """Viewer datafile and trajectory frame"""

    XYZU = traj.Frame.XYZU

    def __init__(self, data_reader, scale=5.):
        """
        :param data_reader `nemd.oplsua.DataFileReader`: datafile reader with
            atom, bond, element and other info.
        :param scale float: scale the LJ dist by this to obtain marker size.
        """
        self.data_reader = data_reader
        self.fig = graph_objects.Figure()
        self.scale = scale

    def setData(self):
        """
        Set data frame with coordinates, elements, marker sizes, and color info.
        """
        if not self.data_reader:
            return
        index = list(self.data_reader.atoms.keys())
        xyz = np.array([x.xyz for x in self.data_reader.atom])
        self.data = pd.DataFrame(xyz, index, columns=self.XYZU)
        self.data_reader.setMinimumDist()
        type_ids = [x.type_id for x in self.data_reader.atom]
        sizes = [self.data_reader.vdws[x].dist * self.scale for x in type_ids]
        element = {x: y.ele for x, y in self.data_reader.masses.items()}
        elements = [element[x] for x in type_ids]
        color = {x: mendeleev.element(y).cpk_color for x, y in element.items()}
        self.color = [color[x] for x in type_ids]
        sz_clr = pd.DataFrame(
            {
                'element': elements,
                'size': sizes,
                'color': self.color
            },
            index=index)
        self.data = pd.concat((self.data, sz_clr), axis=1)

    def plotScatters(self):
        """
        Plot scattered markers for atoms.
        """
        if not self.data_reader:
            return
        ele_vdw = [(x.ele, self.data_reader.vdws[x.id].dist * self.scale)
                   for x in self.data_reader.masses.values()]
        for ele, size in sorted(set(ele_vdw), key=lambda x: x[1],
                                reverse=True):
            idx = (self.data[['element', 'size']] == [ele, size]).all(axis=1)
            data = self.data[idx]
            marker = dict(size=size, color=data['color'].values[0])
            marker = graph_objects.Scatter3d(x=data.xu,
                                             y=data.yu,
                                             z=data.zu,
                                             opacity=0.9,
                                             mode='markers',
                                             name=ele,
                                             marker=marker)
            self.fig.add_trace(marker)

    def plotLines(self):
        """
        Plot lines for bonds.
        """
        if not self.data_reader:
            return
        for bond in self.data_reader.bonds.values():
            atom1 = self.data_reader.atoms[bond.id1]
            atom2 = self.data_reader.atoms[bond.id2]
            pnts = self.data.loc[[atom1.id, atom2.id]][self.XYZU]
            pnts = pd.concat((pnts, pnts.mean().to_frame().transpose()))
            self.plotline(pnts[::2], atom1)
            self.plotline(pnts[1::], atom2)

    def plotline(self, xyz, atom):
        """
        Plot half bond spanning from one atom to the mid point.

        :param xyz `numpy.ndarray`: the bond XYZU span
        :param atom 'types.SimpleNamespace': the bonded atom
        """
        line = dict(width=8, color=self.data.xs(atom.id).color)
        line = graph_objects.Scatter3d(x=xyz.xu,
                                       y=xyz.yu,
                                       z=xyz.zu,
                                       opacity=0.8,
                                       mode='lines',
                                       showlegend=False,
                                       line=line)
        self.fig.add_trace(line)

    def clearPlot(self):
        """
        Clear the atom and bond plots.
        """
        self.fig.data = []

    def updateLayout(self):
        """
        Update the figure layout.
        """
        camera = dict(up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=1.25, y=1.25, z=1.25))
        self.fig.update_layout(template='plotly_dark', scene_camera=camera)

    def show(self):
        """
        Show the figure with plot.
        """
        self.fig.show()
