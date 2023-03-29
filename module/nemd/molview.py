from nemd import traj
import mendeleev
import numpy as np
import pandas as pd
import plotly.graph_objects as graph_objects


class FrameView:
    """Viewer datafile and trajectory frame"""
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
        index = list(self.data_reader.atoms.keys())
        xyz = np.array([x.xyz for x in self.data_reader.atom])
        self.data = pd.DataFrame(xyz, index, columns=['x', 'y', 'z'])
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

    def scatters(self):
        """
        Plot scattered markers for atoms.
        """
        ele_vdw = [(x.ele, self.data_reader.vdws[x.id].dist * self.scale)
                   for x in self.data_reader.masses.values()]
        for ele, size in set(ele_vdw):
            idx = (self.data[['element', 'size']] == [ele, size]).all(axis=1)
            data = self.data[idx]
            marker = dict(size=size, color=data['color'].values[0])
            marker = graph_objects.Scatter3d(x=data.x,
                                             y=data.y,
                                             z=data.z,
                                             opacity=0.9,
                                             mode='markers',
                                             name=ele,
                                             marker=marker)
            self.fig.add_trace(marker)

    def lines(self):
        """
        Plot lines for bonds.
        """
        for bond in self.data_reader.bonds.values():
            atom1 = self.data_reader.atoms[bond.id1]
            atom2 = self.data_reader.atoms[bond.id2]
            pnt1 = np.array(atom1.xyz)
            pnt2 = np.array(atom2.xyz)
            mdp = (pnt1 + pnt2) / 2
            self.line(np.array([pnt1, mdp]), atom1)
            self.line(np.array([mdp, pnt2]), atom2)

    def line(self, xyz, atom):
        """
        Plot half bond spanning from one atom to the mid point.

        :param xyz `numpy.ndarray`: the bond xyz span
        :param atom 'types.SimpleNamespace': the bonded atom
        """
        line = dict(width=8, color=self.data.xs(atom.id).color)
        line = graph_objects.Scatter3d(x=xyz[:, 0],
                                       y=xyz[:, 1],
                                       z=xyz[:, 2],
                                       opacity=0.8,
                                       mode='lines',
                                       showlegend=False,
                                       line=line)
        self.fig.add_trace(line)

    def show(self):
        """
        Show the figure with plot.
        """
        camera = dict(up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=1.25, y=1.25, z=1.25))
        self.fig.update_layout(template='plotly_dark', scene_camera=camera)
        self.fig.show()
