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
        self.data = None

    def setData(self):
        """
        Set data frame with coordinates, elements, marker sizes, and color info.
        """
        if not self.data_reader:
            return
        # Index, XU, YU, ZU
        index = list(self.data_reader.atoms.keys())
        xyz = np.array([x.xyz for x in self.data_reader.atom])
        data = pd.DataFrame(xyz, index, columns=self.XYZU)
        # Element, Size, Color
        self.data_reader.setMinimumDist()
        type_ids = [x.type_id for x in self.data_reader.atom]
        sizes = [self.data_reader.vdws[x].dist * self.scale for x in type_ids]
        element = {x: y.ele for x, y in self.data_reader.masses.items()}
        elements = [element[x] for x in type_ids]
        color = {x: mendeleev.element(y).cpk_color for x, y in element.items()}
        color = [color[x] for x in type_ids]
        sz_clr = pd.DataFrame(
            {
                'element': elements,
                'size': sizes,
                'color': color
            }, index=index)
        self.data = pd.concat((data, sz_clr), axis=1)

    def updateCoords(self, frm):
        """
        Update the coordinate data according to the trajectory frame.

        :param frm 'nemd.traj.Frame': coordinate frame to update with
        """
        self.data[self.XYZU] = frm

    def setFrames(self, frms):
        """
        Set animation from trajectory frames.

        :param frms generator of 'nemd.traj.Frame': the trajectory frames to
            create the animation from.
        """
        import pdb;pdb.set_trace()
        fig_frms = []
        for idx, frm in enumerate(frms):
            self.updateCoords(frm)
            data = self.plotScatters(add_to_fig=False)
            data += self.plotLines(add_to_fig=False)
            fig_frms.append(graph_objects.Frame(data=data, name=f'{idx}'))
        self.fig.update(frames=fig_frms)

    def plotScatters(self, add_to_fig=True):
        """
        Plot scattered markers for atoms.

        :param add_to_fig bool: add to the fig trace if True
        :return markers list of 'plotly.graph_objs._scatter3d.Scatter3d':
            the scatter markers to represent atoms.
        """
        if not self.data_reader:
            return
        markers = []
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
            if add_to_fig:
                self.fig.add_trace(marker)
            markers.append(marker)
        return markers

    def plotLines(self, add_to_fig=True):
        """
        Plot lines for bonds.

        :param add_to_fig bool: add to the fig trace if True
        :return markers list of 'plotly.graph_objs._scatter3d.Scatter3d':
            the line markers to represent bonds.
        """
        if not self.data_reader:
            return
        lines = []
        for bond in self.data_reader.bonds.values():
            atom1 = self.data_reader.atoms[bond.id1]
            atom2 = self.data_reader.atoms[bond.id2]
            pnts = self.data.loc[[atom1.id, atom2.id]][self.XYZU]
            pnts = pd.concat((pnts, pnts.mean().to_frame().transpose()))
            lines.append(self.plotline(pnts[::2], atom1,
                                       add_to_fig=add_to_fig))
            lines.append(self.plotline(pnts[1::], atom2,
                                       add_to_fig=add_to_fig))
        return lines

    def plotline(self, xyz, atom, add_to_fig=True):
        """
        Plot half bond spanning from one atom to the mid point.

        :param xyz `numpy.ndarray`: the bond XYZU span
        :param atom 'types.SimpleNamespace': the bonded atom
        :param add_to_fig bool: add to the fig trace if True
        :return markers 'plotly.graph_objs._scatter3d.Scatter3d':
            the line markers to represent bonds.
        """
        line = dict(width=8, color=self.data.xs(atom.id).color)
        line = graph_objects.Scatter3d(x=xyz.xu,
                                       y=xyz.yu,
                                       z=xyz.zu,
                                       opacity=0.8,
                                       mode='lines',
                                       showlegend=False,
                                       line=line)
        if add_to_fig:
            self.fig.add_trace(line)
        return line

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
        buttons = None
        if self.fig.frames:
            buttons = [dict(label="Play", method="animate", args=[None])]
        self.fig.update_layout(
            template='plotly_dark',
            scene=self.getScene(),
            scene_camera=camera,
            updatemenus=[dict(type="buttons", buttons=buttons)],
        )

    def getScene(self):
        """
        Return the scene with axis range and styles.

        :return dict: keyword arguments for preference.
        """
        if self.fig.frames:
            data = self.fig.frames[0]['data'][0]
            data = np.array([data['x'], data['y'], data['z']]).transpose()
        elif self.data is not None:
            data = self.data[self.XYZU].to_numpy()
        else:
            return
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        dspan = ((dmax - dmin) * 2).max()
        cnt = data.mean(axis=0)
        lbs = cnt - dspan
        hbs = cnt + dspan
        return dict(xaxis=dict(range=[lbs[0], hbs[0]], autorange=False),
                    yaxis=dict(range=[lbs[1], hbs[1]], autorange=False),
                    zaxis=dict(range=[lbs[2], hbs[2]], autorange=False),
                    aspectmode='cube')

    def show(self):
        """
        Show the figure with plot.
        """
        self.fig.show()
