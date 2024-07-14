import itertools
import numpy as np
import pandas as pd
import more_itertools
import plotly.graph_objects as graph_objects

from nemd import traj
from nemd import nmendeleev


class FrameView:
    """Viewer datafile and trajectory frame"""

    XYZU = traj.Frame.XYZU
    X_ELE = 'X'
    X_SIZE = 20
    # Color from https://webmail.life.nthu.edu.tw/~fmhsu/rasframe/CPKCLRS.HTM
    X_COLOR = '#FF1493'
    ELEMENT = traj.Frame.ELEMENT
    SIZE = 'size'
    COLOR = 'color'

    def __init__(self, df_reader=None, scale=5.):
        """
        :param df_reader `nemd.oplsua.DataFileReader`: datafile reader with
            atom, bond, element and other info.
        :param scale float: scale the LJ dist by this to obtain marker size.
        """
        self.df_reader = df_reader
        self.fig = graph_objects.Figure()
        self.scale = scale
        self.data = None
        self.markers = []
        self.lines = []
        self.edges = []

    def setData(self):
        """
        Set data frame with coordinates, elements, marker sizes, and color info.
        """
        if not self.df_reader:
            return
        elements = self.df_reader.elements
        size_map = {i: x for i, x in self.df_reader.pair_coeffs.dist.items()}
        sizes = self.df_reader.atoms.type_id.map(size_map).to_frame()
        unique_elements = set(self.df_reader.masses.element)
        color_map = {
            x: nmendeleev.element(x).cpk_color
            for x in unique_elements
        }
        colors = elements.element.map(color_map).to_frame()
        data = [self.df_reader.atoms[self.XYZU], elements, sizes, colors]
        self.data = traj.Frame(pd.concat(data, axis=1),
                               box=self.df_reader.box,
                               columns=traj.Frame.XYZU_ELE_SZ_CLR)
        breakpoint()

    def updateDataWithFrm(self, frm):
        """
        Update the data according to the trajectory frame.

        :param frm 'nemd.traj.Frame': coordinate frame to update with
        """
        self.data[self.XYZU] = frm[self.XYZU]
        self.data.setBox(frm.getBox())

    def addTraces(self):
        """
        Add traces to the figure.
        """
        self.fig.add_traces(self.markers + self.lines + self.edges)

    def setFrames(self, frms):
        """
        Set animation from trajectory frames.

        :param frms generator of 'nemd.traj.Frame': the trajectory frames to
            create the animation from.
        """
        if self.data is None:
            frm = self.setDataFromTraj(frms)
            frms = itertools.chain([frm], frms)
            self.setEleSz()
            self.setScatters()
            self.setLines()
            self.setEdges()
            self.addTraces()

        fig_frms = []
        for idx, frm in enumerate(frms):
            self.updateDataWithFrm(frm)
            self.setScatters()
            self.setLines()
            self.setEdges()
            data = self.markers + self.lines + self.edges
            fig_frm = graph_objects.Frame(data=data, name=f'{idx}')
            fig_frms.append(fig_frm)
        self.fig.update(frames=fig_frms)

    def setDataFromTraj(self, frms):
        """
        Set the data from trajectory frames.

        :param frms generator of 'nemd.traj.Frame': the trajectory frames to
            create the animation from.

        :return 'nemd.traj.Frame': the first trajectory frame.
        """
        # peekable doesn't work for yield generator
        frm = more_itertools.peekable(frms).peek()
        ele_sz_clr = self.ELE_SZ_CLR.copy()
        try:
            ele_sz_clr[self.ELEMENT] = frm.pop(self.ELEMENT)
        except KeyError:
            ele_sz_clr[self.ELEMENT] = [self.X_ELE] * frm.shape[0]
            ele_sz_clr[self.COLOR] = [self.X_COLOR] * frm.shape[0]
        else:
            element = set(ele_sz_clr[self.ELEMENT])
            color = {
                x:
                self.X_COLOR
                if x == self.X_ELE else nmendeleev.element(x).cpk_color
                for x in element
            }
            ele_sz_clr[self.COLOR] = [
                color[x] for x in ele_sz_clr[self.ELEMENT]
            ]
        finally:
            ele_sz_clr[self.SIZE] = [self.X_SIZE] * frm.shape[0]

        sz_clr = pd.DataFrame(ele_sz_clr, index=range(1, frm.shape[0] + 1))
        data = pd.concat((frm, sz_clr), axis=1)
        self.data = traj.Frame(data,
                               box=frm.getBox(),
                               columns=traj.Frame.XYZU_ELE_SZ_CLR,
                               dtype=None)
        return frm

    def setScatters(self):
        """
        Set scattered markers for atoms.

        :return markers list of 'plotly.graph_objs._scatter3d.Scatter3d':
            the scatter markers to represent atoms.
        """
        if self.data is None:
            return
        self.markers = []
        for ele, size in self.ele_sz:
            idx = (self.data[['element', 'size']] == [ele, size]).all(axis=1)
            data = self.data[idx]
            marker = dict(size=size, color=data['color'].values[0])
            marker = graph_objects.Scatter3d(x=data.xu,
                                             y=data.yu,
                                             z=data.zu,
                                             opacity=0.9,
                                             mode='markers',
                                             name=ele,
                                             marker=marker,
                                             hovertemplate='%{customdata}',
                                             customdata=data.index.values)
            self.markers.append(marker)

    def setEleSz(self):
        """
        Set elements and sizes.
        """
        if self.data is None:
            return
        ele_sz = self.data[[self.ELEMENT, self.SIZE]]
        ele_sz = set([tuple(y.values) for x, y in ele_sz.iterrows()])
        self.ele_sz = sorted(set(ele_sz), key=lambda x: x[1], reverse=True)

    def setLines(self):
        """
        Set lines for bonds.

        :return markers list of 'plotly.graph_objs._scatter3d.Scatter3d':
            the line markers to represent bonds.
        """
        if self.df_reader is None:
            return
        self.lines = []
        for bond in self.df_reader.bonds.values():
            atom1 = self.df_reader.atoms[bond.id1]
            atom2 = self.df_reader.atoms[bond.id2]
            pnts = self.data.loc[[atom1.id, atom2.id]][self.XYZU]
            mid = pnts.mean().to_frame().transpose()
            pnts = np.concatenate((pnts.values, mid.values))
            pnts = pd.DataFrame(pnts, columns=self.XYZU)
            self.setline(pnts[::2], atom1)
            self.setline(pnts[1::], atom2)

    def setline(self, xyz, atom):
        """
        Set half bond spanning from one atom to the mid point.

        :param xyz `numpy.ndarray`: the bond XYZU span
        :param atom 'types.SimpleNamespace': the bonded atom
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
                                       line=line,
                                       hoverinfo='skip')
        self.lines.append(line)

    def setEdges(self):
        """
        Set box edges.
        """
        self.edges = []
        edges = self.data.getEdges()
        for edge in edges:
            self.setEdge(edge)

    def setEdge(self, xyzs):
        """
        Set a box edges.

        :param xyzs list of list: start and end points of the edge.
        """
        xyzs = np.array(xyzs)
        edge = graph_objects.Scatter3d(x=xyzs[:, 0],
                                       y=xyzs[:, 1],
                                       z=xyzs[:, 2],
                                       opacity=0.5,
                                       mode='lines',
                                       showlegend=False,
                                       hoverinfo='skip',
                                       line=dict(width=8, color='#b300ff'))
        self.edges.append(edge)

    def clearData(self):
        """
        Clear the atom and bond plots.
        """
        self.data = None
        self.fig.data = []
        self.fig.frames = []
        self.markers = []
        self.lines = []
        self.edges = []

    def updateLayout(self):
        """
        Update the figure layout.
        """
        buttons = None
        if self.fig.frames:
            buttons = [
                dict(label="Play",
                     method="animate",
                     args=[None, dict(fromcurrent=True)]),
                dict(label='Pause',
                     method="animate",
                     args=[[None], dict(mode='immediate')])
            ]
        updatemenu = dict(type="buttons",
                          buttons=buttons,
                          showactive=False,
                          font={'color': '#000000'},
                          direction="left",
                          pad=dict(r=10, t=87),
                          xanchor="right",
                          yanchor="top",
                          x=0.1,
                          y=0)
        self.fig.update_layout(template='plotly_dark',
                               scene=self.getScene(),
                               sliders=self.getSliders(),
                               updatemenus=[updatemenu],
                               overwrite=True,
                               uirevision=True)

    def getSliders(self):
        """
        Get the sliders for the trajectory frames.

        :return list of dict: add the these slider bars to he menus.
        """
        if not self.fig.frames:
            return []
        slider = dict(active=0,
                      yanchor="top",
                      xanchor="left",
                      x=0.1,
                      y=0,
                      pad=dict(b=10, t=50),
                      len=0.9,
                      transition={
                          "duration": 300,
                          "easing": "cubic-in-out"
                      },
                      currentvalue=dict(prefix='Frame:',
                                        visible=True,
                                        xanchor='right'))
        slider['steps'] = [
            dict(label=x['name'],
                 method='animate',
                 args=[[x['name']], dict(mode='immediate')])
            for x in self.fig.frames
        ]
        return [slider]

    def getScene(self):
        """
        Return the scene with axis range and styles.

        :return dict: keyword arguments for preference.
        """
        data = None
        if self.fig.data:
            data = np.concatenate([
                np.array([i['x'], i['y'], i['z']]).transpose()
                for i in self.fig.data
            ])
        if self.fig.frames:
            datas = np.concatenate([
                np.array([j['x'], j['y'], j['z']]).transpose()
                for i in self.fig.frames for j in i['data']
            ])
            data = np.concatenate(
                (data, datas), axis=0) if self.fig.data else datas
        if data is None:
            return
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        dspan = (dmax - dmin).max()
        cnt = data.mean(axis=0)
        lbs = cnt - dspan
        hbs = cnt + dspan
        return dict(xaxis=dict(range=[lbs[0], hbs[0]], autorange=False),
                    yaxis=dict(range=[lbs[1], hbs[1]], autorange=False),
                    zaxis=dict(range=[lbs[2], hbs[2]], autorange=False),
                    aspectmode='cube')

    def show(self, *arg, **kwargs):
        """
        Show the figure with plot.
        """
        self.fig.show(*arg, **kwargs)
