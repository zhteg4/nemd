import numpy as np
import pandas as pd
import plotly.graph_objects as graph_objects

from nemd import traj
from nemd import nmendeleev


class FrameView:
    """Viewer datafile and trajectory frame"""

    XYZU = traj.Frame.XYZU
    ELEMENT = 'element'
    SIZE = 'size'
    COLOR = 'color'
    X_ELE = 'X'
    X_SIZE = 20
    # Color from https://webmail.life.nthu.edu.tw/~fmhsu/rasframe/CPKCLRS.HTM
    X_COLOR = '#FF1493'
    ELE_SZ_CLR = {ELEMENT: X_ELE, SIZE: X_SIZE, COLOR: X_COLOR}

    def __init__(self, df_reader=None):
        """
        :param df_reader `nemd.oplsua.DataFileReader`: datafile reader with
            atom, bond, element and other info.
        """
        self.df_reader = df_reader
        self.fig = graph_objects.Figure()
        self.data = None
        self.elements = None

    def setData(self, frm=None):
        """
        Set data frame with coordinates, elements, marker sizes, and color info.

        :param frm 'nemd.traj.Frame': the frame to create the data from.
        """
        if not self.df_reader:
            self.data = frm.assign(**self.ELE_SZ_CLR)
            return

        elements = self.df_reader.elements
        smap = {i: x for i, x in self.df_reader.pair_coeffs.dist.items()}
        sizes = self.df_reader.atoms.type_id.map(smap).to_frame(name=self.SIZE)
        uq_elements = set(self.df_reader.masses.element)
        color_map = {x: nmendeleev.element(x).cpk_color for x in uq_elements}
        colors = elements.element.map(color_map).to_frame(name=self.COLOR)
        data = [self.df_reader.atoms[self.XYZU], elements, sizes, colors]
        self.data = traj.Frame(pd.concat(data, axis=1), box=self.df_reader.box)

    def setElements(self):
        """
        Set elements and sizes.
        """
        if self.data is None:
            return

        elements = self.data[[self.ELEMENT, self.SIZE]].drop_duplicates()
        elements.sort_values(by=self.SIZE, ascending=False, inplace=True)
        self.elements = elements.element

    @property
    def scatters(self):
        """
        Set scattered markers for atoms.

        :return list of Scatter3d: the scattered markers to represent atoms.
        """
        if self.data is None:
            return []

        data = []
        for element in self.elements:
            idx = self.data[self.ELEMENT] == element
            selected = self.data[idx]
            sizes = selected[self.SIZE]
            colors = selected[self.COLOR]
            marker = dict(size=sizes.values[0], color=colors.values[0])
            marker = graph_objects.Scatter3d(x=selected.xu,
                                             y=selected.yu,
                                             z=selected.zu,
                                             opacity=0.9,
                                             mode='markers',
                                             name=element,
                                             marker=marker,
                                             hovertemplate='%{customdata}',
                                             customdata=selected.index.values)
            data.append(marker)
        return data

    @property
    def lines(self):
        """
        Set lines for bonds.

        :return list of Scatter3d: the line traces to represent bonds.
        """
        if self.df_reader is None:
            return []

        data = []
        for _, _, atom_id1, atom_id2 in self.df_reader.bonds.itertuples():
            pnts = self.data.loc[[atom_id1, atom_id2]][self.XYZU]
            pnts = pd.concat([pnts, pnts.mean().to_frame().transpose()])
            data.append(self.getLine(pnts[::2], atom_id1))
            data.append(self.getLine(pnts[1::], atom_id2))
        return data

    def getLine(self, xyz, atom_id):
        """
        Set half bond spanning from one atom to the middle point.

        :param xyz `numpy.ndarray`: the bond XYZU span
        :param atom_id int: one bonded atom id
        :return 'Scatter3d': the line markers to represent bonds.
        """
        line = dict(width=8, color=self.data.xs(atom_id).color)
        line = graph_objects.Scatter3d(x=xyz.xu,
                                       y=xyz.yu,
                                       z=xyz.zu,
                                       opacity=0.8,
                                       mode='lines',
                                       showlegend=False,
                                       line=line,
                                       hoverinfo='skip')
        return line

    @property
    def edges(self):
        """
        Set box edges.

        :return list of Scatter3d: the box edges markers.
        """
        data = []
        for edge in self.data.box.edges:
            edge = graph_objects.Scatter3d(x=edge[:, 0],
                                           y=edge[:, 1],
                                           z=edge[:, 2],
                                           opacity=0.5,
                                           mode='lines',
                                           showlegend=False,
                                           hoverinfo='skip',
                                           line=dict(width=8, color='#b300ff'))
            data.append(edge)
        return data

    def addTraces(self):
        """
        Add traces to the figure.
        """
        self.fig.add_traces(self.scatters + self.lines + self.edges)

    def setFrames(self, frms):
        """
        Set animation from trajectory frames.

        :param frms generator of 'nemd.traj.Frame': the trajectory frames to
            create the animation from.
        """

        frames = []
        for idx, frm in enumerate(frms):
            self.data.update(frm)
            data = self.scatters + self.lines + self.edges
            frame = graph_objects.Frame(data=data, name=f'{idx}')
            frames.append(frame)
        self.fig.update(frames=frames)

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

    def getScene(self, autorange=False):
        """
        Return the scene with axis range and styles.

        :param autorange bool: whether to let the axis range be adjusted
            automatically.
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
        dspan = (dmax - dmin).max() / 2
        cnt = data.mean(axis=0)
        lbs = (cnt - dspan)
        hbs = (cnt + dspan)
        return dict(xaxis=dict(range=[lbs[0], hbs[0]], autorange=autorange),
                    yaxis=dict(range=[lbs[1], hbs[1]], autorange=autorange),
                    zaxis=dict(range=[lbs[2], hbs[2]], autorange=autorange),
                    aspectmode='cube')

    def show(self, *arg, outfile=None, inav=False, **kwargs):
        """
        Show the figure with plot.

        :param outfile str: the output file name for the figure.
        :param inav bool: whether to show the figure in an interactive viewer.
        """
        self.fig.write_html(outfile)
        if inav:
            self.fig.show(*arg, **kwargs)

    def reset(self):
        """
        Reset the state.
        """
        self.fig.data = []
        self.fig.frames = []
        self.data = None
        self.elements = None
