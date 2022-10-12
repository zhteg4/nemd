import environutils


class TempEnePlotter(object):

    def __init__(
        self,
        lammp_temp,
        lammps_energy,
        jobname,
    ):
        self.lammp_temp = lammp_temp
        self.lammps_energy = lammps_energy
        self.jobname = jobname
        self.interactive = environutils.is_interactive()
        self.fig_file = self.jobname + '.png'
        self.fig_nrows = 2
        self.fig_ncols = 1

    def load(self):
        self.temp_data = self.lammp_temp.data
        self.ene_data = self.lammps_energy.data
        self.fitted_temp_data = self.lammp_temp.fitted_data
        self.fitted_ene_data = self.lammps_energy.fitted_data
        self.temp_data_nrow, self.temp_data_ncol, self.temp_data_nblock = self.temp_data.shape
        self.ene_names = self.ene_data.dtype.names

    def setFigure(self):
        if self.interactive:
            return

        import matplotlib
        self.old_backed = matplotlib.pyplot.get_backend()
        matplotlib.use("agg", force=True)
        from matplotlib import pyplot as plt

        self.fig = plt.figure()
        self.temp_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols,
                                              1)
        self.ene_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols, 2)

    def plot(self):
        self.load()
        self.setFigure()
        self.plotTemp()
        self.plotEne()
        self.setLayout()
        self.show()
        self.save()
        self.resetMatplotlib()

    def save(self):
        self.fig.savefig(self.fig_file)

    def resetMatplotlib(self):

        if self.interactive:
            return
        import matplotlib
        matplotlib.use(self.old_backed)

    def plotEne(self):
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           -self.ene_data[self.ene_names[2]],
                           label=self.ene_names[2])
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           self.ene_data[self.ene_names[3]],
                           label=self.ene_names[3])
        if self.fitted_ene_data is not None:
            self.ene_axis.plot(self.fitted_ene_data[:, 0],
                               self.fitted_ene_data[:, 1],
                               label='Fitted')
        self.ene_axis.set_xlabel(self.ene_names[0])
        self.ene_axis.set_ylabel(f'Energy {self.ene_names[3].split()[-1]}')
        self.ene_axis.legend(loc='upper left', prop={'size': 6})

    def plotTemp(self):

        for iblock in range(self.temp_data_nblock - 1):
            self.temp_axis.plot(self.temp_data[:, 1, iblock],
                                self.temp_data[:, 3, iblock],
                                '.',
                                label=f'Block {iblock}')
        self.temp_axis.plot(self.temp_data[:, 1, -1],
                            self.temp_data[:, 3, -1],
                            label='Average')
        if self.fitted_temp_data is not None:
            self.temp_axis.plot(self.fitted_temp_data[:, 0],
                                self.fitted_temp_data[:, 1],
                                label='Fitted')
        self.temp_axis.legend(loc='upper right', prop={'size': 6})
        self.temp_axis.set_ylim([270, 330])
        self.temp_axis.set_xlabel('Coordinate (Angstrom)')
        self.temp_axis.set_ylabel('Temperature (K)')

    def setLayout(self):
        self.fig.tight_layout()

    def show(self):
        if not self.interactive:
            return

        self.fig.show()
        input(
            'Showing the temperature profile and energy plots. Press any keys to continue...'
        )
