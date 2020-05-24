import environutils
from matplotlib import pyplot as plt


class TempEnePlotter(object):
    def __init__(self, temp_data, ene_data, jobname):
        self.temp_data = temp_data
        self.ene_data = ene_data
        self.jobname = jobname
        self.interactive = environutils.is_interactive()
        self.fig_file = jobname + '.png'
        self.fig_nrows = 2
        self.fig_ncols = 1

        self.temp_data_nrow, self.temp_data_ncol, self.temp_data_nblock = self.temp_data.shape
        self.ene_names = self.ene_data.dtype.names

    def setup(self):
        self.fig = plt.figure()
        self.temp_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols,
                                              1)
        self.ene_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols, 2)

    def plot(self):
        self.setup()
        self.plotTemp()
        self.plotEne()
        self.setLayout()
        self.show()
        self.save()

    def save(self):
        self.fig.savefig(self.fig_file)

    def plotEne(self):
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           -self.ene_data[self.ene_names[2]],
                           label=self.ene_names[2])
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           self.ene_data[self.ene_names[3]],
                           label=self.ene_names[3])
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
        self.temp_axis.legend(loc='upper right', prop={'size': 6})
        self.temp_axis.set_ylim([270, 330])
        self.temp_axis.set_xlabel('Coordinate (Angstrom)')
        self.temp_axis.set_ylabel('Temperature (K)')

    def setLayout(self):
        plt.tight_layout()

    def show(self):
        if not self.interactive:
            return

        self.fig.show()
        input('Press any keys to continue...')
