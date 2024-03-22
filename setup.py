"""
pip3 install setuptools
pip3 install ".[dev]" -v
"""
import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup
from setuptools.command.install import install


class DarwinInstall:

    LMP = 'lmp_serial'
    LMP_PY = f'{LMP} -h | grep PYTHON'
    USR_LOCAL_BIN = '/usr/local/bin'
    BUILD_LMP = f'submodule/lammps/build/{LMP}'
    USR_LOCAL_LMP = os.path.join(USR_LOCAL_BIN, LMP)

    def __init__(self):
        self.lmp_found = True
        build_lmp = os.path.join(os.getcwd(), self.BUILD_LMP)
        self.ln_lmp = f'ln -sf {build_lmp} {self.USR_LOCAL_LMP}'

    def run(self):
        self.checkLammps()
        self.installLammps()
        self.installQt()

    def checkLammps(self):
        """
        Check whether lammps executable can be found.
        """

        lmp_py = subprocess.run(self.LMP_PY, capture_output=True, shell=True)
        if lmp_py.stdout:
            print('Lammps executable with python package found.')
            return

        if os.path.isfile(self.BUILD_LMP):
            print(f'Creating soft link to lammps executable {self.BUILD_LMP}')
            subprocess.run(self.ln_lmp, shell=True)
            return

        print('Lammps executable with python package not found.')
        self.lmp_found = False

    def installLammps(self):
        """
        Install the lammps with specific packages if not available.
        """
        if self.lmp_found:
            return
        print('Installing lammps...')
        subprocess.run('cd submodule/lammps; bash install.sh', shell=True)
        subprocess.run(self.ln_lmp, shell=True)

    def installQt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        qt = subprocess.run('brew list qt5', capture_output=True, shell=True)
        if qt.stdout:
            print('qt installation found.')
            return
        print('qt installation not found. Installing...')
        subprocess.run('brew install qt5', shell=True)


class LinuxInstall(DarwinInstall):

    def run(self):
        super().run()
        self.installTerm()

    def installLammps(self):
        """
        Install the lammps with specific packages if not available.
        """
        if self.lmp_found:
            return
        print('Installing lammps prerequisites...')
        # zsh for install.sh
        # python3-venv for make install-python
        subprocess.run(
            'sudo apt-get install zsh python3-venv lsb-release gcc '
            'openmpi-bin cmake python3-apt python3-setuptools openmpi-common '
            'libopenmpi-dev libgtk2.0-dev -y',
            shell=True)
        super().installLammps()

    def installQt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        subprocess.run(
            'sudo apt-get install build-essential libgl1-mesa-dev qt6-base-dev -y',
            shell=True)
        subprocess.run(
            "sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev -y",
            shell=True)

    def installTerm(self):
        """
        Install terminal supporting split view.
        """
        subprocess.run('sudo apt install tilix -y', shell=True)


class CustomInstallCommand(install):

    DARWIN = 'darwin'
    LINUX = 'linux'
    INSTALLER = {DARWIN: DarwinInstall, LINUX: LinuxInstall}

    def run(self):
        """
        Main method to run post installation.
        """
        self.setPlatform()
        self.install()
        install.run(self)

    def setPlatform(self):
        """
        Set the platform, such as win32, darwin, linux2 and so on.
        """
        self.platform = sys.platform
        print(f"***** Platform: {self.platform} *****")

    def install(self):
        """
        Install packages outside regular install_requires.
        """
        installer = self.INSTALLER[self.platform]()
        installer.run()


setup(name='nemd',
      version='0.1.0',
      description='A molecular dynamics simulation software',
      url='https://github.com/zhteg4/nemd',
      author='Teng Zhang',
      author_email='zhteg4@gmail.com',
      license='BSD 3-clause',
      packages=['nemd'],
      package_dir={'nemd': 'module/nemd'},
      package_data={'nemd': ['ff/*.prm']},
      scripts=glob.glob('bash_scripts/*') + glob.glob('scripts/*.py') +
      glob.glob('workflows/*.py'),
      install_requires=[
          'numpy == 1.24.3', 'scipy == 1.10.1', 'networkx == 3.1',
          'pandas == 2.0.2', 'more_itertools == 9.1.0', 'chemparse == 0.1.2',
          'mendeleev == 0.14.0', 'rdkit == 2023.3.1', 'signac == 2.0.0',
          'signac-flow == 0.25.1', 'matplotlib == 3.7.1', 'plotly ==5.15.0',
          'dash_bootstrap_components', 'pytest == 7.3.2', 'dash[testing]',
          'pyqt5 == 5.15.7', 'webdriver-manager == 3.8.6', 'flask >= 2.2.5',
          'openpyxl == 3.1.2', 'sh == 2.0.4', 'humanfriendly == 10.0',
          'Pillow == 9.4.0', 'pyvim', 'adjustText'
      ],
      extras_require={
          'dev': [
              'ipdb', 'ipython', 'notebook', 'jupyterlab', 'yapf',
              'RBTools == 4.1', 'snakeviz', 'pyvim'
          ]
      },
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
      ],
      cmdclass={'install': CustomInstallCommand})
