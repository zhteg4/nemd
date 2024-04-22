"""
pip3 install setuptools openvino-telemetry
pip3 install .[dev] -v --break-system-packages
"""
import os
import sys
import glob
import subprocess
from setuptools import setup
from setuptools.command.install import install


class DarwinInstall:

    LMP = 'lmp_serial'
    ALM = 'alm'
    LOCAL_BIN = '/usr/local/bin'
    BUILD_LMP = f'submodule/lammps/build/{LMP}'
    BUILD_ALM = f'submodule/alamode/build/{ALM}/{ALM}'

    def __init__(self):
        build_lmp = os.path.join(os.getcwd(), self.BUILD_LMP)
        local_lmp = os.path.join(self.LOCAL_BIN, self.LMP)
        self.ln_lmp = f'ln -sf {build_lmp} {local_lmp}'
        build_alm = os.path.join(os.getcwd(), self.BUILD_ALM)
        local_alm = os.path.join(self.LOCAL_BIN, self.ALM)
        self.ln_alm = f'ln -sf {build_alm} {local_alm}'

    def run(self):
        """
        Main method to run.
        """
        self.lmp_found = self.checkLammps()
        self.lammpsPrereq()
        self.installLammps()
        self.alm_found = self.checkAlamode()
        self.alamodePrereq()
        self.installAlamode()
        self.installQt()

    def checkLammps(self):
        """
        Check whether lammps executable can be found.

        :return: True if lammps executable with python package found.
        """

        lmp = subprocess.run(f'{self.LMP} -h | grep PYTHON',
                             capture_output=True,
                             shell=True)
        if lmp.stdout:
            print('Lammps executable with python package found.')
            return True

        if os.path.isfile(self.BUILD_LMP):
            print(f'Creating soft link to lammps executable {self.BUILD_LMP}')
            subprocess.run(self.ln_lmp, shell=True)
            return True

        print('Lammps executable with python package not found.')
        return False

    def lammpsPrereq(self):
        """
        Install the packages requird by lammps compilation.
        """
        if self.lmp_found:
            return
        subprocess.run('brew install clang-format', shell=True)

    def installLammps(self, std_cmake_args=None):
        """
        Install the lammps with specific packages if not available.

        :param std_cmake_args: the additional packages or flags.
        """
        if self.lmp_found:
            return
        print('Installing lammps...')
        cmd = 'cd submodule/lammps; bash install.sh'
        if std_cmake_args:
            cmd += f" {std_cmake_args}"
        subprocess.run(cmd, shell=True)
        subprocess.run(self.ln_lmp, shell=True)

    def checkAlamode(self):
        """
        Check whether alamode executable can be found.

        :return: True if lammps executable with python package found.
        """
        alm = subprocess.run(f"which {self.ALM}",
                             capture_output=True,
                             shell=True)
        if alm.stdout:
            print('Alamode executable found.')
            return True

        if os.path.isfile(self.BUILD_ALM):
            print(f'Creating soft link to alamode executable {self.BUILD_ALM}')
            subprocess.run(self.ln_alm, shell=True)
            return True

        print('Alamode executable not found.')
        return False

    def alamodePrereq(self):
        """
        Install the packages required by alamode compilation.
        """

        if self.lmp_found:
            return
        subprocess.run(
            'brew install gcc lapack open-mpi libomp boost eigen spglib fftw cmake llvm',
            shell=True)

    def installAlamode(self, std_cmake_args=None):
        """
        Install the alamode with specific packages if not available.

        :param std_cmake_args: the additional packages or flags.
        """
        if self.alm_found:
            return
        print('Installing alamode...')
        cmd = 'cd submodule/alamode; bash install.sh'
        if std_cmake_args:
            cmd += f" {std_cmake_args}"
        subprocess.run(cmd, shell=True)
        subprocess.run(self.ln_alm, shell=True)

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

    LOCAL_BIN = "$HOME/.local/bin"
    NVDIA = "nvidia-smi | grep 'NVIDIA-SMI'"

    def __init__(self):
        super().__init__()
        self.gpu = False

    def run(self):
        super().run()
        self.installTerm()

    def checkLammps(self):
        """
        See parent class.
        """
        super().checkLammps()
        if self.lmp_found:
            return
        nvdia = subprocess.run(self.NVDIA, capture_output=True, shell=True)
        self.gpu = bool(nvdia.stdout)
        print(f"GPU found as 'nvdia.stdout'")

    def lammpsPrereq(self):
        """
        Install the packages required by lammps compilation.
        """
        if self.lmp_found:
            return
        print('Installing lammps prerequisites...')
        # zsh for install.sh
        # python3-venv, clang-format for make install-python
        # lammps cmake: Found FFMPEG
        packages = (
            "zsh python3-venv clang-format lsb-release gcc openmpi-bin "
            "cmake python3-apt python3-setuptools openmpi-common "
            "libopenmpi-dev libgtk2.0-dev fftw3 fftw3-dev ffmpeg")
        if self.gpu:
            packages += " nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc"
        subprocess.run(f'sudo apt-get install {packages} -y', shell=True)

    def installLammps(self):
        """
        See parent method for docs.
        """
        std_cmake_args = "-D PKG_GPU=on" if self.gpu else ""
        super().installLammps(std_cmake_args=std_cmake_args)

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
          'numpy', 'scipy', 'networkx', 'pandas', 'more_itertools',
          'chemparse', 'mendeleev', 'rdkit', 'signac', 'signac-flow',
          'matplotlib', 'plotly', 'dash_bootstrap_components', 'pytest',
          'dash[testing]', 'pyqt5', 'webdriver-manager', 'flask', 'openpyxl',
          'sh', 'humanfriendly', 'Pillow', 'pyvim', 'adjustText', 'crystals'
      ],
      extras_require={
          'dev': [
              'ipdb', 'ipython', 'notebook', 'jupyterlab', 'yapf',
              'RBTools == 4.1', 'snakeviz', 'pyvim', 'remote_pdb'
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
