"""
pip3 install setuptools
pip3 install ".[dev]" -v
"""
import os
import sys
import shutil
import subprocess
from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    LMP_PY = 'lmp_mpi -h | grep PYTHON'
    STD_CMAKE_ARGS = '*std_cmake_args\n'
    DPKG_PYTHON = '"-DPKG_PYTHON=yes",\n'
    PYTHON_STD = f'{DPKG_PYTHON} {STD_CMAKE_ARGS}'
    DARWIN = 'darwin'
    LINUX = 'linux'
    BUILD = 'build'
    LAMMPS_RB = 'lammps.rb'
    BUILD_LAMMPS_RB = os.path.join(BUILD, LAMMPS_RB)

    def run(self):
        """
        Main method to run post installation.
        """
        install.run(self)
        self.setPlatform()
        self.installQt()
        self.installLammps()
        self.installTerm()

    def setPlatform(self):
        """
        Set the platform, such as win32, darwin, linux2 and so on.
        """
        self.platform = sys.platform
        print(f"***** Platform: {self.platform} *****")

    def installLammps(self):
        """
        Install the lammps with specific packages if not available.
        """
        lmp_py = subprocess.run(self.LMP_PY, capture_output=True, shell=True)
        if lmp_py.stdout:
            print('Lammps executable with python package found.')
            return
        print('Lammps executable with python package not found. Installing...')
        if sys.platform == self.DARWIN:
            subprocess.run('brew remove lammps', shell=True)
            subprocess.run('brew tap homebrew/core', shell=True)
            rb = subprocess.run('brew cat lammps',
                                capture_output=True,
                                shell=True).stdout.decode("utf-8")
            if self.DPKG_PYTHON not in rb:
                rb = rb.replace(self.STD_CMAKE_ARGS, self.PYTHON_STD)
            with open(self.BUILD_LAMMPS_RB, 'w') as fh:
                fh.write(rb)
            subprocess.run(
                f'brew reinstall --build-from-source {self.BUILD_LAMMPS_RB}',
                shell=True)
        if sys.platform == self.LINUX:
            subprocess.run(
                'sudo apt-get install lsb-release gcc openmpi-bin cmake python3-apt python3-setuptools',
                shell=True)
            subprocess.run('sudo apt  install cmake', shell=True)
            subprocess.run(
                'sudo apt-get install openmpi-common libopenmpi-dev libgtk2.0-dev',
                shell=True)
            subprocess.run(
                'cd build; git clone -b stable https://github.com/lammps/lammps.git mylammps; '
                'cd mylammps; cd cmake; rm -f CMakeFiles CMakeCache.txt; rm -rf build ; mkdir build; cd build; '
                'cmake .. -DPKG_PYTHON=yes -DPKG_MOLECULE=yes -DPKG_KSPACE=yes -DPKG_RIGID=yes; cmake --build .',
                shell=True)
            lmp_path = os.path.join('build', 'mylammps', 'cmake', 'build',
                                    'lmp')
            # To be consistent with class Lammps_Driver.PATH in task.py
            subprocess.run(shutil.move(
                lmp_path, os.path.join(self.install_scripts, 'lmp_serial')),
                           shell=True)

    def installQt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        if sys.platform == self.DARWIN:
            qt = subprocess.run('brew list qt5',
                                capture_output=True,
                                shell=True)
            if qt.stdout:
                print('qt installation found.')
                return
            print('qt installation not found. Installing...')
            subprocess.run('brew install qt5', shell=True)
        if sys.platform == self.LINUX:
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
        if sys.platform == self.LINUX:
            subprocess.run('sudo apt install tilix -y', shell=True)


setup(name='nemd',
      version='0.1.0',
      description='A molecular dynamics simulation software',
      url='https://github.com/zhteg4/nemd',
      author='Teng Zhang',
      author_email='2022010236@hust.edu.cn',
      license='BSD 2-clause',
      packages=['nemd'],
      package_dir={'nemd': 'module/nemd'},
      package_data={'nemd': ['ff/*.prm']},
      scripts=['bash_scripts/run_nemd', 'scripts/polymer_builder_driver.py'],
      install_requires=[
          'numpy == 1.24.3', 'scipy == 1.10.1', 'networkx == 3.1',
          'pandas == 2.0.2', 'more_itertools == 9.1.0', 'chemparse == 0.1.2',
          'mendeleev == 0.14.0', 'rdkit == 2023.3.1', 'signac == 2.0.0',
          'signac-flow == 0.25.1', 'lammps == 2022.6.23.4.0',
          'matplotlib == 3.7.1', 'plotly ==5.15.0',
          'dash_bootstrap_components', 'pytest == 7.3.2', 'dash[testing]',
          'pyqt5 == 5.15.7', 'webdriver-manager == 3.8.6', 'flask >= 2.2.5',
          'openpyxl == 3.1.2', 'sh == 2.0.4', 'humanfriendly == 10.0',
          'Pillow == 9.4.0', 'pyvim'
      ],
      extras_require={
          'dev': [
              'ipdb',
              'ipython',
              'notebook',
              'jupyterlab',
              'yapf',
              'RBTools == 4.1',
              'snakeviz',
              'pyvim',
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
