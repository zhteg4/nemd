"""
pip install setuptools
pip3 install ".[dev]"
"""

from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):

    def run(self):
        install.run(self)
        from webdriver_manager.chrome import ChromeDriverManager
        ChromeDriverManager().install()


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
          'numpy', 'scipy', 'networkx', 'pandas', 'more_itertools',
          'chemparse', 'mendeleev', 'rdkit', 'signac', 'signac-flow', 'lammps',
          'matplotlib', 'plotly', 'dash_bootstrap_components', 'pytest',
          'dash[testing]', 'pyqt6', 'webdriver-manager', 'flask>=2.2.2',
          'openpyxl', 'sh', 'humanfriendly', 'panel'
      ],
      extras_require={
          'dev': [
              'ipdb', 'ipython', 'notebook', 'jupyterlab', 'yapf', 'RBTools',
              'snakeviz', 'pyvim'
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
