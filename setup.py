"""
pip install setuptools
pip3 install ".[dev]"
"""

from setuptools import setup

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
          'pyqt6 == 6.5.1', 'webdriver-manager == 3.8.6', 'flask >= 2.2.5',
          'openpyxl == 3.1.2', 'sh == 2.0.4', 'humanfriendly == 10.0'
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
      ])
