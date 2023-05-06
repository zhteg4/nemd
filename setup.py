from setuptools import setup
from setuptools.command.install import install


class post_install(install):
    """
    The run method will be executed as the last installation stage.
    """

    def run(self):
        from webdriver_manager.chrome import ChromeDriverManager
        ChromeDriverManager().install()
        # install.run(self)
        # from subprocess import call
        # call(['python', 'scriptname.py'], cwd=self.install_lib + 'packagename')


setup(name='nemd',
      version='0.1.0',
      description='A polymer dynamics simulation software',
      url='https://github.com/zhteg4/nemd',
      author='Teng Zhang',
      author_email='2022010236@hust.edu.cn',
      license='BSD 2-clause',
      packages=['nemd'],
      package_dir={'nemd': 'module/nemd'},
      package_data={'nemd': ['ff/*.prm']},
      scripts=['bash_scripts/run_nemd', 'scripts/polymer_builder_driver.py'],
      install_requires=[
          'matplotlib', 'numpy', 'chemparse', 'rdkit', 'networkx', 'scipy',
          'pandas', 'signac', 'signac-flow', 'lammps', 'yapf', 'pytest',
          'mendeleev', 'plotly', 'dash', 'flask>=2.2.2', 'openpyxl',
          'dash_bootstrap_components', 'dash-uploader', 'dash[testing]',
          'more_itertools', 'webdriver-manager', 'jupyterlab', 'notebook',
          'snakeviz', 'pyqt5==5.12.0', 'RBTools'
      ],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
      ],
      cmdclass={'install': post_install})
