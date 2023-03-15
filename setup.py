from setuptools import setup

setup(
    name='nemd3',
    version='0.1.0',    
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Stephen Hudson',
    author_email='shudson@anl.gov',
    license='BSD 2-clause',
    packages=['nemd3'],
    package_dir={'nemd3': 'module1/nemd'},
#    package_data={'nemd3': ['ff/*.prm']},
    scripts=['scripts/polymer_builder_driver.py'],
    install_requires=['matplotlib', 'numpy', 'chemparse', 'rdkit', 'networkx', 'scipy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
