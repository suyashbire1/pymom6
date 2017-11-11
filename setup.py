from setuptools import setup

setup(
    name='pymom6',
    version='0.1',
    description='Package to analyze mom6 generated data',
    url='http://github.com/suyashbire1/pymom6',
    author='Suyash Bire',
    author_email='suyash309@gmail.com',
    license='MIT',
    packages=['pymom6'],
    install_requires=[
        'numba',
        'netCDF4',
        'numpy',
        'xarray',
    ],
    zip_safe=False)
