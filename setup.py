from setuptools import setup

setup(name='pym6',
      version='0.1',
      description='Package to analyze mom6 generated data',
      url='http://github.com/suyashbire1/pym6',
      author='Suyash Bire',
      author_email='suyash309@gmail.com',
      license='MIT',
      packages=['pym6'],
      install_requires=[
          'numpy',
          'netCDF4',
      ],
      zip_safe=False)
