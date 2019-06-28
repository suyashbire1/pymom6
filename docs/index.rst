.. pymom6 documentation master file, created by
   sphinx-quickstart on Fri Dec  1 07:29:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pymom6's documentation!
==================================

The aim of this package is to facilitate various mathematical operations on Arakawa C-grid.
On a C-grid, the velocities, tracers, and vorticity lie at different locations.
This makes it tricky to work with subsets of these datasets.
For example, if one takes a subset of a dataset between, say, y=30N and y=40N, the number of gridpoints in the subset of zonal velocity might be different from those in the subset of meridional velocity.

PyMOM6 solves this problem by explicitly requiring a final location, a location (u, v, h, or q) where the data would be located after all the intended operations are performed.
For example, if one needs to calculate vorticity, the final location would be q.
The indices given by subsets at q location are used to take subsets of velocities.
The next step involves extending the velocity datasets because the intended opearation, 1st order differentiation in this case, shortens the subset by one gridpoint.
The extension is achieved by mnemonic methods like xsm, xep, ysm, yep, zsm, and zep.
The method xsm (xsm is an abbreviation for x-start-minus) decreases the starting index of the subset by 1, while xep (x-end-plus) increases the ending index by 1.
The operations ysm/yep and zsm/zep perform the same opeartions in meridional and vertical directions.

Check out the examples to see how pyMOM6 can be used.

.. toctree::
   :maxdepth: 2

   example
   pymom6
