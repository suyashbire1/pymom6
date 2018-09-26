"""
.. module:: pymom6
   :platform: Unix, Windows
   :synopsis: This module implements the Dataset and MOM6Variable classes along with some helper functions.

.. moduleauthor:: Suyash Bire, SUNY Stony Brook


"""
import numpy as np
from functools import partial, partialmethod
from collections import OrderedDict
from netCDF4 import Dataset as dset, MFDataset as mfdset
import xarray as xr
from numba import jit, float32, float64
import copy


class Dataset():
    """This class provides more convinient access to variables of a netcdf file.

    >>> ds = Dataset(filename)
    >>> var = ds.var
    >>> ds.close()

    It can be used as a contextmanager using with statement.

    >>> with Dataset(filename) as ds:
            var = ds.var

    """

    def __init__(self, filename, **initializer):
        """Initializes a Dataset instance

        :param filename: Can be a single netcdf file or a list of netcdf files, wildcards are not accepted
        :returns: A Dataset object containing references to all variables in the file
        :rtype: pymom6.Dataset

        """
        self.filename = filename
        self.initializer = initializer
        self.fh = mfdset(filename) if isinstance(filename,
                                                 list) else dset(filename)

    def close(self):
        """Closes an open Dataset object.

        :returns: None
        :rtype: None

        """
        self.fh.close()
        self.fh = None

    def __getattr__(self, var):
        """Returns a MOM6Variable object or a numpy object if var is a variable in Dataset

        :param var: Name of the netcdf variable
        :returns: MOM6Variable if the variable is 4-dimensional, a numpy array otherwise
        :rtype: pymom6.MOM6Variable or numpy.ndarray

        """
        if self.fh is not None:
            return self._variable_factory(var)
        else:
            raise AttributeError('{} is not open.'.format(self.filename))

    def _variable_factory(self, var):
        """Returns a MOM6Variable object or a numpy object if var is a variable in Dataset

        :param var: Name of the netcdf variable
        :returns: MOM6Variable if the variable is 4-dimensional, a numpy array otherwise
        :rtype: pymom6.MOM6Variable or numpy.ndarray

        """
        try:
            variable = MOM6Variable(var, self.fh, **self.initializer)
        except TypeError:
            variable = self.fh.variables[var][:]
        return variable

    def __enter__(self):
        """Boilerplate for contextmanager functionality """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Boilerplate for contextmanager functionality """
        self.close()


class GridGeometry():
    """This class holds the variables from ocean_geometry.nc file

    :param filename: Name of ocean_geometry file
    :returns: A class with references to variables of ocean_geometry.nc file
    :rtype: pymom6.GridGeometry

    """

    def __init__(self, filename):
        """Creates the GridGeometry class instance.

        :param filename: Name of ocean_geometry file
        :returns: A class with references to variables of ocean_geometry.nc file
        :rtype: pymom6.GridGeometry

        """
        with dset(filename) as fh:
            for var in fh.variables:
                setattr(self, var, fh.variables[var][:])

    def _get_divisor_for_diff(self, loc, axis, weights=None):
        """This method rerturs a divisor to the divide_by, multiply_by, and dbyd methods of MOM6Variable class

        :param loc: grid location of the MOM6Variable (one of u,v,h,q and one of l,i, See docs of MOM6Variable)
        :param axis: Integer between 0 to 3 specifying the axis
        :param weights: can be None or 'area'
        :returns: divisor for divide_by, multiply_by, and dbyd methods of MOM6Variable class
        :rtype: numpy.ndarray

        """
        axis = axis - 2
        divisors = dict(
            u=['dyCu', 'dxCu'],
            v=['dyCv', 'dxCv'],
            h=['dyT', 'dxT'],
            q=['dyBu', 'dxBu'])
        if weights == 'area':
            divisors['h'] = ['Ah', 'Ah']
            divisors['q'] = ['Aq', 'Aq']
        return getattr(self, divisors[loc][axis])


class Domain():
    def _initialize_indices_and_dim_arrays(self):
        """Initializes empty dicts to hold indices and dim_arrays

        :param self: MOM6Variable instance
        :returns: None
        :rtype: None

        """
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}

    @staticmethod
    def _find_index_limits(dimension, start, end, method='lower'):
        """Finds the extreme indices of any dimension between a start and end point

        :param dimension: 1D numpy array
        :param start: start point
        :param end: end point
        :param method: 'lower' or 'higher'
        :returns: tuple with indices corresponding to start and end if start and end are distinct
        :rtype: tuple

        """
        if start == end:
            array = dimension - start
            if method == 'lower':
                useful_index = np.array([1, 1]) * np.argmax(array[array <= 0])
            elif method == 'higher':
                useful_index = np.array(
                    [1, 1]) * (np.argmax(array[array <= 0]) + 1)
            else:
                useful_index = np.array([1, 1]) * np.argmin(np.fabs(array))
        else:
            useful_index = np.nonzero((dimension >= start) &
                                      (dimension <= end))[0]
        lims = useful_index[0], useful_index[-1] + 1
        return lims

    def _get_extremes(self, dim_str, low, high, **initializer):
        """Populates indices and dim_arrays of MOM6Variable

        :param self: MOM6Variable instance
        :param dim_str: string representing the name of the dimension
        :param low: lower bound of the domain along dimension
        :param high: higher bound of the domain along dimension
        :returns: None
        :rtype: None

        """
        self._initialize_indices_and_dim_arrays()
        fh = initializer.get('fh')
        axis_str = dim_str[0].lower()
        stride = initializer.get('stride' + axis_str, 1)
        by_index = initializer.get('by_index', False)
        if dim_str in fh.variables:
            dimension = fh.variables[dim_str][:]
            if by_index:
                self.indices[dim_str] = initializer.get(
                    's' + axis_str, 0), initializer.get(
                        'e' + axis_str, dimension.size), stride
            else:
                low = initializer.get(low, dimension[0])
                high = initializer.get(high, dimension[-1])
                method = initializer.get('method', 'lower')
                self.indices[dim_str] = *self._find_index_limits(
                    dimension, low, high, method=method), stride
            self.dim_arrays[dim_str] = dimension


class _MeridionalDomain(Domain):
    """Initializes meridional domain limits."""

    def __init__(self, **initializer):
        """Initializes meridional domain limits."""
        self._get_extremes('yh', 'south_lat', 'north_lat', **initializer)
        self._get_extremes('yq', 'south_lat', 'north_lat', **initializer)


class _ZonalDomain(Domain):
    """Initializes zonal domain limits."""

    def __init__(self, **initializer):
        """Initializes zonal domain limits."""
        self._get_extremes('xh', 'west_lon', 'east_lon', **initializer)
        self._get_extremes('xq', 'west_lon', 'east_lon', **initializer)


class _HorizontalDomain(_MeridionalDomain, _ZonalDomain):
    """Initializes horizontal domain limits."""

    def __init__(self, **initializer):
        _MeridionalDomain.__init__(self, **initializer)
        _ZonalDomain.__init__(self, **initializer)


class _VerticalDomain(Domain):
    """Initializes vertical domain limits."""

    def __init__(self, **initializer):
        self._get_extremes('zl', 'low_density', 'high_density', **initializer)
        self._get_extremes('zi', 'low_density', 'high_density', **initializer)


class _TemporalDomain(Domain):
    """Initializes temporal domain limits."""

    def __init__(self, **initializer):
        self._get_extremes('Time', 'initial_time', 'final_time', **initializer)


class _txyzDomain(_TemporalDomain, _VerticalDomain, _HorizontalDomain):
    """Initializes temporal, vertical, and horizontal domain limits."""

    def __init__(self, **initializer):
        _TemporalDomain.__init__(self, **initializer)
        _VerticalDomain.__init__(self, **initializer)
        _HorizontalDomain.__init__(self, **initializer)


class _LazyNumpyOperation():
    """This class provides lazy numpy operations for MOM6Variable. This
    class should not be used on its own."""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, array):
        func = self.func
        args = self.args
        kwargs = self.kwargs
        return func(array, *args, **kwargs)


class _BoundaryCondition():
    """This class provides boundary conditions for MOM6Variable. This
    class should not be used on its own."""

    def __init__(self, bc_type, axis, start_or_end):
        self.bc_type = bc_type
        self.axis = axis
        self.start_or_end = start_or_end

    def set_halo_indices(self):
        if self.bc_type == 'dirichletq':
            take_index = 1 if self.start_or_end == 0 else -2
        else:
            take_index = self.start_or_end
        return take_index

    def create_halo(self, array):
        take_index = self.set_halo_indices()
        self.halo = np.take(array, [take_index], axis=self.axis)
        if self.bc_type == 'zeros':
            self.halo = np.zeros(self.halo.shape)

    def boudary_condition_type(self):
        if self.bc_type != 'neumann':
            self.halo = -self.halo

    def append_halo_to_array(self, array):
        self.boudary_condition_type()
        if self.start_or_end == 0:
            array1 = self.halo
            array2 = array
        elif self.start_or_end == -1:
            array1 = array
            array2 = self.halo
        array = np.concatenate((array1, array2), axis=self.axis)
        return array

    def __call__(self, array):
        self.create_halo(array)
        return self.append_halo_to_array(array)


def _get_var_at_z(array, z, e, fillvalue):
    array_out = np.full(
        (array.shape[0], z.shape[0], array.shape[2], array.shape[3]),
        fillvalue)
    for l in range(array.shape[0]):
        for k in range(array.shape[1]):
            for j in range(array.shape[2]):
                for i in range(array.shape[3]):
                    for m in range(z.shape[0]):
                        if (e[l, k, j, i] - z[m]) * (
                                e[l, k + 1, j, i] - z[m]) <= 0:
                            array_out[l, m, j, i] = array[l, k, j, i]
    return array_out


def _get_rho_at_z(array, z, zl, fillvalue):
    array_out = np.full(
        (array.shape[0], z.shape[0], array.shape[2], array.shape[3]),
        fillvalue)
    for l in range(array.shape[0]):
        for k in range(zl.shape[0]):
            for j in range(array.shape[2]):
                for i in range(array.shape[3]):
                    for m in range(z.shape[0]):
                        if (array[l, k, j, i] - z[m]) * (
                                array[l, k + 1, j, i] - z[m]) <= 0:
                            array_out[l, m, j, i] = zl[k]
    return array_out


class MOM6Variable(_txyzDomain):
    """A class to hold a variable from netcdf file generated by MOM6.
    A MOM6 variable that is located at one of the h,u,v,q,l,i points.

    :param var: name of the variable
    :param fh: a handle to an open netCDF file
    :returns: a variable object holding all the data
    :rtype: MOM6Variable

    """

    def __init__(self, var, fh, **initializer):
        """A MOM6 variable that is located at one of the h,u,v,q,l,i points.

        :param var: name of the variable
        :param fh: a handle to an open netCDF file
        :returns: a variable object holding all the data
        :rtype: MOM6Variable

        """

        self._name = initializer.get('name', var)
        self._v = fh.variables[var]
        if len(self._v.dimensions) == 1 or 'nv' in self._v.dimensions:
            raise TypeError('Not a MOM6variable')
        self._initial_dimensions = list(self._v.dimensions)
        self._current_dimensions = self._initial_dimensions
        self._determine_location()
        self.fh = fh
        initializer['fh'] = fh
        _txyzDomain.__init__(self, **initializer)
        self.array = None
        self._polish(**initializer)
        try:
            self._average_DT = fh.variables['average_DT'][:]
        except KeyError:
            self._average_DT = None

    def __getitem__(self, dict_):
        for key, value in list(dict_.items()):
            if key == 'final_loc':
                self.final_loc(value)
                dict_.pop(key)
            elif isinstance(value, slice):
                pass
            else:
                dict_[key] = slice(value, value)
        return self.sel(**dict_)

    def sel(self, **kwargs):
        """Creates a subset of a MOM6Variable object. This only works
        if used before :meth:`read`.

        Possible usage:

        >>> MOM6Variable('u',fh).sel(x=48)
        >>> MOM6Variable('u',fh).sel(x=48,y=25.2)
        >>> MOM6Variable('u',fh).sel(x=slice(48.5,50),y=25)

        Any of 't', 'time', 'T', or 'Time' can be used to slice time dimension.
        Any of 'z', 'zl', or 'zi' can be used to slice vertical dimension.
        Any of 'y', 'yh', or 'yq' can be used to slice meridional dimension.
        Any of 'x', 'xh', or 'xq' can be used to slice zonal dimension.

        :returns: Subset MOM6Variable object
        :rtype: MOM6Variable object

        """
        domain_mapping = {
            ('t', 'Time', 'time', 'T'): _TemporalDomain,
            ('z', 'zl', 'zi'): _VerticalDomain,
            ('y', 'yh', 'yq'): _MeridionalDomain,
            ('x', 'xh', 'xq'): _ZonalDomain
        }
        low_mapping = {
            ('t', 'Time', 'time', 'T'): 'initial_time',
            ('z', 'zl', 'zi'): 'low_density',
            ('y', 'yh', 'yq'): 'south_lat',
            ('x', 'xh', 'xq'): 'west_lon'
        }
        high_mapping = {
            ('t', 'Time', 'time', 'T'): 'final_time',
            ('z', 'zl', 'zi'): 'high_density',
            ('y', 'yh', 'yq'): 'north_lat',
            ('x', 'xh', 'xq'): 'east_lon'
        }
        for key, value in kwargs.items():
            print(key, value)
            for possible_axis_names, domain in domain_mapping.items():
                if key in possible_axis_names:
                    assert isinstance(value, (int, float, slice))
                    if isinstance(value, (int, float)):
                        value = slice(value, value)
                    kwargs_dom = {}
                    if value.start:
                        kwargs_dom[low_mapping[
                            possible_axis_names]] = value.start
                    if value.stop:
                        kwargs_dom[high_mapping[
                            possible_axis_names]] = value.stop
                    if value.step:
                        kwargs_dom['stride' +
                                   possible_axis_names[0]] = value.step
                    kwargs_dom['method'] = kwargs.get('method', 'lower')
                    domain.__init__(self, fh=self.fh, **kwargs_dom)
        return self

    def isel(self, **kwargs):
        """Creates a subset of a MOM6Variable object by index. This
        only works if used before :meth:`read`.

        Possible usage:

        >>> MOM6Variable('u',fh).isel(x=48)
        >>> MOM6Variable('u',fh).isel(x=48,y=25)
        >>> MOM6Variable('u',fh).isel(x=slice(48,50),y=25)

        Any of 't', 'time', 'T', or 'Time' can be used to slice time dimension.
        Any of 'z', 'zl', or 'zi' can be used to slice vertical dimension.
        Any of 'y', 'yh', or 'yq' can be used to slice meridional dimension.
        Any of 'x', 'xh', or 'xq' can be used to slice zonal dimension.

        :returns: Subset MOM6Variable object
        :rtype: MOM6Variable object

        """
        domain_mapping = {
            ('t', 'Time', 'time', 'T'): _TemporalDomain,
            ('z', 'zl', 'zi'): _VerticalDomain,
            ('y', 'yh', 'yq'): _MeridionalDomain,
            ('x', 'xh', 'xq'): _ZonalDomain
        }
        for key, value in kwargs.items():
            for possible_axis_names, domain in domain_mapping.items():
                if key in possible_axis_names:
                    assert isinstance(value, (int, slice))
                    if isinstance(value, int):
                        value = slice(value, value)
                    kwargs_dom = {}
                    if value.start:
                        kwargs_dom['s' + possible_axis_names[0]] = value.start
                    if value.stop:
                        kwargs_dom['e' + possible_axis_names[0]] = value.stop
                    if value.step:
                        kwargs_dom['stride' +
                                   possible_axis_names[0]] = value.step
                    domain.__init__(
                        self, by_index=True, fh=self.fh, **kwargs_dom)
        return self

    def _polish(self, **initializer):
        """This method applies the kwargs from initializer. This
        method should not be directly used.

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self.final_loc(initializer.get('final_loc', None))
        self.fillvalue(initializer.get('fillvalue', 0))
        self.bc_type(initializer.get('bc_type', None))
        self.geometry(initializer.get('geometry', None))
        self._units = initializer.get('units', None)
        self._math = initializer.get('math', None)
        self.operations = []
        return self

    def final_loc(self, final_loc=None):
        """Sets the final location of MOM6Variable

        Possible usage:

        >>> MOM6Variable('u',fh).final_loc('ul').read()

        Same as:

        >>> MOM6Variable('u',fh,final_loc='ul').read()

        :param final_loc: Two alphabet string (One of 'h','u','v', or 'q'
        and one of 'l' or 'i'). Default is the location of the
        variable e.g. u will be at 'ul', e at 'hi', etc.
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        if final_loc:
            self._final_loc = final_loc
        else:
            self.final_loc(self._current_hloc + self._current_vloc)
            self._final_dimensions = tuple(self._current_dimensions)
        self._get_final_location_dimensions()
        return self

    def fillvalue(self, fillvalue):
        """Sets the fillvalue for masked arrays (these are generally
        values at topography, either 0 or np.nan)

        :param fillvalue: 0 (default) or np.nan
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self._fillvalue = fillvalue
        return self

    def bc_type(self, bc_type):
        """Specifies the boundary conditions for the top, bottom,
        south, north, west, and east boundaries.

        Example:

        bc_type = dict(v=['neumann', 'dirichleth', 'zeros', 'dirichletq', 'dirichleth', 'dirichleth'])

        TODO: Simplify specifying BCs

        :param bc_type: Dict indicating six boundary conditions
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self._bc_type = bc_type
        return self

    def geometry(self, geometry):
        """Specifies the GridGeometry object. This is necessary when
        differentiating or moving location (e.g. from u to h points) of the variable.

        :param geometry: pymom6.GridGeometry instance
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self._geometry = geometry
        return self

    def _determine_location(self):
        """Determines the current location of a MOM6Variable based on
        its current dimensions

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        dims = self._current_dimensions
        if 'xh' in dims and 'yh' in dims:
            self._current_hloc = 'h'
        elif 'xq' in dims and 'yq' in dims:
            self._current_hloc = 'q'
        elif 'xq' in dims and 'yh' in dims:
            self._current_hloc = 'u'
        elif 'xh' in dims and 'yq' in dims:
            self._current_hloc = 'v'
        if 'zl' in dims:
            self._current_vloc = 'l'
        elif 'zi' in dims:
            self._current_vloc = 'i'

    def return_dimensions(self):
        """Gives the current dimensions of a MOM6Variable. This method
        is also invoked by MOM6Variable.dimension property.

        :returns: A dictionary of current dimensions
        :rtype: OrderedDict

        """
        dims = self._current_dimensions
        return_dims = OrderedDict()
        for dim in dims:
            dim_array = self.dim_arrays[dim]
            if isinstance(dim_array, np.ndarray):
                start, stop, stride = self.indices[dim]
                return_dims[dim] = dim_array[start:stop:stride]
            else:
                return_dims[dim] = dim_array
        return return_dims

    @staticmethod
    def get_dimensions_by_location(loc):
        """Returns the names of the dimensions corresponding to a
        location, loc.

        :param loc: Location (same format as final_loc, 2 letter string)
        :returns: 4-element tuple containing dimension names
        :rtype: tuple

        """
        loc_registry_hor = dict(
            u=['yh', 'xq'], v=['yq', 'xh'], h=['yh', 'xh'], q=['yq', 'xq'])
        loc_registry_ver = dict(l='zl', i='zi')
        hloc = loc[0]
        vloc = loc[1]
        vdim = loc_registry_ver[vloc]
        hdims = loc_registry_hor[hloc]
        return tuple(['Time', vdim, *hdims])

    def _get_final_location_dimensions(self):
        """Returns dimensions of final location

        :returns: None
        :rtype: Nonetype

        """
        self._final_hloc = self._final_loc[0]
        self._final_vloc = self._final_loc[1]
        self._final_dimensions = self.get_dimensions_by_location(
            self._final_loc)

    def modify_index(self, axis, startend, value):
        """This method modifies the indices associated with the axis

        :param axis: One of 0,1,2,3 corresponding to Time, z, y, or x axis
        :param startend: 0 or 1 indicates whether the starting or the
        ending index is modified, respectively
        :param value: Modify the index by this value
        :returns: None
        :rtype: Nonetype

        """
        dim = self._final_dimensions[axis]
        axis_indices = list(self.indices[dim])
        axis_indices[startend] += value
        self.indices[dim] = tuple(axis_indices)

    def modify_index_return_self(self, axis, startend, value):
        """Same as modify_index. This method returns self at the end.
        This method is used to create convinience methods xsm, xep,
        ysm, yep, zsm, and zep. The first, second, and third alphabets
        stand for the axis, starting or ending index, and plus or
        minus. These methods add or subtract one from the index of the
        given axis.

        For example if you want to lengthen the x-axis domain towards
        the east, you should use xep as this increases the ending
        index of x by 1. Conversely, if you want to extend the x-axis
        domain towrds the west, you should use xsm as this subtracts 1
        from the starting index of x.

        :param axis: One of 0,1,2,3 corresponding to Time, z, y, or x axis
        :param startend: 0 or 1 indicates whether the starting or the
        ending index is modified, respectively
        :param value: Modify the index by this value
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self.modify_index(axis, startend, value)
        return self

    xsm = partialmethod(modify_index_return_self, 3, 0, -1)
    xep = partialmethod(modify_index_return_self, 3, 1, 1)
    ysm = partialmethod(modify_index_return_self, 2, 0, -1)
    yep = partialmethod(modify_index_return_self, 2, 1, 1)
    zsm = partialmethod(modify_index_return_self, 1, 0, -1)
    zep = partialmethod(modify_index_return_self, 1, 1, 1)

    def get_slice(self):
        """Populates the _slice attribute of MOM6Variable based on
        current dimensions. This slice can be used to slice other
        arrays.

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        # assert self._final_dimensions == tuple(self._current_dimensions)
        self._slice = []
        for axis in range(4):
            indices = self._get_slice_by_axis(axis)
            self._slice.append(slice(*indices))
        self._slice = tuple(self._slice)
        return self

    def get_slice_2D(self):
        """Populates the _slice_2D attribute of MOM6Variable based on
        current dimensions. Only x and y slices are populated. This
        slice can be used to slice other arrays.

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self._slice_2D = []
        for axis in range(2, 4):
            indices = self._get_slice_by_axis(axis)
            self._slice_2D.append(slice(*indices))
        self._slice_2D = tuple(self._slice_2D)
        return self

    def _get_slice_by_axis(self, axis):
        """Returns slice for axis. This method is meant for internal
        use.

        :param axis: 0, 1, 2, or 3 for t, z, y, or x
        :returns: list containing beginning and ending index of the axis
        :rtype: List

        """
        dims = self._final_dimensions
        dim = dims[axis]
        indices = list(self.indices[dim])
        if indices[0] < 0:
            indices[0] = 0
        if indices[1] > self.dim_arrays[dim].size:
            if axis != 1 or self._initial_dimensions[1] != 'zi':
                indices[1] = self.dim_arrays[dim].size
            else:
                indices[1] = self.dim_arrays['zi'].size
        return indices

    def read(self):
        """This method reads the data from the disk into memory. Once
        this method is called all the pre-read methods should not be
        used. Post-read methods should be used only after this method
        has been called.

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self.get_slice()

        def lazy_read_and_fill(array):
            array = self._v[self._slice]
            if np.ma.isMaskedArray(array):
                array = array.filled(self._fillvalue)
            return array

        self.operations.append(lazy_read_and_fill)
        self.implement_BC_if_necessary()
        return self

    def multiply_by(self, multiplier, power=1):
        """This is a post-read method that multiplies the MOM6Variable
        by an attribute of GridGeometry.

        :param multiplier: A string indicating any attribute from
        GridGeometry like area, dxT, etc.
        :param power: 1 (default) or -1 (used for divide_by method)
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        self.get_slice_2D()
        multiplier = getattr(self._geometry, multiplier)[self._slice_2D]**power
        multiplier = self.implement_BC_if_necessary_for_multiplier(multiplier)
        self.operations.append(lambda a: a * multiplier)
        return self

    divide_by = partialmethod(multiply_by, power=-1)

    BoundaryCondition = _BoundaryCondition
    _default_bc_type = dict(
        u=[
            'neumann', 'dirichleth', 'dirichleth', 'dirichleth', 'zeros',
            'dirichletq'
        ],
        v=[
            'neumann', 'dirichleth', 'zeros', 'dirichletq', 'dirichleth',
            'dirichleth'
        ],
        h=[
            'neumann', 'dirichleth', 'neumann', 'neumann', 'neumann', 'neumann'
        ],
        q=[
            'neumann', 'dirichleth', 'zeros', 'dirichletq', 'zeros',
            'dirichletq'
        ])

    def implement_BC_if_necessary(self):
        """This post-read method checks if any of the indices extend
        beyond the buondary implements boundary conditions as
        specified by bc_type (or default spedified by _default_bc_type
        attribute)

        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        dims = self._final_dimensions
        if self._bc_type is None:
            self._bc_type = self._default_bc_type
        for i, dim in enumerate(dims[1:]):
            indices = self.indices[dim]
            loc = self._current_hloc
            if indices[0] < 0:
                bc_type = self._bc_type[loc][2 * i]
                self.operations.append(
                    self.BoundaryCondition(bc_type, i + 1, 0))
            if (indices[1] > self.dim_arrays[dim].size):
                if i != 0 or self._initial_dimensions[1] != 'zi':
                    bc_type = self._bc_type[loc][2 * i + 1]
                    self.operations.append(
                        self.BoundaryCondition(bc_type, i + 1, -1))
        return self

    @staticmethod
    def _vertical_move(array):
        """Moves array in the vertical direction from l to i or i to l.

        :param array: np.ndarray
        :returns: moved np.ndarray
        :rtype: np.ndarray

        """
        return 0.5 * (array[:, :-1, :, :] + array[:, 1:, :, :])

    @staticmethod
    def _check_possible_movements_for_move(current_loc,
                                           new_loc=None,
                                           axis=None):
        """Checks where the current MOM6Variable can be moved. This is
        based on the fact the a move can only be accomplished to an
        adjacent location (.e.g. h to u or v to q or u to q or h to v)
        but not in a diagonal direction (e.g. h to q or u to v). This
        method is meant for internal use.

        :param current_loc: Current location (2 letter string)
        :param new_loc: New location (2 letter string)
        :param axis: 0, 1, 2, 3 for t, z, y, or x
        :returns: tuple containing start and end movement (ns and ne)
        :rtype: tuple

        """
        possible_from_to = dict(
            u=['q', 'h'], v=['h', 'q'], h=['v', 'u'], q=['u', 'v'])
        possible_ns = dict(u=[0, 1], v=[1, 0], h=[0, 0], q=[1, 1])
        possible_ne = dict(u=[-1, 0], v=[0, -1], h=[-1, -1], q=[0, 0])

        if new_loc is not None:
            axis = possible_from_to[current_loc].index(new_loc)
            axis += 2
        ns = possible_ns[current_loc][axis - 2]
        ne = possible_ne[current_loc][axis - 2]
        if new_loc is not None:
            return (axis, ns, ne)
        else:
            return (ns, ne)

    @staticmethod
    def _horizontal_move(axis, array):
        """Moves array horizontally

        :param axis: 2 or 3
        :param array: np.ndarray
        :returns: moved np.ndarray
        :rtype: np.ndarray

        """
        return 0.5 * (np.take(array, range(array.shape[axis] - 1), axis=axis) +
                      np.take(array, range(1, array.shape[axis]), axis=axis))

    def _adjust_dimensions_and_indices_for_vertical_move(self):
        """Changes the dimensions from zl to zi or vice versa for
        vertical move.

        :returns: None
        :rtype: NoneType

        """
        self.modify_index(1, 1, -1)
        if self._current_vloc == 'l':
            self.modify_index(1, 0, 1)
            self._current_dimensions[1] = 'zi'
        else:
            self._current_dimensions[1] = 'zl'
        self._determine_location()

    def _adjust_dimensions_and_indices_for_horizontal_move(self, axis, ns, ne):
        """Changes the dimensions and indexes for horizontal move

        :param axis: 2 or 3
        :param ns: 0 or 1
        :param ne: 0 or 1
        :returns: None
        :rtype: NoneType

        """
        self.modify_index(axis, 0, ns)
        self.modify_index(axis, 1, ne)
        current_dimension = list(self._current_dimensions[axis])
        if current_dimension[1] == 'h':
            current_dimension[1] = 'q'
        elif current_dimension[1] == 'q':
            current_dimension[1] = 'h'
        self._current_dimensions[axis] = "".join(current_dimension)
        self._determine_location()

    def move_to(self, new_loc):
        """Moves the MOM6Variable to a new location

        :param new_loc: One letter string (u, v, h, q, l, or i). If l
        or i is given, vertical move is assumed else horizontal move
        is assumed.
        :returns: MOM6Variable at a new location
        :rtype: MOM6Variable

        """
        if new_loc in ['l', 'i'] and new_loc != self._current_vloc:
            self._adjust_dimensions_and_indices_for_vertical_move()
            self.operations.append(self._vertical_move)
        elif new_loc in ['u', 'v', 'h', 'q'] and new_loc != self._current_hloc:
            axis, ns, ne = self._check_possible_movements_for_move(
                self._current_hloc, new_loc=new_loc)
            self._adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            move = partial(self._horizontal_move, axis)
            self.operations.append(move)
        return self

    def implement_BC_if_necessary_for_multiplier(self, multiplier):
        """Same as implement BC if necessary but for multiplier

        :param multiplier: Attribute of GridGeometry (like area, dxT, etc.)
        :returns: multiplier with boundary conditions imposed if necessary
        :rtype: np.ndarray

        """
        dims = self._final_dimensions
        for i, dim in enumerate(dims[2:]):
            indices = self.indices[dim]
            if indices[0] < 0:
                halo = np.take(multiplier, [0], axis=i)
                multiplier = np.concatenate((halo, multiplier), axis=i)
            if (indices[1] > self.dim_arrays[dim].size):
                halo = np.take(multiplier, [-1], axis=i)
                multiplier = np.concatenate((multiplier, halo), axis=i)
        return multiplier

    def dbyd(self, axis, weights=None):
        """This method implements the differentiation operator.

        :param axis: 0, 1, 2, or 3 for t, z, y, or x
        :param weights: If None (deafult) divisor is grid spacing, if
        'area' divisor is cell area
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        if axis > 1:
            ns, ne = self._check_possible_movements_for_move(
                self._current_hloc, axis=axis)
            self._adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            divisor = self._geometry._get_divisor_for_diff(
                self._current_hloc, axis, weights=weights)
            self.get_slice_2D()
            divisor = divisor[self._slice_2D]
            divisor = self.implement_BC_if_necessary_for_multiplier(divisor)
        elif axis == 1:
            divisor = -10 / 1000 * np.diff(
                self.dim_arrays[self._final_dimensions[1]][2:4])
            self._adjust_dimensions_and_indices_for_vertical_move()
        dadx = partial(lambda x, a: np.diff(a, n=1, axis=x) / divisor, axis)
        self.operations.append(dadx)
        return self

    LazyNumpyOperation = _LazyNumpyOperation

    def np_ops(self, npfunc, *args, **kwargs):
        """Implements functionality to apply a numpy operation to
        MOM6Variable. If the numpy operation changes dimension sizes,
        they must be manually adjusted using axis, ns, and ne kwargs.
        If numpy operation changes vertical or horizontal location, it
        should be manually set by using sets_vloc or sets_hloc kwargs.

        :param npfunc: string indicating numpy method (e.g. 'nanmean')
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        sets_hloc = kwargs.get('sets_hloc', None)
        if sets_hloc is not None:
            axis = kwargs.get('axis', None)
            ns = kwargs.get('ns', None)
            ne = kwargs.get('ne', None)
            if ns is not None:
                self.modify_index(axis, 0, ns)
                kwargs.pop('ns')
            if ne is not None:
                self.modify_index(axis, 1, ne)
                kwargs.pop('ne')
            self.hloc = sets_hloc
            kwargs.pop('sets_hloc')
        sets_vloc = kwargs.get('sets_vloc', None)
        if sets_vloc is not None:
            self.modify_index(1, 1, -1)
            if self._current_vloc == 'l' and sets_vloc == 'i':
                self.modify_index(1, 0, 1)
                self.vloc = sets_vloc
                kwargs.pop('sets_vloc')
            elif self._current_vloc == 'i' and sets_vloc == 'l':
                self.vloc = sets_vloc
                kwargs.pop('sets_vloc')
        self.operations.append(
            self.LazyNumpyOperation(npfunc, *args, **kwargs))
        return self

    def where(self, logic_function, other_array, y=None):
        """Implements the numpy.where method for MOM6Variable

        :param logic_function: numpy logical method (e.g. np.greater_equal)
        :param other_array: array of values to be checked against (see
        numpy.where docs)
        :param y: Values from this array are chosen where logic_func
        returns false (see numpy.where docs). If y is None (default)
        MOM6Variable.array.nonzero is returned.
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """

        def wraps_where(array):
            condition = logic_function(array, other_array)
            if y is not None:
                return np.where(condition, array, y)
            else:
                return np.where(condition)

        self.operations.append(wraps_where)
        return self

    @staticmethod
    def meanfunc(array, axis=[1, 2, 3]):
        return np.nanmean(array, axis=axis, keepdims=True)

    @staticmethod
    def meanfunc_time(array, dt=0, axis=0):
        dt = dt[:, np.newaxis, np.newaxis, np.newaxis]
        array *= dt
        return np.nansum(array, axis=axis, keepdims=True) / np.sum(dt)

    def reduce_axis(self, axis, reduce_func):
        axis_string = self._current_dimensions[axis]
        self.dim_arrays[axis_string] = reduce_func(
            self.dim_arrays[axis_string])
        self.indices.pop(axis_string)

    def nanmean(self, axis=[0, 1, 2, 3]):
        """Implements nanmean for MOM6Variable (post-read method)

        :param axis: A single axis or sequence of axes among 0, 1, 2,
        3. The mean is taken along axis/axes supplied here
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        try:
            for ax in axis:
                self.reduce_axis(ax, np.mean)
                if ax == 0:
                    if self._average_DT is not None:
                        self.operations.append(
                            partial(
                                self.meanfunc_time,
                                dt=self._average_DT,
                                axis=ax))
                    else:
                        self.operations.append(partial(self.meanfunc, axis=ax))
                else:
                    self.operations.append(partial(self.meanfunc, axis=ax))
        except TypeError:
            self.reduce_axis(axis, np.mean)
            if axis == 0:
                if self._average_DT is not None:
                    self.operations.append(
                        partial(
                            self.meanfunc_time, dt=self._average_DT,
                            axis=axis))
                else:
                    self.operations.append(partial(self.meanfunc, axis=axis))
            else:
                self.operations.append(partial(self.meanfunc, axis=axis))
        return self

    def reduce_(self, reduce_func, *args, keepdims=True, **kwargs):
        """Implements reduction operations on MOM6Variable. Axes can
        be specified as a single axis or a sequence of 0, 1, 2, 3.

        :param reduce_func: numpy reduction operations like np.mean,
        np.sum, etc.
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        axis = kwargs.get('axis', (0, 1, 2, 3))
        try:
            for ax in axis:
                self.reduce_axis(ax, reduce_func)
        except TypeError:
            self.reduce_axis(axis, reduce_func)
        if keepdims:
            kwargs['keepdims'] = keepdims
        self.operations.append(
            self.LazyNumpyOperation(reduce_func, *args, **kwargs))
        return self

    get_rho_at_z = staticmethod(
        jit(float64[:, :, :, :](float64[:, :, :, :], float64[:], float64[:],
                                float32))(_get_rho_at_z))
    get_var_at_z = staticmethod(
        jit(float64[:, :, :, :](float64[:, :, :, :], float64[:],
                                float64[:, :, :, :], float32))(_get_var_at_z))

    def toz(self, z, e=None, dimstr='z (m)'):
        """Move MOM6Variable to z coordinate from buoyancy coordinates

        :param z: np.ndarray of z locations or a single location
        :param e: MOM6Variable containing e (isopycnal heights)
        :param dimstr: The string to represent new vertical dimension
        :returns: MOM6Variable in z coordinates
        :rtype: MOM6Variable

        """
        new = copy.copy(self)
        if not isinstance(z, np.ndarray):
            z = np.array(
                z, dtype=np.float64) if isinstance(z, list) else np.array(
                    [z], dtype=np.float64)
        new.dim_arrays[dimstr] = z
        new.indices[dimstr] = 0, z.size, 1
        dims = list(new._current_dimensions)
        dims[1] = dimstr
        new._current_dimensions = dims
        if self._name == 'e':
            assert self._current_vloc == 'i'

            def lazy_toz(array):
                return new.get_rho_at_z(
                    array.astype(np.float64), z,
                    self.dim_arrays['zl'].astype(np.float64),
                    float(new._fillvalue))

        else:
            assert e is not None
            assert new._current_hloc == e._current_hloc
            assert e._current_vloc == 'i'

            def lazy_toz(array):
                return new.get_var_at_z(
                    array.astype(np.float64), z, e.array.astype(np.float64),
                    float(new._fillvalue))

        new.operations.append(lazy_toz)
        return new

    def conditional_toz(self, toz, z, e, dimstr='z (m)'):
        """Implements toz conditionally. See toz method documentation
        for definitions of other arguments.

        :param toz: Boolean that determines whether to run toz method
        or not
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        if toz:
            return self.toz(z, e, dimstr=dimstr)
        else:
            return self

    def compute(self, check_loc=True):
        """Sequentially executes all the lazy operations.

        :param check_loc: Checks if the final location is same as the
        current location if true else disables the check.
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        for ops in self.operations:
            self.array = ops(self.array)
        self.operations = []
        if check_loc:
            ch = self._current_hloc
            fh = self._final_hloc
            cv = self._current_vloc
            fv = self._final_vloc
            assert ch == fh, 'Cur hloc = {} is not final hloc = {}'.format(
                ch, fh)
            assert cv == fv, 'Cur vloc = {} is not final vloc = {}'.format(
                cv, fv)
        return self

    def to_DataArray(self, check_loc=True):
        """Converts MOM6Variable to xarray.DataArray instance. This is
        useful for plotting.

        :param check_loc: Checks if the final location is same as the
        current location if true else disables the check.
        :returns: xarray.DataArray of the MOM6Variable
        :rtype: xarray.DataArray

        """
        if len(self.operations) is not 0:
            self.compute(check_loc=check_loc)
        coords = self.return_dimensions()
        coords_squeezed = {}
        dims_squeezed = []
        for i, (coord, value) in enumerate(coords.items()):
            if isinstance(value, np.ndarray):
                coords_squeezed[coord] = value
            else:
                coords_squeezed[coord] = np.array([value])
            dims_squeezed.append(coord)
        da = xr.DataArray(
            self.array, coords=coords_squeezed, dims=dims_squeezed)
        da.name = self._name
        if self._math:
            da.attrs['math'] = self._math
        if self._units:
            da.attrs['units'] = self._units
        return da

    def tokm(self, axis, dim_str=None):
        """Converts x or y axis dimension from degrees to km

        :param axis: one of 2 or 3
        :param dim_str: String representing the new dimension
        :returns: MOM6Variable with axis converted to km
        :rtype: MOM6Variable

        """
        R = 6378
        dim_str_dict = {2: 'y (km)', 3: 'x (km)'}
        new = copy.copy(self)
        if axis == 3:
            ymean = np.mean(list(new.dimensions.items())[2][1])
            dim_array = list(new.dimensions.items())[3][1]
            dim_array = R * np.cos(np.radians(ymean)) * np.radians(dim_array)
        if axis == 2:
            dim_array = list(new.dimensions.items())[2][1]
            dim_array = R * np.radians(dim_array)
        if dim_str is None:
            dim_str = dim_str_dict[axis]
        new.dim_arrays[dim_str] = dim_array
        new.indices[dim_str] = 0, dim_array.size, 1
        dims = list(new._current_dimensions)
        dims[axis] = dim_str
        new._current_dimensions = dims
        return new

    def tob(self, axis, dim_str=None):
        """Converts the vertical dimension to buoyancy from density

        :param axis: 1
        :param dim_str: String representing the newly converted dimension
        :returns: MOM6Variable
        :rtype: MOM6Variable

        """
        new = copy.copy(self)
        dim_array = list(new.dimensions.items())[axis][1]
        drhodt = -0.2
        rho0 = 1000.0
        g = 10
        dbdt = -drhodt * g / rho0
        dim_array = dbdt * (dim_array - dim_array[-1]) / drhodt
        if dim_str is None:
            dim_str = 'b'
        new.dim_arrays[dim_str] = dim_array
        new.indices[dim_str] = 0, dim_array.size, 1
        dims = list(new._current_dimensions)
        dims[axis] = dim_str
        new._current_dimensions = dims
        return new

    @property
    def dimensions(self):
        return self.return_dimensions()

    @property
    def hloc(self):
        return self._current_hloc

    @hloc.setter
    def hloc(self, loc):
        assert loc in ['u', 'v', 'h', 'q']
        self._current_hloc = loc
        self._current_dimensions = list(
            self.get_dimensions_by_location(self._current_hloc +
                                            self._current_vloc))

    @property
    def vloc(self):
        return self._current_vloc

    @vloc.setter
    def vloc(self, loc):
        assert loc in ['l', 'i']
        self._current_vloc = loc
        self._current_dimensions = list(
            self.get_dimensions_by_location(self._current_hloc +
                                            self._current_vloc))

    @property
    def shape(self):
        return self.array.shape if hasattr(self, 'array') else None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name

    @property
    def math(self):
        return self._math

    @math.setter
    def math(self, math):
        assert isinstance(math, str)
        self._math = math

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        assert isinstance(units, str)
        self._units = units

    @property
    def values(self):
        return self.array

    @values.setter
    def values(self, array):
        assert isinstance(array, np.ndarray)
        self.array = array

    def match_location(self, other):
        return (self._current_hloc == other._current_hloc
                and self._current_vloc == other._current_vloc)

    def __add__(self, other):
        to_add = other.values if (hasattr(other, 'array')
                                  and self.match_location(other)) else other
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a + to_add)
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        to_sub = other.values if (hasattr(other, 'array')
                                  and self.match_location(other)) else other
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a - to_sub)
        return new

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        to_mul = other.values if (hasattr(other, 'array')
                                  and self.match_location(other)) else other
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a * to_mul)
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        to_div = other.values if (hasattr(other, 'array')
                                  and self.match_location(other)) else other
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a / to_div)
        return new

    def __rtruediv__(self, other):
        to_div = other.values if (hasattr(other, 'array')
                                  and self.match_location(other)) else other
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: to_div / a)
        return new

    def __pow__(self, other):
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a**other)
        return new

    def __neg__(self):
        new = copy.copy(self)
        self.operations = []
        new.operations.append(lambda a: a * -1)
        return new

    def __repr__(self):
        f_line = """MOM6Variable: {}{}""".format(self._name,
                                                 self._current_dimensions)
        f_line += '\nDimensions:'
        for key, value in self.dimensions.items():
            f_line += '\n{} {}'.format(key, value.size)
        if self.array is not None:
            f_line += """\nArray: {}...""".format(self.values.ravel()[:4])
            f_line += """\n       {}""".format(self.values.ravel()[-4:])
            f_line += """\nShape: {}""".format(self.values.shape)
            f_line += """\nMax, Min: {}, """.format(np.amax(self.values))
            f_line += """{}""".format(np.amin(self.values))
        return f_line
