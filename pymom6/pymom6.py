import numpy as np
from functools import partial, partialmethod
from collections import OrderedDict
from netCDF4 import Dataset as dset, MFDataset as mfdset
import xarray as xr
from numba import jit, float32, float64
import copy


class Dataset():
    def __init__(self, filename, **initializer):
        """Creates a dataset from a single or multiple netcdf files.

        :param filename: Name of the netcdf file
        :returns: Dataset containing references to all variables in
        the file.
        :rtype: pymom6.Dataset

        """
        self.filename = filename
        self.initializer = initializer
        self.fh = mfdset(filename) if isinstance(filename,
                                                 list) else dset(filename)

    def close(self):
        self.fh.close()
        self.fh = None

    def __getattr__(self, var):
        if self.fh is not None:
            return self._variable_factory(var)
        else:
            raise AttributeError(f'{self.filename!r} is not open.')

    def _variable_factory(self, var):
        try:
            variable = MOM6Variable(var, self.fh, **self.initializer)
        except TypeError:
            variable = self.fh.variables[var][:]
        return variable

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class GridGeometry():
    def __init__(self, filename):
        with dset(filename) as fh:
            for var in fh.variables:
                setattr(self, var, fh.variables[var][:])

    def get_divisor_for_diff(self, loc, axis, weights=None):
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


def initialize_indices_and_dim_arrays(obj):
    if hasattr(obj, 'indices') is False:
        obj.indices = {}
    if hasattr(obj, 'dim_arrays') is False:
        obj.dim_arrays = {}


def find_index_limits(dimension, start, end):
    """Finds the extreme indices of the any given dimension of the domain."""
    if start == end:
        array = dimension - start
        useful_index = np.array([1, 1]) * np.argmax(array[array <= 0])
    else:
        useful_index = np.nonzero((dimension >= start) & (dimension <= end))[0]
    lims = useful_index[0], useful_index[-1] + 1
    return lims


def get_extremes(obj, dim_str, low, high, **initializer):
    initialize_indices_and_dim_arrays(obj)
    fh = initializer.get('fh')
    axis_str = dim_str[0].lower()
    stride = initializer.get('stride' + axis_str, 1)
    by_index = initializer.get('by_index', False)
    if dim_str in fh.variables:
        dimension = fh.variables[dim_str][:]
        if by_index:
            obj.indices[dim_str] = initializer.get('s' + axis_str,
                                                   0), initializer.get(
                                                       'e' + axis_str,
                                                       dimension.size), stride
        else:
            low = initializer.get(low, dimension[0])
            high = initializer.get(high, dimension[-1])
            obj.indices[dim_str] = *find_index_limits(dimension, low,
                                                      high), stride
        obj.dim_arrays[dim_str] = dimension


class MeridionalDomain():
    def __init__(self, **initializer):
        """Initializes meridional domain limits."""
        get_extremes(self, 'yh', 'south_lat', 'north_lat', **initializer)
        get_extremes(self, 'yq', 'south_lat', 'north_lat', **initializer)


class ZonalDomain():
    def __init__(self, **initializer):
        """Initializes zonal domain limits."""
        get_extremes(self, 'xh', 'west_lon', 'east_lon', **initializer)
        get_extremes(self, 'xq', 'west_lon', 'east_lon', **initializer)


class HorizontalDomain(MeridionalDomain, ZonalDomain):
    def __init__(self, **initializer):
        MeridionalDomain.__init__(self, **initializer)
        ZonalDomain.__init__(self, **initializer)


class VerticalDomain():
    def __init__(self, **initializer):
        get_extremes(self, 'zl', 'low_density', 'high_density', **initializer)
        get_extremes(self, 'zi', 'low_density', 'high_density', **initializer)


class TemporalDomain():
    def __init__(self, **initializer):
        get_extremes(self, 'Time', 'initial_time', 'final_time', **initializer)


class Domain(TemporalDomain, VerticalDomain, HorizontalDomain):
    def __init__(self, **initializer):
        TemporalDomain.__init__(self, **initializer)
        VerticalDomain.__init__(self, **initializer)
        HorizontalDomain.__init__(self, **initializer)


class LazyNumpyOperation():
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, array):
        func = self.func
        args = self.args
        kwargs = self.kwargs
        return func(array, *args, **kwargs)


class BoundaryCondition():
    def __init__(self, bc_type, axis, start_or_end):
        self.bc_type = bc_type
        self.axis = axis
        self.start_or_end = start_or_end

    def set_halo_indices(self):
        if self.bc_type == 'circsymq':
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
        if self.bc_type != 'mirror':
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


def get_var_at_z(array, z, e, fillvalue):
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


class MOM6Variable(Domain):
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
        self.determine_location()
        self.fh = fh
        initializer['fh'] = fh
        Domain.__init__(self, **initializer)
        self.array = None
        self.polish(**initializer)
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
        domain_mapping = {
            'T': TemporalDomain,
            'z': VerticalDomain,
            'y': MeridionalDomain,
            'x': ZonalDomain
        }
        low_mapping = {
            'T': 'initial_time',
            'z': 'low_density',
            'y': 'south_lat',
            'x': 'west_lon'
        }
        high_mapping = {
            'T': 'final_time',
            'z': 'high_density',
            'y': 'north_lat',
            'x': 'east_lon'
        }
        for key, value in kwargs.items():
            if isinstance(value, slice):
                axis = key[0]
                domain = domain_mapping[axis]
                kwargs = {}
                if value.start:
                    kwargs[low_mapping[axis]] = value.start
                if value.stop:
                    kwargs[high_mapping[axis]] = value.stop
                if value.step:
                    kwargs['stride' + axis.lower()] = value.step
                domain.__init__(self, fh=self.fh, **kwargs)
        return self

    def isel(self, **kwargs):
        domain_mapping = {
            'T': TemporalDomain,
            'z': VerticalDomain,
            'y': MeridionalDomain,
            'x': ZonalDomain
        }
        for key, value in kwargs.items():
            if isinstance(value, slice):
                axis = key[0]
                domain = domain_mapping[axis]
                kwargs = {}
                if value.start:
                    kwargs['s' + axis.lower()] = value.start
                if value.stop:
                    kwargs['e' + axis.lower()] = value.stop
                if value.step:
                    kwargs['stride' + axis.lower()] = value.step
                domain.__init__(self, by_index=True, fh=self.fh, **kwargs)
        return self

    def polish(self, **initializer):
        self.final_loc(initializer.get('final_loc', None))
        self.fillvalue(initializer.get('fillvalue', 0))
        self.bc_type(initializer.get('bc_type', None))
        self.geometry(initializer.get('geometry', None))
        self._units = initializer.get('units', None)
        self._math = initializer.get('math', None)
        self.operations = []
        return self

    def final_loc(self, final_loc=None):
        if final_loc:
            self._final_loc = final_loc
        else:
            self.final_loc(self._current_hloc + self._current_vloc)
            self._final_dimensions = tuple(self._current_dimensions)
        self.get_final_location_dimensions()
        return self

    def fillvalue(self, fillvalue):
        self._fillvalue = fillvalue
        return self

    def bc_type(self, bc_type):
        self._bc_type = bc_type
        return self

    def geometry(self, geometry):
        self._geometry = geometry
        return self

    def determine_location(self):
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
        loc_registry_hor = dict(
            u=['yh', 'xq'], v=['yq', 'xh'], h=['yh', 'xh'], q=['yq', 'xq'])
        loc_registry_ver = dict(l='zl', i='zi')
        hloc = loc[0]
        vloc = loc[1]
        vdim = loc_registry_ver[vloc]
        hdims = loc_registry_hor[hloc]
        return tuple(['Time', vdim, *hdims])

    def get_final_location_dimensions(self):
        self._final_hloc = self._final_loc[0]
        self._final_vloc = self._final_loc[1]
        self._final_dimensions = self.get_dimensions_by_location(
            self._final_loc)

    def modify_index(self, axis, startend, value):
        dim = self._final_dimensions[axis]
        axis_indices = list(self.indices[dim])
        axis_indices[startend] += value
        self.indices[dim] = tuple(axis_indices)

    def modify_index_return_self(self, axis, startend, value):
        self.modify_index(axis, startend, value)
        return self

    xsm = partialmethod(modify_index_return_self, 3, 0, -1)
    xep = partialmethod(modify_index_return_self, 3, 1, 1)
    ysm = partialmethod(modify_index_return_self, 2, 0, -1)
    yep = partialmethod(modify_index_return_self, 2, 1, 1)
    zsm = partialmethod(modify_index_return_self, 1, 0, -1)
    zep = partialmethod(modify_index_return_self, 1, 1, 1)

    def get_slice(self):
        # assert self._final_dimensions == tuple(self._current_dimensions)
        self._slice = []
        for axis in range(4):
            indices = self.get_slice_by_axis(axis)
            self._slice.append(slice(*indices))
        self._slice = tuple(self._slice)
        return self

    def get_slice_2D(self):
        self._slice_2D = []
        for axis in range(2, 4):
            indices = self.get_slice_by_axis(axis)
            self._slice_2D.append(slice(*indices))
        self._slice_2D = tuple(self._slice_2D)
        return self

    def get_slice_by_axis(self, axis):
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
        self.get_slice_2D()
        multiplier = getattr(self._geometry, multiplier)[self._slice_2D]**power
        multiplier = self.implement_BC_if_necessary_for_multiplier(multiplier)
        self.operations.append(lambda a: a * multiplier)
        return self

    divide_by = partialmethod(multiply_by, power=-1)

    BoundaryCondition = BoundaryCondition
    _default_bc_type = dict(
        u=['mirror', 'circsymh', 'circsymh', 'circsymh', 'zeros', 'circsymq'],
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'circsymh'],
        h=['mirror', 'circsymh', 'mirror', 'mirror', 'mirror', 'mirror'],
        q=['mirror', 'circsymh', 'zeros', 'circsymq', 'zeros', 'circsymq'])

    def implement_BC_if_necessary(self):
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
    def vertical_move(array):
        return 0.5 * (array[:, :-1, :, :] + array[:, 1:, :, :])

    @staticmethod
    def check_possible_movements_for_move(current_loc, new_loc=None,
                                          axis=None):
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
    def horizontal_move(axis, array):
        return 0.5 * (np.take(array, range(array.shape[axis] - 1), axis=axis) +
                      np.take(array, range(1, array.shape[axis]), axis=axis))

    def adjust_dimensions_and_indices_for_vertical_move(self):
        self.modify_index(1, 1, -1)
        if self._current_vloc == 'l':
            self.modify_index(1, 0, 1)
            self._current_dimensions[1] = 'zi'
        else:
            self._current_dimensions[1] = 'zl'
        self.determine_location()

    def adjust_dimensions_and_indices_for_horizontal_move(self, axis, ns, ne):
        self.modify_index(axis, 0, ns)
        self.modify_index(axis, 1, ne)
        current_dimension = list(self._current_dimensions[axis])
        if current_dimension[1] == 'h':
            current_dimension[1] = 'q'
        elif current_dimension[1] == 'q':
            current_dimension[1] = 'h'
        self._current_dimensions[axis] = "".join(current_dimension)
        self.determine_location()

    def move_to(self, new_loc):
        if new_loc in ['l', 'i'] and new_loc != self._current_vloc:
            self.adjust_dimensions_and_indices_for_vertical_move()
            self.operations.append(self.vertical_move)
        elif new_loc in ['u', 'v', 'h', 'q'] and new_loc != self._current_hloc:
            axis, ns, ne = self.check_possible_movements_for_move(
                self._current_hloc, new_loc=new_loc)
            self.adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            move = partial(self.horizontal_move, axis)
            self.operations.append(move)
        return self

    def implement_BC_if_necessary_for_multiplier(self, multiplier):
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
        if axis > 1:
            ns, ne = self.check_possible_movements_for_move(
                self._current_hloc, axis=axis)
            self.adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            divisor = self._geometry.get_divisor_for_diff(
                self._current_hloc, axis, weights=weights)
            self.get_slice_2D()
            divisor = divisor[self._slice_2D]
            divisor = self.implement_BC_if_necessary_for_multiplier(divisor)
        elif axis == 1:
            divisor = 9.8 / 1031 * np.diff(
                self.dim_arrays[self._final_dimensions[1]][2:4])
            self.adjust_dimensions_and_indices_for_vertical_move()
        dadx = partial(lambda x, a: np.diff(a, n=1, axis=x) / divisor, axis)
        self.operations.append(dadx)
        return self

    LazyNumpyOperation = LazyNumpyOperation

    def np_ops(self, npfunc, *args, **kwargs):
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

    get_var_at_z = staticmethod(
        jit(float64[:, :, :, :](float64[:, :, :, :], float64[:],
                                float64[:, :, :, :], float32))(get_var_at_z))

    def toz(self, z, e, dimstr='z (m)'):
        assert self._current_hloc == e._current_hloc
        assert e._current_vloc == 'i'
        if not isinstance(z, np.ndarray):
            z = np.array(
                z, dtype=np.float64) if isinstance(z, list) else np.array(
                    [z], dtype=np.float64)
        self.dim_arrays[dimstr] = z
        self.indices[dimstr] = 0, z.size, 1
        dims = list(self._current_dimensions)
        dims[1] = dimstr
        self._current_dimensions = dims

        def lazy_toz(array):
            return self.get_var_at_z(
                array.astype(np.float64), z,
                e.array.astype(np.float64), float(self._fillvalue))

        self.operations.append(lazy_toz)
        return self

    def compute(self, check_loc=True):
        for ops in self.operations:
            self.array = ops(self.array)
        self.operations = []
        if check_loc:
            ch = self._current_hloc
            fh = self._final_hloc
            cv = self._current_vloc
            fv = self._final_vloc
            assert ch == fh, f'Cur hloc = {ch} is not final hloc = {fh}'
            assert cv == fv, f'Cur vloc = {cv} is not final vloc = {fv}'
        return self

    def to_DataArray(self, check_loc=True):
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
        R = 6378
        dim_str_dict = {2: 'y (km)', 3: 'x (km)'}
        if axis == 3:
            ymean = np.mean(list(self.dimensions.items())[2][1])
            dim_array = list(self.dimensions.items())[3][1]
            dim_array = R * np.cos(np.radians(ymean)) * np.radians(dim_array)
        if axis == 2:
            dim_array = list(self.dimensions.items())[2][1]
            dim_array = R * np.radians(dim_array)
        if dim_str is None:
            dim_str = dim_str_dict[axis]
        self.dim_arrays[dim_str] = dim_array
        self.indices[dim_str] = 0, dim_array.size, 1
        dims = list(self._current_dimensions)
        dims[axis] = dim_str
        self._current_dimensions = dims
        return self

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
        f_line = f"""MOM6Variable: {self._name}{self._current_dimensions}"""
        f_line += '\nDimensions:'
        for key, value in self.dimensions.items():
            f_line += f'\n{key} {value.size}'
        if self.array is not None:
            f_line += f"""\nArray: {self.values.ravel()[:4]}..."""
            f_line += f"""\n       {self.values.ravel()[-4:]}"""
            f_line += f"""\nShape: {self.values.shape}"""
            f_line += f"""\nMax, Min: {np.amax(self.values)}, """
            f_line += f"""{np.amin(self.values)}"""
        return f_line
