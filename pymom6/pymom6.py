import numpy as np
from functools import partial, partialmethod
from collections import OrderedDict
from netCDF4 import Dataset as dset, MFDataset as mfdset
import xarray as xr
from numba import jit
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def Dataset(fil, **initializer):
    fh = mfdset(fil) if isinstance(fil, list) else dset(fil)
    ds = SimpleNamespace()
    for var in fh.variables:
        try:
            setattr(ds, var, MOM6Variable(var, fh, **initializer))
        except AttributeError:
            pass
    yield ds
    fh.close()


class GridGeometry():
    def __init__(self, filename):
        with dset(filename) as fh:
            for var in fh.variables:
                setattr(self, var, fh.variables[var][:])

    def get_divisor_for_diff(self, loc, axis, weights=None):
        axis = axis - 2
        divisors = dict(
            u=['dyBu', 'dxT'],
            v=['dyT', 'dxBu'],
            h=['dyCv', 'dxCu'],
            q=['dyCu', 'dxCv'])
        if weights == 'area':
            divisors['u'] = ['Aq', 'Ah']
            divisors['v'] = ['Ah', 'Aq']
        return getattr(self, divisors[loc][axis])


def initialize_indices_and_dim_arrays(obj):
    if hasattr(obj, 'indices') is False:
        obj.indices = {}
    if hasattr(obj, 'dim_arrays') is False:
        obj.dim_arrays = {}


def find_index_limits(dimension, start, end):
    """Finds the extreme indices of the any given dimension of the domain."""
    useful_index = np.nonzero((dimension >= start) & (dimension <= end))[0]
    lims = useful_index[0], useful_index[-1] + 1
    return lims


def get_extremes(obj, dim_str, low, high, **initializer):
    initialize_indices_and_dim_arrays(obj)
    fh = initializer.get('fh')
    stride = initializer.get('stride' + dim_str[0].lower(), 1)
    try:
        dimension = fh.variables[dim_str][:]
        low = initializer.get(low, dimension[0])
        high = initializer.get(high, dimension[-1])
        obj.indices[dim_str] = *find_index_limits(dimension, low, high), stride
        obj.dim_arrays[dim_str] = dimension
    except KeyError:
        pass


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
        return self.func(array, *self.args, **self.kwargs)


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


class MOM6Variable(Domain):
    def __init__(self, var, fh, **initializer):
        self._name = initializer.get('name', var)
        self._v = fh.variables[var]
        self._initial_dimensions = list(self._v.dimensions)
        self._current_dimensions = list(self._v.dimensions)
        self.determine_location()
        initializer['fh'] = fh
        self.polish(**initializer)

    def polish(self, **initializer):
        Domain.__init__(self, **initializer)
        final_loc = initializer.get('final_loc', None)
        if final_loc:
            self._final_loc = final_loc
        else:
            self._final_loc = self._current_hloc + self._current_vloc
            self._final_dimensions = tuple(self._current_dimensions)
        self.get_final_location_dimensions()
        self._fillvalue = initializer.get('fillvalue', 0)
        self._bc_type = initializer.get('bc_type', None)
        self.geometry = initializer.get('geometry', None)
        self._units = initializer.get('units', None)
        self._math = initializer.get('math', None)
        self.array = None
        self.operations = []

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
        dims = self._final_dimensions
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

    def get_current_location_dimensions(self, loc):
        self._current_hloc = loc[0]
        self._current_vloc = loc[1]
        self._current_dimensions = list(self.get_dimensions_by_location(loc))

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
        def lazy_read_and_fill(array):
            array = self._v[self._slice]
            if np.ma.isMaskedArray(array):
                array = array.filled(self._fillvalue)
            return array

        self.operations.append(lazy_read_and_fill)
        self.implement_BC_if_necessary()
        return self

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
                # if i != 0 or self._current_vloc != 'i':
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

    def dbyd(self, axis, weights=None):
        if axis > 1:
            ns, ne = self.check_possible_movements_for_move(
                self._current_hloc, axis=axis)
            self.adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            divisor = self.geometry.get_divisor_for_diff(
                self._current_hloc, axis, weights=weights)
            self.get_slice_2D()
            divisor = divisor[self._slice_2D]
        elif axis == 1:
            divisor = 9.8 / 1031 * np.diff(
                self.dim_arrays[self._final_dimensions[1]][2:4])
            self.adjust_dimensions_and_indices_for_vertical_move()
        dadx = partial(lambda x, a: np.diff(a, n=1, axis=x) / divisor, axis)
        self.operations.append(dadx)
        return self

    LazyNumpyOperation = LazyNumpyOperation

    def np_ops(self, npfunc, *args, **kwargs):
        self.operations.append(
            self.LazyNumpyOperation(npfunc, *args, **kwargs))
        return self

    def nanmean(self, axis=[0, 1, 2, 3], keepdims=False):
        try:
            for ax in axis:
                axis_string = self._current_dimensions[ax]
                self.dim_arrays[axis_string] = np.mean(
                    self.dim_arrays[axis_string])
                self.indices.pop(axis_string)
        except TypeError:
            axis_string = self._current_dimensions[axis]
            self.dim_arrays[axis_string] = np.mean(
                self.dim_arrays[axis_string])

        def meanfunc(array):
            return np.nanmean(array, axis=axis, keepdims=keepdims)

        self.operations.append(meanfunc)
        return self

    @staticmethod
    @jit
    def get_var_at_z(array, z, e, fillvalue=0.0):
        array_out = np.full(
            (array.shape[0], z.size, array.shape[2], array.shape[3]),
            fillvalue)
        for l in range(array.shape[0]):
            for k in range(array.shape[1]):
                for j in range(array.shape[2]):
                    for i in range(array.shape[3]):
                        for m in range(z.size):
                            if (e[l, k, j, i] - z[m]) * (
                                    e[l, k + 1, j, i] - z[m]) <= 0:
                                array_out[l, m, j, i] = array[l, k, j, i]
        return array_out

    def toz(self, z, e):
        assert self._current_hloc == e._current_hloc
        assert e._current_vloc == 'i'

        def lazy_toz(array):
            return self.get_var_at_z(
                array, z, e.array, fillvalue=self.fillvalue)

        self.operations.append(lazy_toz)
        return self

    def compute(self):
        for ops in self.operations:
            self.array = ops(self.array)
        self.operations = []
        assert self._current_hloc == self._final_hloc
        assert self._current_vloc == self._final_vloc
        return self

    def to_DataArray(self):
        if len(self.operations) is not 0:
            self.compute()
        da = xr.DataArray(
            self.array,
            coords=self.return_dimensions(),
            dims=self._final_dimensions)
        da.name = self._name
        if self._math:
            da.attrs['math'] = self._math
        if self._units:
            da.attrs['units'] = self._units
        return da

    @property
    def dimensions(self):
        return self.return_dimensions()

    @property
    def hloc(self):
        return self._current_hloc

    @property
    def vloc(self):
        return self._current_vloc

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

    def match_location(self, other):
        return (self._current_hloc == other._current_hloc
                and self._current_vloc == other._current_vloc)

    def __add__(self, other):
        if hasattr(other, 'array') and self.match_location(other):
            self.array += other.values
        else:
            self.array += other
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if hasattr(other, 'array') and self.match_location(other):
            self.array -= other.values
        else:
            self.array -= other
        return self

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if hasattr(other, 'array') and self.match_location(other):
            self.array *= other.values
        else:
            self.array *= other
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if hasattr(other, 'array') and self.match_location(other):
            self.array /= other.values
        else:
            self.array /= other
        return self

    def __rtruediv__(self, other):
        if hasattr(other, 'array') and self.match_location(other):
            self.array = other.values / self.array
        else:
            self.array = other / self.array
        return self

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __neg__(self):
        self.array *= -1
        return self
