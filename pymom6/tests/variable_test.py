import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import pymom6
from netCDF4 import Dataset as dset, MFDataset as mfdset
import numpy as np
import xarray as xr
import unittest
import os.path
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
gv3 = pymom6.MOM6Variable
geom = pymom6.GridGeometry
pdset = pymom6.Dataset

# pdset2 = pymom6.Dataset2


class test_variable(unittest.TestCase):
    def setUp(self):
        self.south_lat, self.north_lat = 30, 40
        self.west_lon, self.east_lon = -10, -5
        path = os.path.dirname(__file__) + '/data/'
        self.fil1 = path + 'output__0001_12_009.nc'
        self.fil2 = path + 'output__0001_11_019.nc'
        self.fh = dset(self.fil1)
        self.geom = geom(path + 'ocean_geometry.nc')
        self.mfh = mfdset([self.fil2, self.fil1])
        self.initializer = dict(
            geometry=self.geom,
            south_lat=self.south_lat,
            north_lat=self.north_lat,
            west_lon=self.west_lon,
            east_lon=self.east_lon)
        self.vars = ['e', 'u', 'v', 'wparam', 'RV']

    def tearDown(self):
        self.fh.close()
        self.mfh.close()

    def test_locations(self):
        hlocs = ['h', 'u', 'v', 'h', 'q']
        vlocs = ['i', 'l', 'l', 'l', 'l']
        for i, var in enumerate(self.vars):
            gvvar = gv3(var, self.fh, **self.initializer)
            self.assertEqual(gvvar.hloc, hlocs[i])
            self.assertEqual(gvvar.vloc, vlocs[i])

    def test_has_indices(self):
        for i, var in enumerate(self.vars):
            gvvar = gv3(var, self.fh, **self.initializer)
            for dim in gvvar.dimensions:
                self.assertIn(dim, gvvar.indices)

    def test_array(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh,
                        **self.initializer).get_slice().read().compute()
            slices = gvvar._slice
            array = self.fh.variables[var][slices]
            gvvar = gvvar.array
            with pdset(self.fil1, **self.initializer) as pdset_sub:
                pdvar = getattr(pdset_sub, var).read().compute().array
            with pdset(self.fil1) as pdset_sub:
                pdvar2 = getattr(pdset_sub, var).sel(
                    xh=slice(self.west_lon, self.east_lon, 1),
                    yh=slice(self.south_lat,
                             self.north_lat)).read().compute().array
                pdvar3 = getattr(pdset_sub, var).isel(
                    Time=slices[0],
                    zl=slices[1],
                    xh=slices[3],
                    yh=slice(slices[2].start,
                             slices[2].stop)).read().compute()
                xh = pdset_sub.xh
            self.assertIsInstance(xh, np.ndarray)
            self.assertIsInstance(gvvar, np.ndarray)
            self.assertTrue(np.allclose(gvvar, array))
            self.assertTrue(np.allclose(gvvar, pdvar))
            self.assertTrue(np.allclose(gvvar, pdvar2))
            self.assertTrue(np.allclose(gvvar, pdvar3.array))
            self.assertIsInstance(repr(pdvar3), str)
            with pdset(self.fil1, **self.initializer) as pdset_sub:
                pdvar4 = getattr(pdset_sub, var)
            self.assertIsInstance(repr(pdvar4), str)

    def test_array_check_loc(self):
        gvvar = gv3(
            'u', self.fh, final_loc='vl',
            **self.initializer).read().compute(check_loc=False)
        self.assertIsInstance(gvvar.array, np.ndarray)

    def test_array_divideby(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh,
                        **self.initializer).read().divide_by('dxT').compute()
            var_array = self.fh.variables[var][gvvar._slice]
            divisor = gvvar._geometry.dxT[gvvar._slice_2D]
            var_array /= divisor
            self.assertTrue(np.allclose(gvvar.array, var_array))

    def test_array_multiplyby(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh,
                **self.initializer).read().multiply_by('dxT').compute()
            var_array = self.fh.variables[var][gvvar._slice]
            multiplier = gvvar._geometry.dxT[gvvar._slice_2D]
            var_array *= multiplier
            self.assertTrue(np.allclose(gvvar.array, var_array))

    def test_multifile_array(self):
        for var in self.vars:
            gvvar = gv3(var, self.mfh,
                        **self.initializer).get_slice().read().compute()
            self.assertIsInstance(gvvar.array, np.ndarray)
            gvvar.name = 'temp'
            gvvar.units = 'unts'
            gvvar.math = 'mat'
            gvvar.vloc = 'i'
            gvvar.hloc = 'u'
            self.assertTrue(gvvar.hloc == 'u' and gvvar._current_vloc == 'i'
                            and gvvar.name == 'temp' and gvvar.units == 'unts'
                            and gvvar.math == 'mat')
            gvvar.vloc = 'l'
            gvvar.hloc = 'v'
            self.assertTrue(gvvar._current_hloc == 'v'
                            and gvvar._current_vloc == 'l')

    def test_array_full(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh).get_slice().read().compute().array
            var_array = self.fh.variables[var][:]
            with pdset(self.fil1) as pdset_full:
                pdvar = getattr(pdset_full,
                                var).get_slice().read().compute().array
            self.assertTrue(np.allclose(gvvar, var_array))
            self.assertTrue(np.allclose(gvvar, pdvar))

    def test_array_full_fillvalue(self):
        for i, fill in enumerate([np.nan, 0]):
            gvvar = gv3(
                'u', self.fh,
                fillvalue=fill).get_slice().read().compute().array
            if i == 0:
                self.assertTrue(np.all(np.isnan(gvvar[:, :, :, -1])))
            else:
                self.assertTrue(np.all(gvvar[:, :, :, -1] == 0))

    def test_numpy_func(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, fillvalue=0).get_slice().read().np_ops(
                    np.mean, keepdims=True).compute().array
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.mean(var_array, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar, var_array), msg=f'{gvvar,var_array}')

    def test_numpy_func_with_move(self):
        gvvar = gv3(
            'e', self.fh, fillvalue=0, final_loc='hl').zep().read().np_ops(
                np.diff, axis=1, sets_vloc='l').compute()
        var_array = self.fh.variables['e'][:]
        var_array = np.diff(var_array, axis=1)
        self.assertTrue(np.allclose(gvvar.array, var_array))
        self.assertTrue(gvvar.vloc == 'l')

    def test_numpy_func_with_move_wparam(self):
        gvvar = gv3(
            'wparam', self.fh, fillvalue=0,
            final_loc='hi').zsm().zep().read().np_ops(
                np.diff, axis=1, sets_vloc='i').compute()
        zi = self.fh.variables['zi'][:]
        var_array = self.fh.variables['wparam'][:]
        var_array = np.concatenate(
            (var_array[:, :1], var_array, -var_array[:, -1:]), axis=1)
        var_array = np.diff(var_array, axis=1)
        self.assertTrue(gvvar.vloc == 'i')
        self.assertTrue(np.allclose(gvvar.array, var_array))
        self.assertTrue(np.allclose(gvvar.dimensions['zi'], zi))

    def test_numpy_func_with_move_wparam_u(self):
        gvvar = gv3('wparam', self.fh, fillvalue=0).read().compute()
        gvvar = gvvar.np_ops(
            np.take,
            np.arange(0, gvvar.shape[3] - 1),
            axis=3,
            sets_hloc='h',
            ns=0,
            ne=-1).compute()
        var_array = self.fh.variables['wparam'][:, :, :, :-1]
        xh = self.fh.variables['xh'][:]
        self.assertTrue(np.allclose(gvvar.array, var_array))
        self.assertTrue(np.allclose(gvvar.dimensions['xh'], xh[:-1]))

    def test_where(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, fillvalue=0).read().where(
                    np.less_equal, 0, y=0).compute()
            array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(array):
                array.filled(0)
            array[array > 0] = 0
            self.assertTrue(np.allclose(gvvar.array, array), msg=f'{var}')
            gvvar = gv3(
                var, self.fh, fillvalue=0).read().where(np.less_equal,
                                                        0).compute()
            array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(array):
                array.filled(0)
            i = np.where(array <= 0)
            self.assertTrue(np.allclose(gvvar.array, i), msg=f'{var}')

    def test_boundary_conditions(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh).xsm().xep().ysm().yep().get_slice().read(
            ).compute().array
            var_array = self.fh.variables[var][:]
            shape1 = gvvar.shape
            shape2 = var_array.shape
            self.assertTrue(shape1[0] == shape2[0])
            self.assertTrue(shape1[1] == shape2[1])
            self.assertTrue(shape1[2] == shape2[2] + 2)
            self.assertTrue(shape1[3] == shape2[3] + 2)

    def test_final_locs(self):
        hlocs = ['h', 'u', 'v', 'q']
        vlocs = ['l', 'i']
        vdims = ['zl', 'zi']
        ydims = ['yh', 'yh', 'yq', 'yq']
        xdims = ['xh', 'xq', 'xh', 'xq']
        for var in self.vars:
            for i, hloc in enumerate(hlocs):
                for j, vloc in enumerate(vlocs):
                    gvvar = gv3(
                        var,
                        self.fh,
                        final_loc=hloc + vloc,
                        **self.initializer)
                    dims = gvvar._final_dimensions
                    self.assertTrue(dims[0] == 'Time')
                    self.assertTrue(dims[1] == vdims[j])
                    self.assertTrue(dims[2] == ydims[i])
                    self.assertTrue(dims[3] == xdims[i])

    def test_modify_indices(self):
        plusminus = [-1, 1]
        for i, dim in enumerate(['z', 'y', 'x']):
            for j, op in enumerate(['sm', 'ep']):
                for var in self.vars:
                    gvvar = gv3(var, self.fh, **self.initializer)
                    a = gvvar.indices[gvvar._final_dimensions[i + 1]]
                    gvvar = getattr(gvvar, dim + op)()
                    b = gvvar.indices[gvvar._final_dimensions[i + 1]]
                    self.assertEqual(a[j] + plusminus[j], b[j])

    def test_nanmean_tz(self):
        for var in self.vars:
            gvvar = (gv3(var, self.fh).get_slice().read()
                     .nanmean(axis=(0, 1)).compute())
            dims = gvvar.dimensions
            self.assertTrue(dims['Time'].size == 1)
            self.assertTrue(list(dims.items())[1][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array.filled(0)
            var_array = np.nanmean(var_array, axis=(0, 1), keepdims=True)
            self.assertTrue(np.allclose(gvvar.array, var_array))
            gvvar = (gv3(var, self.fh).get_slice().read()
                     .nanmean(axis=1).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[1][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nanmean(var_array, axis=1, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar.array, var_array),
                msg=f'{var, gvvar.array-var_array}')

    def test_nansum_reduce_tz(self):
        for var in self.vars:
            gvvar = (gv3(var, self.fh).read().reduce_(np.nansum,
                                                      axis=(0, 1)).compute())
            dims = gvvar.dimensions
            #self.assertTrue(list(dims.items())[0][1].size == 1)
            self.assertTrue(dims['Time'].size == 1)
            self.assertTrue(list(dims.items())[1][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array.filled(0)
            var_array = np.nansum(var_array, axis=(0, 1), keepdims=True)
            self.assertTrue(np.allclose(gvvar.array, var_array))
            gvvar = (gv3(var, self.fh).read().reduce_(np.nansum,
                                                      axis=1).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[1][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nansum(var_array, axis=1, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar.array, var_array),
                msg=f'{var, gvvar.array-var_array}')

    def test_nanmean_xy(self):
        for var in self.vars:
            gvvar = (gv3(
                var,
                self.fh).get_slice().read().nanmean(axis=(2, 3)).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[2][1].size == 1)
            self.assertTrue(list(dims.items())[3][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nanmean(var_array, axis=(2, 3), keepdims=True)
            self.assertTrue(np.allclose(gvvar.array, var_array), )
            gvvar = (gv3(var, self.fh).get_slice().read()
                     .nanmean(axis=2).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[2][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nanmean(var_array, axis=2, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar.array, var_array),
                msg=f'{var, gvvar.array-var_array}')

    def test_nansum_reduce_xy(self):
        for var in self.vars:
            gvvar = (gv3(var, self.fh).read().reduce_(np.nansum,
                                                      axis=(2, 3)).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[2][1].size == 1)
            self.assertTrue(list(dims.items())[3][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nansum(var_array, axis=(2, 3), keepdims=True)
            self.assertTrue(np.allclose(gvvar.array, var_array), )
            gvvar = (gv3(var, self.fh).read().reduce_(np.nansum,
                                                      axis=2).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[2][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nansum(var_array, axis=2, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar.array, var_array),
                msg=f'{var, gvvar.array-var_array}')
            gvvar = (gv3(var, self.fh).read().reduce_(
                np.nansum, keepdims=False, axis=2).compute())
            dims = gvvar.dimensions
            self.assertTrue(list(dims.items())[2][1].size == 1)
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.nansum(var_array, axis=2, keepdims=False)
            self.assertTrue(
                np.allclose(gvvar.array, var_array),
                msg=f'{var, gvvar.array-var_array}')

    def test_xarray(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, units='m', math='1',
                **self.initializer).get_slice().read().to_DataArray()
            self.assertIsInstance(gvvar, xr.DataArray)
            self.assertTrue(gvvar.name == var)
            self.assertTrue(gvvar.attrs['math'] == '1')
            self.assertTrue(gvvar.attrs['units'] == 'm')

    def test_xarray_dimensions(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, **self.initializer).get_slice().read().nanmean(
                    axis=0).to_DataArray()
            shp = gvvar.shape
            self.assertIsInstance(gvvar, xr.DataArray)
            self.assertTrue(len(shp) == 4)
            self.assertTrue(shp[0] == 1)
            gvvar = gv3(var, self.fh,
                        **self.initializer).get_slice().read().nanmean(
                            axis=(0, 2)).to_DataArray()
            shp = gvvar.shape
            self.assertIsInstance(gvvar, xr.DataArray)
            self.assertTrue(len(shp) == 4)
            self.assertTrue(shp[0] == 1)
            self.assertTrue(shp[2] == 1)

    def test_get_var_at_z(self):
        array = np.full((1, 3, 5, 5), 1)
        array[:, 0, :, :] = 2
        array[:, 2, :, :] = 0
        e = np.full((1, 4, 5, 5), -2500)
        e[:, 0] = 0
        e[:, 1] = -1000
        e[:, 2] = -2000
        z = np.array([-2500, -1250, -750, -1])
        array_at_z = pymom6.get_var_at_z(array, z, e, 0)
        self.assertTrue(np.all(array_at_z[:, 0] == 0))
        self.assertTrue(np.all(array_at_z[:, 1] == 1))
        self.assertTrue(np.all(array_at_z[:, 2] == 2))
        self.assertTrue(np.all(array_at_z[:, 3] == 2))

    def test_var_get_atz(self):
        gvvar = gv3('wparam', self.fh,
                    **self.initializer).get_slice().read().compute()
        e = gv3('e', self.fh, **self.initializer).get_slice().read().compute()
        array = gvvar.values
        array[:, -1] = 1e-2
        gvvar.values = array
        self.assertTrue(np.all(gvvar.array[:, -1] == 1e-2))
        gvvar = gvvar.toz(-2400, e).compute()
        self.assertTrue(gvvar.shape[1] == 1)
        self.assertTrue(gvvar._current_dimensions[1] == 'z (m)')
        self.assertTrue(np.allclose(gvvar.array[:, -1], 1e-2))

    def test_var_get_atz_xarray(self):
        gvvar = gv3('wparam', self.fh,
                    **self.initializer).get_slice().read().compute()
        e = gv3('e', self.fh, **self.initializer).get_slice().read().compute()
        array = gvvar.values
        array[:, -1] = 123
        gvvar.values = array
        self.assertTrue(np.all(gvvar.array[:, -1] == 123))
        gvvar = gvvar.toz(-2400, e).to_DataArray()
        self.assertTrue(gvvar.shape[1] == 1)
        self.assertTrue(gvvar.dims[1] == 'z (m)')
        self.assertTrue(np.all(gvvar.values[:, -1] == 123))

    def test_var_get_atz_plot(self):
        e = gv3(
            'e', self.fh, final_loc='ui',
            **self.initializer).zep().xep().read().move_to('u').nanmean(
                axis=(0, 2)).compute()
        gvvar = gv3('u', self.fh,
                    **self.initializer).read().nanmean(axis=(0, 2)).toz(
                        np.linspace(-2400, -1, 5), e).to_DataArray()
        fig, ax = plt.subplots(1, 1)
        im = gvvar.plot(ax=ax)
        self.assertIsInstance(im, mpl.collections.QuadMesh)
        e = gv3(
            'e', self.fh, final_loc='ui',
            **self.initializer).zep().xep().read().move_to('u').nanmean(
                axis=2).compute()
        gvvar = gv3('u',
                    self.fh, **self.initializer).read().nanmean(axis=2).toz(
                        -1, e).to_DataArray()
        im = gvvar.plot(ax=ax)
        self.assertIsInstance(im, mpl.collections.QuadMesh)

    def test_var_get_atz_withnan(self):
        e = gv3('e', self.fh, **self.initializer).get_slice().read().compute()
        gvvar = gv3(
            'wparam', self.fh, fillvalue=np.nan,
            **self.initializer).get_slice().read().toz(0, e).compute()
        self.assertTrue(gvvar.shape[1] == 1)
        self.assertTrue(gvvar._current_dimensions[1] == 'z (m)')
        self.assertTrue(np.any(np.isnan(gvvar.array)), msg=f'{gvvar.array}')

    def test_tokm(self):
        dim_str = ['x (km)', 'y (km)']
        axis = [3, 2]
        for i, dstr in enumerate(dim_str):
            gvvar = gv3('wparam', self.fh, **self.initializer).read().tokm(
                axis[i]).compute()
            slc = gvvar.get_slice_2D()._slice_2D
            self.assertTrue(list(gvvar.dimensions.keys())[axis[i]] == dstr)
            xh = self.fh.variables['xh'][:]
            xh = xh[slc[1]]
            yh = self.fh.variables['yh'][:]
            yh = yh[slc[0]]
            if i == 0:
                x = 6378 * np.cos(np.radians(yh.mean())) * np.radians(xh)
                self.assertTrue(
                    np.allclose(list(gvvar.dimensions.values())[axis[i]], x),
                    msg=f'{x}')
            if i == 1:
                y = 6378 * np.radians(yh)
                self.assertTrue(
                    np.allclose(list(gvvar.dimensions.values())[axis[i]], y))
            gvvar = gv3('wparam', self.fh, **self.initializer).read().tokm(
                axis[i], dim_str='test').compute()
            slc = gvvar.get_slice_2D()._slice_2D
            self.assertTrue(list(gvvar.dimensions.keys())[axis[i]] == 'test')

    def test_getitem(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh, units='m', math='1')[dict(x=-20,y=35)].read().compute()
            self.assertTrue(gvvar.shape[2] == 1)
            self.assertTrue(gvvar.shape[3] == 1)
            gvvar = gv3(var, self.fh, units='m', math='1')[dict(final_loc='vl',Time=slice(342,357),x=-20,y=35)].read().compute(check_loc=False)
            self.assertTrue(gvvar.shape[0] == 3)
            self.assertTrue(gvvar.shape[2] == 1)
            self.assertTrue(gvvar.shape[3] == 1)
            self.assertTrue(gvvar._final_hloc == 'v')
            self.assertTrue(gvvar._final_vloc == 'l')
