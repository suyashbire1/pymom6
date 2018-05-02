from context import pymom6
from netCDF4 import Dataset as dset
import numpy as np
import unittest
import os.path
gv3 = pymom6.MOM6Variable


class test_move(unittest.TestCase):
    def setUp(self):
        self.south_lat, self.north_lat = 30, 40
        self.west_lon, self.east_lon = -10, -5
        path = os.path.dirname(__file__) + '/data/'
        self.fh = dset(path + 'output__0001_12_009.nc')
        self.fhgeo = dset(path + 'ocean_geometry.nc')
        self.geometry = pymom6.GridGeometry(path + 'ocean_geometry.nc')
        self.initializer = dict(
            south_lat=self.south_lat,
            north_lat=self.north_lat,
            west_lon=self.west_lon,
            east_lon=self.east_lon,
            geometry=self.geometry)
        self.vars = ['e', 'u', 'v', 'wparam', 'RV']

    def tearDown(self):
        self.fh.close()
        self.fhgeo.close()

    def test_move_u(self):
        ops = ['yep', 'xsm']
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('u', self.fh, final_loc=new_loc[i] + 'l'),
                            op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            u = self.fh.variables['u'][:]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if i == 0:
                u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
                u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            elif i == 1:
                u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
                u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
            self.assertTrue(np.allclose(u, gvvar))

    def test_move_u_subset(self):
        ops = ['yep', 'xsm']
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('u',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'yep':
                xq = self.fh.variables['xq'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                iy = np.append(iy, iy[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                ix = np.insert(ix, 0, ix[0] - 1)
            u = self.fh.variables['u'][:, :, iy, ix]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if op == 'yep':
                u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            else:
                u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
            self.assertTrue(np.allclose(u, gvvar))

    def test_move_v(self):
        ops = ['ysm', 'xep']
        new_loc = ['h', 'q']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('v', self.fh, final_loc=new_loc[i] + 'l'),
                            op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'v')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            v = self.fh.variables['v'][:]
            if np.ma.isMaskedArray(v):
                v = v.filled(0)
            if i == 0:
                v = np.concatenate((np.zeros(v[:, :, :1, :].shape), v), axis=2)
                v = 0.5 * (v[:, :, :-1] + v[:, :, 1:])
            elif i == 1:
                v = np.concatenate((v, -v[:, :, :, -1:]), axis=3)
                v = 0.5 * (v[:, :, :, :-1] + v[:, :, :, 1:])
            self.assertTrue(np.allclose(v, gvvar))

    def test_move_v_subset(self):
        ops = ['ysm', 'xep']
        new_loc = ['h', 'q']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('v',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'v')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xep':
                xq = self.fh.variables['xq'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                ix = np.append(ix, ix[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                iy = np.insert(iy, 0, iy[0] - 1)
            v = self.fh.variables['v'][:, :, iy, ix]
            if np.ma.isMaskedArray(v):
                v = v.filled(0)
            if op == 'xep':
                v = 0.5 * (v[:, :, :, :-1] + v[:, :, :, 1:])
            else:
                v = 0.5 * (v[:, :, :-1] + v[:, :, 1:])
            self.assertTrue(np.allclose(v, gvvar))

    def test_move_h(self):
        ops = ['xep', 'yep']
        new_loc = ['u', 'v']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('wparam', self.fh, final_loc=new_loc[i] + 'l'),
                op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'h')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            w = self.fh.variables['wparam'][:]
            if np.ma.isMaskedArray(w):
                w = w.filled(0)
            if i == 0:
                w = np.concatenate((w, w[:, :, :, -1:]), axis=3)
                w = 0.5 * (w[:, :, :, :-1] + w[:, :, :, 1:])
            elif i == 1:
                w = np.concatenate((w, w[:, :, -1:, :]), axis=2)
                w = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
            self.assertTrue(np.allclose(w, gvvar))

    def test_move_h_subset(self):
        ops = ['xep', 'yep']
        new_loc = ['u', 'v']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('wparam',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'h')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xep':
                xq = self.fh.variables['xq'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                ix = np.append(ix, ix[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                iy = np.append(iy, iy[-1] + 1)
            w = self.fh.variables['wparam'][:, :, iy, ix]
            if np.ma.isMaskedArray(w):
                w = w.filled(0)
            if op == 'xep':
                w = 0.5 * (w[:, :, :, :-1] + w[:, :, :, 1:])
            else:
                w = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
            self.assertTrue(np.allclose(w, gvvar))

    def test_move_q(self):
        ops = ['xsm', 'ysm']
        new_loc = ['v', 'u']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('RV', self.fh, final_loc=new_loc[i] + 'l'),
                op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'q')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j], msg=f'{value}')
            rv = self.fh.variables['RV'][:]
            if np.ma.isMaskedArray(rv):
                rv = rv.filled(0)
            if i == 0:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :, :1].shape), rv), axis=3)
                rv = 0.5 * (rv[:, :, :, :-1] + rv[:, :, :, 1:])
            elif i == 1:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :1, :].shape), rv), axis=2)
                rv = 0.5 * (rv[:, :, :-1] + rv[:, :, 1:])
            self.assertTrue(np.allclose(rv, gvvar))

    def test_move_q_subset(self):
        ops = ['xsm', 'ysm']
        new_loc = ['v', 'u']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('RV',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'q')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xsm':
                xh = self.fh.variables['xh'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                ix = np.insert(ix, 0, ix[0] - 1)
            else:
                xq = self.fh.variables['xq'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                iy = np.insert(iy, 0, iy[0] - 1)
            r = self.fh.variables['RV'][:, :, iy, ix]
            if np.ma.isMaskedArray(r):
                r = r.filled(0)
            if op == 'xsm':
                r = 0.5 * (r[:, :, :, :-1] + r[:, :, :, 1:])
            else:
                r = 0.5 * (r[:, :, :-1] + r[:, :, 1:])
            self.assertTrue(np.allclose(r, gvvar))

    def test_move_vertical_e(self):
        gvvar = gv3('e', self.fh, final_loc='hl').zep().get_slice().read()
        self.assertTrue(gvvar.vloc == 'i')
        gvvar = gvvar.move_to('l')
        self.assertTrue(gvvar.vloc == 'l')
        dims = gvvar.dimensions
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        e = self.fh.variables['e'][:]
        e = 0.5 * (e[:, :-1] + e[:, 1:])
        self.assertTrue(np.allclose(e, gvvar))

    def test_move_vertical_u(self):
        gvvar = gv3(
            'u', self.fh, final_loc='ui').zsm().zep().get_slice().read()
        self.assertTrue(gvvar.vloc == 'l')
        gvvar = gvvar.move_to('i')
        self.assertTrue(gvvar.vloc == 'i')
        dims = gvvar.dimensions
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        u = self.fh.variables['u'][:]
        if np.ma.isMaskedArray(u):
            u = u.filled(0)
        u = np.concatenate((u[:, :1, :, :], u, -u[:, -1:, :, :]), axis=1)
        u = 0.5 * (u[:, :-1] + u[:, 1:])
        self.assertTrue(np.allclose(u, gvvar))

    def test_move_u_twice(self):
        gvvar = gv3(
            'u', self.fh, final_loc='vl').xsm().yep().get_slice().read()
        self.assertTrue(gvvar.hloc == 'u')
        gvvar = gvvar.move_to('h')
        self.assertTrue(gvvar.hloc == 'h')
        gvvar = gvvar.move_to('v')
        self.assertTrue(gvvar.hloc == 'v')
        dims = gvvar.dimensions
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        u = self.fh.variables['u'][:]
        if np.ma.isMaskedArray(u):
            u = u.filled(0)
        u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
        u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
        u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
        u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        self.assertTrue(np.allclose(u, gvvar))

    def test_ddx_twice(self):
        gvvar = gv3('u', self.fh, **self.initializer).xep().xsm().read().dbyd(
            3, weights='area').move_to('u').compute()
        self.assertTrue(isinstance(gvvar.array, np.ndarray))
        gvvar = gv3(
            'u', self.fh, geometry=self.geometry).xep().xsm().read().dbyd(
                3, weights='area').move_to('u').compute()
        self.assertTrue(isinstance(gvvar.array, np.ndarray))

    def test_ddx_u_subset(self):
        ops = ['yep', 'xsm']
        axis = [2, 3]
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('u',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'yep':
                xq = self.fh.variables['xq'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                divisor = self.fhgeo.variables['dyBu'][iy, ix]
                iy = np.append(iy, iy[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                divisor = self.fhgeo.variables['dxT'][iy, ix]
                ix = np.insert(ix, 0, ix[0] - 1)
            u = self.fh.variables['u'][:, :, iy, ix]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            du = np.diff(u, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(du, gvvar))

    def test_ddx_v_subset(self):
        ops = ['ysm', 'xep']
        new_loc = ['h', 'q']
        axis = [2, 3]
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('v',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'v')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xep':
                xq = self.fh.variables['xq'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                divisor = self.fhgeo.variables['dxBu'][iy, ix]
                ix = np.append(ix, ix[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                divisor = self.fhgeo.variables['dyT'][iy, ix]
                iy = np.insert(iy, 0, iy[0] - 1)
            v = self.fh.variables['v'][:, :, iy, ix]
            if np.ma.isMaskedArray(v):
                v = v.filled(0)
            dv = np.diff(v, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dv, gvvar), msg=f'{dv-gvvar}')

    def test_ddx_h_subset(self):
        ops = ['xep', 'yep']
        axis = [3, 2]
        new_loc = ['u', 'v']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('wparam',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'h')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xep':
                xq = self.fh.variables['xq'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                divisor = self.fhgeo.variables['dxCu'][iy, ix]
                ix = np.append(ix, ix[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                divisor = self.fhgeo.variables['dyCv'][iy, ix]
                iy = np.append(iy, iy[-1] + 1)
            w = self.fh.variables['wparam'][:, :, iy, ix]
            if np.ma.isMaskedArray(w):
                w = w.filled(0)
            dw = np.diff(w, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dw, gvvar))

    def test_ddx_q_subset(self):
        ops = ['xsm', 'ysm']
        axis = [3, 2]
        new_loc = ['v', 'u']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('RV',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'q')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'xsm':
                xh = self.fh.variables['xh'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                divisor = self.fhgeo.variables['dxCv'][iy, ix]
                ix = np.insert(ix, 0, ix[0] - 1)
            else:
                xq = self.fh.variables['xq'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                divisor = self.fhgeo.variables['dyCu'][iy, ix]
                iy = np.insert(iy, 0, iy[0] - 1)
            r = self.fh.variables['RV'][:, :, iy, ix]
            if np.ma.isMaskedArray(r):
                r = r.filled(0)
            dr = np.diff(r, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dr, gvvar))

    def test_ddx_u(self):
        ops = ['yep', 'xsm']
        axis = [2, 3]
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('u',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    geometry=self.geometry), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            u = self.fh.variables['u'][:]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if i == 0:
                u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
                divisor1 = self.fhgeo.variables['dyCu'][:]
                divisor = self.fhgeo.variables['dyBu'][:]
                self.assertTrue(np.allclose(divisor, divisor1))
            elif i == 1:
                u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
                divisor = self.fhgeo.variables['dxT'][:]
            du = np.diff(u, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(du, gvvar))

    def test_ddx_u_weights_area(self):
        ops = ['yep', 'xsm']
        axis = [2, 3]
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('u',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    geometry=self.geometry), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.dbyd(axis[i], weights='area')
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            u = self.fh.variables['u'][:]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if i == 0:
                u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
                divisor = self.fhgeo.variables['Aq'][:]
            elif i == 1:
                u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
                divisor = self.fhgeo.variables['Ah'][:]
            du = np.diff(u, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(du, gvvar), msg=f'{i}')

    def test_ddx_v(self):
        ops = ['ysm', 'xep']
        new_loc = ['h', 'q']
        axis = [2, 3]
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('v',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    geometry=self.geometry), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'v')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            v = self.fh.variables['v'][:]
            if np.ma.isMaskedArray(v):
                v = v.filled(0)
            if i == 0:
                v = np.concatenate((np.zeros(v[:, :, :1, :].shape), v), axis=2)
                divisor = self.fhgeo.variables['dyT'][:]
                dv = np.diff(v, axis=axis[i]) / divisor
            elif i == 1:
                v = np.concatenate((v, -v[:, :, :, -1:]), axis=3)
                divisor = self.fhgeo.variables['dxBu'][:]
            dv = np.diff(v, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dv, gvvar), msg=f'{dv-gvvar}')

    def test_ddx_h(self):
        ops = ['xep', 'yep']
        axis = [3, 2]
        new_loc = ['u', 'v']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('wparam',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    geometry=self.geometry), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'h')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            w = self.fh.variables['wparam'][:]
            if np.ma.isMaskedArray(w):
                w = w.filled(0)
            if i == 0:
                w = np.concatenate((w, w[:, :, :, -1:]), axis=3)
                divisor = self.fhgeo.variables['dxCu'][:]
            elif i == 1:
                w = np.concatenate((w, w[:, :, -1:, :]), axis=2)
                divisor = self.fhgeo.variables['dyCv'][:]
            dw = np.diff(w, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dw, gvvar))

    def test_ddx_q(self):
        ops = ['xsm', 'ysm']
        axis = [3, 2]
        new_loc = ['v', 'u']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('RV',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    geometry=self.geometry), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'q')
            gvvar = gvvar.dbyd(axis[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            rv = self.fh.variables['RV'][:]
            if np.ma.isMaskedArray(rv):
                rv = rv.filled(0)
            if i == 0:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :, :1].shape), rv), axis=3)
                divisor = self.fhgeo.variables['dxCv'][:]
            elif i == 1:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :1, :].shape), rv), axis=2)
                divisor = self.fhgeo.variables['dyCu'][:]
            dr = np.diff(rv, axis=axis[i]) / divisor
            self.assertTrue(np.allclose(dr, gvvar))

    def test_ddx_vertical_e(self):
        gvvar = gv3('e', self.fh, final_loc='hl').zep().get_slice().read()
        self.assertTrue(gvvar.vloc == 'i')
        gvvar = gvvar.dbyd(1)
        self.assertTrue(gvvar.vloc == 'l')
        dims = gvvar.dimensions
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        e = self.fh.variables['e'][:]
        zi = self.fh.variables['zi'][:]
        db = 9.8 / 1031 * (zi[3] - zi[2])
        de = np.diff(e, axis=1) / db
        self.assertTrue(np.allclose(de, gvvar))

    def test_ddx_vertical_u(self):
        gvvar = gv3(
            'u', self.fh, final_loc='ui').zsm().zep().get_slice().read()
        self.assertTrue(gvvar.vloc == 'l')
        gvvar = gvvar.dbyd(1)
        self.assertTrue(gvvar.vloc == 'i')
        dims = gvvar.dimensions
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        u = self.fh.variables['u'][:]
        if np.ma.isMaskedArray(u):
            u = u.filled(0)
        u = np.concatenate((u[:, :1, :, :], u, -u[:, -1:, :, :]), axis=1)
        zi = self.fh.variables['zi'][:]
        db = 9.8 / 1031 * (zi[3] - zi[2])
        du = np.diff(u, axis=1) / db
        self.assertTrue(np.allclose(du, gvvar))

    def test_move_u_subset_xarray(self):
        ops = ['yep', 'xsm']
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(
                gv3('u',
                    self.fh,
                    final_loc=new_loc[i] + 'l',
                    **self.initializer), op)().get_slice().read().move_to(
                        new_loc[i])
            dims = gvvar.dimensions
            gvvar = gvvar.to_DataArray()
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(
                    value.size == gvvar.shape[j],
                    msg=f'{value.size,gvvar.shape[j]}')
            if op == 'yep':
                xq = self.fh.variables['xq'][:]
                yq = self.fh.variables['yq'][:]
                ix = np.where((xq > self.west_lon) & (xq < self.east_lon))[0]
                iy = np.where((yq > self.south_lat) & (yq < self.north_lat))[0]
                iy = np.append(iy, iy[-1] + 1)
            else:
                xh = self.fh.variables['xh'][:]
                yh = self.fh.variables['yh'][:]
                ix = np.where((xh > self.west_lon) & (xh < self.east_lon))[0]
                iy = np.where((yh > self.south_lat) & (yh < self.north_lat))[0]
                ix = np.insert(ix, 0, ix[0] - 1)
            u = self.fh.variables['u'][:, :, iy, ix]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if op == 'yep':
                u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            else:
                u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
            self.assertTrue(np.allclose(u, gvvar.values))
