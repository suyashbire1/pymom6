# from pymom6 import pymom6
import pymom6
from netCDF4 import Dataset as dset
import numpy as np
import unittest
import os.path
gv3 = pymom6.MOM6Variable


class test_domain(unittest.TestCase):
    def setUp(self):
        self.slat, self.nlat = 30, 40
        self.wlon, self.elon = -10, -5
        path = os.path.dirname(__file__) + '/data/'
        self.fh = dset(path + 'output__0001_12_009.nc')
        self.initializer = dict(
            fh=self.fh,
            south_lat=self.slat,
            north_lat=self.nlat,
            west_lon=self.wlon,
            east_lon=self.elon)

    def tearDown(self):
        self.fh.close()

    def test_meridional_domain(self):
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = pymom6.MeridionalDomain(
                **self.initializer).indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_zonal_domain(self):
        for loc in ['h', 'q']:
            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = pymom6.ZonalDomain(**self.initializer).indices['x' + loc]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_stride_meridional_domain(self):
        for stride in range(2, 4):
            self.initializer['stridey'] = stride
            for loc in ['h', 'q']:
                var = 'y' + loc
                lat = self.fh.variables[var][:]
                lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
                lat_restricted = lat_restricted[::stride]
                a, b, c = pymom6.MeridionalDomain(
                    **self.initializer).indices['y' + loc]
                self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_stride_zonal_domain(self):
        for stride in range(2, 4):
            self.initializer['stridex'] = stride
            for loc in ['h', 'q']:
                var = 'x' + loc
                lon = self.fh.variables[var][:]
                lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
                lon_restricted = lon_restricted[::stride]
                a, b, c = pymom6.ZonalDomain(**self.initializer).indices['x'
                                                                         + loc]
                self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_horizontal_domain(self):
        hdomain = pymom6.HorizontalDomain(**self.initializer)
        self.initializer['stridex'] = 1
        self.initializer['stridey'] = 1
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = hdomain.indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = hdomain.indices['x' + loc]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_vertical_domain(self):
        zl = self.fh.variables['zl'][:]
        zi = self.fh.variables['zi'][:]
        a, b, c = pymom6.VerticalDomain(**self.initializer).indices['zl']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(zl))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(zl, zl[a:b:c]))
        a, b, c = pymom6.VerticalDomain(**self.initializer).indices['zi']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(zi))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(zi, zi[a:b:c]))

    def test_temporal_domain(self):
        Time = self.fh.variables['Time'][:]
        a, b, c = pymom6.TemporalDomain(**self.initializer).indices['Time']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(Time))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(Time, Time[a:b:c]))

    def test_domain(self):
        zl = self.fh.variables['zl'][:]
        zi = self.fh.variables['zi'][:]
        Time = self.fh.variables['Time'][:]
        domain = pymom6.Domain(**self.initializer)
        self.assertEqual(domain.indices['Time'], (0, len(Time), 1))
        self.assertEqual(domain.indices['zl'], (0, len(zl), 1))
        self.assertEqual(domain.indices['zi'], (0, len(zi), 1))
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = domain.indices[var]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = domain.indices[var]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))
