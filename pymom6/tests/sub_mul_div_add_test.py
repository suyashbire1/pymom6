from context import pymom6
import os.path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def test_sub():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = getattr(pdset, var).read() - 2
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()
        momvar3 = getattr(pdset, var).read().compute()
    assert np.all(momvar.array == -(2 - momvar1).compute().array)
    assert np.allclose((momvar3 - momvar).compute().array, 2.0)


def test_mul():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = getattr(pdset, var).read() * 2
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()
        momvar3 = getattr(pdset, var).read().compute()
        temp = (momvar3 / momvar).compute().array
        temp = temp[np.isfinite(temp)]
    assert np.all(momvar.array == (2 * momvar1).compute().array)
    assert np.allclose(temp, 0.5)


def test_div():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = getattr(pdset, var).read() / 2
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()
        momvar3 = getattr(pdset, var).read().compute()
        temp = (momvar3 / momvar).compute().array
        temp = temp[np.isfinite(temp)]
    assert np.all(momvar1.array == (2 * momvar).compute().array)
    assert np.allclose(temp, 2.0)


def test_add():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = getattr(pdset, var).read() + 2
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()
        momvar3 = getattr(pdset, var).read().compute()

    assert np.all(momvar.array == (2 + momvar1).compute().array)
    assert np.allclose((momvar3 - momvar).compute().array, -2.0)


def test_pow():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = getattr(pdset, var).read()**2
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()

    assert np.all(momvar.array == momvar1.array**2)


def test_rtruediv():
    var = 'u'
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = 1 / getattr(pdset, var).read()
        momvar = momvar.compute()
        momvar1 = getattr(pdset, var).read().compute()

    assert np.all(momvar.array == 1 / momvar1.array)


def test_add_mul_view_or_copy():
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        momvar = pdset.u.read().compute()
        momvar1 = pdset.u.read().compute()
        momvar2 = pdset.u.read().compute()
        assert id(momvar) != id(momvar1)
        assert id(momvar) != id(momvar2)
        assert id(momvar1) != id(momvar2)
        momvar3 = (momvar * momvar1 * momvar2)
        momvar3.name = '3'
        momvar3 = momvar3.compute()
        momvar4 = (momvar + momvar1 + momvar2)
        momvar4.name = '4'
        momvar4 = momvar4.compute()
        assert np.any(momvar3.array != momvar.array)
        assert id(momvar3.array) != id(momvar.array)
        assert np.any(momvar4.array != momvar.array)
        assert id(momvar4.array) != id(momvar.array)


def test_add_more_than_two():
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset(fil2) as pdset:
        u = pdset.u.read().compute()
        u1 = pdset.u.read().compute()
        u2 = pdset.u.read().compute()
        u4 = (u + u1 + u2).compute()
        u5 = (u * 3).compute()
        assert np.allclose(u.array, u1.array)
        assert np.allclose(u5.array, u4.array)
