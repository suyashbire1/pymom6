import pymom6
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
