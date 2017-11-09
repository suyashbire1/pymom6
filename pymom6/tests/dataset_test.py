import pymom6
import pytest
import os.path
import numpy as np


@pytest.fixture(scope='function', params=['e', 'u', 'v', 'wparam', 'RV'])
def var(request):
    return request.param


path = os.path.dirname(__file__) + '/data/'
fil1 = path + 'output__0001_11_019.nc'
fil2 = path + 'output__0001_12_009.nc'
fil_list = [fil1, fil2]


@pytest.mark.parametrize("fil", [fil2, fil_list])
def test_variable_from_dataset(var, fil):
    with pymom6.Dataset2(fil) as pdset:
        momvar = getattr(pdset, var)()
    assert isinstance(momvar, pymom6.MOM6Variable)


def test_variable_same_from_dataset(var):
    path = os.path.dirname(__file__) + '/data/'
    fil2 = path + 'output__0001_12_009.nc'
    with pymom6.Dataset2(fil2) as pdset:
        momvar = getattr(pdset, var)().read().compute() * 2
        momvar1 = getattr(pdset, var)().read().compute() * 3
    assert not np.allclose(momvar.array, momvar1.array)
