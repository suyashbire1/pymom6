import pymom6
import pytest
import os.path


@pytest.fixture(params=['e', 'u', 'v', 'wparam', 'RV'])
def var(request):
    return request.param


path = os.path.dirname(__file__) + '/data/'
fil1 = path + 'output__0001_11_019.nc'
fil2 = path + 'output__0001_12_009.nc'
fil_list = [fil1, fil2]


@pytest.mark.parametrize("fil", [fil2, fil_list])
def test_variable_from_dataset(var, fil):
    with pymom6.Dataset(fil) as pdset:
        assert isinstance(getattr(pdset, var), pymom6.MOM6Variable)
