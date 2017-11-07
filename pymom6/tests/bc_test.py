import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import pymom6
import numpy as np
import pytest
gv3 = pymom6.MOM6Variable


@pytest.fixture(params=range(4))
def axis(request):
    return request.param


@pytest.fixture(params=[0, -1])
def start_or_end(request):
    return request.param


@pytest.fixture(params=['zeros', 'mirror', 'circsymh', 'circsymq'])
def bc_type(request):
    return request.param


def test_create_halo(bc_type, axis, start_or_end):
    dummy_array = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    dummy_BC = pymom6.BoundaryCondition(bc_type, axis, start_or_end)
    dummy_BC.create_halo(dummy_array)
    if dummy_BC.bc_type == 'circsymq':
        take_index = 1 if start_or_end == 0 else -2
        compare_array = dummy_array.take([take_index], axis=axis)
    elif dummy_BC.bc_type == 'zeros':
        compare_array = np.zeros(
            dummy_array.take([start_or_end], axis=axis).shape)
    else:
        compare_array = dummy_array.take([start_or_end], axis=axis)
    assert np.all(dummy_BC.halo == compare_array)


def test_dummy_BC_append_halo(bc_type, axis, start_or_end):
    dummy_array = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    dummy_BC = pymom6.BoundaryCondition(bc_type, axis, start_or_end)
    dummy_BC.create_halo(dummy_array)
    array = dummy_BC.append_halo_to_array(dummy_array)
    if start_or_end == 0:
        if bc_type == 'circsymq':
            array1 = -dummy_array.take([1], axis=axis)
        elif bc_type == 'circsymh':
            array1 = -dummy_array.take([start_or_end], axis=axis)
        elif dummy_BC.bc_type == 'zeros':
            array1 = np.zeros(
                dummy_array.take([start_or_end], axis=axis).shape)
        else:
            array1 = dummy_array.take([start_or_end], axis=axis)
        array2 = dummy_array
    elif start_or_end == -1:
        array1 = dummy_array
        if bc_type == 'circsymq':
            array2 = -dummy_array.take([-2], axis=axis)
        elif bc_type == 'circsymh':
            array2 = -dummy_array.take([start_or_end], axis=axis)
        elif dummy_BC.bc_type == 'zeros':
            array2 = np.zeros(
                dummy_array.take([start_or_end], axis=axis).shape)
        else:
            array2 = dummy_array.take([start_or_end], axis=axis)
    dummy_array = np.concatenate((array1, array2), axis=axis)
    assert np.all(array == dummy_array)
