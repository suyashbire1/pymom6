import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import pymom6
import os.path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def test_find_index_limits():
    dimension = np.array([1, 2, 3])
    lims = pymom6.find_index_limits(dimension, 1.5, 1.5)
    assert lims[0] == 0
    assert lims[1] == 1
    lims = pymom6.find_index_limits(dimension, 1.5, 1.5, method='higher')
    assert lims[0] == 1
    assert lims[1] == 2
    lims = pymom6.find_index_limits(dimension, 1.5, 1.5, method='nearest')
    assert lims[0] == 0
    assert lims[1] == 1

    lims = pymom6.find_index_limits(dimension, 1.6, 1.6)
    assert lims[0] == 0
    assert lims[1] == 1
    lims = pymom6.find_index_limits(dimension, 1.6, 1.6, method='higher')
    assert lims[0] == 1
    assert lims[1] == 2
    lims = pymom6.find_index_limits(dimension, 1.6, 1.6, method='nearest')
    assert lims[0] == 1
    assert lims[1] == 2
