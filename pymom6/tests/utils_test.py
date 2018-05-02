from context import pymom6
import os.path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def test_find_index_limits():
    dimension = np.arange(6)
    lims = pymom6._find_index_limits(dimension, 1.5, 1.5)
    assert lims[0] == 1
    assert lims[1] == 2
    lims = pymom6._find_index_limits(dimension, 1.5, 1.5, method='higher')
    assert lims[0] == 2
    assert lims[1] == 3
    lims = pymom6._find_index_limits(dimension, 1.5, 1.5, method='nearest')
    assert lims[0] == 1
    assert lims[1] == 2

    start = 2.8
    lims = pymom6._find_index_limits(dimension, start, start)
    assert lims[0] == 2
    assert lims[1] == 3
    lims = pymom6._find_index_limits(dimension, start, start, method='higher')
    assert lims[0] == 3
    assert lims[1] == 4
    lims = pymom6._find_index_limits(dimension, start, start, method='nearest')
    assert lims[0] == 3
    assert lims[1] == 4
    lims = pymom6._find_index_limits(dimension, 2.3, 2.3, method='nearest')
    assert lims[0] == 2
    assert lims[1] == 3
