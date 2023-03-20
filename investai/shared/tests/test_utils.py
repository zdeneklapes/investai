# -*- coding: utf-8 -*-
from shared.utils import calculate_return_from_weights
import numpy as np


def test_calculate_return_from_weights():
    """This test that rewards are calculated correctly"""
    e1 = calculate_return_from_weights(t_now=np.array([2, 2, 2]),
                                       t_prev=np.array([1, 1, 1]),
                                       weights=np.array([0.5, 0.4, 0.1]))
    assert e1 == 1.0
