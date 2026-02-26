import pytest
from kernelbench.score import *
import math

'''
Usage:
pytest test_score.py
'''

abs_tol = 0.0000001

def test_geometric_mean_speed_ratio_correct_only():

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    '''
    Geometric mean of the speed ratio for correct samples
    '''
    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 1.185631101, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 0.801816719, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 0, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_and_faster_only():

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    '''
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    '''

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 5, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 1.033333333, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 0, abs_tol=abs_tol)

def test_fastp():

    '''
    Rate of samples within a threshold p (1.0)
    '''

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.2, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.2, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0, abs_tol=abs_tol)

    '''
    Rate of samples within a threshold p (0.5)
    '''

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.4, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.6, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0, abs_tol=abs_tol)

