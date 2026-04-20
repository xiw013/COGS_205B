import math
import pytest
import numpy as np
from signal_detection import SignalDetection


# ----------------------------
# Constructor tests
# ----------------------------

def test_init_valid_case1():
    s = SignalDetection(40, 10, 5, 45)
    assert s.hits == 40
    assert s.misses == 10
    assert s.false_alarms == 5
    assert s.correct_rejections == 45


def test_init_rejects_boolean_case2():
    with pytest.raises(ValueError):
        SignalDetection(False, 1, 0, 2)


def test_init_rejects_negative_case3():
    with pytest.raises((TypeError, ValueError)):
        SignalDetection(1, 4, 5, -1)


def test_init_rejects_float_case4():
    with pytest.raises((TypeError, ValueError)):
        SignalDetection(0.05, 1, 1, 5)


def test_init_rejects_infinity_case5():
    with pytest.raises((TypeError, ValueError)):
        SignalDetection(np.inf, 0, 0, 1)


# ----------------------------
# hit_rate tests
# ----------------------------

def test_hit_rate_case1():
    s = SignalDetection(40, 10, 5, 45)
    assert s.hit_rate() == 40 / 50


def test_hit_rate_all_hits_case2():
    s = SignalDetection(10, 0, 3, 7)
    assert s.hit_rate() == 1.0


def test_hit_rate_no_hits_case3():
    s = SignalDetection(0, 10, 3, 7)
    assert s.hit_rate() == 0.0

def test_hit_rate_all_zero():
    with pytest.raises(ValueError):
        s = SignalDetection(0, 0, 0, 0)
        s.hit_rate() == 0.0


# ----------------------------
# false_alarm_rate tests
# ----------------------------

def test_false_alarm_rate_case1():
    s = SignalDetection(40, 10, 5, 45)
    assert s.false_alarm_rate() == 5 / 50


def test_false_alarm_rate_all_false_alarms_case2():
    s = SignalDetection(10, 0, 10, 0)
    assert s.false_alarm_rate() == 1.0


def test_false_alarm_rate_no_false_alarms_case3():
    s = SignalDetection(10, 0, 0, 10)
    assert s.false_alarm_rate() == 0.0

def test_false_alarm_all_zero():
    with pytest.raises(ValueError):
        s = SignalDetection(0, 0, 0, 0)
        s.false_alarm_rate() == 0.0

# ----------------------------
# d_prime tests
# ----------------------------

def test_d_prime_case1():
    s = SignalDetection(40, 10, 5, 45)
    d = s.d_prime()
    assert isinstance(d, float)
    assert d > 0


def test_d_prime_perfect_detector_case2():
    with pytest.raises(ValueError):
        s = SignalDetection(50, 0, 0, 50)
        d = s.d_prime()


def test_d_prime_chance_case3():
    s = SignalDetection(25, 25, 25, 25)
    d = s.d_prime()
    assert pytest.approx(d, abs=1e-8) == 0


# ----------------------------
# criterion tests
# ----------------------------

def test_criterion_case1():
    s = SignalDetection(40, 10, 5, 45)
    c = s.criterion()
    assert isinstance(c, float)


def test_criterion_chance_case2():
    s = SignalDetection(25, 25, 25, 25)
    c = s.criterion()
    assert pytest.approx(c, abs=1e-8) == 0


# ----------------------------
# operator tests
# ----------------------------

def test_add_case1():
    s1 = SignalDetection(40, 10, 5, 45)
    s2 = SignalDetection(10, 5, 2, 8)
    s3 = s1 + s2

    assert s3.hits == 50
    assert s3.misses == 15
    assert s3.false_alarms == 7
    assert s3.correct_rejections == 53

def test_add_incomptatible_case1():
    with pytest.raises(TypeError):
        s1 = SignalDetection(40, 10, 5, 45)
        s2 = 10
        s3 = s1 + s2

def test_multiply_case2():
    s = SignalDetection(10, 5, 2, 8)
    s2 = s * 2

    assert s2.hits == 20
    assert s2.misses == 10
    assert s2.false_alarms == 4
    assert s2.correct_rejections == 16


# ----------------------------
# plotting tests
# ----------------------------

def test_plot_sdt_case1():
    s = SignalDetection(40, 10, 5, 45)
    fig, ax = s.plot_sdt()

    assert fig is not None
    assert ax is not None


def test_plot_roc_case2():
    sdt_list = [
        SignalDetection(50, 10, 5, 45),
        SignalDetection(45, 15, 10, 40),
        SignalDetection(40, 20, 15, 35),
    ]
    fig, ax = SignalDetection.plot_roc(sdt_list)

    assert fig is not None
    assert ax is not None