import pytest

from stats import average, median


def test_average_simple():
    assert average([2, 4, 6]) == 4


def test_average_single_value():
    assert average([5]) == 5


def test_average_empty_raises():
    with pytest.raises(ValueError):
        average([])


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5
