import pytest

from cart import apply_discount, cart_total, line_total


def test_line_total_uses_quantity():
    assert line_total(2.5, 4) == 10


def test_discount_ten_percent():
    assert apply_discount(200, 10) == 180


def test_discount_zero_keeps_total():
    assert apply_discount(50, 0) == 50


def test_discount_out_of_range():
    with pytest.raises(ValueError):
        apply_discount(100, 120)


def test_cart_total_without_discount():
    assert cart_total([(2, 3), (1, 4)]) == 10


def test_cart_total_with_discount():
    assert cart_total([(10, 2)], discount_percent=50) == 10
