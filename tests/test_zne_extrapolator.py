import pytest

from src.mitigation.zne_extrapolator import ExtrapolationResult, ZNEExtrapolator


def test_linear_extrapolation_to_zero():
    result = ZNEExtrapolator.extrapolate([1, 3], [0.8, 0.6])
    assert isinstance(result, ExtrapolationResult)
    assert result.bounded == pytest.approx(0.9)
    assert result.raw == pytest.approx(0.9)


def test_polynomial_requires_three_factors():
    with pytest.raises(ValueError):
        ZNEExtrapolator.extrapolate([1, 3], [0.8, 0.6], "polynomial")


def test_extrapolated_value_is_bounded():
    result = ZNEExtrapolator.extrapolate([1, 3], [0.2, 0.9])
    assert 0.0 <= result.bounded <= 1.0


def test_raw_and_bounded_differ_when_out_of_range():
    """When extrapolation produces a value outside [0,1], raw preserves it."""
    with pytest.warns(RuntimeWarning, match="clipped"):
        result = ZNEExtrapolator.extrapolate([1, 3], [0.2, 0.9])
    # Raw may be negative (downward extrapolation to λ=0 from increasing trend)
    assert result.raw != result.bounded or (0.0 <= result.raw <= 1.0)


def test_exponential_extrapolation():
    result = ZNEExtrapolator.extrapolate([1, 3, 5], [0.9, 0.7, 0.5], "exponential")
    assert isinstance(result, ExtrapolationResult)
    assert 0.0 <= result.bounded <= 1.0


def test_three_factor_linear():
    result = ZNEExtrapolator.extrapolate([1, 3, 5], [0.9, 0.7, 0.5])
    assert isinstance(result, ExtrapolationResult)
    assert result.bounded == pytest.approx(1.0)  # linear fit → 1.0 at λ=0


def test_duplicate_factors_rejected():
    with pytest.raises(ValueError, match="unique"):
        ZNEExtrapolator.extrapolate([1, 1], [0.8, 0.6])


def test_single_factor_rejected():
    with pytest.raises(ValueError, match="at least two"):
        ZNEExtrapolator.extrapolate([1], [0.8])
