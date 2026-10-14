"""Rare-event and numerical-limit checks independent of CDF subtraction."""

import math

import pytest

from lab.calculations import (
    ValidationError, format_moment, log_probability, moments,
    multinomial_joint, multinomial_logjoint, probability, quantile,
    validate_multinomial, validate_parameters,
)


def test_laplace_extreme_left_interval_matches_closed_form():
    event = {"kind": "interval", "a": -50, "b": -49}
    expected = 0.5 * math.exp(-49) * -math.expm1(-1)
    actual = probability("laplace", {"mu": 0, "b": 1}, event)
    assert actual == pytest.approx(expected, rel=1e-13, abs=0)


def test_gamma_shape_two_tail_matches_erlang_closed_form():
    event = {"kind": "gt", "x": 50}
    expected = math.exp(-50) * 51
    assert probability("gamma", {"alpha": 2, "theta": 1}, event) == pytest.approx(expected, rel=1e-13, abs=0)


def test_normal_underflow_remains_available_as_log_probability():
    params = {"mu": 0, "sigma": 1}
    event = {"kind": "gt", "x": 40}
    assert probability("norm", params, event) == 0
    assert math.isfinite(log_probability("norm", params, event))
    assert log_probability("norm", params, event) < -800


def test_exponential_underflow_interval_log_matches_closed_form():
    params = {"rate": 1}
    event = {"kind": "interval", "a": 1000, "b": 1001}
    assert probability("expon", params, event) == 0
    expected_log = -1000 + math.log(-math.expm1(-1))
    assert log_probability("expon", params, event) == pytest.approx(expected_log)


def test_unavailable_scipy_log_tail_is_reported_as_numerical_limit():
    # Gamma's logsf may itself underflow: that must not imply impossibility.
    with pytest.raises(ValidationError, match="precisión numérica"):
        probability("gamma", {"alpha": 2, "theta": 1}, {"kind": "gt", "x": 1000})


@pytest.mark.parametrize("id,params,event", [
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "interval", "a": 2, "b": 2}),
    ("expon", {"rate": 1}, {"kind": "le", "x": -1}),
    ("binom", {"n": 10, "p": 0}, {"kind": "eq", "x": 1}),
    ("geom", {"p": 1}, {"kind": "gt", "x": 1}),
])
def test_impossible_events_have_exact_zero_and_negative_infinite_log(id, params, event):
    assert probability(id, params, event) == 0
    assert log_probability(id, params, event) == -math.inf


def test_zero_probability_multinomial_category_is_not_repaired():
    assert multinomial_joint(3, [1, 0], [2, 1]) == 0
    assert multinomial_logjoint(3, [1, 0], [2, 1]) == -math.inf
    assert multinomial_joint(3, [1, 0], [3, 0]) == pytest.approx(1)


def test_multinomial_float_roundoff_accepted_but_conceptual_error_rejected():
    validate_multinomial(10, [0.1, 0.2, 0.7])
    with pytest.raises(ValidationError, match="sumar 1"):
        validate_multinomial(10, [0.1, 0.2, 0.700001])
    with pytest.raises(ValidationError, match="distintos"):
        validate_multinomial(10, [0.5, 0.5], names=["A", " a "])


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.5"])
def test_parameters_reject_nonreal_or_nonfinite_values(value):
    with pytest.raises(ValidationError):
        validate_parameters("bernoulli", {"p": value})


def test_mathematical_limits_are_separate_from_ui_limits():
    validate_parameters("poisson", {"mu": 10000})
    validate_parameters("norm", {"mu": -100000, "sigma": 10000})


def test_finite_moment_overflow_is_not_mathematical_infinity():
    result = moments("lognorm", {"mu_log": 20, "sigma_log": 40})
    assert result["mean"].status == "numerical_error"
    assert result["variance"].status == "numerical_error"
    for moment in result.values():
        assert "nan" not in format_moment(moment).lower()
        assert "inf" not in format_moment(moment).lower()


def test_geometric_degenerate_quantiles_use_actual_support_all_q():
    for q in (0, 0.01, 0.5, 0.99, 1):
        assert quantile("geom", {"p": 1}, q) == 1


def test_hypergeometric_single_population_variance_is_exact_zero():
    result = moments("hypergeom", {"M": 1, "K": 1, "n": 1})
    assert result["mean"].value == 1
    assert result["variance"].value == 0
    assert result["variance"].status == "finite"
