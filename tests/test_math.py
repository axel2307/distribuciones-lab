"""Numerical contracts and independent checks for the teaching calculator."""

import math

import numpy as np
import pytest
from scipy import stats

from lab.catalog import REGISTRY, defaults
from lab.calculations import (
    ValidationError,
    event_mask,
    format_moment,
    frozen_dist,
    log_probability,
    moments,
    multinomial_joint,
    probability,
    quantile,
    validate_event,
    validate_multinomial,
    validate_parameters,
)
from lab.state import get_model, new_model, replace_model


IDS = (
    "bernoulli", "binom", "poisson", "geom", "nbinom", "hypergeom",
    "multinomial", "randint", "norm", "uniform", "expon", "gamma", "beta",
    "lognorm", "t", "chi2", "f", "weibull_min", "pareto", "cauchy",
    "laplace", "rayleigh", "gumbel_r", "genextreme",
)
UNIVARIATE = tuple(key for key in IDS if key != "multinomial")
DISCRETE = ("bernoulli", "binom", "poisson", "geom", "nbinom", "hypergeom", "randint")
CONTINUOUS = tuple(key for key in UNIVARIATE if key not in DISCRETE)


def test_exactly_required_24_families():
    assert set(REGISTRY) == set(IDS)
    assert len(REGISTRY) == 24
    assert REGISTRY["multinomial"].kind == "multivariate"


@pytest.mark.parametrize("distribution_id", IDS)
def test_default_parameters_are_valid_and_fresh(distribution_id):
    params = defaults(distribution_id)
    validate_parameters(distribution_id, params)
    other = defaults(distribution_id)
    assert params == other
    assert params is not other
    params["__test_mutation__"] = 1
    assert "__test_mutation__" not in defaults(distribution_id)


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
def test_probabilities_bounded_and_cdf_monotone_for_defaults(distribution_id):
    params = defaults(distribution_id)
    rv = frozen_dist(distribution_id, params)
    lower, upper = rv.ppf([0.001, 0.999])
    assert np.isfinite(lower) and np.isfinite(upper)
    if lower == upper:
        lower, upper = lower - 1, upper + 1
    xs = np.linspace(lower, upper, 61)
    cdf = np.array([probability(distribution_id, params, {"kind": "le", "x": float(x)}) for x in xs])
    tail = np.array([probability(distribution_id, params, {"kind": "gt", "x": float(x)}) for x in xs])
    assert np.all(np.isfinite(cdf))
    assert np.all((cdf >= 0) & (cdf <= 1))
    assert np.all((tail >= 0) & (tail <= 1))
    assert np.all(np.diff(cdf) >= -1e-14)
    np.testing.assert_allclose(cdf + tail, 1, atol=1e-13)
    interval = probability(distribution_id, params, {"kind": "interval", "a": float(lower), "b": float(upper)})
    assert 0 <= interval <= 1


@pytest.mark.parametrize("distribution_id", CONTINUOUS)
@pytest.mark.parametrize("q", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_continuous_quantile_inverts_cdf(distribution_id, q):
    params = defaults(distribution_id)
    x = quantile(distribution_id, params, q)
    assert math.isfinite(x)
    assert probability(distribution_id, params, {"kind": "le", "x": x}) == pytest.approx(q, abs=3e-9)


@pytest.mark.parametrize("distribution_id", DISCRETE)
@pytest.mark.parametrize("q", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_discrete_quantile_has_correct_inequalities(distribution_id, q):
    params = defaults(distribution_id)
    k = quantile(distribution_id, params, q)
    assert math.isfinite(k) and k == math.floor(k)
    at_k = probability(distribution_id, params, {"kind": "le", "x": k})
    before_k = probability(distribution_id, params, {"kind": "le", "x": k - 1})
    assert at_k >= q - 1e-13
    assert before_k < q + 1e-13


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
def test_default_moments_are_classified_and_displayable(distribution_id):
    values = moments(distribution_id, defaults(distribution_id))
    assert set(values) == {"mean", "median", "variance", "std"}
    for moment in values.values():
        assert moment.status in {"finite", "undefined", "infinite"}
        label = format_moment(moment)
        assert isinstance(label, str) and label.strip()
        assert "nan" not in label.lower()
        assert label.lower() != "inf"
        if moment.status == "finite":
            assert moment.value is not None and math.isfinite(moment.value)
    if values["variance"].status == "finite":
        assert values["std"].status == "finite"
        assert values["std"].value ** 2 == pytest.approx(values["variance"].value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
@pytest.mark.parametrize("q", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_quantile_probability_rejected(distribution_id, q):
    with pytest.raises(ValidationError):
        quantile(distribution_id, defaults(distribution_id), q)


@pytest.mark.parametrize("event", [
    {"kind": "interval", "a": 3, "b": 2},
    {"kind": "le", "x": float("nan")},
    {"kind": "gt", "x": float("nan")},
    {"kind": "interval", "a": 0, "b": float("nan")},
    {"kind": "does_not_exist", "x": 0},
    {"kind": "quantile", "q": -0.1},
])
def test_invalid_events_rejected(event):
    with pytest.raises(ValidationError):
        validate_event(event)


def test_continuous_point_event_is_rejected():
    with pytest.raises(ValidationError):
        validate_event({"kind": "eq", "x": 1}, discrete=False)


def test_fractional_discrete_equality_is_rejected():
    with pytest.raises(ValidationError):
        validate_event({"kind": "eq", "x": 1.5}, discrete=True)


def test_event_mask_respects_inclusive_and_exclusive_limits():
    x = np.array([-1, 0, 1, 2, 3], dtype=float)
    np.testing.assert_array_equal(event_mask(x, {"kind": "le", "x": 1}), [True, True, True, False, False])
    np.testing.assert_array_equal(event_mask(x, {"kind": "gt", "x": 1}), [False, False, False, True, True])
    np.testing.assert_array_equal(event_mask(x, {"kind": "interval", "a": 0, "b": 2}), [False, True, True, True, False])
    np.testing.assert_array_equal(event_mask(x, {"kind": "eq", "x": 2}), [False, False, False, True, False])


@pytest.mark.parametrize("probabilities", [
    [0.2], [0.2] * 6, [0.3, 0.6], [0.3, 0.71], [-0.1, 1.1],
    [float("nan"), 1.0], [float("inf"), 0],
])
def test_multinomial_bad_probabilities_rejected(probabilities):
    with pytest.raises(ValidationError):
        validate_multinomial(10, probabilities)


@pytest.mark.parametrize("n", [-1, 2.5, float("nan"), float("inf")])
def test_multinomial_invalid_n_rejected(n):
    with pytest.raises(ValidationError):
        validate_multinomial(n, [0.3, 0.7])


@pytest.mark.parametrize("counts", [[3, 6], [-1, 11], [3.5, 6.5], [10], [1, 2, 7], [float("nan"), 10]])
def test_multinomial_bad_count_vectors_rejected(counts):
    with pytest.raises(ValidationError):
        multinomial_joint(10, [0.3, 0.7], counts)


def test_multinomial_joint_matches_independent_combinatorial_formula():
    n, probabilities, counts = 10, [0.2, 0.3, 0.5], [2, 3, 5]
    expected = math.factorial(n)
    for count, p in zip(counts, probabilities):
        expected *= p ** count / math.factorial(count)
    assert multinomial_joint(n, probabilities, counts) == pytest.approx(expected, rel=1e-13)


def test_multinomial_zero_probability_category_boundary():
    assert multinomial_joint(4, [0.0, 1.0], [0, 4]) == pytest.approx(1)
    assert multinomial_joint(4, [0.0, 1.0], [1, 3]) == pytest.approx(0)


def test_multinomial_requires_name_vector_to_match_categories():
    with pytest.raises(ValidationError):
        validate_multinomial(10, [0.3, 0.7], names=["Única"])


@pytest.mark.parametrize("distribution_id, params, event, expected", [
    ("bernoulli", {"p": 0.3}, {"kind": "eq", "x": 1}, 0.3),
    ("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5}, 0.24609375),
    ("randint", {"low": 1, "high": 6}, {"kind": "le", "x": 3}, 0.5),
    ("randint", {"low": 1, "high": 6}, {"kind": "eq", "x": 6}, 1 / 6),
    ("randint", {"low": 1, "high": 6}, {"kind": "interval", "a": 2, "b": 4}, 0.5),
    ("randint", {"low": 1, "high": 6}, {"kind": "interval", "a": 2.01, "b": 4.99}, 2 / 6),
    ("randint", {"low": 1, "high": 6}, {"kind": "interval", "a": 2.1, "b": 2.9}, 0),
    ("randint", {"low": 1, "high": 6}, {"kind": "gt", "x": 3}, 0.5),
    ("randint", {"low": 1, "high": 6}, {"kind": "gt", "x": 3.9}, 0.5),
    ("geom", {"p": 0.5}, {"kind": "eq", "x": 1}, 0.5),
    ("geom", {"p": 0.5}, {"kind": "gt", "x": 3}, 0.125),
    ("nbinom", {"r": 2, "p": 0.5}, {"kind": "eq", "x": 1}, 0.25),
    ("hypergeom", {"M": 52, "K": 4, "n": 5}, {"kind": "eq", "x": 1}, math.comb(4, 1) * math.comb(48, 4) / math.comb(52, 5)),
    ("expon", {"rate": 2}, {"kind": "le", "x": 1}, 1 - math.exp(-2)),
    ("pareto", {"alpha": 2, "xmin": 3}, {"kind": "gt", "x": 6}, 0.25),
])
def test_independent_exact_probability_results(distribution_id, params, event, expected):
    assert probability(distribution_id, params, event) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("distribution_id, params, event, expected", [
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "gt", "x": 10}, 7.619853024160526e-24),
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "interval", "a": 9, "b": 10}, 1.128512207423599e-19),
    ("expon", {"rate": 1}, {"kind": "interval", "a": 50, "b": 51}, 1.2192024317354474e-22),
    ("geom", {"p": 0.5}, {"kind": "interval", "a": 60, "b": 61}, 3 * 2 ** -61),
])
def test_rare_tail_events_avoid_cdf_cancellation(distribution_id, params, event, expected):
    actual = probability(distribution_id, params, event)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=3e-13, abs=0)
    assert log_probability(distribution_id, params, event) == pytest.approx(math.log(expected), abs=1e-11)


def test_underflow_retains_distinct_log_probability():
    params = {"rate": 1}
    event = {"kind": "gt", "x": 1000}
    assert probability("expon", params, event) == 0
    assert log_probability("expon", params, event) == pytest.approx(-1000)
    assert log_probability("expon", params, {"kind": "interval", "a": -3, "b": -2}) == -math.inf


@pytest.mark.parametrize("distribution_id, params, point", [
    ("bernoulli", {"p": 0}, 0), ("bernoulli", {"p": 1}, 1),
    ("binom", {"n": 0, "p": 0.3}, 0), ("binom", {"n": 10, "p": 0}, 0),
    ("binom", {"n": 10, "p": 1}, 10), ("poisson", {"mu": 0}, 0),
    ("geom", {"p": 1}, 1), ("nbinom", {"r": 3, "p": 1}, 0),
    ("hypergeom", {"M": 1, "K": 1, "n": 1}, 1),
    ("hypergeom", {"M": 20, "K": 0, "n": 5}, 0),
    ("randint", {"low": 4, "high": 4}, 4),
])
def test_degenerate_distributions_have_correct_moments_and_quantile_endpoints(distribution_id, params, point):
    value = moments(distribution_id, params)
    assert all(moment.status == "finite" for moment in value.values())
    assert value["mean"].value == point
    assert value["median"].value == point
    assert value["variance"].value == 0
    assert value["std"].value == 0
    for q in (0, 0.1, 0.5, 0.9, 1):
        assert quantile(distribution_id, params, q) == point
    assert probability(distribution_id, params, {"kind": "eq", "x": point}) == pytest.approx(1)


@pytest.mark.parametrize("distribution_id, updates, mean_status, variance_status", [
    ("cauchy", {}, "undefined", "undefined"),
    ("t", {"df": 0.5}, "undefined", "undefined"),
    ("t", {"df": 1}, "undefined", "undefined"),
    ("t", {"df": 1.5}, "finite", "infinite"),
    ("t", {"df": 2}, "finite", "infinite"),
    ("t", {"df": 2.1}, "finite", "finite"),
    ("f", {"dfd": 1}, "infinite", "undefined"),
    ("f", {"dfd": 2}, "infinite", "undefined"),
    ("f", {"dfd": 3}, "finite", "infinite"),
    ("f", {"dfd": 4}, "finite", "infinite"),
    ("f", {"dfd": 4.1}, "finite", "finite"),
    ("pareto", {"alpha": 0.5}, "infinite", "undefined"),
    ("pareto", {"alpha": 1}, "infinite", "undefined"),
    ("pareto", {"alpha": 1.5}, "finite", "infinite"),
    ("pareto", {"alpha": 2}, "finite", "infinite"),
    ("pareto", {"alpha": 2.1}, "finite", "finite"),
    ("genextreme", {"xi": -1}, "finite", "finite"),
    ("genextreme", {"xi": 0}, "finite", "finite"),
    ("genextreme", {"xi": 0.49}, "finite", "finite"),
    ("genextreme", {"xi": 0.5}, "finite", "infinite"),
    ("genextreme", {"xi": 0.75}, "finite", "infinite"),
    ("genextreme", {"xi": 1}, "infinite", "undefined"),
    ("genextreme", {"xi": 1.5}, "infinite", "undefined"),
])
def test_heavy_tail_moments_are_classified_mathematically(distribution_id, updates, mean_status, variance_status):
    params = {**defaults(distribution_id), **updates}
    value = moments(distribution_id, params)
    assert value["mean"].status == mean_status
    assert value["variance"].status == variance_status
    assert value["std"].status == variance_status
    assert value["median"].status == "finite"


def test_gev_shape_mapping_and_finite_mean_independently():
    params = {"xi": 0.75, "mu": 3, "sigma": 2}
    x = 5
    expected_cdf = math.exp(-(1 + params["xi"] * (x - params["mu"]) / params["sigma"]) ** (-1 / params["xi"]))
    assert probability("genextreme", params, {"kind": "le", "x": x}) == pytest.approx(expected_cdf)
    expected_mean = params["mu"] + params["sigma"] * (math.gamma(1 - params["xi"]) - 1) / params["xi"]
    assert moments("genextreme", params)["mean"].value == pytest.approx(expected_mean)


@pytest.mark.parametrize("distribution_id, updates", [
    ("bernoulli", {"p": -0.1}), ("binom", {"n": -1}), ("binom", {"n": 1.5}),
    ("poisson", {"mu": -1}), ("geom", {"p": 0}), ("nbinom", {"r": 0}),
    ("nbinom", {"r": 2.5}), ("hypergeom", {"M": 0}),
    ("hypergeom", {"M": 10, "K": 11}), ("hypergeom", {"M": 10, "n": 11}),
    ("randint", {"low": 7, "high": 6}), ("randint", {"high": 6.5}),
    ("norm", {"sigma": 0}), ("uniform", {"a": 1, "b": 1}),
    ("expon", {"rate": 0}), ("gamma", {"alpha": 0}), ("gamma", {"theta": -1}),
    ("beta", {"alpha": 0}), ("beta", {"beta": -1}), ("lognorm", {"sigma_log": 0}),
    ("t", {"df": 0}), ("chi2", {"df": -1}), ("f", {"dfn": 0}), ("f", {"dfd": 0}),
    ("weibull_min", {"shape": 0}), ("weibull_min", {"scale": 0}),
    ("pareto", {"alpha": 0}), ("pareto", {"xmin": 0}),
    ("cauchy", {"gamma": 0}), ("laplace", {"b": 0}), ("rayleigh", {"sigma": 0}),
    ("gumbel_r", {"beta": 0}), ("genextreme", {"sigma": 0}),
])
def test_mathematically_invalid_parameters_are_rejected(distribution_id, updates):
    with pytest.raises(ValidationError):
        validate_parameters(distribution_id, {**defaults(distribution_id), **updates})


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
def test_nonfinite_parameters_are_rejected(distribution_id):
    for parameter in REGISTRY[distribution_id].parameters:
        for value in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValidationError):
                validate_parameters(distribution_id, {**defaults(distribution_id), parameter.key: value})


def test_mathematical_validation_is_not_confused_with_practical_ui_limits():
    validate_parameters("binom", {"n": 501, "p": 0.5})
    validate_parameters("norm", {"mu": 20000, "sigma": 0.001})


# These expressions independently encode the equations displayed in the lesson,
# rather than calling scipy.stats again. Non-default, asymmetric parameters help
# expose swapped arguments and rate/scale or logarithm/moment confusion.
FORMULA_CASES = [
    ("bernoulli", {"p": 0.3}, 1),
    ("binom", {"n": 10, "p": 0.3}, 4),
    ("poisson", {"mu": 3.2}, 2),
    ("geom", {"p": 0.3}, 4),
    ("nbinom", {"r": 3, "p": 0.3}, 4),
    ("hypergeom", {"M": 52, "K": 4, "n": 5}, 1),
    ("randint", {"low": 2, "high": 7}, 7),
    ("norm", {"mu": 3, "sigma": 2}, 4),
    ("uniform", {"a": -2, "b": 3}, 0),
    ("expon", {"rate": 2}, 0.7),
    ("gamma", {"alpha": 2.5, "theta": 3}, 4),
    ("beta", {"alpha": 2.5, "beta": 0.7}, 0.3),
    ("lognorm", {"mu_log": 0.8, "sigma_log": 0.4}, 1.4),
    ("t", {"df": 3.5}, 0.6),
    ("chi2", {"df": 4.5}, 4),
    ("f", {"dfn": 3, "dfd": 8}, 1.2),
    ("weibull_min", {"shape": 2, "scale": 4}, 3),
    ("pareto", {"alpha": 2.5, "xmin": 3}, 6),
    ("cauchy", {"x0": -2, "gamma": 3}, 1),
    ("laplace", {"mu": 3, "b": 2}, 4),
    ("rayleigh", {"sigma": 3}, 2),
    ("gumbel_r", {"mu": 3, "beta": 2}, 4),
    ("genextreme", {"xi": 0.2, "mu": 3, "sigma": 2}, 4),
    ("genextreme", {"xi": -0.2, "mu": 3, "sigma": 2}, 4),
    ("genextreme", {"xi": 0, "mu": 3, "sigma": 2}, 4),
]


def _displayed_formula_value(distribution_id, p, x):
    if distribution_id == "bernoulli":
        return p["p"] if x == 1 else 1 - p["p"]
    if distribution_id == "binom":
        return math.comb(p["n"], x) * p["p"] ** x * (1 - p["p"]) ** (p["n"] - x)
    if distribution_id == "poisson":
        return math.exp(-p["mu"]) * p["mu"] ** x / math.factorial(x)
    if distribution_id == "geom":
        return (1 - p["p"]) ** (x - 1) * p["p"]
    if distribution_id == "nbinom":
        return math.comb(x + p["r"] - 1, p["r"] - 1) * p["p"] ** p["r"] * (1 - p["p"]) ** x
    if distribution_id == "hypergeom":
        return math.comb(p["K"], x) * math.comb(p["M"] - p["K"], p["n"] - x) / math.comb(p["M"], p["n"])
    if distribution_id == "randint":
        return 1 / (p["high"] - p["low"] + 1)
    if distribution_id == "norm":
        return math.exp(-((x - p["mu"]) / p["sigma"]) ** 2 / 2) / (p["sigma"] * math.sqrt(2 * math.pi))
    if distribution_id == "uniform":
        return 1 / (p["b"] - p["a"])
    if distribution_id == "expon":
        return p["rate"] * math.exp(-p["rate"] * x)
    if distribution_id == "gamma":
        return x ** (p["alpha"] - 1) * math.exp(-x / p["theta"]) / (math.gamma(p["alpha"]) * p["theta"] ** p["alpha"])
    if distribution_id == "beta":
        beta_function = math.gamma(p["alpha"]) * math.gamma(p["beta"]) / math.gamma(p["alpha"] + p["beta"])
        return x ** (p["alpha"] - 1) * (1 - x) ** (p["beta"] - 1) / beta_function
    if distribution_id == "lognorm":
        return math.exp(-((math.log(x) - p["mu_log"]) / p["sigma_log"]) ** 2 / 2) / (x * p["sigma_log"] * math.sqrt(2 * math.pi))
    if distribution_id == "t":
        nu = p["df"]
        return math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2)) * (1 + x * x / nu) ** (-(nu + 1) / 2)
    if distribution_id == "chi2":
        return x ** (p["df"] / 2 - 1) * math.exp(-x / 2) / (2 ** (p["df"] / 2) * math.gamma(p["df"] / 2))
    if distribution_id == "f":
        d1, d2 = p["dfn"], p["dfd"]
        beta_function = math.gamma(d1 / 2) * math.gamma(d2 / 2) / math.gamma((d1 + d2) / 2)
        return (d1 / d2) ** (d1 / 2) * x ** (d1 / 2 - 1) / (beta_function * (1 + d1 * x / d2) ** ((d1 + d2) / 2))
    if distribution_id == "weibull_min":
        return p["shape"] / p["scale"] * (x / p["scale"]) ** (p["shape"] - 1) * math.exp(-(x / p["scale"]) ** p["shape"])
    if distribution_id == "pareto":
        return p["alpha"] * p["xmin"] ** p["alpha"] / x ** (p["alpha"] + 1)
    if distribution_id == "cauchy":
        return 1 / (math.pi * p["gamma"] * (1 + ((x - p["x0"]) / p["gamma"]) ** 2))
    if distribution_id == "laplace":
        return math.exp(-abs(x - p["mu"]) / p["b"]) / (2 * p["b"])
    if distribution_id == "rayleigh":
        return x / p["sigma"] ** 2 * math.exp(-x * x / (2 * p["sigma"] ** 2))
    if distribution_id == "gumbel_r":
        z = (x - p["mu"]) / p["beta"]
        return math.exp(-z - math.exp(-z)) / p["beta"]
    if distribution_id == "genextreme":
        z = (x - p["mu"]) / p["sigma"]
        if p["xi"] == 0:
            return math.exp(-z - math.exp(-z)) / p["sigma"]
        t = 1 + p["xi"] * z
        return math.exp(-t ** (-1 / p["xi"])) * t ** (-1 / p["xi"] - 1) / p["sigma"]
    raise AssertionError(distribution_id)


@pytest.mark.parametrize("distribution_id, params, x", FORMULA_CASES)
def test_implemented_parametrization_matches_displayed_equation(distribution_id, params, x):
    rv = frozen_dist(distribution_id, params)
    actual = rv.pmf(x) if distribution_id in DISCRETE else rv.pdf(x)
    assert actual == pytest.approx(_displayed_formula_value(distribution_id, params, x), rel=2e-13, abs=1e-15)


def test_negative_gev_shape_matches_reversed_exponential_oracle():
    # xi=-1, mu=2, sigma=3 gives X=5-E where E~Exp(scale=3).
    params = {"xi": -1, "mu": 2, "sigma": 3}
    value = moments("genextreme", params)
    assert value["mean"].value == pytest.approx(2)
    assert value["variance"].value == pytest.approx(9)
    assert value["median"].value == pytest.approx(5 - 3 * math.log(2))
    assert quantile("genextreme", params, 1) == 5
    for q in (0.01, 0.1, 0.5, 0.9):
        assert quantile("genextreme", params, q) == pytest.approx(5 + 3 * math.log(q))
    for x in (-3, 0, 2, 5):
        assert probability("genextreme", params, {"kind": "le", "x": x}) == pytest.approx(math.exp(-(5 - x) / 3))
    assert probability("genextreme", params, {"kind": "gt", "x": 5}) == 0


def test_finite_moment_overflow_is_numerical_error_not_nonexistence():
    # All lognormal moments exist mathematically, even when floats overflow.
    value = moments("lognorm", {"mu_log": 709, "sigma_log": 3})
    assert value["mean"].status == "numerical_error"
    assert value["variance"].status == "numerical_error"
    assert value["std"].status == "numerical_error"
    assert value["median"].status == "finite"


def test_documented_quantile_endpoints_for_non_degenerate_supports():
    assert quantile("randint", {"low": 1, "high": 6}, 0) == 1
    assert quantile("randint", {"low": 1, "high": 6}, 1) == 6
    assert quantile("binom", {"n": 10, "p": 0.3}, 0) == 0
    assert quantile("binom", {"n": 10, "p": 0.3}, 1) == 10
    assert quantile("geom", {"p": 0.3}, 0) == 1
    assert quantile("geom", {"p": 0.3}, 1) == math.inf
    assert quantile("norm", {"mu": 0, "sigma": 1}, 0) == -math.inf
    assert quantile("norm", {"mu": 0, "sigma": 1}, 1) == math.inf


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
def test_each_required_parameter_is_validated(distribution_id):
    for parameter in REGISTRY[distribution_id].parameters:
        params = defaults(distribution_id)
        del params[parameter.key]
        with pytest.raises(ValidationError):
            validate_parameters(distribution_id, params)


@pytest.mark.parametrize("distribution_id", IDS)
def test_each_distribution_state_initializes_valid_parameters_and_events(distribution_id):
    model = new_model(distribution_id)
    validate_parameters(distribution_id, model["parameters"])
    if distribution_id == "multinomial":
        assert model["event"]["kind"] == "joint"
        validate_multinomial(model["parameters"]["n"], model["parameters"]["probabilities"], counts=model["event"]["counts"])
    else:
        validate_event(model["event"], REGISTRY[distribution_id].kind == "discrete")


def test_distribution_models_and_case_models_preserve_separate_state():
    models = {}
    normal = get_model(models, "explore:norm", "norm")
    geometric = get_model(models, "explore:geom", "geom")
    case = get_model(models, "case:norm", "norm", {"mu": 50, "sigma": 2}, {"kind": "gt", "x": 52})
    normal["parameters"]["mu"] = 5
    geometric["parameters"]["p"] = 0.6
    assert get_model(models, "explore:norm", "norm")["parameters"]["mu"] == 5
    assert get_model(models, "explore:geom", "geom")["parameters"]["p"] == 0.6
    assert case["parameters"]["mu"] == 50
    assert "p" not in normal["parameters"]
    assert "mu" not in geometric["parameters"]
    replacement = replace_model(models, "explore:norm", "norm")
    assert replacement["parameters"] == defaults("norm")
    assert geometric["parameters"]["p"] == 0.6
    assert case["parameters"]["mu"] == 50


def test_multinomial_defaults_and_loaded_examples_are_deep_copies():
    first, second = defaults("multinomial"), defaults("multinomial")
    first["probabilities"][0] = 0
    first["names"][0] = "Modificada"
    assert second == defaults("multinomial")
    parameters = {"n": 10, "probabilities": [0.2, 0.3, 0.5], "names": ["A", "B", "C"]}
    event = {"kind": "joint", "counts": [2, 3, 5]}
    model = new_model("multinomial", parameters, event)
    model["parameters"]["probabilities"][0] = 0
    model["parameters"]["names"][0] = "Otra"
    model["event"]["counts"][0] = 4
    assert parameters["probabilities"] == [0.2, 0.3, 0.5]
    assert parameters["names"] == ["A", "B", "C"]
    assert event["counts"] == [2, 3, 5]
