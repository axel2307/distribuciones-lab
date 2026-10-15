"""Numerical explanations must preserve the meaning and precision of events."""

import math
import re

import pytest

from lab.calculations import ValidationError
from lab.case_math import calculation_steps


def _latex(steps):
    return "\n".join(step["latex"] for step in steps)


@pytest.mark.parametrize("distribution_id,params,event,samples,expected", [
    ("t", {"df": 15}, {"kind": "gt", "x": 2},
     {"n": 16, "mean": 52, "std": 4, "null": 50},
     (r"\frac{4}{\sqrt{16}}\approx 1", r"\frac{52-(50)}{1}\approx 2", "0.031972504")),
    ("chi2", {"df": 9}, {"kind": "gt", "x": 14.4},
     {"n": 10, "variance": 1.6, "null_variance": 1},
     (r"(10-1)(1.6)\approx 14.4", r"\frac{14.4}{1}\approx 14.4", "0.10879096")),
    ("f", {"dfn": 9, "dfd": 11}, {"kind": "gt", "x": 2},
     {"n1": 10, "n2": 12, "variance1": 8, "variance2": 4},
     (r"\frac{8}{4}\approx 2", r"\nu_1=10-1=9,\quad\nu_2=12-1=11", "0.13856623")),
])
def test_measured_data_are_transformed_before_null_tail(distribution_id, params, event, samples, expected):
    steps = calculation_steps(distribution_id, params, event, samples)
    assert len(steps) == 4
    latex = _latex(steps)
    for text in expected:
        assert text in latex
    assert r"\mid H_0" in steps[-2]["latex"]
    assert "No es la probabilidad de que H₀ sea verdadera" in steps[-1]["explanation"]


def test_f_inverting_samples_requires_inverting_degrees_and_tail():
    steps = calculation_steps("f", {"dfn": 11, "dfd": 9}, {"kind": "le", "x": .5},
                              {"n1": 12, "n2": 10, "variance1": 4, "variance2": 8})
    assert r"\frac{4}{8}\approx 0.5" in steps[0]["latex"]
    assert "0.13856623" in steps[-1]["latex"]
    assert r"F_{R}(0.5)" in steps[-2]["latex"]


def test_multinomial_explicitly_combines_orders_and_their_probability():
    steps = calculation_steps("multinomial", {"n": 10, "probabilities": [.2, .3, .5]},
                              {"kind": "joint", "counts": [2, 3, 5]})
    assert r"\frac{10!}{2!\,3!\,5!}=2520" in steps[0]["latex"]
    assert r"(0.2)^{2}\,(0.3)^{3}\,(0.5)^{5}" in steps[1]["latex"]
    assert r"3.375\times10^{-5}" in steps[1]["latex"]
    assert r"2520\cdot 3.375\times10^{-5}\approx 0.08505" in steps[2]["latex"]
    assert r"\left(2,3,5\right)" in steps[2]["latex"]


def test_multinomial_zero_power_is_one_and_not_an_impossible_event():
    steps = calculation_steps("multinomial", {"n": 3, "probabilities": [0., 1.]},
                              {"kind": "joint", "counts": [0, 3]})
    assert r"(0)^{0}" in steps[1]["latex"]
    assert r"\approx 1=100" in steps[-1]["latex"]


def test_combinatorial_explanations_do_not_expand_large_factorials():
    steps = calculation_steps("binom", {"n": 10000, "p": .5}, {"kind": "eq", "x": 5000})
    assert len(_latex(steps)) < 1500
    assert r"\binom{10000}{5000}" in steps[1]["latex"]
    steps = calculation_steps("multinomial", {"n": 1000, "probabilities": [.2, .3, .5]},
                              {"kind": "joint", "counts": [200, 300, 500]})
    assert len(_latex(steps)) < 1500
    assert "1000!" in steps[0]["latex"]


def test_underflow_is_described_as_positive_instead_of_zero():
    steps = calculation_steps("norm", {"mu": 0., "sigma": 1.}, {"kind": "gt", "x": 40.})
    assert "positiva" in steps[-1]["explanation"]
    assert r"\log P(X>40)\approx -804.60844" in steps[-1]["latex"]
    assert r">0" in steps[-1]["latex"]
    assert r"\approx 0" not in steps[-1]["latex"]


def test_right_tail_interval_explains_survival_difference():
    steps = calculation_steps("norm", {"mu": 0., "sigma": 1.}, {"kind": "interval", "a": 9., "b": 10.})
    assert r"\operatorname{SF}_{X}(9)-\operatorname{SF}_{X}(10)" in steps[-2]["latex"]
    assert r"1.1285122\times10^{-19}" in steps[-1]["latex"]


def test_real_discrete_interval_uses_ceil_floor_and_empty_event():
    params = {"n": 20, "p": .3}
    steps = calculation_steps("binom", params, {"kind": "interval", "a": 2.1, "b": 4.9})
    assert r"\sum_{k=3}^{4}" in steps[1]["latex"]
    assert r"F_X(4)-F_X(2)" in steps[1]["latex"]
    steps = calculation_steps("binom", params, {"kind": "interval", "a": 2.1, "b": 2.9})
    assert "vacío" in steps[1]["explanation"]
    assert steps[-1]["latex"].endswith("=0")


@pytest.mark.parametrize("event,expected", [
    ({"kind": "gt", "x": 1e20}, "=0"),
    ({"kind": "le", "x": 1e20}, "=1"),
    ({"kind": "le", "x": -1e20}, "=0"),
    ({"kind": "interval", "a": -1e20, "b": 1e20}, "=1"),
])
def test_huge_bounded_events_remain_explainable(event, expected):
    steps = calculation_steps("binom", {"n": 20, "p": .3}, event)
    assert 2 <= len(steps) <= 4
    # Results are approximate when the evaluated floating-point probability is 1.
    assert steps[-1]["latex"].endswith("=0") if expected == "=0" else r"\approx 1=100" in steps[-1]["latex"]


def test_unresolved_huge_unbounded_event_preserves_precision_error():
    with pytest.raises(ValidationError, match="precisión"):
        calculation_steps("poisson", {"mu": 2}, {"kind": "gt", "x": 1e20})


def test_quantile_explains_discrete_jump_and_infinite_endpoint():
    steps = calculation_steps("binom", {"n": 2, "p": .5}, {"kind": "quantile", "q": .5})
    assert r"F_X(Q(q)-1)<q\le F_X(Q(q))" in steps[-1]["latex"]
    assert r"F_X(0)\approx 0.25,\quad q=0.5,\quad F_X(1)\approx 0.75" in steps[-1]["latex"]
    steps = calculation_steps("norm", {"mu": 0., "sigma": 1.}, {"kind": "quantile", "q": 1.})
    assert len(steps) == 2
    assert r"Q(1)=\infty" in steps[-1]["latex"]


def test_quantile_rounded_to_support_boundary_does_not_claim_cdf_is_target():
    steps = calculation_steps("beta", {"alpha": 1e-10, "beta": 1e-10}, {"kind": "quantile", "q": .1})
    assert "difiere" in steps[-1]["explanation"]
    assert "=0.1" not in steps[-1]["latex"]


@pytest.mark.parametrize("event", [{"kind": "quantile"}, {"kind": "invalid"}, None])
def test_invalid_event_has_user_facing_validation_error(event):
    with pytest.raises(ValidationError):
        calculation_steps("norm", {"mu": 0, "sigma": 1}, event)


@pytest.mark.parametrize("samples", [
    {"n": 15, "mean": 52, "std": 4, "null": 50},
    {"n": 16, "mean": 52, "std": 0, "null": 50},
    {"n": 1, "mean": 52, "std": 4, "null": 50},
    {"n": 16, "mean": math.inf, "std": 4, "null": 50},
    {"n": 16, "std": 4, "null": 50},
])
def test_incoherent_or_invalid_sample_summaries_are_rejected(samples):
    with pytest.raises(ValidationError):
        calculation_steps("t", {"df": 15}, {"kind": "gt", "x": 2}, samples)


def test_underflowed_standard_error_gives_validation_error():
    with pytest.raises(ValidationError, match="error estándar"):
        calculation_steps("t", {"df": 1e308}, {"kind": "gt", "x": 2},
                          {"n": 1e308, "mean": 52, "std": 1e-308, "null": 50})


def test_logarithm_is_not_taken_for_nonpositive_lognormal_bounds():
    steps = calculation_steps("lognorm", {"mu_log": 1., "sigma_log": .5},
                              {"kind": "interval", "a": -2., "b": 4.})
    assert len(steps) == 3
    assert r"\ln(-2)" not in _latex(steps)


def test_gev_zero_shape_uses_gumbel_limit_instead_of_dividing_by_zero():
    steps = calculation_steps("genextreme", {"mu": 2., "sigma": 3., "xi": 0.},
                              {"kind": "gt", "x": 5.})
    assert r"e^{-z-e^{-z}}" in steps[0]["latex"]
    assert "1/(0)" not in _latex(steps)


@pytest.mark.parametrize("distribution_id,params,event,substitution", [
    ("uniform", {"a": 0., "b": 10.}, {"kind": "le", "x": 3.},
     r"\frac{3-(0)}{10-(0)}\approx\frac{3}{10}"),
    ("expon", {"rate": .2}, {"kind": "gt", "x": 10.},
     r"e^{-((0.2)(10))}\approx e^{-2}"),
    ("weibull_min", {"shape": 2., "scale": 100.}, {"kind": "le", "x": 80.},
     r"1-e^{-(\left(\frac{80}{100}\right)^{2})}\approx 1-e^{-0.64}"),
    ("rayleigh", {"sigma": 2.}, {"kind": "gt", "x": 4.},
     r"e^{-(\frac{(4)^2}{2(2)^2})}\approx e^{-2}"),
])
def test_simple_families_show_readable_event_arithmetic(distribution_id, params, event, substitution):
    steps = calculation_steps(distribution_id, params, event)
    assert len(steps) == 3
    assert substitution in steps[1]["latex"]
    assert r"\int" not in _latex(steps)
    assert "f_X" not in _latex(steps)


SIMPLE_MODELS = [
    ("uniform", {"a": -3., "b": 7.}),
    ("expon", {"rate": .4}),
    ("weibull_min", {"shape": .7, "scale": 3.}),
    ("rayleigh", {"sigma": 1.7}),
]


def _closed_survival(distribution_id, params, x):
    """Independent elementary oracle, without the app engine or SciPy."""
    if distribution_id == "uniform":
        return min(1., max(0., (params["b"] - x) / (params["b"] - params["a"])))
    if x <= 0:
        return 1.
    if distribution_id == "expon":
        exponent = params["rate"] * x
    elif distribution_id == "weibull_min":
        exponent = (x / params["scale"]) ** params["shape"]
    else:
        exponent = x * x / (2 * params["sigma"] ** 2)
    return math.exp(-exponent)


def _displayed_probability(step):
    if step["latex"].endswith("=0"):
        return 0.
    match = re.search(r"\\approx ([\d.+-]+)(?:\\times10\^\{(-?\d+)\})?", step["latex"])
    assert match, step["latex"]
    return float(match.group(1)) * 10 ** int(match.group(2) or 0)


@pytest.mark.parametrize("distribution_id,params", SIMPLE_MODELS)
@pytest.mark.parametrize("event", [
    {"kind": "le", "x": 1.5},
    {"kind": "gt", "x": 2.3},
    {"kind": "interval", "a": -2., "b": 4.},
    {"kind": "interval", "a": 1.25, "b": 3.75},
    {"kind": "interval", "a": -5., "b": -2.},
])
def test_simple_event_results_match_independent_length_or_survival(distribution_id, params, event):
    survival = lambda x: _closed_survival(distribution_id, params, x)
    if event["kind"] == "le":
        expected = 1 - survival(event["x"])
    elif event["kind"] == "gt":
        expected = survival(event["x"])
    else:
        expected = survival(event["a"]) - survival(event["b"])
    steps = calculation_steps(distribution_id, params, event)
    assert _displayed_probability(steps[-1]) == pytest.approx(expected, rel=5e-8, abs=1e-14)


@pytest.mark.parametrize("distribution_id,params", SIMPLE_MODELS[1:])
@pytest.mark.parametrize("kind,expected", [("le", 0.), ("gt", 1.)])
def test_negative_time_never_enters_exponential_or_fractional_power(distribution_id, params, kind, expected):
    steps = calculation_steps(distribution_id, params, {"kind": kind, "x": -2.5})
    assert "soporte" in steps[1]["title"]
    assert "e^{" not in steps[1]["latex"]
    assert _displayed_probability(steps[-1]) == expected


@pytest.mark.parametrize("distribution_id,params", SIMPLE_MODELS)
def test_simple_quantile_uses_inverse_with_current_parameters(distribution_id, params):
    q = .75
    if distribution_id == "uniform":
        expected = params["a"] + q * (params["b"] - params["a"])
        expression = r"-3+(0.75)(7-(-3))"
    elif distribution_id == "expon":
        expected = -math.log1p(-q) / params["rate"]
        expression = r"\frac{-\ln(1-0.75)}{0.4}"
    elif distribution_id == "weibull_min":
        expected = params["scale"] * (-math.log1p(-q)) ** (1 / params["shape"])
        expression = r"3[-\ln(1-0.75)]^{1/(0.7)}"
    else:
        expected = params["sigma"] * math.sqrt(-2 * math.log1p(-q))
        expression = r"1.7\sqrt{-2\ln(1-0.75)}"
    steps = calculation_steps(distribution_id, params, {"kind": "quantile", "q": q})
    assert expression in steps[1]["latex"]
    assert _displayed_probability(steps[1]) == pytest.approx(expected, rel=5e-8)
    assert "f_X" not in _latex(steps)
    assert "0.75" in steps[-1]["latex"]


@pytest.mark.parametrize("distribution_id,params", SIMPLE_MODELS)
@pytest.mark.parametrize("q", [0., 1.])
def test_simple_quantile_support_endpoints_remain_explicit(distribution_id, params, q):
    steps = calculation_steps(distribution_id, params, {"kind": "quantile", "q": q})
    assert 2 <= len(steps) <= 3
    assert "extremo" in steps[1]["explanation"]
    if distribution_id != "uniform" and q == 1:
        assert r"\infty" in steps[1]["latex"]


def test_simple_tail_underflow_is_not_rounded_to_zero_in_derivation():
    steps = calculation_steps("expon", {"rate": 1.}, {"kind": "gt", "x": 1000.})
    assert r"e^{-1000}" in steps[1]["latex"]
    assert ">0" in steps[-1]["latex"]
    assert "positiva" in steps[-1]["explanation"]


def test_tiny_left_tail_preserves_complement_without_false_zero():
    steps = calculation_steps("expon", {"rate": 1e-10}, {"kind": "le", "x": 1e-10})
    assert r"1-e^{-" in steps[1]["latex"]
    assert _displayed_probability(steps[-1]) == pytest.approx(1e-20, rel=1e-8)


@pytest.mark.parametrize("distribution_id,params", SIMPLE_MODELS)
def test_close_interval_bounds_stay_distinct_in_written_calculation(distribution_id, params):
    lower, upper = 2., 2.000000001
    steps = calculation_steps(distribution_id, params, {"kind": "interval", "a": lower, "b": upper})
    assert "2.000000001" in steps[1]["latex"]
    assert _displayed_probability(steps[-1]) > 0


@pytest.mark.parametrize("distribution_id,params,lower,upper", [
    ("norm", {"mu": 50., "sigma": 2.}, 50., 50.00000001),
    ("lognorm", {"mu_log": 0., "sigma_log": 1.}, 2., 2.000000001),
    ("gamma", {"alpha": 2., "theta": 1.}, 2., 2.000000001),
    ("beta", {"alpha": 2., "beta": 5.}, .2, .2000000001),
    ("t", {"df": 15}, 2., 2.000000001),
    ("chi2", {"df": 9}, 2., 2.000000001),
    ("f", {"dfn": 9, "dfd": 11}, 2., 2.000000001),
    ("pareto", {"alpha": 3., "xmin": 1.}, 2., 2.000000001),
    ("cauchy", {"x0": 0., "gamma": 1.}, 2., 2.000000001),
    ("laplace", {"mu": 0., "b": 2.}, 2., 2.000000001),
    ("gumbel_r", {"mu": 0., "beta": 1.}, 2., 2.000000001),
    ("genextreme", {"mu": 0., "sigma": 1., "xi": .1}, 2., 2.000000001),
])
def test_general_area_and_transform_preserve_narrow_interval_bounds(distribution_id, params, lower, upper):
    """Displayed limits must identify the same positive-area event as the engine."""
    steps = calculation_steps(distribution_id, params, {"kind": "interval", "a": lower, "b": upper})
    lower_text = repr(lower).removesuffix(".0")
    upper_text = repr(upper).removesuffix(".0")
    area = steps[-2]["latex"]
    assert rf"\int_{{{lower_text}}}^{{{upper_text}}}" in area
    assert f"({upper_text})" in area  # CDF/SF is evaluated at the original upper bound.
    assert rf"P({lower_text}\le X\le {upper_text})" in steps[-1]["latex"]
    assert _displayed_probability(steps[-1]) > 0
    if len(steps) == 4:
        transformed = steps[1]["latex"]
        assert f"({lower_text})" in transformed
        assert f"({upper_text})" in transformed
        assert upper_text in transformed.split(r"\approx")[0] or upper_text in transformed
        # The two symbolic thresholds remain distinct, independently of how
        # their approximate transformed values are rounded for reading.
        label = "y" if distribution_id == "gamma" else "z"
        assert rf"{label}({lower_text})\le {label.upper()}\le {label}({upper_text})" in transformed


def test_normal_substitution_does_not_cancel_distinct_mean_and_threshold():
    steps = calculation_steps("norm", {"mu": 50.00000001, "sigma": 2.000000001},
                              {"kind": "gt", "x": 50.00000002})
    assert "50.00000001" in steps[0]["latex"]
    assert r"\frac{50.00000002-(50.00000001)}{2.000000001}" in steps[1]["latex"]
    assert "50.00000002" in steps[-2]["latex"]


def test_lognormal_substitution_keeps_log_parameters_and_bound_distinct():
    steps = calculation_steps("lognorm", {"mu_log": .9999999999, "sigma_log": .5000000001},
                              {"kind": "gt", "x": 2.000000001})
    assert "0.9999999999" in steps[0]["latex"]
    assert r"\frac{\ln(2.000000001)-(0.9999999999)}{0.5000000001}" in steps[1]["latex"]
    assert "2.000000001" in steps[-2]["latex"]


def test_t_numerator_keeps_distinct_measured_mean_and_null_reference():
    mean, null = 50.00000001, 50.
    statistic = (mean - null) / (4 / math.sqrt(16))
    steps = calculation_steps("t", {"df": 15}, {"kind": "gt", "x": statistic},
                              {"n": 16, "mean": mean, "std": 4, "null": null})
    assert r"\frac{50.00000001-(50)}{1}" in steps[1]["latex"]
    assert "50-(50)" not in steps[1]["latex"]


def test_probability_parameter_near_one_is_not_shown_as_one():
    steps = calculation_steps("bernoulli", {"p": .9999999999}, {"kind": "eq", "x": 0})
    assert "1-0.9999999999" in steps[0]["latex"]
    assert _displayed_probability(steps[-1]) > 0
