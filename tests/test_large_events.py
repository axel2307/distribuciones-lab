"""Huge finite event bounds must resolve safely, without SciPy object errors."""

import math

import pytest

from lab.calculations import ValidationError, log_probability, probability


FINITE_LAWS = [
    ("bernoulli", {"p": .3}),
    ("binom", {"n": 20, "p": .2}),
    ("hypergeom", {"M": 52, "K": 4, "n": 5}),
    ("randint", {"low": 1, "high": 6}),
]
UNBOUNDED_LAWS = [
    ("poisson", {"mu": 5}),
    ("geom", {"p": .25}),
    ("nbinom", {"r": 3, "p": .4}),
]


@pytest.mark.parametrize("id,params", FINITE_LAWS)
@pytest.mark.parametrize("huge", [1e20, 1e100])
def test_huge_bounds_beyond_finite_support_resolve_exactly(id, params, huge):
    for event, expected in [
        ({"kind": "le", "x": huge}, 1),
        ({"kind": "gt", "x": huge}, 0),
        ({"kind": "eq", "x": huge}, 0),
        ({"kind": "le", "x": -huge}, 0),
        ({"kind": "gt", "x": -huge}, 1),
        ({"kind": "eq", "x": -huge}, 0),
        ({"kind": "interval", "a": -huge, "b": huge}, 1),
        ({"kind": "interval", "a": huge, "b": 2 * huge}, 0),
        ({"kind": "interval", "a": -2 * huge, "b": -huge}, 0),
    ]:
        assert probability(id, params, event) == expected
        assert log_probability(id, params, event) == (0 if expected == 1 else -math.inf)


@pytest.mark.parametrize("id,params", UNBOUNDED_LAWS)
def test_huge_negative_bounds_outside_nonnegative_support_are_exact(id, params):
    for event, expected in [
        ({"kind": "le", "x": -1e20}, 0),
        ({"kind": "gt", "x": -1e20}, 1),
        ({"kind": "eq", "x": -1e20}, 0),
        ({"kind": "interval", "a": -1e20, "b": -1e19}, 0),
    ]:
        assert probability(id, params, event) == expected
        assert log_probability(id, params, event) == (0 if expected == 1 else -math.inf)


@pytest.mark.parametrize("id,params", UNBOUNDED_LAWS)
@pytest.mark.parametrize("event", [
    {"kind": "le", "x": 1e20},
    {"kind": "gt", "x": 1e20},
    {"kind": "eq", "x": 1e20},
    {"kind": "interval", "a": 0, "b": 1e20},
    {"kind": "interval", "a": 1e20, "b": 2e20},
])
def test_unresolved_enormous_lattice_values_report_precision(id, params, event):
    for calculate in (probability, log_probability):
        with pytest.raises(ValidationError, match="precisión.*enteros consecutivos"):
            calculate(id, params, event)


def test_huge_interval_bound_reduces_to_finite_support_without_changing_event():
    params = {"n": 20, "p": .2}
    left_event = {"kind": "interval", "a": -1e20, "b": 4}
    right_event = {"kind": "interval", "a": 4, "b": 1e20}
    left_expected = sum(math.comb(20, k) * .2**k * .8**(20-k) for k in range(5))
    right_expected = sum(math.comb(20, k) * .2**k * .8**(20-k) for k in range(4, 21))
    assert probability("binom", params, left_event) == pytest.approx(left_expected)
    assert probability("binom", params, right_event) == pytest.approx(right_expected)
    assert left_event["a"] == -1e20
    assert right_event["b"] == 1e20


@pytest.mark.parametrize("id,params,point", [
    ("poisson", {"mu": 0}, 0),
    ("geom", {"p": 1}, 1),
    ("nbinom", {"r": 10, "p": 1}, 0),
])
def test_point_masses_resolve_huge_positive_bounds_exactly(id, params, point):
    assert probability(id, params, {"kind": "le", "x": 1e20}) == 1
    assert probability(id, params, {"kind": "gt", "x": 1e20}) == 0
    assert probability(id, params, {"kind": "eq", "x": 1e20}) == 0
    assert probability(id, params, {"kind": "interval", "a": -1e20, "b": point}) == 1


def test_huge_negative_interval_lower_bound_preserves_geometric_tail():
    params = {"p": .5}
    assert probability("geom", params, {"kind": "interval", "a": -1e20, "b": 3}) == pytest.approx(.875)
    tail = {"kind": "interval", "a": 60, "b": 61}
    assert probability("geom", params, tail) == pytest.approx(3 * 2**-61, rel=1e-13, abs=0)
    assert math.isfinite(log_probability("geom", params, tail))
