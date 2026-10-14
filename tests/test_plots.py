"""Chart values, highlighting and truncation must match the actual law."""

import numpy as np
import pytest

from lab.calculations import ValidationError, frozen_dist
from lab.catalog import REGISTRY, defaults
from lab.plots import MAX_CONTINUOUS_POINTS, MAX_DISCRETE_POINTS, category_chart, make_charts


UNIVARIATE = [key for key, value in REGISTRY.items() if value.kind != "multivariate"]


@pytest.mark.parametrize("distribution_id", UNIVARIATE)
def test_all_univariate_defaults_have_finite_charts(distribution_id):
    first, second, metadata = make_charts(distribution_id, defaults(distribution_id), {"kind": "le", "x": 2})
    for figure in (first, second):
        assert len(figure.data) >= 1
        for trace in figure.data:
            assert np.all(np.isfinite(np.asarray(trace.x, dtype=float)))
            assert np.all(np.isfinite(np.asarray(trace.y, dtype=float)))
    assert metadata["left"] <= metadata["right"]
    assert 0 <= metadata["excluded_probability"] <= 1
    assert np.all(np.diff(np.asarray(second.data[0].y)) >= -1e-12)


@pytest.mark.parametrize("event,selected", [
    ({"kind": "le", "x": 3}, [1, 2, 3]),
    ({"kind": "gt", "x": 3}, [4, 5, 6]),
    ({"kind": "gt", "x": 3.5}, [4, 5, 6]),
    ({"kind": "interval", "a": 2, "b": 4}, [2, 3, 4]),
    ({"kind": "interval", "a": 2.1, "b": 4.9}, [3, 4]),
    ({"kind": "eq", "x": 3}, [3]),
])
def test_discrete_highlights_match_inclusive_exclusive_event(event, selected):
    first, second, metadata = make_charts("randint", {"low": 1, "high": 6}, event)
    assert list(first.data[1].x) == selected
    assert list(second.data[1].x) == selected
    assert second.data[0].line.shape == "hv"
    assert list(second.data[0].x) == list(range(0, 8))
    assert list(second.data[0].y) == pytest.approx([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1])
    assert metadata["excluded_probability"] == 0


def test_large_discrete_range_has_no_subsampling_or_normalization():
    params = {"r": 100, "p": 0.001}
    first, _, metadata = make_charts("nbinom", params, {"kind": "le", "x": 100000}, coverage=0.9999)
    x = np.sort(np.concatenate([np.asarray(trace.x) for trace in first.data]))
    mass = np.concatenate([np.asarray(trace.y) for trace in first.data])
    assert len(x) == MAX_DISCRETE_POINTS
    assert np.all(np.diff(x) == 1)
    rv = frozen_dist("nbinom", params)
    assert metadata["excluded_probability"] == pytest.approx(rv.cdf(x[0] - 1) + rv.sf(x[-1]))
    assert mass.sum() == pytest.approx(1 - metadata["excluded_probability"], abs=1e-11)
    assert metadata["truncated"]
    assert any("sin saltear" in note for note in metadata["notes"])


@pytest.mark.parametrize("distribution_id,params,point", [
    ("geom", {"p": 1}, 1),
    ("nbinom", {"r": 3, "p": 1}, 0),
    ("poisson", {"mu": 0}, 0),
    ("binom", {"n": 100000, "p": 1}, 100000),
    ("hypergeom", {"M": 1, "K": 1, "n": 1}, 1),
])
def test_point_masses_use_effective_support(distribution_id, params, point):
    first, second, metadata = make_charts(distribution_id, params, {"kind": "eq", "x": point})
    assert list(first.data[1].x) == [point]
    assert list(first.data[1].y) == [1]
    assert metadata["left"] == metadata["right"] == point
    assert metadata["excluded_probability"] == 0
    assert second.data[0].line.shape == "hv"


@pytest.mark.parametrize("event,lower,upper", [
    ({"kind": "interval", "a": -0.12345, "b": 0.54321}, -0.12345, 0.54321),
    ({"kind": "gt", "x": -0.12345}, -0.12345, None),
    ({"kind": "le", "x": 0.54321}, None, 0.54321),
])
def test_continuous_shading_stops_at_exact_event_boundaries(event, lower, upper):
    first, _, metadata = make_charts("norm", {"mu": 0, "sigma": 1}, event)
    shaded = first.data[0]
    assert shaded.fill == "tozeroy"
    assert shaded.x[0] == pytest.approx(metadata["left"] if lower is None else lower)
    assert shaded.x[-1] == pytest.approx(metadata["right"] if upper is None else upper)
    assert len(shaded.x) <= MAX_CONTINUOUS_POINTS
    assert metadata["excluded_probability"] == pytest.approx(0.01)


@pytest.mark.parametrize("distribution_id,params", [
    ("beta", {"alpha": 0.2, "beta": 0.4}),
    ("gamma", {"alpha": 0.2, "theta": 1}),
    ("weibull_min", {"shape": 0.2, "scale": 1}),
])
def test_singular_density_endpoint_is_omitted_and_explained(distribution_id, params):
    first, _, metadata = make_charts(distribution_id, params, {"kind": "le", "x": 0.5})
    assert any("diverge" in note for note in metadata["notes"])
    for trace in first.data:
        assert np.all(np.isfinite(trace.y))
    rv = frozen_dist(distribution_id, params)
    assert metadata["excluded_probability"] == pytest.approx(rv.cdf(metadata["left"]) + rv.sf(metadata["right"]))


def test_cauchy_window_is_bounded_and_honest_even_without_moments():
    _, _, metadata = make_charts("cauchy", {"x0": 0, "gamma": 1}, {"kind": "le", "x": 0})
    assert metadata["left"] == pytest.approx(-30)
    assert metadata["right"] == pytest.approx(-metadata["left"])
    assert metadata["excluded_probability"] == pytest.approx(1 - (2 / np.pi) * np.arctan(30))
    assert any("cobertura real" in note for note in metadata["notes"])


def test_remote_finite_gev_endpoint_does_not_consume_plot_resolution():
    _, _, metadata = make_charts("genextreme", {"xi": 0.00001, "mu": 0, "sigma": 1}, {"kind": "le", "x": 0})
    assert metadata["left"] > -10
    assert metadata["excluded_probability"] == pytest.approx(0.01)


def test_multinomial_chart_explicitly_displays_category_probabilities():
    figure = category_chart(["Compra", "Consulta", "Sale"], [0.2, 0.3, 0.5], [2, 3, 5])
    assert list(figure.data[0].y) == [0.2, 0.3, 0.5]
    assert "un ensayo" in figure.layout.title.text
    assert "Probabilidad" in figure.layout.yaxis.title.text
    with pytest.raises(ValidationError):
        category_chart(["A", "B"], [0.6, 0.6], [6, 6])
    with pytest.raises(ValidationError):
        make_charts("multinomial", defaults("multinomial"), {"kind": "le", "x": 1})


@pytest.mark.parametrize("coverage", [0, 1, -0.1, np.inf, np.nan])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValidationError):
        make_charts("norm", defaults("norm"), {"kind": "le", "x": 0}, coverage)


def test_noninteger_discrete_equality_is_rejected():
    with pytest.raises(ValidationError):
        make_charts("randint", {"low": 1, "high": 6}, {"kind": "eq", "x": 3.2})


@pytest.mark.parametrize("distribution_id,params", [
    ("f", {"dfn": 0.1, "dfd": 0.1}),
    ("t", {"df": 0.1}),
    ("pareto", {"alpha": 0.1, "xmin": 1}),
    ("weibull_min", {"shape": 0.1, "scale": 10}),
])
def test_heavy_tails_have_practical_windows_and_adaptive_cdf(distribution_id, params):
    widths, excluded = [], []
    rv = frozen_dist(distribution_id, params)
    for coverage in (0.99, 0.9999):
        first, second, metadata = make_charts(distribution_id, params, {"kind": "le", "x": 2}, coverage)
        widths.append(metadata["right"] - metadata["left"])
        excluded.append(metadata["excluded_probability"])
        assert widths[-1] < 10000
        assert metadata["excluded_probability"] == pytest.approx(rv.cdf(metadata["left"]) + rv.sf(metadata["right"]))
        assert any("cobertura real" in note for note in metadata["notes"])
        cdf_values = np.asarray(second.data[0].y)
        assert np.count_nonzero((cdf_values > 0.01) & (cdf_values < 0.99)) >= 100
        assert len(second.data[0].x) <= MAX_CONTINUOUS_POINTS
        assert len(first.data[-1].x) <= MAX_CONTINUOUS_POINTS
    assert widths[1] > widths[0]
    assert excluded[1] <= excluded[0]


def test_event_next_to_singular_endpoint_has_shading_even_in_wide_window():
    first, _, metadata = make_charts("weibull_min", {"shape": 0.1, "scale": 10}, {"kind": "le", "x": 0.5}, 0.9999)
    assert first.data[0].fill == "tozeroy"
    assert first.data[0].x[-1] == 0.5
    assert len(first.data[0].x) >= 20
    assert first.data[0].x[0] > 0
    assert any("mucha probabilidad" in note for note in metadata["notes"])
