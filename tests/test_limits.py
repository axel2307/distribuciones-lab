"""Audit every UI boundary and the corners most prone to numeric failures."""

from itertools import product
import math

import numpy as np
import pytest

from lab.calculations import (
    ValidationError, log_probability, moments, probability,
    quantile, validate_parameters,
)
from lab.catalog import REGISTRY, defaults
from lab.plots import category_chart, make_charts
from lab.state import initial_event


def boundary_cases():
    cases = []
    for id, spec in REGISTRY.items():
        if spec.kind == "multivariate":
            continue
        cases.append(pytest.param(id, defaults(id), id=f"{id}-default"))
        for p in spec.parameters:
            for edge in ("minimum", "maximum"):
                params = defaults(id)
                params[p.key] = getattr(p, edge)
                cases.append(pytest.param(id, params, id=f"{id}-{p.key}-{edge}"))
        for edges in product(("minimum", "maximum"), repeat=len(spec.parameters)):
            params = {p.key: getattr(p, edge) for p, edge in zip(spec.parameters, edges)}
            cases.append(pytest.param(id, params, id=f"{id}-corner-{'-'.join(edges)}"))
    return cases


@pytest.mark.parametrize("id,params", boundary_cases())
def test_ui_boundary_parameters_calculate_and_render_or_reject_invalid_combination(id, params):
    try:
        validate_parameters(id, params)
    except ValidationError:
        # Independent controls can create e.g. a >= b or K > M. The UI must
        # reject these combinations, rather than let SciPy generate NaNs.
        with pytest.raises(ValidationError):
            probability(id, params, {"kind": "le", "x": 0})
        return
    event = initial_event(id, params)
    p = probability(id, params, event)
    assert 0 <= p <= 1
    assert math.isfinite(log_probability(id, params, event))
    assert all(m.status != "numerical_error" for m in moments(id, params).values())
    for coverage in (0.99, 0.9999):
        density, cdf, metadata = make_charts(id, params, event, coverage)
        assert 0 <= metadata["excluded_probability"] <= 1
        assert metadata["left"] <= metadata["right"]
        for figure in (density, cdf):
            for trace in figure.data:
                assert np.all(np.isfinite(np.asarray(trace.x, dtype=float)))
                assert np.all(np.isfinite(np.asarray(trace.y, dtype=float)))
        accumulated = np.asarray(cdf.data[0].y, dtype=float)
        assert np.all(np.diff(accumulated) >= -1e-12)
        if REGISTRY[id].kind == "discrete":
            assert sum(len(trace.x) for trace in density.data) <= 1200
        else:
            assert max(len(trace.x) for trace in density.data) <= 1000


@pytest.mark.parametrize("n", [0, 500])
@pytest.mark.parametrize("ps", [[0, 1], [1, 0], [.5, .5], [.1, .2, .3, .4, 0]])
def test_multinomial_zero_categories_and_trial_limits(n, ps):
    names = [f"Categoría {i + 1}" for i in range(len(ps))]
    params = {"n": n, "probabilities": ps, "names": names}
    validate_parameters("multinomial", params)
    event = initial_event("multinomial", params)
    assert sum(event["counts"]) == n
    assert 0 <= probability("multinomial", params, event) <= 1
    figure = category_chart(names, ps, [n * p for p in ps])
    assert np.all(np.isfinite(figure.data[0].y))


def test_beta_quantile_rounded_to_endpoint_retains_honest_achieved_probability():
    # The UI warns when an interior requested quantile rounds to a support
    # endpoint in double precision. Returning the achieved CDF prevents the
    # result from falsely claiming exactly q; plotting finite support is safe.
    params = {"alpha": .1, "beta": .1}
    q = .999
    x = quantile("beta", params, q)
    assert x == 1
    achieved = probability("beta", params, {"kind": "le", "x": x})
    assert achieved == 1
    assert abs(achieved - q) > 1e-8
    density, cdf, metadata = make_charts("beta", params, {"kind": "le", "x": x})
    assert metadata["excluded_probability"] == 0
    assert all(np.all(np.isfinite(trace.y)) for trace in density.data)
