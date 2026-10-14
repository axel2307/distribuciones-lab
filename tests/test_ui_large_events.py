"""Large discrete event inputs must produce results or explained errors."""

from pathlib import Path
import tempfile

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=30).run()


def choose(app, distribution_id):
    app.radio(key="distribution_type").set_value("Discretas").run()
    app.selectbox(key="distribution::Discretas").select(distribution_id).run()
    assert not app.exception


def set_event(app, distribution_id, event):
    prefix = f"explore::{distribution_id}"
    app.selectbox(key=f"{prefix}::event_kind").select(event["kind"]).run()
    if event["kind"] == "interval":
        app.number_input(key=f"{prefix}::a").set_value(float(event["a"]))
        app.number_input(key=f"{prefix}::b").set_value(float(event["b"])).run()
    else:
        app.number_input(key=f"{prefix}::x").set_value(float(event["x"])).run()


@pytest.mark.parametrize("distribution_id", ["bernoulli", "binom", "hypergeom", "randint"])
@pytest.mark.parametrize("event, expected", [
    ({"kind": "le", "x": 1e20}, 1),
    ({"kind": "gt", "x": 1e20}, 0),
    ({"kind": "eq", "x": 1e20}, 0),
    ({"kind": "interval", "a": 1e20, "b": 2e20}, 0),
    ({"kind": "le", "x": -1e20}, 0),
    ({"kind": "gt", "x": -1e20}, 1),
])
def test_huge_event_outside_finite_support_is_exact_and_does_not_raise(app, distribution_id, event, expected):
    choose(app, distribution_id)
    set_event(app, distribution_id, event)
    assert not app.exception
    assert not app.tabs[0].error
    metrics = [metric.value for metric in app.tabs[0].metric if metric.label == "Probabilidad"]
    assert metrics == [str(expected)]
    assert len(app.tabs[0].get("plotly_chart")) == 2


@pytest.mark.parametrize("distribution_id", ["poisson", "geom", "nbinom"])
@pytest.mark.parametrize("event", [
    {"kind": "le", "x": 1e20},
    {"kind": "gt", "x": 1e20},
    {"kind": "eq", "x": 1e20},
    {"kind": "interval", "a": 1e20, "b": 2e20},
])
def test_huge_unbounded_discrete_event_is_explained_without_traceback_and_recovers(app, distribution_id, event):
    choose(app, distribution_id)
    set_event(app, distribution_id, event)
    assert not app.exception
    assert app.tabs[0].error
    explanation = " ".join(element.value for element in app.tabs[0].error).lower()
    assert "precisión" in explanation or "rango numérico" in explanation or "límite" in explanation
    assert "traceback" not in explanation and "overflowerror" not in explanation
    assert not any(metric.label == "Probabilidad" for metric in app.tabs[0].metric)
    set_event(app, distribution_id, {"kind": "le", "x": 3})
    assert not app.exception and not app.tabs[0].error
    assert len(app.tabs[0].get("plotly_chart")) == 2
