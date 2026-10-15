"""Exercise the real Streamlit script and its callbacks with AppTest.

These tests verify emitted LaTeX and Plotly elements, not browser pixels or
KaTeX/Plotly's client-side drawing. Visual checks must be performed separately.
"""

from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

from lab.catalog import REGISTRY, defaults
from lab.content import LESSONS


APP = Path(__file__).resolve().parents[1] / "app.py"
IDS = tuple(REGISTRY)


@pytest.fixture
def app(tmp_path, monkeypatch):
    # AppTest writes a temporary entrypoint. Reuse pytest's temporary directory,
    # which also makes the tests usable in restricted Windows workspaces.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=30).run()


def choose(app, distribution_id):
    kind = "Continuas" if REGISTRY[distribution_id].kind == "continuous" else "Discretas"
    if app.radio(key="distribution_type").value != kind:
        app.radio(key="distribution_type").set_value(kind).run()
    app.selectbox(key=f"distribution::{kind}").select(distribution_id).run()
    return app


def assert_no_exception(app):
    assert not app.exception, [element.value for element in app.exception]


def explorer_model(app, distribution_id):
    return app.session_state["models"][f"explore::{distribution_id}"]


def displayed_probability(app):
    values = [metric.value for metric in app.tabs[0].metric if metric.label == "Probabilidad"]
    assert values, "The explorer did not display a probability result"
    return float(values[0].replace(",", "."))


@pytest.mark.parametrize("distribution_id", IDS)
def test_every_distribution_has_working_default_screen(app, distribution_id):
    choose(app, distribution_id)
    assert_no_exception(app)
    assert not app.error, [element.value for element in app.error]
    assert len(IDS) == 24
    assert [tab.label for tab in app.tabs] == ["Explorar", "Entender", "Caso práctico"]
    assert any(REGISTRY[distribution_id].formula in element.value for element in app.latex)
    assert any(REGISTRY[distribution_id].support in element.value for element in app.latex)
    assert explorer_model(app, distribution_id)["parameters"] == defaults(distribution_id)
    assert len(app.tabs[0].get("plotly_chart")) >= 2
    assert len(app.tabs[2].subheader) == 5
    assert len(app.tabs[2].get("plotly_chart")) >= 2
    assert any(metric.label == "Probabilidad del evento" for metric in app.tabs[2].metric)
    assert not app.button(key=f"case::{distribution_id}::load").disabled


def test_parameter_change_updates_numeric_result_and_percentage(app):
    choose(app, "bernoulli")
    app.selectbox(key="explore::bernoulli::event_kind").select("eq")
    app.number_input(key="explore::bernoulli::x").set_value(1.0)
    app.number_input(key="explore::bernoulli::param::p").set_value(0.8).run()
    assert_no_exception(app)
    assert displayed_probability(app) == pytest.approx(0.8)
    assert any(metric.label == "Porcentaje" and metric.value == "80 %" for metric in app.tabs[0].metric)
    app.number_input(key="explore::bernoulli::param::p").set_value(0.25).run()
    assert_no_exception(app)
    assert displayed_probability(app) == pytest.approx(0.25)
    assert any("25 de cada 100" in element.value for element in app.tabs[0].markdown)


def test_parameter_and_event_state_survive_distribution_switches_without_leaking(app):
    choose(app, "nbinom")
    app.number_input(key="explore::nbinom::param::r").set_value(9)
    app.number_input(key="explore::nbinom::param::p").set_value(0.65)
    app.selectbox(key="explore::nbinom::event_kind").select("gt").run()
    app.number_input(key="explore::nbinom::x").set_value(7.0).run()
    saved = deepcopy(explorer_model(app, "nbinom"))

    choose(app, "geom")
    assert app.number_input(key="explore::geom::param::p").value == defaults("geom")["p"]
    app.number_input(key="explore::geom::param::p").set_value(0.2).run()
    choose(app, "norm")
    assert explorer_model(app, "norm")["parameters"] == defaults("norm")
    choose(app, "nbinom")
    assert_no_exception(app)
    assert explorer_model(app, "nbinom") == saved
    assert app.number_input(key="explore::nbinom::param::r").value == 9
    assert app.number_input(key="explore::nbinom::param::p").value == pytest.approx(0.65)
    assert app.selectbox(key="explore::nbinom::event_kind").value == "gt"
    assert app.number_input(key="explore::nbinom::x").value == 7.0
    choose(app, "geom")
    assert app.number_input(key="explore::geom::param::p").value == pytest.approx(0.2)


@pytest.mark.parametrize("distribution_id", IDS)
def test_load_every_example_copies_parameters_and_event_into_explorer(app, distribution_id):
    choose(app, distribution_id)
    case = LESSONS[distribution_id]["example"]
    app.button(key=f"case::{distribution_id}::load").click().run()
    assert_no_exception(app)
    assert not app.error, [element.value for element in app.error]
    assert explorer_model(app, distribution_id)["parameters"] == case["parameters"]
    assert explorer_model(app, distribution_id)["event"] == case["event"]
    assert app.success
    assert len(app.tabs[0].get("plotly_chart")) >= 2


def test_loading_edited_case_uses_current_variant_and_recomputes_result(app):
    choose(app, "binom")
    app.number_input(key="case::binom::param::n").set_value(10)
    app.number_input(key="case::binom::param::p").set_value(0.5)
    app.selectbox(key="case::binom::event_kind").select("eq").run()
    app.number_input(key="case::binom::x").set_value(5.0).run()
    app.button(key="case::binom::load").click().run()
    assert_no_exception(app)
    assert explorer_model(app, "binom")["parameters"] == {"n": 10, "p": 0.5}
    assert explorer_model(app, "binom")["event"] == {"kind": "eq", "x": 5.0}
    # Independent reference: C(10,5)/2**10.
    assert displayed_probability(app) == pytest.approx(0.24609375)


def test_reset_restores_active_defaults_and_leaves_other_distribution_state(app):
    choose(app, "geom")
    app.number_input(key="explore::geom::param::p").set_value(0.7).run()
    other = deepcopy(explorer_model(app, "geom"))
    choose(app, "binom")
    app.number_input(key="explore::binom::param::n").set_value(80)
    app.number_input(key="explore::binom::param::p").set_value(0.75).run()
    app.sidebar.button[0].click().run()
    assert_no_exception(app)
    assert explorer_model(app, "binom")["parameters"] == defaults("binom")
    assert explorer_model(app, "geom") == other
    assert app.number_input(key="explore::binom::param::n").value == defaults("binom")["n"]
    assert app.success


def test_invalid_interval_is_explained_without_exception_and_can_be_corrected(app):
    choose(app, "norm")
    app.selectbox(key="explore::norm::event_kind").select("interval").run()
    app.number_input(key="explore::norm::a").set_value(3.0)
    app.number_input(key="explore::norm::b").set_value(2.0).run()
    assert_no_exception(app)
    assert app.tabs[0].error
    assert not app.tabs[0].get("plotly_chart")
    app.number_input(key="explore::norm::b").set_value(4.0).run()
    assert_no_exception(app)
    assert not app.tabs[0].error
    assert len(app.tabs[0].get("plotly_chart")) == 2


def test_multinomial_invalid_probability_sum_is_not_repaired(app):
    choose(app, "multinomial")
    app.number_input(key="explore::multinomial::prob::0").set_value(0.7).run()
    assert_no_exception(app)
    assert app.tabs[0].error
    params = explorer_model(app, "multinomial")["parameters"]
    assert params["probabilities"] == [0.7, 0.3, 0.2]
    assert not app.tabs[0].get("plotly_chart")
    app.number_input(key="explore::multinomial::prob::0").set_value(0.5).run()
    assert_no_exception(app)
    assert not app.tabs[0].error
    assert len(app.tabs[0].get("plotly_chart")) == 3


def test_multinomial_invalid_counts_and_invalid_case_disable_loading(app):
    choose(app, "multinomial")
    app.number_input(key="explore::multinomial::count::0").set_value(0).run()
    assert_no_exception(app)
    assert app.tabs[0].error
    # The independently valid marginal remains available after a joint error.
    assert len(app.tabs[0].get("plotly_chart")) == 3
    app.number_input(key="case::multinomial::count::0").set_value(3).run()
    assert_no_exception(app)
    assert app.tabs[2].error
    assert app.button(key="case::multinomial::load").disabled


def test_multinomial_two_to_five_categories_names_and_marginal_selection(app):
    choose(app, "multinomial")
    app.number_input(key="explore::multinomial::category_count").set_value(5).run()
    assert_no_exception(app)
    params = explorer_model(app, "multinomial")["parameters"]
    assert len(params["probabilities"]) == len(params["names"]) == 5
    assert params["probabilities"][-2:] == [0.0, 0.0]
    app.text_input(key="explore::multinomial::name::4").set_value("Otros").run()
    app.selectbox(key="explore::multinomial::category").select(4).run()
    assert_no_exception(app)
    assert explorer_model(app, "multinomial")["category"] == 4
    assert explorer_model(app, "multinomial")["parameters"]["names"][4] == "Otros"
    assert any("Otros" in element.value and "marginal" in element.value for element in app.tabs[0].markdown)
    app.number_input(key="explore::multinomial::category_count").set_value(2)
    app.number_input(key="explore::multinomial::prob::1").set_value(0.5).run()
    assert_no_exception(app)
    params = explorer_model(app, "multinomial")["parameters"]
    assert len(params["probabilities"]) == len(params["names"]) == 2
    assert params["probabilities"] == [0.5, 0.5]
    assert explorer_model(app, "multinomial")["category"] == 0
    assert app.selectbox(key="explore::multinomial::category").value == 0


def test_exact_noninteger_discrete_result_is_rejected_and_tail_event_is_exclusive(app):
    choose(app, "randint")
    app.selectbox(key="explore::randint::event_kind").select("eq").run()
    app.number_input(key="explore::randint::x").set_value(2.5).run()
    assert_no_exception(app)
    assert app.tabs[0].error
    app.selectbox(key="explore::randint::event_kind").select("gt").run()
    app.number_input(key="explore::randint::x").set_value(3.0).run()
    assert_no_exception(app)
    assert not app.tabs[0].error
    assert displayed_probability(app) == pytest.approx(0.5)
    assert any(r"P(X>3)" in element.value for element in app.tabs[0].latex)
