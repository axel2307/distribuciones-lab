"""Independent regressions for the educational, live worked-case journey.

The examples below use closed forms or explicit SciPy reference laws; they do
not obtain their expected answers from the application's distribution factory.
Browser rendering is checked separately from these content/Streamlit checks.
"""

from copy import deepcopy
import math
import re
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from lab.calculations import ValidationError, probability
from lab.catalog import REGISTRY
from lab.content import LESSONS
from lab.plots import make_charts
from lab.case_math import calculation_steps
from lab.case_studies import case_details
from lab.case_inputs import sample_model


IDS = tuple(REGISTRY)
DETAIL_FIELDS = (
    "title", "context", "question", "variable", "outcomes", "unit",
    "rationale", "parameter_rows", "event_words", "interpretation",
    "limitation", "experiment",
)
SAMPLES = {
    "t": {"n": 16, "mean": 52.0, "std": 4.0, "null": 50.0},
    "chi2": {"n": 10, "variance": 1.6, "null_variance": 1.0},
    "f": {"n1": 10, "n2": 12, "variance1": 8.0, "variance2": 4.0},
}


def flattened(value):
    if isinstance(value, dict):
        return " ".join(flattened(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return " ".join(flattened(item) for item in value)
    return str(value)


def text_number(value):
    return f"{value:.8g}".replace(".", ",")


@pytest.mark.parametrize("distribution_id", IDS)
def test_every_live_case_has_a_complete_guided_journey(distribution_id):
    example = LESSONS[distribution_id]["example"]
    params, event = deepcopy(example["parameters"]), deepcopy(example["event"])
    samples = deepcopy(SAMPLES.get(distribution_id))
    details = case_details(distribution_id, params, event, samples=samples)
    assert len(IDS) == 24
    assert set(DETAIL_FIELDS) <= details.keys()
    for field in DETAIL_FIELDS:
        if field == "unit":
            # Indicators, scores and standardized statistics are dimensionless.
            assert isinstance(details[field], str)
        else:
            assert details[field], (distribution_id, field)
    assert "ilustrativ" in details["context"].lower()
    assert "?" in details["question"]
    assert "próximamente" not in flattened(details).lower()
    assert params == example["parameters"] and event == example["event"]

    steps = calculation_steps(distribution_id, params, event, samples=samples)
    assert 2 <= len(steps) <= 4
    for step in steps:
        assert set(("title", "explanation", "latex")) <= step.keys()
        assert all(step[key].strip() for key in ("title", "explanation", "latex"))
    # Every journey ends with an actual numerical substitution/result.
    assert any(character.isdigit() for character in steps[-1]["latex"])
    assert not re.search(r"\bnan\b", flattened(steps).lower())


# These variants deliberately differ from all original examples. The expected
# probabilities come from short independently checkable calculations.
VARIANTS = [
    ("bernoulli", {"p": 0.7}, {"kind": "eq", "x": 1}, 0.7),
    ("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5}, 252 / 1024),
    ("poisson", {"mu": 2}, {"kind": "le", "x": 2}, 5 * math.exp(-2)),
    ("geom", {"p": 0.5}, {"kind": "le", "x": 3}, 1 - 0.5 ** 3),
    ("nbinom", {"r": 2, "p": 0.5}, {"kind": "eq", "x": 1}, 0.25),
    ("hypergeom", {"M": 20, "K": 5, "n": 4}, {"kind": "gt", "x": 0}, 1 - math.comb(15, 4) / math.comb(20, 4)),
    ("multinomial", {"n": 8, "probabilities": [0.25, 0.75], "names": ["Correo nuevo", "Sucursal nueva"]}, {"kind": "joint", "counts": [2, 6]}, 28 * 0.25 ** 2 * 0.75 ** 6),
    ("randint", {"low": 2, "high": 9}, {"kind": "interval", "a": 3, "b": 5}, 3 / 8),
    ("norm", {"mu": 3, "sigma": 2}, {"kind": "le", "x": 3}, 0.5),
    ("uniform", {"a": -1, "b": 3}, {"kind": "interval", "a": 0, "b": 2}, 0.5),
    ("expon", {"rate": 2}, {"kind": "gt", "x": 1}, math.exp(-2)),
    ("gamma", {"alpha": 2, "theta": 3}, {"kind": "le", "x": 3}, 1 - 2 * math.exp(-1)),
    ("beta", {"alpha": 2, "beta": 2}, {"kind": "le", "x": 0.5}, 0.5),
    ("lognorm", {"mu_log": 1, "sigma_log": 0.5}, {"kind": "le", "x": math.e}, 0.5),
    ("t", {"df": 7}, {"kind": "le", "x": 0}, 0.5),
    ("chi2", {"df": 2}, {"kind": "le", "x": 2}, 1 - math.exp(-1)),
    ("f", {"dfn": 6, "dfd": 6}, {"kind": "le", "x": 1}, 0.5),
    ("weibull_min", {"shape": 2, "scale": 3}, {"kind": "le", "x": 3}, 1 - math.exp(-1)),
    ("pareto", {"alpha": 2, "xmin": 4}, {"kind": "gt", "x": 8}, 0.25),
    ("cauchy", {"x0": -1, "gamma": 3}, {"kind": "le", "x": -1}, 0.5),
    ("laplace", {"mu": 2, "b": 3}, {"kind": "le", "x": 2}, 0.5),
    ("rayleigh", {"sigma": 3}, {"kind": "le", "x": 3}, 1 - math.exp(-0.5)),
    ("gumbel_r", {"mu": 2, "beta": 3}, {"kind": "le", "x": 2}, math.exp(-1)),
    ("genextreme", {"xi": -1, "mu": 2, "sigma": 3}, {"kind": "le", "x": 2}, math.exp(-1)),
]


@pytest.mark.parametrize("distribution_id,params,event,expected", VARIANTS)
def test_edited_journeys_keep_a_contextual_question_and_independent_answer(distribution_id, params, event, expected):
    details = case_details(distribution_id, params, event)
    assert details["context"] and details["question"] and details["event_words"]
    assert details["context"] != LESSONS[distribution_id]["example"]["context"]
    assert probability(distribution_id, params, event) == pytest.approx(expected, abs=1e-14)
    assert text_number(expected) in details["interpretation"]
    steps = calculation_steps(distribution_id, params, event)
    assert 2 <= len(steps) <= 4
    assert flattened(steps) != flattened(calculation_steps(
        distribution_id, LESSONS[distribution_id]["example"]["parameters"],
        LESSONS[distribution_id]["example"]["event"]))


def test_binomial_question_retains_context_after_editing():
    details = case_details("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5})
    text = flattened(details)
    assert "visitant" in text.lower() or "convers" in text.lower()
    assert "10" in details["context"] and "5" in details["question"]
    assert "0,5" in text
    assert "20 visitantes" not in text
    assert "0,24609375" in details["interpretation"]


def test_multinomial_names_counts_and_combinatorial_substitution_stay_live():
    params = {"n": 10, "probabilities": [0.2, 0.3, 0.5], "names": ["Local", "Teléfono", "Web"]}
    event = {"kind": "joint", "counts": [2, 3, 5]}
    steps = calculation_steps("multinomial", params, event)
    latex = " ".join(step["latex"] for step in steps)
    assert "2520" in latex
    assert "3.375" in latex or "0.00003375" in latex
    assert "0.08505" in latex
    assert probability("multinomial", params, event) == pytest.approx(0.08505)
    assert stats.binom(10, 0.5).pmf(5) == pytest.approx(0.24609375)
    renamed = {"n": 8, "probabilities": [0.25, 0.75], "names": ["Correo nuevo", "Sucursal nueva"]}
    details = case_details("multinomial", renamed, {"kind": "joint", "counts": [2, 6]})
    text = flattened(details)
    assert all(name in text for name in renamed["names"])
    assert "Local" not in text and "Teléfono" not in text and "Web" not in text


def test_reference_statistics_start_from_measurements_and_keep_hypotheses_clear():
    fixtures = [
        ("t", {"df": 15}, {"kind": "gt", "x": 2.0}, stats.t(15).sf(2), ("52", "50", "4", "16")),
        ("chi2", {"df": 9}, {"kind": "gt", "x": 14.4}, stats.chi2(9).sf(14.4), ("1.6", "14.4", "9")),
        ("f", {"dfn": 9, "dfd": 11}, {"kind": "gt", "x": 2.0}, stats.f(9, 11).sf(2), ("8", "4", "9", "11")),
    ]
    for distribution_id, params, event, expected, substitutions in fixtures:
        samples = SAMPLES[distribution_id]
        details = case_details(distribution_id, params, event, samples=samples)
        steps = calculation_steps(distribution_id, params, event, samples=samples)
        latex = " ".join(step["latex"] for step in steps)
        assert all(value in latex for value in substitutions)
        assert text_number(expected) in details["interpretation"]
        text = flattened(details).lower()
        assert "muestra" in text
        assert "normal" in text
        assert "hipótesis" in text or "h₀" in text or "h_0" in text or "h0" in text
        assert any(phrase in text for phrase in ("no es la probabilidad", "no indica", "no significa", "no es p(h0 verdadera)"))
    assert 4 / math.sqrt(16) == 1
    assert (52 - 50) / (4 / math.sqrt(16)) == 2
    assert stats.t(15).sf(2) == pytest.approx(0.031972503642998)
    assert 2 * stats.t(15).sf(2) == pytest.approx(0.063945007285996)
    assert stats.chi2(9).sf(9 * 1.6) == pytest.approx(0.1087909552559986)
    # Swapping samples changes both degrees of freedom and the relevant tail.
    assert stats.f(11, 9).cdf(0.5) == pytest.approx(stats.f(9, 11).sf(2))


def test_beta_answer_is_uncertainty_about_a_rate_not_visitor_frequency():
    details = case_details("beta", {"alpha": 9.0, "beta": 41.0}, {"kind": "gt", "x": 0.2})
    text = details["interpretation"].lower()
    assert "tasa" in text or "proporción" in text
    assert "incertidumbre" in flattened(details).lower()
    assert "tasa desconocida" in text
    assert "de cada 100 visitantes" not in text


@pytest.mark.parametrize("distribution_id", [item for item in IDS if item != "multinomial"])
def test_case_chart_highlight_matches_the_numeric_event(distribution_id):
    example = LESSONS[distribution_id]["example"]
    params, event = example["parameters"], example["event"]
    first, _, metadata = make_charts(distribution_id, params, event)
    if REGISTRY[distribution_id].kind == "discrete":
        selected = np.asarray(first.data[1].x)
        if event["kind"] == "gt":
            assert np.all(selected > event["x"])
        elif event["kind"] == "le":
            assert np.all(selected <= event["x"])
        elif event["kind"] == "eq":
            assert np.all(selected == event["x"])
        else:
            assert np.all((event["a"] <= selected) & (selected <= event["b"]))
        visible_mass = sum(first.data[1].y)
        assert abs(visible_mass - probability(distribution_id, params, event)) <= metadata["excluded_probability"] + 1e-12
    else:
        shaded = [trace for trace in first.data if trace.fill == "tozeroy"]
        assert shaded, distribution_id
        x = np.asarray(shaded[0].x)
        if event["kind"] == "gt":
            assert x[0] == pytest.approx(event["x"])
        elif event["kind"] == "le":
            assert x[-1] == pytest.approx(event["x"])
        else:
            assert x[0] == pytest.approx(event["a"])
            assert x[-1] == pytest.approx(event["b"])


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    entrypoint = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(entrypoint, default_timeout=30).run()


def choose(app, distribution_id):
    kind = "Continuas" if REGISTRY[distribution_id].kind == "continuous" else "Discretas"
    if app.radio(key="distribution_type").value != kind:
        app.radio(key="distribution_type").set_value(kind).run()
    app.selectbox(key=f"distribution::{kind}").select(distribution_id).run()
    assert not app.exception, [element.value for element in app.exception]


def case_probability(app):
    return float(next(metric.value for metric in app.tabs[2].metric
                      if metric.label == "Probabilidad del evento").replace(",", "."))


def case_text(app):
    return "\n".join(element.value for kind in ("markdown", "success", "caption")
                     for element in app.tabs[2].get(kind))


@pytest.mark.parametrize("distribution_id,params,event,expected", VARIANTS)
def test_all_24_case_editors_recompute_question_steps_result_and_graph_in_place(app, distribution_id, params, event, expected):
    choose(app, distribution_id)
    prefix = f"case::{distribution_id}"
    if distribution_id in SAMPLES:
        app.radio(key=f"{prefix}::input_mode").set_value("Parámetros del estadístico").run()
    if distribution_id == "multinomial":
        app.number_input(key=f"{prefix}::category_count").set_value(2).run()
        app.number_input(key=f"{prefix}::param::n").set_value(params["n"])
        for i, (name, p, count) in enumerate(zip(params["names"], params["probabilities"], event["counts"])):
            app.text_input(key=f"{prefix}::name::{i}").set_value(name)
            app.number_input(key=f"{prefix}::prob::{i}").set_value(p)
            app.number_input(key=f"{prefix}::count::{i}").set_value(count)
        app.run()
    else:
        for key, value in params.items():
            app.number_input(key=f"{prefix}::param::{key}").set_value(value)
        app.selectbox(key=f"{prefix}::event_kind").select(event["kind"]).run()
        for key, value in event.items():
            if key != "kind":
                app.number_input(key=f"{prefix}::{key}").set_value(float(value))
        app.run()
    assert not app.exception, [element.value for element in app.exception]
    assert not app.tabs[2].error
    assert case_probability(app) == pytest.approx(expected, rel=8e-8, abs=1e-14)
    details = case_details(distribution_id, params, event)
    text = case_text(app)
    assert details["question"] in text
    assert details["interpretation"] in text
    expected_steps = calculation_steps(distribution_id, params, event)
    emitted = [element.value for element in app.tabs[2].latex]
    assert all(any(step["latex"] in element for element in emitted) for step in expected_steps)
    assert len(app.tabs[2].get("plotly_chart")) >= 2
    assert not app.button(key=f"{prefix}::load").disabled


@pytest.mark.parametrize("distribution_id,samples,expected_params,expected_stat,expected", [
    ("t", {"n": 25, "mean": 53.0, "std": 5.0, "null": 50.0}, {"df": 24.0}, 3.0, stats.t(24).sf(3)),
    ("chi2", {"n": 6, "variance": 2.0, "null_variance": 1.0}, {"df": 5.0}, 10.0, stats.chi2(5).sf(10)),
    ("f", {"n1": 8, "n2": 11, "variance1": 3.0, "variance2": 2.0}, {"dfn": 7.0, "dfd": 10.0}, 1.5, stats.f(7, 10).sf(1.5)),
])
def test_sample_controls_derive_current_reference_law_statistic_and_answer(app, distribution_id, samples, expected_params, expected_stat, expected):
    choose(app, distribution_id)
    prefix = f"case::{distribution_id}"
    for key, value in samples.items():
        app.number_input(key=f"{prefix}::sample::{key}").set_value(value)
    app.run()
    assert not app.exception and not app.tabs[2].error
    model = app.session_state["models"][prefix]
    assert model["parameters"] == expected_params
    assert model["event"] == {"kind": "gt", "x": expected_stat}
    assert case_probability(app) == pytest.approx(expected, rel=8e-8)
    details = case_details(distribution_id, expected_params, model["event"], samples=samples)
    assert details["question"] in case_text(app)
    assert len(app.tabs[2].get("plotly_chart")) == 2
    # Loading a sample-derived example copies both the law and the statistic.
    app.button(key=f"{prefix}::load").click().run()
    assert app.session_state["models"][f"explore::{distribution_id}"]["parameters"] == expected_params
    assert app.session_state["models"][f"explore::{distribution_id}"]["event"] == {"kind": "gt", "x": expected_stat}


@pytest.mark.parametrize("distribution_id,key", [("t", "std"), ("chi2", "null_variance"), ("f", "variance2")])
def test_invalid_sample_does_not_leave_a_stale_answer_and_recovers(app, distribution_id, key):
    choose(app, distribution_id)
    prefix = f"case::{distribution_id}"
    valid = app.number_input(key=f"{prefix}::sample::{key}").value
    app.number_input(key=f"{prefix}::sample::{key}").set_value(0.0).run()
    assert not app.exception and app.tabs[2].error
    assert app.button(key=f"{prefix}::load").disabled
    assert not app.tabs[2].success and not app.tabs[2].metric
    assert not app.tabs[2].latex and not app.tabs[2].get("plotly_chart")
    app.number_input(key=f"{prefix}::sample::{key}").set_value(valid).run()
    assert not app.exception and not app.tabs[2].error
    assert app.tabs[2].success and app.tabs[2].metric
    assert len(app.tabs[2].get("plotly_chart")) == 2


def test_multinomial_primary_chart_is_the_exact_joint_event_and_marginal_is_separate(app):
    choose(app, "multinomial")
    app.selectbox(key="case::multinomial::joint_category").select(2).run()
    charts = app.tabs[2].get("plotly_chart")
    figure = json.loads(charts[0].proto.spec)
    assert figure["data"][0]["y"] == pytest.approx([0.08505, 0.91495])
    assert "0,24609375" in case_text(app)
    assert "marginal" in case_text(app).lower()
    assert case_probability(app) == pytest.approx(0.08505)


def test_case_reset_and_distribution_switch_preserve_independent_work(app):
    choose(app, "binom")
    app.number_input(key="case::binom::param::n").set_value(10)
    app.number_input(key="case::binom::param::p").set_value(0.5).run()
    saved = deepcopy(app.session_state["models"]["case::binom"])
    choose(app, "t")
    app.number_input(key="case::t::sample::mean").set_value(54.0).run()
    assert case_probability(app) == pytest.approx(stats.t(15).sf(4), rel=8e-8)
    choose(app, "binom")
    assert app.session_state["models"]["case::binom"] == saved
    choose(app, "t")
    assert app.number_input(key="case::t::sample::mean").value == 54.0
    app.button(key="case::t::reset").click().run()
    assert not app.exception and not app.tabs[2].error
    assert app.number_input(key="case::t::sample::mean").value == 52.0
    assert case_probability(app) == pytest.approx(stats.t(15).sf(2), rel=8e-8)
    assert app.session_state["models"]["case::binom"] == saved


def test_f_sample_swap_requires_reciprocal_statistic_degrees_and_opposite_tail():
    params, event = sample_model("f", {"n1": 10, "n2": 12, "variance1": 8.0, "variance2": 4.0})
    swapped_params, swapped_event = sample_model("f", {"n1": 12, "n2": 10, "variance1": 4.0, "variance2": 8.0}, tail="le")
    assert params == {"dfn": 9.0, "dfd": 11.0} and event == {"kind": "gt", "x": 2.0}
    assert swapped_params == {"dfn": 11.0, "dfd": 9.0} and swapped_event == {"kind": "le", "x": 0.5}
    assert probability("f", params, event) == pytest.approx(probability("f", swapped_params, swapped_event))


def test_underflowing_standard_error_is_a_validation_error():
    with pytest.raises(ValidationError):
        sample_model("t", {"n": 16, "mean": 52.0, "std": 5e-324, "null": 50.0})


def test_case_reference_mode_and_selected_tail_survive_distribution_switches(app):
    choose(app, "f")
    app.selectbox(key="case::f::sample_tail").select("le").run()
    left_probability = case_probability(app)
    choose(app, "norm")
    choose(app, "f")
    assert app.selectbox(key="case::f::sample_tail").value == "le"
    assert case_probability(app) == left_probability
    app.radio(key="case::f::input_mode").set_value("Parámetros del estadístico").run()
    app.number_input(key="case::f::param::dfn").set_value(6.0)
    app.number_input(key="case::f::param::dfd").set_value(6.0)
    app.selectbox(key="case::f::event_kind").select("gt").run()
    app.number_input(key="case::f::x").set_value(1.0).run()
    choose(app, "chi2")
    choose(app, "f")
    assert app.radio(key="case::f::input_mode").value == "Parámetros del estadístico"
    assert app.number_input(key="case::f::param::dfn").value == 6.0
    assert app.number_input(key="case::f::param::dfd").value == 6.0
    assert case_probability(app) == pytest.approx(0.5)
    app.button(key="case::f::reset").click().run()
    assert app.radio(key="case::f::input_mode").value == "Mediciones de la muestra"
    assert app.selectbox(key="case::f::sample_tail").value == "gt"
    assert case_probability(app) == pytest.approx(stats.f(9, 11).sf(2), rel=8e-8)


def test_case_discrete_quantile_reports_cdf_jump_and_shades_the_achieved_event(app):
    choose(app, "binom")
    app.number_input(key="case::binom::param::n").set_value(10)
    app.number_input(key="case::binom::param::p").set_value(0.5)
    app.selectbox(key="case::binom::event_kind").select("quantile").run()
    app.number_input(key="case::binom::q").set_value(0.5).run()
    assert not app.exception and not app.tabs[2].error
    assert any(metric.label == "Umbral del caso" and metric.value == "5" for metric in app.tabs[2].metric)
    text = case_text(app)
    assert "0,62304688" in text
    assert "acumulad" in text.lower()
    latex = " ".join(element.value for element in app.tabs[2].latex)
    assert "0.62304688" in latex and "F_X" in latex
    figure = json.loads(app.tabs[2].get("plotly_chart")[0].proto.spec)
    assert len(figure["data"]) >= 2


def test_case_tiny_positive_probability_retains_logarithm_instead_of_false_zero(app):
    choose(app, "expon")
    app.number_input(key="case::expon::param::rate").set_value(1.0)
    app.selectbox(key="case::expon::event_kind").select("gt").run()
    app.number_input(key="case::expon::x").set_value(1000.0).run()
    assert not app.exception and not app.tabs[2].error
    warning = " ".join(element.value for element in app.tabs[2].warning)
    assert "positiva" in warning and "-1000" in warning
    assert "no es un cero exacto" in case_text(app).lower() or "no es un cero exacto" in warning.lower()
    latex = " ".join(element.value for element in app.tabs[2].latex)
    assert r"\log" in latex and "-1000" in latex
    assert len(app.tabs[2].get("plotly_chart")) == 2
