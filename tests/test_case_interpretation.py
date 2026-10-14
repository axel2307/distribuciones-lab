"""Regressions: editable case explanations must describe the current scenario."""

import math
from pathlib import Path
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

from lab.calculations import ValidationError
from lab.case_narrative import case_interpretation
from lab.catalog import REGISTRY
from lab.content import LESSONS


APP = Path(__file__).resolve().parents[1] / "app.py"


def number(value):
    """Public display precision, independently formatted for test assertions."""
    return f"{value:.8g}".replace(".", ",")


VARIANTS = [
    ("bernoulli", {"p": 0.7}, {"kind": "eq", "x": 1}, 0.7),
    ("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5}, 0.24609375),
    ("poisson", {"mu": 2}, {"kind": "le", "x": 2}, 5 * math.exp(-2)),
    ("geom", {"p": 0.5}, {"kind": "le", "x": 3}, 0.875),
    ("nbinom", {"r": 2, "p": 0.5}, {"kind": "eq", "x": 1}, 0.25),
    ("hypergeom", {"M": 20, "K": 5, "n": 4}, {"kind": "gt", "x": 0}, 1 - math.comb(15, 4) / math.comb(20, 4)),
    ("multinomial", {"n": 8, "probabilities": [0.25, 0.75], "names": ["Nuevo A", "Nuevo B"]}, {"kind": "joint", "counts": [2, 6]}, math.comb(8, 2) * 0.25 ** 2 * 0.75 ** 6),
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


def test_variant_oracles_cover_every_family():
    assert {item[0] for item in VARIANTS} == set(REGISTRY)
    assert len(VARIANTS) == 24


@pytest.mark.parametrize("distribution_id, params, event, expected", VARIANTS)
def test_current_case_interpretation_has_live_parameters_and_independent_result(distribution_id, params, event, expected):
    text = case_interpretation(distribution_id, params, event)
    assert isinstance(text, str) and text
    assert "Modelo actual" in text
    assert number(expected) in text
    assert number(100 * expected) in text
    for parameter in REGISTRY[distribution_id].parameters:
        assert parameter.name in text
        assert number(params[parameter.key]) in text
    assert text != LESSONS[distribution_id]["example"]["interpretation"]


@pytest.mark.parametrize("event, expected_phrase, expected", [
    ({"kind": "le", "x": 3}, "X sea como máximo 3", 0.3),
    ({"kind": "gt", "x": 3}, "X sea mayor que 3", 0.7),
    ({"kind": "interval", "a": 2, "b": 6}, "X esté entre 2 y 6", 0.4),
])
def test_event_wording_matches_current_limits_and_probability(event, expected_phrase, expected):
    text = case_interpretation("uniform", {"a": 0, "b": 10}, event)
    assert expected_phrase in text
    assert number(expected) in text
    if event["kind"] == "interval":
        assert "ambos incluidos" in text


def test_exact_binomial_variant_has_new_trials_event_and_probability():
    text = case_interpretation("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5})
    assert "X sea exactamente 5" in text
    assert "éxitos en 10 ensayos" in text
    assert "0,5 por ensayo" in text
    assert "0,24609375" in text
    assert "24,609375" in text
    assert "20 visitantes" not in text
    assert "5 o más" not in text
    assert "al menos 5" not in text


def test_multinomial_joint_variant_has_live_names_vector_and_trials():
    params = {"n": 8, "probabilities": [0.25, 0.75], "names": ["Correo nuevo", "Sucursal nueva"]}
    text = case_interpretation("multinomial", params, {"kind": "joint", "counts": [2, 6]})
    assert "Correo nuevo" in text and "Sucursal nueva" in text
    assert "2" in text and "6" in text and "8" in text
    assert "0,25" in text and "0,75" in text
    for stale in ("Local", "Teléfono", "Web", "(2,3,5)"):
        assert stale not in text


def test_quantile_variant_reports_requested_and_achieved_probability():
    text = case_interpretation("uniform", {"a": 0, "b": 10}, {"kind": "quantile", "q": 0.25})
    assert "0,25" in text and "2,5" in text
    assert "cuantil" in text.lower()
    assert "acumulada" in text.lower()


def test_quantile_rounding_does_not_claim_requested_probability_was_achieved():
    text = case_interpretation("beta", {"alpha": 0.1, "beta": 0.1}, {"kind": "quantile", "q": 0.999})
    assert "0,999" in text
    assert "alcanzada" in text.lower() or "lograda" in text.lower()
    assert "es 1 (100 %)" in text
    assert "precisión" in text.lower() or "redonde" in text.lower()


def test_discrete_quantile_interpretation_reports_probability_after_cdf_jump():
    text = case_interpretation("binom", {"n": 10, "p": 0.5}, {"kind": "quantile", "q": 0.5})
    assert "q = 0,5" in text
    assert "cuantil es 5" in text
    # Independent sum C(10,k)/2**10 for k=0..5 is 638/1024.
    assert "0,62304688" in text and "62,304688" in text


def test_underflow_variant_interpretation_explains_positive_probability():
    text = case_interpretation("expon", {"rate": 1}, {"kind": "gt", "x": 1000})
    assert "mayor que 1000" in text
    assert "probabilidad es positiva" in text
    assert "logaritmo natural es -1000" in text
    assert "probabilidad cero" not in text


def test_impossible_current_event_is_explained_as_exact_zero():
    text = case_interpretation("uniform", {"a": 0, "b": 1}, {"kind": "interval", "a": 2, "b": 3})
    assert "X esté entre 2 y 3" in text
    assert "probabilidad cero" in text
    assert "positiva" not in text


def test_quantile_extreme_has_support_explanation_without_fake_finite_threshold():
    text = case_interpretation("norm", {"mu": 3, "sigma": 2}, {"kind": "quantile", "q": 1})
    assert "q = 1" in text
    assert "superior infinito" in text
    assert "no hay un umbral finito" in text


@pytest.mark.parametrize("distribution_id, params, event", [
    ("binom", {"n": -1, "p": 0.5}, {"kind": "eq", "x": 5}),
    ("binom", {"n": 10, "p": 0.5}, {"kind": "eq", "x": 5.2}),
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "interval", "a": 3, "b": 2}),
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "quantile", "q": 1.1}),
    ("norm", {"mu": 0, "sigma": 1}, {"kind": "joint", "counts": [1]}),
    ("norm", {"mu": 0, "sigma": 1}, {}),
    ("multinomial", {"n": 8, "probabilities": [0.2, 0.7], "names": ["A", "B"]}, {"kind": "joint", "counts": [2, 6]}),
    ("multinomial", {"n": 8, "probabilities": [0.25, 0.75], "names": ["A", "B"]}, {"kind": "joint", "counts": [2, 5]}),
])
def test_invalid_variants_do_not_return_plausible_interpretations(distribution_id, params, event):
    with pytest.raises(ValidationError):
        case_interpretation(distribution_id, params, event)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=30).run()


def choose(app, distribution_id):
    kind = "Continuas" if REGISTRY[distribution_id].kind == "continuous" else "Discretas"
    if app.radio(key="distribution_type").value != kind:
        app.radio(key="distribution_type").set_value(kind).run()
    app.selectbox(key=f"distribution::{kind}").select(distribution_id).run()
    assert not app.exception
    return app


def case_text(app):
    return "\n".join(element.value for element in app.tabs[2].markdown)


def test_ui_edited_binomial_result_and_step_seven_describe_current_case(app):
    choose(app, "binom")
    app.number_input(key="case::binom::param::n").set_value(10)
    app.number_input(key="case::binom::param::p").set_value(0.5)
    app.selectbox(key="case::binom::event_kind").select("eq").run()
    app.number_input(key="case::binom::x").set_value(5.0).run()
    assert not app.exception and not app.tabs[2].error
    text = case_text(app)
    assert "X sea exactamente 5" in text
    assert "éxitos en 10 ensayos" in text
    assert "0,24609375" in text
    assert LESSONS["binom"]["example"]["interpretation"] not in text
    assert any("variante" in element.value.lower() or "modificad" in element.value.lower() for element in app.tabs[2].info)


def test_ui_joint_case_uses_edited_category_names_and_count_vector(app):
    choose(app, "multinomial")
    app.text_input(key="case::multinomial::name::0").set_value("Correo nuevo")
    app.number_input(key="case::multinomial::count::0").set_value(1)
    app.number_input(key="case::multinomial::count::2").set_value(6).run()
    assert not app.exception and not app.tabs[2].error
    text = case_text(app)
    assert "Correo nuevo" in text
    assert LESSONS["multinomial"]["example"]["interpretation"] not in text
    assert "vector" in text.lower()


def test_ui_invalid_case_suppresses_active_interpretation_and_recovers(app):
    choose(app, "norm")
    app.selectbox(key="case::norm::event_kind").select("interval").run()
    app.number_input(key="case::norm::a").set_value(3)
    app.number_input(key="case::norm::b").set_value(2).run()
    assert not app.exception and app.tabs[2].error
    assert app.button(key="case::norm::load").disabled
    text = case_text(app)
    assert "Modelo actual" not in text
    assert LESSONS["norm"]["example"]["interpretation"] not in text
    app.number_input(key="case::norm::b").set_value(4).run()
    assert not app.exception and not app.tabs[2].error
    assert "Modelo actual" in case_text(app)
