from pathlib import Path
import tempfile
from streamlit.testing.v1 import AppTest


def test_beta_rounded_quantile_warns_and_keeps_achieved_probability(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=30).run()
    app.selectbox(key="distribution::Continuas").select("beta").run()
    app.number_input(key="explore::beta::param::alpha").set_value(0.1)
    app.number_input(key="explore::beta::param::beta").set_value(0.1)
    app.selectbox(key="explore::beta::event_kind").select("quantile").run()
    app.number_input(key="explore::beta::q").set_value(0.999).run()
    assert not app.exception
    assert any("El cuantil está demasiado cerca" in e.value for e in app.warning)
    assert any("100 %" in e.value for e in app.tabs[0].markdown)


def test_positive_underflow_shows_logarithm_instead_of_exact_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=30).run()
    app.selectbox(key="explore::norm::event_kind").select("gt").run()
    app.number_input(key="explore::norm::x").set_value(40.0).run()
    assert not app.exception
    assert any("No es un cero exacto" in e.value for e in app.warning)
    assert any("Logaritmo natural" in e.value for e in app.tabs[0].markdown)
