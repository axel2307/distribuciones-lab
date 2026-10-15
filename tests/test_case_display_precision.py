"""Close event endpoints must stay distinct across the whole guided case."""
from pathlib import Path
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

from lab.case_studies import case_details


@pytest.mark.parametrize("distribution_id,params,a,b,expected", [
    ("norm", {"mu":50., "sigma":2.}, 50., 50.00000001, "50,00000001"),
    ("beta", {"alpha":9., "beta":41.}, .2, .20000000001, "20,000000001"),
])
def test_close_limits_remain_distinct_in_the_question(distribution_id, params, a, b, expected):
    details = case_details(distribution_id, params, {"kind":"interval", "a":a, "b":b})
    assert expected in details["question"]
    assert expected in details["event_words"]


def test_probability_near_one_is_not_described_as_a_certain_parameter():
    details = case_details("bernoulli", {"p":.9999999999}, {"kind":"eq", "x":0})
    assert "0,9999999999" in details["context"]
    assert details["parameter_rows"][0]["Valor"] == "0,9999999999"


def test_close_limits_remain_distinct_in_streamlit_notation(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.selectbox(key="case::norm::event_kind").select("interval").run()
    app.number_input(key="case::norm::a").set_value(50.)
    app.number_input(key="case::norm::b").set_value(50.00000001).run()
    assert not app.exception and not app.tabs[2].error
    assert any("50.00000001" in item.value for item in app.tabs[2].latex)
    assert any("50,00000001" in item.value for item in app.tabs[2].markdown)
    probability = next(item.value for item in app.tabs[2].metric if item.label == "Probabilidad del evento")
    assert float(probability.replace(",", ".")) > 0
