"""Completeness and independently calculated worked-example checks."""

import math

import pytest
from scipy import stats

from lab.catalog import REGISTRY, defaults
from lab.calculations import multinomial_joint, probability, quantile, validate_parameters
from lab.lessons_continuous import LESSONS as CONTINUOUS_LESSONS
from lab.lessons_discrete import LESSONS as DISCRETE_LESSONS


LESSONS = {**DISCRETE_LESSONS, **CONTINUOUS_LESSONS}
LESSON_FIELDS = (
    "intuition", "variable", "assumptions", "applications", "inappropriate",
    "error", "effects", "example",
)
EXAMPLE_FIELDS = (
    "context", "question", "variable", "justification", "parameters", "event",
    "interpretation", "limitations",
)


def test_catalog_and_educational_content_cover_the_same_24_families():
    assert len(LESSONS) == 24
    assert set(LESSONS) == set(REGISTRY)
    assert not set(CONTINUOUS_LESSONS) & set(DISCRETE_LESSONS)


@pytest.mark.parametrize("distribution_id", tuple(LESSONS))
def test_every_lesson_and_example_is_complete(distribution_id):
    lesson = LESSONS[distribution_id]
    for field in LESSON_FIELDS:
        assert field in lesson and lesson[field], (distribution_id, field)
    for field in EXAMPLE_FIELDS:
        assert field in lesson["example"] and lesson["example"][field] is not None, (distribution_id, field)
    text = " ".join(str(value) for value in lesson["example"].values()).lower()
    assert "ilustrativ" in text, "Invented examples must identify themselves as illustrative."
    assert "próximamente" not in str(lesson).lower()
    assert isinstance(lesson["assumptions"], (list, tuple))
    assert isinstance(lesson["applications"], (list, tuple))
    assert isinstance(lesson["effects"], dict)
    if distribution_id != "multinomial":
        assert set(lesson["effects"]) == {parameter.key for parameter in REGISTRY[distribution_id].parameters}


@pytest.mark.parametrize("distribution_id", tuple(REGISTRY))
def test_formula_support_symbols_and_parameter_descriptions_are_present(distribution_id):
    spec = REGISTRY[distribution_id]
    assert spec.formula.strip()
    assert spec.symbols
    assert spec.support
    assert spec.scipy_name == distribution_id
    assert spec.source_url == f"https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.{distribution_id}.html"
    for parameter in spec.parameters:
        assert parameter.name and parameter.symbol and parameter.explanation and parameter.restriction
        assert parameter.minimum <= parameter.default <= parameter.maximum
        assert parameter.step > 0


@pytest.mark.parametrize("distribution_id", tuple(LESSONS))
def test_example_parameters_are_valid(distribution_id):
    validate_parameters(distribution_id, LESSONS[distribution_id]["example"]["parameters"])


def test_multinomial_example_joint_probability_is_combinatorial():
    example = LESSONS["multinomial"]["example"]
    assert example["event"]["kind"] == "joint"
    params = example["parameters"]
    counts = example["event"]["counts"]
    expected = math.factorial(params["n"])
    for p, count in zip(params["probabilities"], counts):
        expected *= p ** count / math.factorial(count)
    actual = multinomial_joint(params["n"], params["probabilities"], counts)
    assert actual == pytest.approx(expected, rel=1e-12)


def _independent_reference_rv(distribution_id, p):
    """Explicit documentation mappings; deliberately do not use app factories."""
    factories = {
        "bernoulli": lambda: stats.bernoulli(p["p"]),
        "binom": lambda: stats.binom(p["n"], p["p"]),
        "poisson": lambda: stats.poisson(p["mu"]),
        "geom": lambda: stats.geom(p["p"]),
        "nbinom": lambda: stats.nbinom(p["r"], p["p"]),
        "hypergeom": lambda: stats.hypergeom(p["M"], p["K"], p["n"]),
        "randint": lambda: stats.randint(p["low"], p["high"] + 1),
        "norm": lambda: stats.norm(p["mu"], p["sigma"]),
        "uniform": lambda: stats.uniform(p["a"], p["b"] - p["a"]),
        "expon": lambda: stats.expon(scale=1 / p["rate"]),
        "gamma": lambda: stats.gamma(p["alpha"], scale=p["theta"]),
        "beta": lambda: stats.beta(p["alpha"], p["beta"]),
        "lognorm": lambda: stats.lognorm(p["sigma_log"], scale=math.exp(p["mu_log"])),
        "t": lambda: stats.t(p["df"]),
        "chi2": lambda: stats.chi2(p["df"]),
        "f": lambda: stats.f(p["dfn"], p["dfd"]),
        "weibull_min": lambda: stats.weibull_min(p["shape"], scale=p["scale"]),
        "pareto": lambda: stats.pareto(p["alpha"], scale=p["xmin"]),
        "cauchy": lambda: stats.cauchy(p["x0"], p["gamma"]),
        "laplace": lambda: stats.laplace(p["mu"], p["b"]),
        "rayleigh": lambda: stats.rayleigh(scale=p["sigma"]),
        "gumbel_r": lambda: stats.gumbel_r(p["mu"], p["beta"]),
        "genextreme": lambda: stats.genextreme(-p["xi"], p["mu"], p["sigma"]),
    }
    return factories[distribution_id]()


@pytest.mark.parametrize("distribution_id", tuple(key for key in LESSONS if key != "multinomial"))
def test_worked_example_result_matches_documented_scipy_mapping(distribution_id):
    example = LESSONS[distribution_id]["example"]
    params, event = example["parameters"], example["event"]
    rv = _independent_reference_rv(distribution_id, params)
    discrete = REGISTRY[distribution_id].kind == "discrete"
    if event["kind"] == "quantile":
        assert quantile(distribution_id, params, event["q"]) == pytest.approx(rv.ppf(event["q"]))
        return
    if event["kind"] == "eq":
        expected = rv.pmf(event["x"])
    elif event["kind"] == "le":
        expected = rv.cdf(event["x"])
    elif event["kind"] == "gt":
        expected = rv.sf(event["x"])
    else:
        lower, upper = event["a"], event["b"]
        if discrete:
            lower, upper = math.ceil(lower), math.floor(upper)
            # Explicitly summing the finite event also checks inclusive endpoints.
            expected = sum(float(rv.pmf(k)) for k in range(lower, upper + 1))
        elif rv.cdf(lower) >= 0.5:
            expected = rv.sf(lower) - rv.sf(upper)
        else:
            expected = rv.cdf(upper) - rv.cdf(lower)
    actual = probability(distribution_id, params, event)
    assert 0 <= actual <= 1
    assert actual == pytest.approx(float(expected), rel=1e-11, abs=1e-14)


@pytest.mark.parametrize("distribution_id", ["expon", "gamma", "lognorm", "randint", "nbinom", "genextreme", "pareto"])
def test_important_conventions_are_explained_in_catalog_symbols(distribution_id):
    symbols_text = " ".join(REGISTRY[distribution_id].symbols.values()).lower()
    expected_words = {
        "expon": ("tasa", "scale"), "gamma": ("escala", "tasa"),
        "lognorm": ("ln(x)", "scale"), "randint": ("b+1", "high"),
        "nbinom": ("fracasos", "k+r"), "genextreme": ("scipy", "c"),
        "pareto": ("scale", "loc"),
    }
    for word in expected_words[distribution_id]:
        assert word in symbols_text
