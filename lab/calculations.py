"""Validated SciPy parametrizations and probability calculations.

Public parameter names follow the formulas in the catalog. No interface bounds
are enforced here: this module checks mathematical domains only. Probabilities
for discrete intervals include both endpoints, with ceil/floor for real bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
from scipy import stats


class ValidationError(ValueError):
    """An invalid mathematical input, with a message suitable for the UI."""


@dataclass(frozen=True)
class Moment:
    value: float | None
    status: str = "finite"


DISCRETE_IDS = frozenset(
    {"bernoulli", "binom", "poisson", "geom", "nbinom", "hypergeom", "randint"}
)
CONTINUOUS_IDS = frozenset(
    {"norm", "uniform", "expon", "gamma", "beta", "lognorm", "t", "chi2", "f",
     "weibull_min", "pareto", "cauchy", "laplace", "rayleigh", "gumbel_r", "genextreme"}
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValidationError(f"{name} debe ser un número real finito.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"{name} debe ser un número real finito.") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} debe ser un número real finito.")
    return result


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValidationError(f"{name} debe ser un entero.")
    number = int(number)
    if minimum is not None and number < minimum:
        raise ValidationError(f"{name} debe ser mayor o igual que {minimum}.")
    return number


def _get(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise ValidationError(f"Falta el parámetro {key}.")
    return params[key]


def _positive(params: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if _number(_get(params, key), key) <= 0:
            raise ValidationError(f"{key} debe ser estrictamente positivo.")


def _probability_param(params: Mapping[str, Any], key: str = "p", allow_zero: bool = True) -> None:
    p = _number(_get(params, key), key)
    if not (0 <= p <= 1) or (not allow_zero and p == 0):
        domain = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValidationError(f"{key} debe pertenecer a {domain}.")


def validate_multinomial(
    n: Any, probabilities: Sequence[float], counts: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
) -> None:
    """Reject invalid vectors before SciPy can replace the last probability.

    A 1e-12 absolute tolerance allows floating-point summation roundoff only;
    values are never rescaled or otherwise repaired by this application.
    """
    trials = _integer(n, "El número de ensayos", 0)
    if isinstance(probabilities, (str, bytes)):
        raise ValidationError("Ingresá una lista de probabilidades por categoría.")
    try:
        size = len(probabilities)
    except TypeError as exc:
        raise ValidationError("Ingresá una lista de probabilidades por categoría.") from exc
    if not 2 <= size <= 5:
        raise ValidationError("La multinomial necesita entre 2 y 5 categorías.")
    values = [_number(p, f"La probabilidad de la categoría {i + 1}") for i, p in enumerate(probabilities)]
    if any(p < 0 or p > 1 for p in values):
        raise ValidationError("Cada probabilidad debe pertenecer a [0, 1].")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError("Las probabilidades deben sumar 1. No se normalizan automáticamente.")
    if names is not None:
        if len(names) != size or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValidationError("Cada categoría debe tener un nombre no vacío.")
        if len({name.strip().casefold() for name in names}) != size:
            raise ValidationError("Los nombres de las categorías deben ser distintos.")
    if counts is not None:
        if isinstance(counts, (str, bytes)) or not hasattr(counts, "__len__") or len(counts) != size:
            raise ValidationError("Ingresá un conteo por categoría.")
        integers = [_integer(k, f"El conteo de la categoría {i + 1}", 0) for i, k in enumerate(counts)]
        if sum(integers) != trials:
            raise ValidationError("La suma de los conteos debe ser igual al número de ensayos n.")


def validate_parameters(distribution_id: str, params: Mapping[str, Any]) -> None:
    if not isinstance(params, Mapping):
        raise ValidationError("Los parámetros deben especificarse por nombre.")
    if distribution_id == "multinomial":
        validate_multinomial(_get(params, "n"), _get(params, "probabilities"), names=params.get("names"))
        return
    if distribution_id not in DISCRETE_IDS | CONTINUOUS_IDS:
        raise ValidationError("La distribución seleccionada no pertenece al catálogo.")
    if distribution_id in {"bernoulli", "binom"}:
        _probability_param(params)
        if distribution_id == "binom":
            _integer(_get(params, "n"), "n", 0)
    elif distribution_id == "poisson":
        if _number(_get(params, "mu"), "mu") < 0:
            raise ValidationError("La cantidad media de eventos debe ser no negativa.")
    elif distribution_id in {"geom", "nbinom"}:
        _probability_param(params, allow_zero=False)
        if distribution_id == "nbinom":
            _integer(_get(params, "r"), "El número de éxitos r", 1)
    elif distribution_id == "hypergeom":
        population = _integer(_get(params, "M"), "El tamaño de población M", 1)
        successes = _integer(_get(params, "K"), "Los éxitos en la población K", 0)
        draws = _integer(_get(params, "n"), "El tamaño de extracción n", 0)
        if successes > population or draws > population:
            raise ValidationError("K y n no pueden superar el tamaño de población M.")
    elif distribution_id == "randint":
        low = _integer(_get(params, "low"), "El límite inferior")
        high = _integer(_get(params, "high"), "El límite superior")
        if low > high:
            raise ValidationError("El límite inferior no puede superar al superior.")
    elif distribution_id == "uniform":
        if _number(_get(params, "a"), "a") >= _number(_get(params, "b"), "b"):
            raise ValidationError("La uniforme continua requiere a < b.")
    else:
        real_keys = {
            "norm": ("mu",), "lognorm": ("mu_log",), "cauchy": ("x0",),
            "laplace": ("mu",), "gumbel_r": ("mu",), "genextreme": ("xi", "mu"),
        }
        positive_keys = {
            "norm": ("sigma",), "expon": ("rate",), "gamma": ("alpha", "theta"),
            "beta": ("alpha", "beta"), "lognorm": ("sigma_log",), "t": ("df",),
            "chi2": ("df",), "f": ("dfn", "dfd"), "weibull_min": ("shape", "scale"),
            "pareto": ("alpha", "xmin"), "cauchy": ("gamma",), "laplace": ("b",),
            "rayleigh": ("sigma",), "gumbel_r": ("beta",), "genextreme": ("sigma",),
        }
        for key in real_keys.get(distribution_id, ()):
            _number(_get(params, key), key)
        _positive(params, *positive_keys[distribution_id])


def frozen_dist(distribution_id: str, params: Mapping[str, Any]):
    """Return the matching SciPy RV; public keys are deliberately pedagogical."""
    validate_parameters(distribution_id, params)
    p = params
    if distribution_id == "bernoulli":
        return stats.bernoulli(p["p"])
    if distribution_id == "binom":
        return stats.binom(int(p["n"]), p["p"])
    if distribution_id == "poisson":
        return stats.poisson(p["mu"])
    if distribution_id == "geom":
        return stats.geom(p["p"])
    if distribution_id == "nbinom":
        return stats.nbinom(int(p["r"]), p["p"])
    if distribution_id == "hypergeom":
        return stats.hypergeom(int(p["M"]), int(p["K"]), int(p["n"]))
    if distribution_id == "randint":
        return stats.randint(int(p["low"]), int(p["high"]) + 1)
    if distribution_id == "multinomial":
        return stats.multinomial(int(p["n"]), p["probabilities"])
    if distribution_id == "norm":
        return stats.norm(loc=p["mu"], scale=p["sigma"])
    if distribution_id == "uniform":
        return stats.uniform(loc=p["a"], scale=p["b"] - p["a"])
    if distribution_id == "expon":
        return stats.expon(scale=1 / p["rate"])
    if distribution_id == "gamma":
        return stats.gamma(p["alpha"], scale=p["theta"])
    if distribution_id == "beta":
        return stats.beta(p["alpha"], p["beta"])
    if distribution_id == "lognorm":
        try:
            scale = math.exp(p["mu_log"])
        except OverflowError as exc:
            raise ValidationError("La escala de la lognormal supera la precisión numérica disponible.") from exc
        if scale == 0:
            raise ValidationError("La escala de la lognormal queda por debajo de la precisión numérica disponible.")
        return stats.lognorm(p["sigma_log"], scale=scale)
    if distribution_id == "t":
        return stats.t(p["df"])
    if distribution_id == "chi2":
        return stats.chi2(p["df"])
    if distribution_id == "f":
        return stats.f(p["dfn"], p["dfd"])
    if distribution_id == "weibull_min":
        return stats.weibull_min(p["shape"], scale=p["scale"])
    if distribution_id == "pareto":
        return stats.pareto(p["alpha"], scale=p["xmin"])
    if distribution_id == "cauchy":
        return stats.cauchy(loc=p["x0"], scale=p["gamma"])
    if distribution_id == "laplace":
        return stats.laplace(loc=p["mu"], scale=p["b"])
    if distribution_id == "rayleigh":
        return stats.rayleigh(scale=p["sigma"])
    if distribution_id == "gumbel_r":
        return stats.gumbel_r(loc=p["mu"], scale=p["beta"])
    # Conventional xi has the opposite sign to SciPy genextreme's shape c.
    return stats.genextreme(-p["xi"], loc=p["mu"], scale=p["sigma"])


def validate_event(event: Mapping[str, Any], discrete: bool = False) -> None:
    if not isinstance(event, Mapping):
        raise ValidationError("Ingresá un evento válido.")
    kind = event.get("kind")
    if kind in {"le", "gt"}:
        _number(_get(event, "x"), "El límite x")
    elif kind == "eq":
        if not discrete:
            raise ValidationError("En una distribución continua la probabilidad de un punto exacto es cero; usá un intervalo.")
        _integer(_get(event, "x"), "El resultado k")
    elif kind == "interval":
        lower = _number(_get(event, "a"), "El límite a")
        upper = _number(_get(event, "b"), "El límite b")
        if lower > upper:
            raise ValidationError("El intervalo requiere a ≤ b.")
    elif kind == "quantile":
        q = _number(_get(event, "q"), "La probabilidad del cuantil")
        if not 0 <= q <= 1:
            raise ValidationError("La probabilidad del cuantil debe pertenecer a [0, 1].")
    else:
        raise ValidationError("Elegí un evento univariado válido.")


def _checked_probability(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError("No se pudo obtener una probabilidad con precisión numérica suficiente.")
    if not 0 <= value <= 1:
        # Only accept floating-point roundoff, never repair a conceptual input.
        if -1e-14 <= value < 0:
            return 0.0
        if 1 < value <= 1 + 1e-14:
            return 1.0
        raise ValidationError("El cálculo numérico produjo una probabilidad fuera de [0, 1].")
    return value


def _log_difference(log_big: float, log_small: float) -> float:
    """exp(big) - exp(small), avoiding catastrophic tail cancellation."""
    if log_big == -math.inf:
        return 0.0
    if log_small == -math.inf:
        return math.exp(log_big)
    if log_small > log_big:
        raise ValidationError("No se pudo calcular el intervalo con precisión numérica suficiente.")
    return math.exp(log_big) * -math.expm1(log_small - log_big)


def _log_difference_log(log_big: float, log_small: float) -> float:
    """Log(exp(big) - exp(small)), retaining probabilities below float range."""
    if log_big == -math.inf or log_big == log_small:
        return -math.inf
    if log_small == -math.inf:
        return log_big
    if log_small > log_big:
        raise ValidationError("No se pudo calcular el intervalo con precisión numérica suficiente.")
    return log_big + math.log(-math.expm1(log_small - log_big))


def _interval(rv, lower: float, upper: float) -> float:
    """Probability between CDF cutoffs (the caller handles lattice inclusion)."""
    lower_cdf = float(rv.cdf(lower))
    upper_cdf = float(rv.cdf(upper))
    if upper_cdf <= 0.5:
        result = _log_difference(float(rv.logcdf(upper)), float(rv.logcdf(lower)))
    elif lower_cdf >= 0.5:
        result = _log_difference(float(rv.logsf(lower)), float(rv.logsf(upper)))
    else:
        result = upper_cdf - lower_cdf
    return _checked_probability(result)


def _probability_value(distribution_id: str, params: Mapping[str, Any], event: Mapping[str, Any]) -> float:
    if distribution_id == "multinomial":
        if event.get("kind") != "joint":
            raise ValidationError("La multinomial requiere un vector de conteos; para una CDF usá una marginal binomial.")
        validate_parameters(distribution_id, params)
        return multinomial_joint(params["n"], params["probabilities"], _get(event, "counts"))
    discrete = distribution_id in DISCRETE_IDS
    validate_event(event, discrete)
    rv = frozen_dist(distribution_id, params)
    kind = event["kind"]
    if kind == "quantile":
        raise ValidationError("Usá la función cuantil para obtener un valor de X.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "le":
            x = math.floor(event["x"]) if discrete else event["x"]
            return _checked_probability(rv.cdf(x))
        if kind == "gt":
            x = math.floor(event["x"]) if discrete else event["x"]
            # sf avoids cancellation from 1 - CDF for rare upper-tail events.
            result = float(rv.sf(x))
            if result == 0:
                result = math.exp(float(rv.logsf(x)))
            return _checked_probability(result)
        if kind == "eq":
            return _checked_probability(rv.pmf(int(event["x"])))
        if discrete:
            low, high = math.ceil(event["a"]), math.floor(event["b"])
            if low > high:
                return 0.0
            if low == high:
                return _checked_probability(rv.pmf(low))
            return _interval(rv, low - 1, high)
        if event["a"] == event["b"]:
            return 0.0
        return _interval(rv, event["a"], event["b"])


def log_probability(distribution_id: str, params: Mapping[str, Any], event: Mapping[str, Any]) -> float:
    """Natural logarithm of an event probability, including extremely rare tails.

    A finite logarithm paired with probability()==0 identifies floating-point
    underflow. Negative infinity denotes an impossible event under the model.
    This scalar is for a diagnostic UI label, not a probability percentage.
    """
    if distribution_id == "multinomial":
        if event.get("kind") != "joint":
            raise ValidationError("La multinomial requiere un vector de conteos; elegí una marginal para una CDF.")
        validate_parameters(distribution_id, params)
        return multinomial_logjoint(params["n"], params["probabilities"], _get(event, "counts"))
    discrete = distribution_id in DISCRETE_IDS
    validate_event(event, discrete)
    rv = frozen_dist(distribution_id, params)
    kind = event["kind"]
    if kind == "quantile":
        raise ValidationError("Usá la función cuantil para obtener un valor de X.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind in {"le", "gt"}:
            x = math.floor(event["x"]) if discrete else event["x"]
            value = float(rv.logcdf(x) if kind == "le" else rv.logsf(x))
        elif kind == "eq":
            value = float(rv.logpmf(int(event["x"])))
        else:
            low, high = event["a"], event["b"]
            if discrete:
                low, high = math.ceil(low), math.floor(high)
                if low > high:
                    return -math.inf
                if low == high:
                    return float(rv.logpmf(low))
                low -= 1
            elif low == high:
                return -math.inf
            lower_cdf, upper_cdf = float(rv.cdf(low)), float(rv.cdf(high))
            if upper_cdf <= 0.5:
                value = _log_difference_log(float(rv.logcdf(high)), float(rv.logcdf(low)))
            elif lower_cdf >= 0.5:
                value = _log_difference_log(float(rv.logsf(low)), float(rv.logsf(high)))
            else:
                value = math.log(upper_cdf - lower_cdf)
    if math.isnan(value) or value == math.inf or value > 1e-14:
        raise ValidationError("No se pudo obtener la probabilidad logarítmica con precisión numérica suficiente.")
    if value == -math.inf and _event_possible(distribution_id, params, event, rv):
        raise ValidationError("La probabilidad es positiva, pero incluso su logaritmo queda fuera de la precisión numérica disponible. No se representa como cero exacto.")
    return value


def probability(distribution_id: str, params: Mapping[str, Any], event: Mapping[str, Any]) -> float:
    """Probability as a float; consult log_probability when it underflows.

    If SciPy cannot retain even a logarithm of a positive probability, report a
    numerical limitation instead of returning an apparent exact zero.
    """
    value = _probability_value(distribution_id, params, event)
    if value == 0:
        log_probability(distribution_id, params, event)
    return value


def _point_mass(distribution_id: str, params: Mapping[str, Any]) -> float | None:
    """Recognize valid degeneracies before SciPy ppf/stats edge cases."""
    if distribution_id == "bernoulli" and params["p"] in (0, 1):
        return float(params["p"])
    if distribution_id == "binom":
        if params["n"] == 0 or params["p"] == 0:
            return 0.0
        if params["p"] == 1:
            return float(params["n"])
    if distribution_id == "poisson" and params["mu"] == 0:
        return 0.0
    if distribution_id == "geom" and params["p"] == 1:
        return 1.0
    if distribution_id == "nbinom" and params["p"] == 1:
        return 0.0
    if distribution_id == "hypergeom":
        lower = max(0, params["n"] - (params["M"] - params["K"]))
        upper = min(params["n"], params["K"])
        if lower == upper:
            return float(lower)
    if distribution_id == "randint" and params["low"] == params["high"]:
        return float(params["low"])
    return None


def _event_possible(distribution_id: str, params: Mapping[str, Any], event: Mapping[str, Any], rv) -> bool:
    """Does an event contain positive-mass/probability values in the support?"""
    point = _point_mass(distribution_id, params)
    if point is not None:
        return bool(event_mask([point], event)[0])
    lower, upper = rv.support()
    discrete = distribution_id in DISCRETE_IDS
    if event["kind"] == "le":
        x = math.floor(event["x"]) if discrete else event["x"]
        return x >= lower if discrete else x > lower
    if event["kind"] == "gt":
        x = math.floor(event["x"]) if discrete else event["x"]
        return x < upper
    if event["kind"] == "eq":
        return lower <= event["x"] <= upper
    if discrete:
        return max(math.ceil(event["a"]), lower) <= min(math.floor(event["b"]), upper)
    return max(event["a"], lower) < min(event["b"], upper)


def quantile(distribution_id: str, params: Mapping[str, Any], q: float) -> float:
    """Smallest discrete support value whose CDF is ≥ q; inverse CDF otherwise.

    At q=0 this API returns the lower support endpoint, not SciPy's discrete
    lower-support-minus-one convention. At q=1 it returns the upper endpoint.
    Unbounded endpoints are mathematical infinities; UI uses interior q values.
    """
    validate_event({"kind": "quantile", "q": q})
    if distribution_id == "multinomial":
        raise ValidationError("No existe un cuantil escalar conjunto de la multinomial; elegí una marginal.")
    rv = frozen_dist(distribution_id, params)
    point = _point_mass(distribution_id, params)
    if point is not None:
        return point
    if q == 0:
        return float(rv.support()[0])
    if q == 1:
        return float(rv.support()[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(rv.ppf(q))
    if not math.isfinite(value):
        raise ValidationError("El cuantil excede la precisión numérica disponible; elegí una probabilidad menos extrema.")
    return value


def _finite_moment(value: Any) -> Moment:
    value = float(value)
    # Analytic moment existence is classified separately below. Nonfinite SciPy
    # results for a mathematically finite moment indicate numerical limitations.
    if not math.isfinite(value):
        return Moment(None, "numerical_error")
    return Moment(value)


def moments(distribution_id: str, params: Mapping[str, Any]) -> dict[str, Moment]:
    """Population moments, with existence checked before invoking SciPy.

    Variance means E[(X-E[X])²]. Without a finite mean, this central moment and
    its square-root standard deviation are undefined. If the mean is finite
    but the central second moment diverges, variance/std are infinite. A
    mathematically finite result lost to floating-point overflow is classified
    as numerical_error, never as mathematical infinity.
    """
    if distribution_id == "multinomial":
        raise ValidationError("Los momentos de la multinomial son vectores; elegí una marginal para estadísticas escalares.")
    rv = frozen_dist(distribution_id, params)
    point = _point_mass(distribution_id, params)
    if point is not None:
        return {"mean": Moment(point), "median": Moment(point), "variance": Moment(0.0), "std": Moment(0.0)}
    statuses: dict[str, str] = {}
    if distribution_id == "cauchy":
        statuses = {"mean": "undefined", "variance": "undefined"}
    elif distribution_id == "t":
        if params["df"] <= 1:
            statuses = {"mean": "undefined", "variance": "undefined"}
        elif params["df"] <= 2:
            statuses = {"variance": "infinite"}
    elif distribution_id == "pareto":
        if params["alpha"] <= 1:
            statuses["mean"] = "infinite"
            statuses["variance"] = "undefined"
        elif params["alpha"] <= 2:
            statuses["variance"] = "infinite"
    elif distribution_id == "f":
        if params["dfd"] <= 2:
            statuses["mean"] = "infinite"
            statuses["variance"] = "undefined"
        elif params["dfd"] <= 4:
            statuses["variance"] = "infinite"
    elif distribution_id == "genextreme":
        if params["xi"] >= 1:
            statuses["mean"] = "infinite"
            statuses["variance"] = "undefined"
        elif params["xi"] >= 0.5:
            statuses["variance"] = "infinite"
    result: dict[str, Moment] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, function in (("mean", rv.mean), ("variance", rv.var)):
            if key in statuses:
                result[key] = Moment(None, statuses[key])
            else:
                try:
                    result[key] = _finite_moment(function())
                    if key == "variance" and result[key].value is not None and result[key].value < 0:
                        result[key] = Moment(None, "numerical_error")
                except (ValueError, OverflowError, FloatingPointError):
                    result[key] = Moment(None, "numerical_error")
    try:
        result["median"] = _finite_moment(quantile(distribution_id, params, 0.5))
    except (ValidationError, ValueError, OverflowError, FloatingPointError):
        result["median"] = Moment(None, "numerical_error")
    variance = result["variance"]
    if variance.status == "finite" and variance.value is not None and variance.value >= 0:
        result["std"] = Moment(math.sqrt(variance.value))
    elif variance.status == "finite":
        result["std"] = Moment(None, "numerical_error")
    else:
        result["std"] = Moment(None, variance.status)
    return {key: result[key] for key in ("mean", "median", "variance", "std")}


def format_moment(moment: Moment) -> str:
    labels = {
        "undefined": "No existe para estos parámetros",
        "infinite": "Infinita",
        "numerical_error": "No disponible por precisión numérica",
    }
    if moment.status != "finite":
        return labels.get(moment.status, "No disponible por precisión numérica")
    if moment.value is None or not math.isfinite(moment.value):
        return labels["numerical_error"]
    return f"{moment.value:.6g}"


def multinomial_joint(n: int, probabilities: Sequence[float], counts: Sequence[int]) -> float:
    return _checked_probability(math.exp(multinomial_logjoint(n, probabilities, counts)))


def multinomial_logjoint(n: int, probabilities: Sequence[float], counts: Sequence[int]) -> float:
    validate_multinomial(n, probabilities, counts=counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = stats.multinomial.logpmf(counts, int(n), probabilities)
    value = float(value)
    if math.isnan(value) or value == math.inf or value > 1e-14:
        raise ValidationError("No se pudo calcular la probabilidad conjunta con precisión numérica suficiente.")
    return value


def event_mask(x: Any, event: Mapping[str, Any]) -> np.ndarray:
    """Mask the same mathematical event computed by probability()."""
    values = np.asarray(x)
    kind = event.get("kind")
    if kind == "le":
        return values <= event["x"]
    if kind == "gt":
        return values > event["x"]
    if kind == "interval":
        return (values >= event["a"]) & (values <= event["b"])
    if kind == "eq":
        return values == event["x"]
    raise ValidationError("El gráfico necesita un evento con límites explícitos.")
