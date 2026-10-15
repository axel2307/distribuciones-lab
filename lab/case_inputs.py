"""Sample summaries that determine the reference law and observed statistic.

The summaries are illustrative inputs, not fitted data. The numerical engine
remains responsible for validating and calculating probabilities.
"""
from copy import deepcopy
import math

from .calculations import ValidationError

SAMPLE_DEFAULTS = {
    "t": {"n": 16, "mean": 52.0, "std": 4.0, "null": 50.0},
    "chi2": {"n": 10, "variance": 1.6, "null_variance": 1.0},
    "f": {"n1": 10, "n2": 12, "variance1": 8.0, "variance2": 4.0},
}


def sample_defaults(distribution_id):
    return deepcopy(SAMPLE_DEFAULTS[distribution_id])


def sample_model(distribution_id, samples, tail="gt"):
    """Derive degrees of freedom and an event at the observed statistic."""
    if tail not in {"gt", "le"}:
        raise ValidationError("Elegí una cola derecha o izquierda del estadístico observado.")
    if distribution_id not in SAMPLE_DEFAULTS:
        raise ValidationError("Este caso no utiliza un resumen muestral.")
    for key in SAMPLE_DEFAULTS[distribution_id]:
        value = samples.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError("Los datos muestrales deben ser números finitos.")
        if key in {"n", "n1", "n2"} and (int(value) != value or value < 2):
            raise ValidationError("Cada tamaño muestral debe ser un entero de al menos 2 mediciones.")
    if distribution_id == "t":
        if samples["std"] <= 0:
            raise ValidationError("El desvío muestral debe ser positivo para calcular t.")
        params = {"df": float(samples["n"] - 1)}
        standard_error = samples["std"] / math.sqrt(samples["n"])
        if standard_error == 0:
            raise ValidationError("El error estándar es demasiado pequeño para la precisión numérica disponible.")
        observed = (samples["mean"] - samples["null"]) / standard_error
    elif distribution_id == "chi2":
        if samples["variance"] < 0 or samples["null_variance"] <= 0:
            raise ValidationError("La varianza muestral debe ser no negativa y la varianza de H₀ positiva.")
        params = {"df": float(samples["n"] - 1)}
        observed = (samples["n"] - 1) * samples["variance"] / samples["null_variance"]
    else:
        if samples["variance1"] < 0 or samples["variance2"] <= 0:
            raise ValidationError("La primera varianza debe ser no negativa y la segunda positiva para formar el cociente.")
        params = {"dfn": float(samples["n1"] - 1), "dfd": float(samples["n2"] - 1)}
        observed = samples["variance1"] / samples["variance2"]
    if not math.isfinite(observed):
        raise ValidationError("El estadístico excede la precisión numérica disponible.")
    return params, {"kind": tail, "x": float(observed)}
