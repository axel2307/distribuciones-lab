"""Worked-case interpretations derived from the current model and event."""
import math
from collections.abc import Mapping

from .catalog import REGISTRY
from .calculations import (ValidationError, validate_parameters, validate_event, probability, log_probability,
                           quantile, multinomial_logjoint)


def _number(value):
    return f"{float(value):.8g}".replace(".", ",")


def _model_description(distribution_id, params):
    spec = REGISTRY[distribution_id]
    entries = [f"{p.name} ({p.symbol}) = {_number(params[p.key])}" for p in spec.parameters]
    if distribution_id == "multinomial":
        entries += [f"Probabilidad de «{name}» = {_number(p)}"
                    for name, p in zip(params["names"], params["probabilities"])]
    description = f"Modelo actual: {spec.name}. " + "; ".join(entries) + "."
    if distribution_id == "binom":
        description += (f" X cuenta éxitos en {params['n']} ensayos, con probabilidad de éxito "
                        f"{_number(params['p'])} por ensayo.")
    elif distribution_id == "nbinom":
        description += (f" X cuenta fracasos antes de alcanzar {params['r']} éxitos; "
                        f"el total de intentos es T = X + {params['r']}.")
    return description


def case_interpretation(distribution_id, params, event):
    """Return a current interpretation, or raise the engine's ValidationError.

    Historical example prose is never reused here. Both the numerical result
    and the event description use the same live parameters as the calculator.
    """
    validate_parameters(distribution_id, params)
    if distribution_id == "multinomial":
        if not isinstance(event, Mapping) or event.get("kind") != "joint":
            raise ValidationError("La multinomial requiere un evento conjunto con un vector de conteos.")
        # This also validates vector length, integer counts and their sum.
        joint_value = probability(distribution_id, params, event)
    else:
        validate_event(event, discrete=REGISTRY[distribution_id].kind == "discrete")
    model = _model_description(distribution_id, params)
    kind = event["kind"]
    if kind == "quantile":
        x = quantile(distribution_id, params, event["q"])
        if not math.isfinite(x):
            end = "inferior" if x < 0 else "superior"
            return (f"{model} Para q = {_number(event['q'])}, el cuantil es el extremo "
                    f"{end} infinito del soporte: no hay un umbral finito que mostrar.")
        achieved = probability(distribution_id, params, {"kind":"le", "x":x})
        return (f"{model} Para la probabilidad acumulada objetivo q = {_number(event['q'])}, "
                f"el cuantil es {_number(x)}. La probabilidad lograda P(X ≤ {_number(x)}) "
                f"es {_number(achieved)} ({_number(100*achieved)} %). "
                "La probabilidad lograda corresponde al umbral representable; puede diferir "
                "de q por los saltos de una variable discreta o por precisión numérica.")
    if kind == "joint":
        value = joint_value
        log_value = multinomial_logjoint(params["n"], params["probabilities"], event["counts"])
        counts = ", ".join(f"«{name}»: {k}" for name,k in zip(params["names"],event["counts"]))
        description = (f"el vector exacto de conteos ({counts}), en {params['n']} ensayos. "
                       "Se incluyen todos los órdenes de llegada que producen este vector; "
                       "no se multiplican probabilidades marginales")
    else:
        value = probability(distribution_id, params, event)
        log_value = log_probability(distribution_id, params, event)
        if kind == "interval":
            description = f"que X esté entre {_number(event['a'])} y {_number(event['b'])} (ambos incluidos)"
        elif kind == "le":
            description = f"que X sea como máximo {_number(event['x'])} (incluido)"
        elif kind == "gt":
            description = f"que X sea mayor que {_number(event['x'])} (excluido)"
        else:
            description = f"que X sea exactamente {_number(event['x'])}"
    lead = f"{model} El evento actual es {description}. "
    if value == 0 and math.isfinite(log_value):
        return (lead + "La probabilidad es positiva, pero no puede representarse en precisión doble. "
                f"Su logaritmo natural es {_number(log_value)}; no es un cero exacto.")
    if value == 0:
        return lead + "Bajo estos supuestos, este evento tiene probabilidad cero."
    return (lead + f"La probabilidad es {_number(value)} ({_number(100*value)} %). "
            f"Al repetir este modelo, aproximadamente {_number(100*value)} de cada 100 casos "
            "cumplirían este evento. Es una frecuencia esperada, no una garantía para una muestra de 100.")
