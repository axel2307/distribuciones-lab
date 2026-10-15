"""Short numerical derivations for the live worked cases.

This is a presentation layer: the existing calculation engine remains the
source of every probability, logarithm and quantile.  In particular, sums and
integrals below explain the event; they do not replace its robust evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .calculations import (
    DISCRETE_IDS, ValidationError, frozen_dist, log_probability, probability,
    quantile, validate_event, validate_parameters,
)
from .catalog import REGISTRY


def _n(value):
    """Compact LaTeX number; scientific notation never becomes a rounded zero."""
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isinf(value):
        return r"-\infty" if value < 0 else r"\infty"
    if math.isnan(value):
        raise ValidationError("No se puede mostrar un resultado numérico indeterminado.")
    text = f"{value:.8g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def _given(value):
    """Keep distinct input bounds distinct, even in very narrow intervals."""
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return _n(value)
    text = repr(value).removesuffix(".0")
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def _step(title, explanation, latex):
    return {"title": title, "explanation": explanation, "latex": latex}


def _event(event, variable="X"):
    kind = event["kind"]
    if kind == "interval":
        return rf"P({_given(event['a'])}\le {variable}\le {_given(event['b'])})"
    symbol = {"le": r"\le", "gt": ">", "eq": "="}[kind]
    return rf"P({variable}{symbol}{_given(event['x'])})"


def _result(distribution_id, params, event, value, log_value, *, samples=False):
    variable = {"t": "T", "chi2": "Q", "f": "R"}.get(distribution_id, "X") if samples else "X"
    expression = _event(event, variable)
    if value == 0 and math.isfinite(log_value):
        return _step(
            "Conservar una probabilidad muy pequeña",
            "El evento es posible. Su probabilidad positiva es demasiado pequeña para "
            "representarla como un decimal de precisión doble; se conserva su logaritmo.",
            rf"\log {expression}\approx {_n(log_value)}\quad\Longrightarrow\quad"
            rf"{expression}=e^{{\log P(E)}}>0",
        )
    if value == 0:
        return _step("Leer el resultado", "El evento no contiene resultados con probabilidad positiva bajo este modelo.",
                     expression + "=0")
    explanation = "La probabilidad del evento es el área sombreada o la suma de las barras seleccionadas."
    if samples:
        explanation = ("Bajo H₀, esta es la probabilidad de la región indicada al repetir muestras "
                       "completas con los mismos tamaños. No es la probabilidad de que H₀ sea verdadera.")
    return _step("Leer el resultado", explanation,
                 rf"{expression}\approx {_n(value)}={_n(100 * value)}\,\%")


def _mass(distribution_id, p, k="k"):
    """PMF with current parameters substituted, on the distribution's support."""
    if distribution_id == "bernoulli":
        return rf"({_given(p['p'])})^{{{k}}}(1-{_given(p['p'])})^{{1-{k}}}"
    if distribution_id == "binom":
        return (rf"\binom{{{_given(p['n'])}}}{{{k}}}({_given(p['p'])})^{{{k}}}"
                rf"(1-{_given(p['p'])})^{{{_given(p['n'])}-{k}}}")
    if distribution_id == "poisson":
        return rf"e^{{-{_given(p['mu'])}}}\frac{{({_given(p['mu'])})^{{{k}}}}}{{{k}!}}"
    if distribution_id == "geom":
        return rf"(1-{_given(p['p'])})^{{{k}-1}}\,({_given(p['p'])})"
    if distribution_id == "nbinom":
        return (rf"\binom{{{k}+{_given(p['r'])}-1}}{{{_given(p['r'])}-1}}"
                rf"({_given(p['p'])})^{{{_given(p['r'])}}}(1-{_given(p['p'])})^{{{k}}}")
    if distribution_id == "hypergeom":
        return (rf"\frac{{\binom{{{_n(p['K'])}}}{{{k}}}"
                rf"\binom{{{_n(p['M'] - p['K'])}}}{{{_n(p['n'])}-{k}}}}}"
                rf"{{\binom{{{_n(p['M'])}}}{{{_n(p['n'])}}}}}")
    return rf"\frac{{1}}{{{_n(p['high'])}-{_n(p['low'])}+1}}"


def _mass_setup(distribution_id, params):
    explanations = {
        "bernoulli": "Un ensayo tiene dos resultados: 1 (éxito) y 0 (fracaso). Sustituimos la probabilidad de éxito.",
        "binom": "El coeficiente cuenta las posiciones de los éxitos; las potencias dan la probabilidad de cada orden.",
        "poisson": "La media corresponde al intervalo completo de observación. Cada k es un número entero de llegadas.",
        "geom": "El primer éxito ocurre en el intento k: antes hubo exactamente k − 1 fracasos.",
        "nbinom": "El último intento es el éxito r. Antes se distribuyen r − 1 éxitos y k fracasos; hay k + r intentos en total.",
        "hypergeom": "Contamos extracciones sin reemplazo: elegimos k éxitos y n − k elementos del resto de la población.",
        "randint": "Contamos los enteros posibles incluyendo ambos extremos. Cada uno tiene la misma probabilidad.",
    }
    note = " En estas masas, una potencia 0⁰ aporta 1." if (
        params.get("p") in (0, 1) or distribution_id == "poisson" and params["mu"] == 0
    ) else ""
    return _step("Sustituir en la probabilidad de un resultado",
                 explanations[distribution_id] + note + " Fuera del soporte la masa es cero.",
                 rf"m(k)=P(X=k)={_mass(distribution_id, params)}")


def _coefficient(n, counts):
    """Show small exact coefficients without expanding enormous factorials."""
    n, counts = int(n), [int(k) for k in counts]
    if n <= 1000:
        remaining, result = n, 1
        for k in counts:
            result *= math.comb(remaining, k)
            remaining -= k
        if result.bit_length() <= 128:
            return "=" + str(result)
    # A symbolic coefficient is more useful than hundreds of factorial digits.
    return ""


def _discrete_event(distribution_id, params, event, rv):
    kind = event["kind"]
    lo, hi = rv.support()
    if kind == "eq":
        k = int(event["x"])
        if k < lo or k > hi:
            return _step("Comprobar si el resultado es posible", "El valor pedido está fuera del soporte.",
                         _event(event) + "=0")
        latex = rf"P(X={_n(k)})={_mass(distribution_id, params, _n(k))}"
        if distribution_id == "binom":
            coefficient = _coefficient(params["n"], [k, int(params["n"]) - k])
            if coefficient:
                latex = (rf"\binom{{{_n(params['n'])}}}{{{_n(k)}}}{coefficient},\qquad " + latex)
        return _step("Elegir el resultado exacto", "Sustituimos k por el conteo pedido, sin sumar otros resultados.", latex)
    if kind == "interval":
        lower, upper = math.ceil(event["a"]), math.floor(event["b"])
        if lower > upper:
            return _step("Identificar los enteros incluidos", "Entre estos límites no hay ningún entero; el evento es vacío.",
                         rf"\lceil {_given(event['a'])}\rceil={_n(lower)}>{_n(upper)}=\lfloor {_given(event['b'])}\rfloor")
        lower = max(lower, int(lo)) if math.isfinite(lo) else lower
        upper = min(upper, int(hi)) if math.isfinite(hi) else upper
        if lower > upper:
            return _step("Comprobar el soporte", "El intervalo no contiene resultados posibles de esta distribución.",
                         _event(event) + "=0")
        return _step("Sumar sólo los enteros del evento",
                     "Redondeamos el límite inferior hacia arriba y el superior hacia abajo. Los dos enteros se incluyen; se omiten valores imposibles.",
                     rf"{_event(event)}=\sum_{{k={_n(lower)}}}^{{{_n(upper)}}}m(k)"
                     rf"=F_X({_n(upper)})-F_X({_n(lower - 1)})")
    cutoff = math.floor(event["x"])
    if (kind == "le" and cutoff < lo) or (kind == "gt" and cutoff >= hi):
        return _step("Comprobar el soporte", "El límite deja todos los resultados posibles fuera del evento.",
                     _event(event) + "=0")
    if (kind == "le" and cutoff >= hi) or (kind == "gt" and cutoff < lo):
        return _step("Comprobar el soporte", "El evento incluye todos los resultados posibles del modelo.",
                     _event(event) + "=1")
    if kind == "le":
        upper = min(cutoff, int(hi)) if math.isfinite(hi) else cutoff
        lower = int(lo) if math.isfinite(lo) else -math.inf
        summation = rf"\sum_{{k={_n(lower)}}}^{{{_n(upper)}}}m(k)"
        explanation = "Como X es entero, 'como máximo x' incluye hasta el entero inferior a x. Sumamos las masas de esos resultados."
        if distribution_id == "geom" and cutoff >= 1:
            summation = rf"1-(1-{_given(params['p'])})^{{{_n(cutoff)}}}"
            explanation = "El complemento de lograr el primer éxito hasta este intento es fracasar en todos esos intentos."
        return _step("Acumular hasta el límite incluido", explanation,
                     rf"{_event(event)}={summation}=F_X({_n(cutoff)})")
    lower = max(cutoff + 1, int(lo)) if math.isfinite(lo) else cutoff + 1
    upper = int(hi) if math.isfinite(hi) else math.inf
    explanation = ("'Más de x' empieza en el entero siguiente a su parte entera. "
                   "La supervivencia SF evalúa esta cola directamente, sin restar un acumulado casi igual a 1.")
    return _step("Seleccionar la cola derecha", explanation,
                 rf"{_event(event)}=\sum_{{k={_n(lower)}}}^{{{_n(upper)}}}m(k)"
                 rf"=\operatorname{{SF}}_X({_n(cutoff)})")


def _density(distribution_id, p):
    """Density with the model's numerical parameters, restricted to its support."""
    v = {parameter.key: _given(p[parameter.key]) for parameter in REGISTRY[distribution_id].parameters}
    if distribution_id == "norm":
        return (rf"\frac{{1}}{{{v['sigma']}\sqrt{{2\pi}}}}"
                rf"\exp\!\left[-\frac{{(u-({v['mu']}))^2}}{{2({v['sigma']})^2}}\right]")
    if distribution_id == "uniform":
        return rf"\frac{{1}}{{{v['b']}-({v['a']})}}\quad({_given(p['a'])}\le u\le {_given(p['b'])})"
    if distribution_id == "expon":
        return rf"{v['rate']}\,e^{{-({v['rate']})u}}"
    if distribution_id == "gamma":
        return rf"\frac{{u^{{{v['alpha']}-1}}e^{{-u/{v['theta']}}}}}{{\Gamma({v['alpha']})({v['theta']})^{{{v['alpha']}}}}}"
    if distribution_id == "beta":
        return rf"\frac{{u^{{{v['alpha']}-1}}(1-u)^{{{v['beta']}-1}}}}{{B({v['alpha']},{v['beta']})}}"
    if distribution_id == "lognorm":
        return (rf"\frac{{1}}{{u\,{v['sigma_log']}\sqrt{{2\pi}}}}"
                rf"\exp\!\left[-\frac{{(\ln u-({v['mu_log']}))^2}}{{2({v['sigma_log']})^2}}\right]")
    if distribution_id == "t":
        return (rf"\frac{{\Gamma(({v['df']}+1)/2)}}{{\sqrt{{{v['df']}\pi}}\,\Gamma({v['df']}/2)}}"
                rf"\left(1+\frac{{u^2}}{{{v['df']}}}\right)^{{-({v['df']}+1)/2}}")
    if distribution_id == "chi2":
        return rf"\frac{{u^{{{v['df']}/2-1}}e^{{-u/2}}}}{{2^{{{v['df']}/2}}\Gamma({v['df']}/2)}}"
    if distribution_id == "f":
        return (rf"\frac{{({v['dfn']}/{v['dfd']})^{{{v['dfn']}/2}}u^{{{v['dfn']}/2-1}}}}"
                rf"{{B({v['dfn']}/2,{v['dfd']}/2)(1+{v['dfn']}u/{v['dfd']})^{{({v['dfn']}+{v['dfd']})/2}}}}")
    if distribution_id == "weibull_min":
        return (rf"\frac{{{v['shape']}}}{{{v['scale']}}}\left(\frac{{u}}{{{v['scale']}}}\right)^{{{v['shape']}-1}}"
                rf"\exp\!\left[-\left(\frac{{u}}{{{v['scale']}}}\right)^{{{v['shape']}}}\right]")
    if distribution_id == "pareto":
        return rf"\frac{{{v['alpha']}({v['xmin']})^{{{v['alpha']}}}}}{{u^{{{v['alpha']}+1}}}}"
    if distribution_id == "cauchy":
        return rf"\frac{{1}}{{\pi\,{v['gamma']}\left[1+\left((u-({v['x0']}))/{v['gamma']}\right)^2\right]}}"
    if distribution_id == "laplace":
        return rf"\frac{{1}}{{2({v['b']})}}\exp\!\left[-\frac{{|u-({v['mu']})|}}{{{v['b']}}}\right]"
    if distribution_id == "rayleigh":
        return rf"\frac{{u}}{{({v['sigma']})^2}}\exp\!\left[-\frac{{u^2}}{{2({v['sigma']})^2}}\right]"
    scale = p["beta"] if distribution_id == "gumbel_r" else p["sigma"]
    z = rf"z=\frac{{u-({_given(p['mu'])})}}{{{_given(scale)}}}"
    if distribution_id == "gumbel_r" or p["xi"] == 0:
        return rf"\frac{{1}}{{{_given(scale)}}}e^{{-z-e^{{-z}}}},\quad {z}"
    xi = v["xi"]
    return (rf"\frac{{1}}{{{v['sigma']}}}(1+({xi})z)^{{-1/({xi})-1}}"
            rf"e^{{-(1+({xi})z)^{{-1/({xi})}}}},\quad {z},\quad 1+({xi})z>0")


def _density_setup(distribution_id, params):
    details = {
        "norm": "La desviación σ describe mediciones individuales, en las unidades originales; no es el error estándar de una media.",
        "uniform": "La densidad es constante entre los dos extremos y cero fuera: la probabilidad es longitud relativa.",
        "expon": "La tasa λ está en eventos por unidad de tiempo; la espera media es 1/λ.",
        "gamma": "La escala θ está en unidades de tiempo y no es una tasa. Con forma entera, X puede ser una espera de varias llegadas.",
        "beta": "La variable es una proporción desconocida entre 0 y 1. Integramos incertidumbre sobre esa proporción.",
        "lognorm": "Los parámetros describen ln(X), no X. El modelo sólo permite mediciones positivas.",
        "t": "La t central modela un estadístico sin unidades bajo la hipótesis nula, con los grados de libertad elegidos.",
        "chi2": "La chi-cuadrado central modela un estadístico no negativo y sin unidades, con los grados de libertad elegidos.",
        "f": "La F central describe un cociente sin unidades; importa el orden de los grados del numerador y denominador.",
        "weibull_min": "La escala η está en unidades de vida útil: F(η) = 1 − exp(−1), aproximadamente 63,2 %.",
        "pareto": "El modelo empieza en el mínimo xₘ. La densidad es cero por debajo de ese valor.",
        "cauchy": "El centro es una mediana y la escala no es una desviación estándar: la media y la varianza no existen.",
        "laplace": "La escala b controla la caída de la densidad a ambos lados del centro.",
        "rayleigh": "σ es la desviación de cada componente normal, no la desviación de la magnitud X.",
        "gumbel_r": "La ubicación y la escala describen máximos por bloque; la ubicación no coincide con su media.",
        "genextreme": "La forma ξ fija la cola y el soporte: 1 + ξz > 0. Para ξ = 0 se usa el límite de Gumbel.",
    }
    return _step("Sustituir los parámetros del modelo", details[distribution_id] +
                 " La densidad se integra; su altura no es una probabilidad.",
                 rf"f_X(u)={_density(distribution_id, params)}")


def _transform(distribution_id, p, event):
    """A small scale conversion when it materially helps read the event."""
    if event["kind"] == "quantile":
        return None
    bounds = [event["a"], event["b"]] if event["kind"] == "interval" else [event["x"]]
    entries = []
    for x in bounds:
        try:
            if distribution_id == "norm":
                value = (x - p["mu"]) / p["sigma"]
                formula = rf"\frac{{{_given(x)}-({_given(p['mu'])})}}{{{_given(p['sigma'])}}}"
            elif distribution_id == "lognorm" and x > 0:
                value = (math.log(x) - p["mu_log"]) / p["sigma_log"]
                formula = rf"\frac{{\ln({_given(x)})-({_given(p['mu_log'])})}}{{{_given(p['sigma_log'])}}}"
            elif distribution_id == "expon":
                value = p["rate"] * x
                formula = rf"({_given(p['rate'])})({_given(x)})"
            elif distribution_id == "gamma":
                value = x / p["theta"]
                formula = rf"\frac{{{_given(x)}}}{{{_given(p['theta'])}}}"
            elif distribution_id in {"gumbel_r", "genextreme"}:
                scale = p["beta"] if distribution_id == "gumbel_r" else p["sigma"]
                value = (x - p["mu"]) / scale
                formula = rf"\frac{{{_given(x)}-({_given(p['mu'])})}}{{{_given(scale)}}}"
            else:
                return None
            if not math.isfinite(value):
                return None  # Never label numerical overflow as a mathematical infinity.
        except (OverflowError, ValueError):
            return None
        label = "z" if distribution_id in {"norm", "lognorm", "gumbel_r", "genextreme"} else "y"
        entries.append(rf"{label}({_given(x)})={formula}\approx {_n(value)}")
    meaning = {
        "norm": "Restamos la media y dividimos por σ: el umbral queda expresado en desviaciones estándar de una medición individual.",
        "lognorm": "Tomamos el logaritmo natural y lo estandarizamos. En esta escala Z sigue una normal estándar.",
        "expon": "Multiplicar la espera por la tasa produce Y = λX, una exponencial de tasa 1.",
        "gamma": "Dividir por la escala produce Y = X/θ, una gamma con la misma forma y escala 1.",
        "gumbel_r": "Restamos la ubicación y dividimos por la escala para expresar el máximo en la escala de la curva.",
        "genextreme": "Expresamos el máximo en la escala z; el motor también comprueba el soporte impuesto por ξ.",
    }
    variable = label.upper()
    if event["kind"] == "interval":
        converted_expression = (rf"P({label}({_given(event['a'])})\le {variable}"
                                rf"\le {label}({_given(event['b'])}))")
    else:
        relation = r"\le" if event["kind"] == "le" else ">"
        converted_expression = rf"P({variable}{relation}{label}({_given(event['x'])}))"
    equivalence = _event(event) + "=" + converted_expression
    return _step("Ubicar el umbral en la escala del modelo", meaning[distribution_id],
                 r"\begin{gathered}" + r"\qquad ".join(entries) + r"\\" + equivalence + r"\end{gathered}")


def _area(distribution_id, params, event, variable="X", null=False):
    condition = r"\mid H_0" if null else ""
    kind = event["kind"]
    x = _given(event.get("x", 0))
    density = rf"f_{{{variable}}}(u)"
    if kind == "le":
        formula = (rf"P({variable}\le {x}{condition})=\int_{{-\infty}}^{{{x}}}{density}\,du"
                   rf"=F_{{{variable}}}({x})")
        explanation = "Acumulamos el área a la izquierda del umbral, contando sólo el soporte del modelo."
    elif kind == "gt":
        formula = (rf"P({variable}>{x}{condition})=\int_{{{x}}}^{{\infty}}{density}\,du"
                   rf"=\operatorname{{SF}}_{{{variable}}}({x})")
        explanation = "Tomamos el área a la derecha del umbral. Se evalúa con supervivencia (SF) para conservar precisión en colas pequeñas."
    else:
        a, b = _given(event["a"]), _given(event["b"])
        # Use the same stable identity as the engine. Do not show 1 - 1 for a
        # positive interval in an extreme right tail.
        left = probability(distribution_id, params, {"kind": "le", "x": event["a"]})
        right = probability(distribution_id, params, {"kind": "le", "x": event["b"]})
        if left >= .5:
            difference = rf"\operatorname{{SF}}_{{{variable}}}({a})-\operatorname{{SF}}_{{{variable}}}({b})"
        else:
            difference = rf"F_{{{variable}}}({b})-F_{{{variable}}}({a})"
        formula = (rf"P({a}\le {variable}\le {b}{condition})=\int_{{{a}}}^{{{b}}}{density}\,du={difference}")
        explanation = ("Integramos entre los dos límites. En la cola derecha restamos supervivencias; "
                       "si hace falta, el motor usa sus logaritmos para evitar cancelación.")
        if left == right and event["a"] != event["b"]:
            explanation += " Los acumulados redondeados pueden coincidir aunque el área sea positiva."
    return _step("Seleccionar el área del evento", explanation, formula)


SIMPLE_CONTINUOUS_IDS = frozenset({"uniform", "expon", "weibull_min", "rayleigh"})


def _simple_setup(distribution_id, p):
    """Use lengths or survival, rather than a density, for elementary cases."""
    if distribution_id == "uniform":
        return _step("Medir el intervalo completo",
                     "En una uniforme continua, intervalos de igual longitud tienen la misma probabilidad. "
                     "Primero identificamos toda la longitud disponible.",
                     rf"X\in[{_given(p['a'])},{_given(p['b'])}],\qquad "
                     rf"L_{{total}}=b-a={_given(p['b'])}-({_given(p['a'])})")
    if distribution_id == "expon":
        return _step("Relacionar espera y tasa",
                     "La tasa cuenta eventos por unidad de tiempo. La supervivencia es la probabilidad "
                     "de que la espera todavía no termine; el tiempo sólo puede ser no negativo.",
                     rf"\lambda={_given(p['rate'])},\qquad S(x)=P(X>x)=e^{{-\lambda x}},\quad x\ge0")
    if distribution_id == "weibull_min":
        return _step("Relacionar vida útil, forma y escala",
                     "La escala η está en unidades de vida útil. Al llegar a η ya ocurrió aproximadamente "
                     "el 63,2 % de las fallas; la forma k cambia cómo crece el riesgo.",
                     rf"k={_given(p['shape'])},\quad\eta={_given(p['scale'])},\qquad "
                     rf"P(X\le\eta)=1-e^{{-1}}\approx0.63212056")
    return _step("Relacionar magnitud y componentes",
                 "σ es la desviación de cada componente normal independiente. La magnitud X es no negativa; "
                 "su supervivencia da la probabilidad de superar el umbral.",
                 rf"\sigma={_given(p['sigma'])},\qquad S(x)=P(X>x)="
                 rf"e^{{-x^2/(2\sigma^2)}},\quad x\ge0")


def _simple_exponent(distribution_id, p, x):
    """A substituted survival exponent and, when representable, its value."""
    if distribution_id == "expon":
        formula = rf"({_given(p['rate'])})({_given(x)})"
        operation = lambda: float(p["rate"]) * float(x)
    elif distribution_id == "weibull_min":
        formula = rf"\left(\frac{{{_given(x)}}}{{{_given(p['scale'])}}}\right)^{{{_given(p['shape'])}}}"
        operation = lambda: (float(x) / float(p["scale"])) ** float(p["shape"])
    else:
        formula = rf"\frac{{({_given(x)})^2}}{{2({_given(p['sigma'])})^2}}"
        operation = lambda: (float(x) / float(p["sigma"])) ** 2 / 2
    try:
        value = operation()
    except OverflowError:
        value = None
    if value is not None and not math.isfinite(value):
        value = None
    return formula, value


def _simple_event(distribution_id, p, event):
    # This is a symbolic derivation only; probability/log_probability have
    # already validated and evaluated the original, unchanged event.
    kind = event["kind"]
    if distribution_id == "uniform":
        if kind == "le":
            lower, upper = p["a"], min(p["b"], event["x"])
        elif kind == "gt":
            lower, upper = max(p["a"], event["x"]), p["b"]
        else:
            lower, upper = max(p["a"], event["a"]), min(p["b"], event["b"])
        if upper <= lower:
            return _step("Medir la parte favorable",
                         "El evento no contiene un intervalo de longitud positiva dentro del soporte.",
                         rf"L_E=0,\qquad {_event(event)}=0")
        lower, upper = max(lower, p["a"]), min(upper, p["b"])
        favorable, total = upper - lower, p["b"] - p["a"]
        substituted = rf"\frac{{{_given(upper)}-({_given(lower)})}}{{{_given(p['b'])}-({_given(p['a'])})}}"
        # Do not turn overflow in an optional displayed subtraction into a
        # mathematical infinity. The engine retains responsibility for errors.
        if math.isfinite(favorable) and math.isfinite(total):
            substituted += rf"\approx\frac{{{_n(favorable)}}}{{{_n(total)}}}"
        return _step("Comparar longitud favorable y longitud total",
                     "Tomamos sólo la parte del evento dentro del soporte. Dividimos su longitud "
                     "por la longitud total; las unidades se cancelan.",
                     rf"{_event(event)}=\frac{{L_E}}{{L_{{total}}}}={substituted}")
    if kind != "interval" and event["x"] <= 0:
        value = 0 if kind == "le" else 1
        return _step("Comprobar el soporte",
                     "X es no negativa y continua: no hay área por debajo de cero ni masa en el punto cero. "
                     "Resolvemos el evento con el soporte antes de usar potencias o exponentes.",
                     _event(event) + f"={value}")
    if kind == "interval":
        lower, upper = max(0., event["a"]), event["b"]
        if upper <= lower:
            return _step("Comprobar el soporte",
                         "El intervalo no contiene una región de longitud positiva dentro del soporte no negativo.",
                         _event(event) + "=0")
        lower_formula, lower_value = _simple_exponent(distribution_id, p, lower)
        upper_formula, upper_value = _simple_exponent(distribution_id, p, upper)
        formula = (rf"{_event(event)}=S({_given(lower)})-S({_given(upper)})"
                   rf"=e^{{-({lower_formula})}}-e^{{-({upper_formula})}}")
        # Never display rounded exponents as identical terms of a positive
        # difference. The unrounded input substitutions above stay distinct.
        if (lower_value is not None and upper_value is not None
                and _n(lower_value) != _n(upper_value)):
            formula += rf"\approx e^{{-{_n(lower_value)}}}-e^{{-{_n(upper_value)}}}"
        return _step("Restar las supervivencias de los dos límites",
                     "Superar el límite inferior incluye dos grupos: caer dentro del intervalo o superar "
                     "también el superior. Restamos el segundo grupo. Los límites fuera del soporte "
                     "no aportan área y el motor conserva precisión cuando la diferencia es muy pequeña.", formula)
    exponent, value = _simple_exponent(distribution_id, p, event["x"])
    tail = rf"e^{{-({exponent})}}"
    if value is not None:
        short_tail = rf"e^{{-{_n(value)}}}"
    else:
        short_tail = None
    if kind == "gt":
        quantity = {"expon": "la espera", "weibull_min": "la vida útil", "rayleigh": "la magnitud"}[distribution_id]
        explanation = f"Queremos que {quantity} supere el umbral: calculamos directamente la supervivencia."
        title = "Calcular la probabilidad de superar el umbral"
        formula = _event(event) + "=" + tail
        if short_tail is not None:
            formula += r"\approx " + short_tail
    else:
        explanation = "Tener un valor como máximo igual al umbral es el complemento de superarlo. Restamos esa supervivencia de 1; el motor conserva precisión incluso para probabilidades muy pequeñas."
        title = "Calcular el complemento de superar el umbral"
        formula = _event(event) + "=1-" + tail
        if short_tail is not None:
            formula += r"\approx 1-" + short_tail
    return _step(title, explanation, formula)


def _simple_quantile(distribution_id, p, q):
    """Inverse survival algebra; the numerical answer still comes from ppf."""
    if distribution_id == "uniform":
        return (rf"a+q(b-a)={_given(p['a'])}+({_given(q)})"
                rf"({_given(p['b'])}-({_given(p['a'])}))")
    if distribution_id == "expon":
        return rf"\frac{{-\ln(1-q)}}{{\lambda}}=\frac{{-\ln(1-{_given(q)})}}{{{_given(p['rate'])}}}"
    if distribution_id == "weibull_min":
        return (rf"\eta[-\ln(1-q)]^{{1/k}}={_given(p['scale'])}"
                rf"[-\ln(1-{_given(q)})]^{{1/({_given(p['shape'])})}}")
    return (rf"\sigma\sqrt{{-2\ln(1-q)}}={_given(p['sigma'])}"
            rf"\sqrt{{-2\ln(1-{_given(q)})}}")


def _sample_number(samples, key, *, positive=False, nonnegative=False, integer=False):
    try:
        raw = samples[key]
        if isinstance(raw, (bool, str, bytes)):
            raise ValueError
        value = float(raw)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"El dato muestral {key} debe ser un número real finito.") from exc
    if not math.isfinite(value) or positive and value <= 0 or nonnegative and value < 0:
        raise ValidationError(f"El dato muestral {key} no pertenece al dominio permitido.")
    if integer and (not value.is_integer() or value < 2):
        raise ValidationError(f"El tamaño muestral {key} debe ser un entero de al menos 2.")
    return int(value) if integer else value


def _sample_steps(distribution_id, params, samples):
    if not isinstance(samples, Mapping):
        raise ValidationError("Los datos muestrales deben indicarse por nombre.")
    if distribution_id == "t":
        n = _sample_number(samples, "n", integer=True)
        mean, null = (_sample_number(samples, key) for key in ("mean", "null"))
        std = _sample_number(samples, "std", positive=True)
        se = std / math.sqrt(n)
        if se == 0 or not math.isfinite(se):
            raise ValidationError("El error estándar excede la precisión numérica disponible.")
        statistic = (mean - null) / se
        expected = {"df": n - 1}
        steps = [
            _step("Pasar de dispersión individual a error estándar",
                  "s es la desviación de las mediciones de la muestra; s/√n estima la dispersión de su media. Conserva las unidades de las mediciones.",
                  rf"\widehat{{\operatorname{{SE}}}}(\bar x)=\frac{{s}}{{\sqrt n}}"
                  rf"=\frac{{{_given(std)}}}{{\sqrt{{{n}}}}}\approx {_n(se)}"),
            _step("Comparar la media con la referencia de H₀",
                  "El cociente queda sin unidades. Para mediciones independientes de una población normal y σ desconocida, bajo H₀ sigue una t central.",
                  rf"t_{{obs}}=\frac{{\bar x-\mu_0}}{{s/\sqrt n}}"
                  rf"=\frac{{{_given(mean)}-({_given(null)})}}{{{_given(se)}}}\approx {_n(statistic)},"
                  rf"\qquad\nu={n}-1={n-1}"),
        ]
    elif distribution_id == "chi2":
        n = _sample_number(samples, "n", integer=True)
        variance = _sample_number(samples, "variance", nonnegative=True)
        null_variance = _sample_number(samples, "null_variance", positive=True)
        squared_sum = (n - 1) * variance
        statistic = squared_sum / null_variance
        expected = {"df": n - 1}
        steps = [
            _step("Recuperar la suma de desviaciones al cuadrado",
                  "La varianza muestral s² usa divisor n − 1. La suma conserva unidades al cuadrado, por ejemplo mm².",
                  rf"\sum_{{i=1}}^{{{n}}}(x_i-\bar x)^2=(n-1)s^2"
                  rf"=({n}-1)({_given(variance)})\approx {_n(squared_sum)}"),
            _step("Comparar la dispersión con la referencia de H₀",
                  "Dividir por la varianza poblacional propuesta cancela las unidades. Con observaciones independientes normales, el estadístico sigue exactamente una chi-cuadrado bajo H₀.",
                  rf"Q_{{obs}}=\frac{{(n-1)s^2}}{{\sigma_0^2}}"
                  rf"=\frac{{{_given(squared_sum)}}}{{{_given(null_variance)}}}\approx {_n(statistic)},"
                  rf"\qquad\nu={n}-1={n-1}"),
        ]
    elif distribution_id == "f":
        n1, n2 = (_sample_number(samples, key, integer=True) for key in ("n1", "n2"))
        v1 = _sample_number(samples, "variance1", nonnegative=True)
        v2 = _sample_number(samples, "variance2", positive=True)
        statistic = v1 / v2
        expected = {"dfn": n1 - 1, "dfd": n2 - 1}
        steps = [
            _step("Formar el cociente de las varianzas muestrales",
                  "Usamos el orden de las dos muestras definido de antemano. Ambas varianzas tienen las mismas unidades al cuadrado; su cociente no tiene unidades.",
                  rf"R_{{obs}}=\frac{{s_1^2}}{{s_2^2}}=\frac{{{_given(v1)}}}{{{_given(v2)}}}\approx {_n(statistic)}"),
            _step("Asignar los grados de libertad en ese mismo orden",
                  "Bajo H₀ de varianzas poblacionales iguales, dos muestras independientes de poblaciones normales dan una F central. Si se invierte el cociente, también deben intercambiarse los grados y la cola.",
                  rf"\nu_1={n1}-1={n1-1},\quad\nu_2={n2}-1={n2-1},"
                  rf"\qquad R\mid H_0\sim F_{{{n1-1},{n2-1}}}"),
        ]
    else:
        raise ValidationError("Los datos muestrales de este caso corresponden a t, chi-cuadrado o F.")
    if not math.isfinite(statistic):
        raise ValidationError("El estadístico muestral excede la precisión numérica disponible.")
    for key, value in expected.items():
        if not math.isclose(float(params[key]), value, rel_tol=0, abs_tol=1e-12):
            raise ValidationError("Los grados de libertad deben corresponder a los tamaños muestrales del caso.")
    return steps


def _multinomial_steps(params, event, value, log_value):
    n, probabilities, counts = params["n"], params["probabilities"], event["counts"]
    denominator = r"\,".join(rf"{int(k)}!" for k in counts)
    coefficient_value = _coefficient(n, counts)
    coefficient = rf"C=\frac{{{int(n)}!}}{{{denominator}}}" + coefficient_value
    terms = r"\,".join(rf"({_given(p)})^{{{int(k)}}}" for p, k in zip(probabilities, counts))
    log_order = 0.0
    for p, k in zip(probabilities, counts):
        if k:
            log_order = -math.inf if p == 0 else log_order + k * math.log(p)
    order_value = ("0" if log_order == -math.inf else
                   rf"e^{{{_n(log_order)}}}" if log_order < -700 else _n(math.exp(log_order)))
    vector = r"\left(" + ",".join(str(int(k)) for k in counts) + r"\right)"
    if value == 0 and math.isfinite(log_value):
        result = rf"\log P(\mathbf X={vector})\approx {_n(log_value)},\quad P(\mathbf X={vector})>0"
        conclusion = "La probabilidad conjunta es positiva, pero demasiado pequeña para un decimal de precisión doble; conservamos su logaritmo."
    else:
        factor = coefficient_value.removeprefix("=") if coefficient_value else "C"
        result = (rf"P(\mathbf X={vector})=C\,P(\text{{un orden}})"
                  rf"\approx {factor}\cdot {order_value}\approx {_n(value)}={_n(100*value)}\,\%")
        conclusion = "Sumamos todos los órdenes que producen exactamente este vector. Los conteos esperados no garantizan ese resultado; no multiplicamos probabilidades marginales."
    return [
        _step("Contar los órdenes que producen el vector", "Cada permutación de las categorías puede dar los mismos conteos. El coeficiente multinomial cuenta esas formas.", coefficient),
        _step("Calcular la probabilidad de un orden", "Por independencia, multiplicamos las probabilidades de sus categorías. Una categoría de probabilidad cero con conteo cero aporta 0⁰ = 1.",
              rf"P(\text{{un orden}})={terms}\approx {order_value}"),
        _step("Reunir todos esos órdenes", conclusion, result),
    ]


def _quantile_steps(distribution_id, params, event, rv):
    q = event["q"]
    x = quantile(distribution_id, params, q)
    setup = (_mass_setup(distribution_id, params) if distribution_id in DISCRETE_IDS
             else _simple_setup(distribution_id, params) if distribution_id in SIMPLE_CONTINUOUS_IDS
             else _density_setup(distribution_id, params))
    if not math.isfinite(x):
        return [setup, _step("Encontrar el extremo del soporte",
                            "Para q = 0 o q = 1 se muestra el extremo real del soporte. Aquí es infinito: no existe un umbral finito que mostrar.",
                            rf"Q({_given(q)})={_n(x)}")]
    definition = (r"\min\{k\text{ del soporte}:F_X(k)\ge q\}" if distribution_id in DISCRETE_IDS
                  else r"F_X^{-1}(q)")
    explanation = ("Buscamos el menor entero del soporte que alcanza la acumulación objetivo."
                   if distribution_id in DISCRETE_IDS else "Buscamos el umbral que alcanza la acumulación objetivo.")
    if q in (0, 1):
        definition = r"\text{extremo del soporte}"
        explanation = "En q = 0 o q = 1 la convención de la app devuelve el extremo correspondiente del soporte."
    elif distribution_id in SIMPLE_CONTINUOUS_IDS:
        definition = _simple_quantile(distribution_id, params, q)
        explanation = "Despejamos el umbral en la fórmula de acumulación y sustituimos la probabilidad objetivo."
    steps = [setup, _step("Buscar el umbral de acumulación", explanation,
                          rf"Q({_given(q)})={definition}\approx {_n(x)}")]
    achieved = probability(distribution_id, params, {"kind": "le", "x": x})
    latex = rf"F_X({_given(x)})\approx {_n(achieved)}"
    explanation = "Comprobamos la probabilidad realmente alcanzada en el umbral representable."
    if distribution_id in DISCRETE_IDS and 0 < q < 1:
        below = probability(distribution_id, params, {"kind": "le", "x": x - 1})
        latex = (r"\begin{gathered}F_X(Q(q)-1)<q\le F_X(Q(q))\\"
                 rf"F_X({_given(x - 1)})\approx {_n(below)},\quad q={_given(q)},\quad "
                 rf"F_X({_given(x)})\approx {_n(achieved)}\end{{gathered}}")
        explanation += " Los saltos de la acumulada pueden hacer que supere q; no exigimos que sea igual."
    elif abs(achieved - q) > 1e-8:
        explanation += " La acumulada lograda difiere de q por los extremos del soporte o por precisión numérica; no se presenta como igualdad."
    steps.append(_step("Comprobar la acumulada lograda", explanation, latex))
    return steps


def calculation_steps(distribution_id, params, event, samples=None):
    """Return 2–4 Spanish steps with current numerical substitutions.

    ``samples`` optionally supplies raw summaries for t (n, mean, std, null),
    chi2 (n, variance, null_variance), or f (n1, n2, variance1, variance2).
    Their sample sizes must agree with the distribution's degrees of freedom.
    Invalid or unresolvable numerical input raises the engine's ValidationError.
    """
    validate_parameters(distribution_id, params)
    if not isinstance(event, Mapping):
        raise ValidationError("Ingresá un evento válido.")
    if event.get("kind") == "quantile":
        validate_event(event, discrete=distribution_id in DISCRETE_IDS)
        rv = frozen_dist(distribution_id, params)
        return _quantile_steps(distribution_id, params, event, rv)
    # Validate and calculate first, before presentation code touches thresholds
    # or combinatorial arguments. This preserves the engine's edge handling.
    value = probability(distribution_id, params, event)
    log_value = log_probability(distribution_id, params, event)
    if distribution_id == "multinomial":
        return _multinomial_steps(params, event, value, log_value)
    rv = frozen_dist(distribution_id, params)
    if samples is not None:
        steps = _sample_steps(distribution_id, params, samples)
        steps += [_area(distribution_id, params, event,
                        {"t": "T", "chi2": "Q", "f": "R"}[distribution_id], null=True),
                  _result(distribution_id, params, event, value, log_value, samples=True)]
        return steps
    if distribution_id in DISCRETE_IDS:
        return [_mass_setup(distribution_id, params),
                _discrete_event(distribution_id, params, event, rv),
                _result(distribution_id, params, event, value, log_value)]
    if distribution_id in SIMPLE_CONTINUOUS_IDS:
        return [_simple_setup(distribution_id, params),
                _simple_event(distribution_id, params, event),
                _result(distribution_id, params, event, value, log_value)]
    steps = [_density_setup(distribution_id, params)]
    transform = _transform(distribution_id, params, event)
    if transform is not None:
        steps.append(transform)
    steps += [_area(distribution_id, params, event),
              _result(distribution_id, params, event, value, log_value)]
    return steps
