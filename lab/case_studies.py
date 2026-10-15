"""Live Spanish case stories, kept separate from Streamlit and the engine.

The story, units and interpretation always use the same inputs as the case
calculation. Values are illustrative, never estimates from an actual dataset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .calculations import (
    ValidationError, log_probability, probability, quantile, validate_event,
    validate_parameters,
)
from .catalog import REGISTRY


def _n(value):
    return f"{float(value):.8g}".replace(".", ",")


def _row(symbol, value, meaning):
    return {"Parámetro": symbol, "Valor": value, "Significado": meaning}


def _given(value):
    """Preserve distinct event inputs, including close interval endpoints."""
    if isinstance(value, int):
        return str(value)
    return repr(float(value)).removesuffix(".0").replace(".", ",")


def _value(value, unit, precise=False):
    render = _given if precise else _n
    if unit == "proporción":
        return f"{render(100 * value)} % de conversión"
    if unit == "sin unidades":
        return render(value)
    return f"{render(value)} {unit}".strip()


def _event_words(event, noun, unit, discrete=False):
    """Translate the actual event; integer thresholds follow engine rounding."""
    kind = event["kind"]
    if kind == "quantile":
        return (f"hallar el umbral de {noun} que acumula al menos "
                f"{_given(100 * event['q'])} % de probabilidad")
    if kind == "interval":
        a, b = event["a"], event["b"]
        if discrete:
            low, high = math.ceil(a), math.floor(b)
            if low > high:
                return (f"que {noun} esté entre {_given(a)} y {_value(b, unit, True)}; "
                        "ese intervalo no contiene ningún resultado entero")
            a, b = low, high
        return f"que {noun} esté entre {_value(a, unit, True)} y {_value(b, unit, True)}, incluidos los extremos"
    x = event["x"]
    if discrete and kind in {"le", "gt"}:
        x = math.floor(x)
    if kind == "le":
        return f"que {noun} sea como máximo {_value(x, unit, True)}"
    if kind == "gt":
        text = f"que {noun} sea mayor que {_value(x, unit, True)}"
        if discrete:
            text += f", es decir, al menos {_value(x + 1, unit, True)}"
        return text
    return f"que {noun} sea exactamente {_value(x, unit, True)}"


def _result(distribution_id, params, event, unit):
    if event["kind"] == "quantile":
        x = quantile(distribution_id, params, event["q"])
        if not math.isfinite(x):
            side = "inferior" if x < 0 else "superior"
            return (f"El cuantil pedido es el extremo {side} infinito del soporte: "
                    "no hay un umbral finito para esa probabilidad acumulada.")
        achieved = probability(distribution_id, params, {"kind": "le", "x": x})
        return (f"El cuantil es {_value(x, unit)}. Hasta ese umbral se acumula "
                f"{_n(100 * achieved)} % (objetivo: {_n(100 * event['q'])} %). "
                "En una variable discreta, un salto de la acumulada puede superar el objetivo; "
                "en una continua, un valor extremo puede quedar limitado por el redondeo numérico.")
    value = probability(distribution_id, params, event)
    if value == 0:
        log_value = log_probability(distribution_id, params, event)
        if math.isfinite(log_value):
            return (f"La probabilidad es positiva, pero demasiado pequeña para representarla "
                    f"en precisión doble; su logaritmo natural es {_n(log_value)}. No es un cero exacto.")
        return "El evento tiene probabilidad cero bajo este modelo y estos límites."
    return f"La probabilidad del evento es {_n(value)}, equivalente a {_n(100 * value)} %."


def _sample_rows(distribution_id, params, samples):
    """Validate that displayed measurements really produce the displayed model."""
    if not samples:
        return None
    required = {
        "t": ("n", "mean", "std", "null"),
        "chi2": ("n", "variance", "null_variance"),
        "f": ("n1", "n2", "variance1", "variance2"),
    }.get(distribution_id)
    if required is None:
        return None
    if not isinstance(samples, Mapping) or any(key not in samples for key in required):
        raise ValidationError("Faltan mediciones para explicar el estadístico del caso.")
    if any(isinstance(samples[key], (str, bool)) or not math.isfinite(float(samples[key])) for key in required):
        raise ValidationError("Las mediciones del caso deben ser números finitos.")
    for key in ("n1", "n2") if distribution_id == "f" else ("n",):
        if samples[key] < 2 or not float(samples[key]).is_integer():
            raise ValidationError("Cada muestra debe contener al menos dos mediciones.")
    positive = {"t": ("std",), "chi2": ("null_variance",), "f": ("variance2",)}[distribution_id]
    if any(samples[key] <= 0 for key in positive):
        raise ValidationError("La escala usada para dividir debe ser positiva.")
    if distribution_id == "chi2" and samples["variance"] < 0:
        raise ValidationError("La varianza muestral no puede ser negativa.")
    if distribution_id == "f" and samples["variance1"] < 0:
        raise ValidationError("La varianza muestral no puede ser negativa.")
    expected = ({"dfn": samples["n1"] - 1, "dfd": samples["n2"] - 1}
                if distribution_id == "f" else {"df": samples["n"] - 1})
    if any(not math.isclose(params[key], value, rel_tol=1e-12, abs_tol=1e-12)
           for key, value in expected.items()):
        raise ValidationError("Los grados de libertad deben coincidir con los tamaños de las muestras.")
    return samples


def case_details(distribution_id, params, event, samples=None):
    """Return concise, contextual content for all 24 editable worked cases.

    ``samples`` contains observed summaries for t, chi-square or F. It is absent
    in the advanced parameter mode, which explains the central reference law
    without inventing sample sizes for arbitrary real degrees of freedom.
    """
    validate_parameters(distribution_id, params)
    spec = REGISTRY[distribution_id]
    if distribution_id != "multinomial":
        validate_event(event, discrete=spec.kind == "discrete")
    s = _sample_rows(distribution_id, params, samples)
    p = params
    rows = []
    noun = "el resultado"
    unit = "sin unidades"
    repetition = "El resultado describe el evento elegido bajo los supuestos del modelo."

    if distribution_id == "bernoulli":
        title = "Una visita: ¿compra o no compra?"
        context = f"Una tienda recibe un visitante. Su probabilidad de comprar se fija en {_given(p['p'])}."
        variable = "X = 1 si compra y X = 0 si no compra. Se observa una sola visita."
        outcomes = "0: no compra · 1: compra. No hay un tercer resultado."
        noun = "el indicador de compra"
        rationale = "Bernoulli describe una única decisión con dos resultados excluyentes que cubren todas las posibilidades."
        rows = [_row("p", _given(p['p']), "Probabilidad de compra de este visitante; 1 − p es la de no comprar.")]
        limitation = "La probabilidad elegida es ilustrativa. Visitantes con perfiles diferentes pueden necesitar probabilidades diferentes."
        experiment = "¿Qué pasaría si subís p a 0,70? Antes de calcular, anticipá cuál de las dos barras debería crecer."
        repetition = "El porcentaje corresponde a la decisión de una visita; no asegura lo que hará una persona concreta."

    elif distribution_id == "binom":
        title = "¿Cuántas compras producirá la campaña?"
        context = (f"Se fija un grupo de {_given(p['n'])} visitantes comparables; cada uno compra "
                   f"con probabilidad {_given(p['p'])}, de manera independiente.")
        variable = f"X cuenta compras en ese grupo fijo de {_given(p['n'])} visitas."
        outcomes = f"Enteros de 0 a {_given(p['n'])} compras."
        noun, unit = "la cantidad de compras", "compras"
        rationale = "Binomial: número fijo de visitas, dos resultados por visita, independencia y una misma probabilidad de compra."
        rows = [_row("n", f"{_given(p['n'])} visitas", "Tamaño fijado antes de observar las compras."),
                _row("p", _given(p['p']), "Probabilidad de compra por visita; no es el número de compras.")]
        limitation = "Clientes repetidos o canales con probabilidades distintas rompen los supuestos. Sin reemplazo en una población pequeña conviene hipergeométrica."
        experiment = "¿Qué pasaría si duplicás n y mantenés p y el evento? Compará el conteo esperado np con el umbral antes de calcular."
        repetition = f"El experimento completo es un grupo de {_given(p['n'])} visitas, no una visita aislada."

    elif distribution_id == "poisson":
        title = "¿Alcanza la capacidad de la mesa de ayuda?"
        context = (f"Durante una hora estable llegan consultas a una tasa ilustrativa de {_given(p['mu'])} por hora. "
                   f"La ventana es 1 hora: μ = {_given(p['mu'])} × 1 = {_given(p['mu'])} consultas esperadas.")
        variable = "X cuenta consultas recibidas en la próxima ventana de una hora."
        outcomes = "0, 1, 2, … consultas; el modelo no fija un máximo."
        noun, unit = "la cantidad de consultas en esa hora", "consultas"
        rationale = "Poisson cuenta llegadas en una ventana fija con tasa estable e incrementos independientes."
        rows = [_row("μ", f"{_given(p['mu'])} consultas/hora × 1 hora", "Media del conteo de la ventana; coincide numéricamente con la tasa porque la ventana dura una hora.")]
        limitation = "Después de un incidente pueden llegar ráfagas dependientes. Una tasa que cambia durante la hora exige otro modelo."
        experiment = "¿Qué pasaría si se duplica la tasa durante la misma hora? Duplicá μ y anticipá cómo cambia la cola; no dupliques la probabilidad final."
        repetition = "El porcentaje se refiere a ventanas completas de una hora bajo el mismo régimen de llegadas."

    elif distribution_id == "geom":
        title = "¿Cuántos intentos hasta la primera respuesta?"
        context = f"Cada intento de contacto logra respuesta con probabilidad {_given(p['p'])}; se continúa hasta la primera respuesta."
        variable = "X cuenta todos los intentos, incluido el que consigue la primera respuesta."
        outcomes = "1, 2, 3, … intentos. Cero intentos nunca puede producir una respuesta."
        noun, unit = "el número de intentos hasta responder", "intentos"
        rationale = "Geométrica: se espera un primer éxito con intentos independientes y p constante; no se fija cuántos intentos habrá."
        rows = [_row("p", _given(p['p']), "Probabilidad de respuesta en cada intento.")]
        limitation = "La insistencia, el aprendizaje o un límite de contactos pueden cambiar el mecanismo. No confundir intentos con fracasos: fracasos = X − 1."
        experiment = "¿Qué pasaría si duplicás p? Compará la espera media 1/p y la posibilidad de seguir sin respuesta al quinto intento."
        repetition = "Cada repetición comienza de nuevo y termina con la primera respuesta; el intento exitoso está incluido."

    elif distribution_id == "nbinom":
        title = "¿Cuántos contactos hasta completar la meta?"
        context = (f"Un equipo busca {_given(p['r'])} ventas. Cada contacto independiente convierte "
                   f"con probabilidad {_given(p['p'])}; se detiene justo al completar la meta.")
        variable = f"X cuenta contactos fallidos antes de la venta número {_given(p['r'])}; los contactos totales son T = X + {_given(p['r'])}."
        outcomes = "0, 1, 2, … contactos fallidos. Las ventas no se incluyen en X."
        noun, unit = "el número de contactos fallidos antes de la meta", "contactos fallidos"
        rationale = "Binomial negativa: la meta de ventas es fija y el número de contactos es aleatorio; X cuenta solo los fracasos."
        rows = [_row("r", f"{_given(p['r'])} ventas", "Éxitos necesarios para detener el proceso."),
                _row("p", _given(p['p']), "Probabilidad constante de venta por contacto.")]
        limitation = "Un presupuesto que fija los contactos de antemano corresponde a otro problema. La calidad variable de los contactos invalida una p común."
        experiment = "¿Qué pasaría si exigís una venta más con la misma p? Anticipá qué ocurre con los fracasos esperados r(1 − p)/p."
        repetition = f"Una repetición termina al lograr {_given(p['r'])} ventas. Para pasar de fracasos a contactos, sumá siempre {_given(p['r'])}."

    elif distribution_id == "hypergeom":
        title = "Inspeccionar un lote sin devolver las piezas"
        context = (f"Un lote tiene {_given(p['M'])} piezas, de las cuales {_given(p['K'])} son defectuosas. "
                   f"Se eligen {_given(p['n'])} al azar, sin reemplazo.")
        variable = "X cuenta piezas defectuosas en la muestra extraída."
        low, high = max(0, p['n'] - p['M'] + p['K']), min(p['K'], p['n'])
        outcomes = f"Enteros de {_n(low)} a {_n(high)} piezas defectuosas; los límites salen de lo que hay en el lote."
        noun, unit = "la cantidad de defectuosas extraídas", "piezas"
        rationale = "Hipergeométrica: población finita conocida y extracción sin reemplazo. Las extracciones no son independientes."
        rows = [_row("M", f"{_given(p['M'])} piezas", "Total del lote."),
                _row("K", f"{_given(p['K'])} defectuosas", "Piezas de interés que existen en ese lote."),
                _row("n", f"{_given(p['n'])} piezas", "Cantidad extraída sin devolver piezas.")]
        limitation = "Si devolvés cada pieza y mezclás, corresponde binomial con p = K/M. Aquí K se supone conocido; no se está estimando."
        experiment = "¿Qué pasaría si inspeccionás todo el lote, n = M? Comprobá que X deja de ser aleatorio y vale K."
        repetition = "Cada repetición parte de un lote completo con la misma composición y selecciona una nueva muestra sin reemplazo."

    elif distribution_id == "multinomial":
        title = "¿Cómo se repartirán los pedidos entre canales?"
        names = p.get('names', [f"Canal {i + 1}" for i in range(len(p['probabilities']))])
        categories = "; ".join(f"{name}: {_n(prob)}" for name, prob in zip(names, p['probabilities']))
        context = f"Se reciben {_given(p['n'])} pedidos. Cada pedido usa exactamente un canal, con probabilidades {categories}."
        variable = "X es el vector de cantidades por canal, en este orden: " + ", ".join(names) + "."
        outcomes = f"Vectores de enteros no negativos cuya suma es {_given(p['n'])}; los conteos están ligados por ese total."
        noun, unit = "el reparto de pedidos", "pedidos"
        rationale = "Multinomial: total fijo, categorías excluyentes y exhaustivas, pedidos independientes y probabilidades constantes que suman 1."
        rows = [_row("n", f"{_given(p['n'])} pedidos", "Total fijo entre todos los canales.")]
        rows += [_row(f"p{i + 1} · {name}", _n(prob), f"Probabilidad del canal en un pedido; su conteo esperado es n × p = {_n(p['n'] * prob)} pedidos.")
                 for i, (name, prob) in enumerate(zip(names, p['probabilities']))]
        limitation = "Los conteos no son independientes: deben sumar n. La probabilidad de un vector no es el producto de probabilidades marginales; un conteo esperado no garantiza ese reparto."
        experiment = "¿Qué pasaría si mirás un solo canal? Su marginal binomial incluye todos los repartos posibles entre los demás canales y suele tener mayor probabilidad que un vector exacto."
        repetition = f"La unidad del experimento es el grupo completo de {_given(p['n'])} pedidos, con todos sus conteos simultáneamente."

    elif distribution_id == "randint":
        title = "Un sorteo entre etiquetas enteras"
        count = p['high'] - p['low'] + 1
        context = f"Un sorteo elige una etiqueta entera desde {_given(p['low'])} hasta {_given(p['high'])}, ambas incluidas, con igual probabilidad."
        variable = "X es la etiqueta numérica que sale en un único sorteo."
        outcomes = f"Los {_n(count)} enteros de {_given(p['low'])} a {_given(p['high'])}; hay b − a + 1 opciones, no b − a."
        noun = "la etiqueta sorteada"
        rationale = "Uniforme discreta: el mecanismo asigna la misma probabilidad a cada entero permitido."
        rows = [_row("a", _given(p['low']), "Primera etiqueta incluida."), _row("b", _given(p['high']), "Última etiqueta incluida; también puede salir.")]
        limitation = "Desconocer las probabilidades no las hace uniformes. Un sorteo sesgado o etiquetas faltantes necesitan otro modelo."
        experiment = "¿Qué pasaría si agregás una etiqueta al extremo superior? Contá otra vez las opciones y anticipá la nueva masa de cada entero."

    elif distribution_id == "norm":
        title = "¿La pieza supera el diámetro permitido?"
        context = f"Una fábrica modela el diámetro de sus piezas con media poblacional {_given(p['mu'])} mm y desviación estándar {_given(p['sigma'])} mm."
        variable = "X es el diámetro de una pieza elegida al azar, en mm; no es la media de varias piezas."
        outcomes = "Valores continuos. La normal permite cualquier real; los diámetros negativos deben tener probabilidad despreciable para que el modelo físico sea razonable."
        noun, unit = "el diámetro de una pieza", "mm"
        rationale = "Normal: mediciones aproximadamente simétricas alrededor de un centro estable, con una escala de dispersión poblacional."
        rows = [_row("μ", f"{_given(p['mu'])} mm", "Diámetro medio poblacional."),
                _row("σ", f"{_given(p['sigma'])} mm", "Desviación entre piezas; no es el error estándar de una media.")]
        limitation = "Una forma sesgada, lotes mezclados o probabilidad apreciable de diámetros negativos hacen inadecuado este modelo."
        experiment = "¿Qué pasaría si reducís σ a la mitad sin mover μ? Observá cómo cambia el área de piezas que exceden el mismo límite."
        repetition = "El porcentaje corresponde a piezas individuales producidas bajo el mismo proceso."

    elif distribution_id == "uniform":
        title = "¿En qué momento llegará el vehículo?"
        context = (f"Se toma una hora de referencia como minuto 0. La llegada de un vehículo se supone "
                   f"uniforme entre los minutos {_given(p['a'])} y {_given(p['b'])} respecto de esa referencia.")
        variable = "X es el minuto de llegada relativo a la referencia: negativo significa antes; positivo, después."
        outcomes = f"Cualquier valor real entre {_given(p['a'])} y {_given(p['b'])} minutos, no solo minutos enteros."
        noun, unit = "el momento de llegada", "minutos respecto de la referencia"
        rationale = "Uniforme continua: intervalos de igual duración tienen igual probabilidad dentro de la ventana. La probabilidad es longitud favorable / longitud total."
        rows = [_row("a", f"{_given(p['a'])} min", "Primera llegada posible."),
                _row("b", f"{_given(p['b'])} min", "Última llegada posible; la ventana dura b − a minutos.")]
        limitation = "Se necesita justificar una densidad constante. La falta de información o la puntualidad aproximada no bastan para suponer uniformidad."
        experiment = "¿Qué pasaría si duplicás la amplitud b − a y conservás un intervalo favorable dentro de la ventana? Anticipá el cambio de su área."

    elif distribution_id == "expon":
        title = "¿Cuánto habrá que esperar la próxima consulta?"
        context = f"Una mesa de ayuda recibe consultas según un proceso de Poisson estable de tasa {_given(p['rate'])} consultas por minuto."
        variable = "X es la espera, en minutos desde ahora, hasta la próxima consulta."
        outcomes = "Tiempos continuos no negativos, sin un máximo fijado."
        noun, unit = "la espera hasta la próxima consulta", "minutos"
        rationale = "Exponencial: tiempo hasta la siguiente llegada de un proceso de tasa constante. La espera no conserva memoria de cuánto tiempo pasó."
        rows = [_row("λ", f"{_given(p['rate'])} consultas/min", f"Tasa de llegada; la espera media es 1/λ = {_n(1 / p['rate'])} minutos.")]
        limitation = "Horarios, ráfagas o una tasa que depende de la espera rompen el modelo sin memoria. La tasa no es la espera media."
        experiment = "¿Qué pasaría si duplicás λ? Antes de calcular, anticipá qué sucede con la espera media y con la cola de esperas largas."

    elif distribution_id == "gamma":
        title = "¿Terminará el trámite dentro del plazo?"
        if float(p['alpha']).is_integer():
            context = (f"Un trámite suma {_given(p['alpha'])} etapas independientes, cada una con duración "
                       f"exponencial de media {_given(p['theta'])} minutos.")
        else:
            context = (f"La duración positiva de un trámite se representa con una Gamma de forma "
                       f"{_given(p['alpha'])} y escala {_given(p['theta'])} minutos. La forma no entera se elige directamente como modelo.")
        variable = "X es la duración total del trámite, en minutos."
        outcomes = "Tiempos continuos no negativos. No hay un máximo fijo."
        noun, unit = "la duración total", "minutos"
        rationale = "Gamma permite esperas positivas con asimetría. Si la forma es entera, también surge al sumar ese número de esperas exponenciales de igual escala e independientes."
        rows = [_row("α", _given(p['alpha']), "Forma; solo si es entera admite la lectura como número de etapas exponenciales."),
                _row("θ", f"{_given(p['theta'])} min", f"Escala, no tasa. La duración media total es αθ = {_n(p['alpha'] * p['theta'])} minutos.")]
        limitation = "Etapas dependientes o con escalas diferentes no producen esta Gamma exacta. Una forma fraccionaria no representa una fracción literal de etapa."
        experiment = "¿Qué pasaría si duplicás θ con la misma α? Compará el plazo original con otro también duplicado."

    elif distribution_id == "beta":
        title = "¿La tasa de conversión supera el objetivo?"
        if p['alpha'] == 9 and p['beta'] == 41:
            context = ("Una tienda quiere evaluar su tasa de conversión. Se parte de una previa Beta(1, 1) y se observan 8 compras y 40 no compras "
                       "en 48 visitas hipotéticas. La actualización da α = 1 + 8 = 9 y β = 1 + 40 = 41.")
        else:
            context = (f"Una tienda quiere evaluar su tasa de conversión. Se elige una Beta({_given(p['alpha'])}, {_given(p['beta'])}) para representar incertidumbre "
                       "sobre la tasa de conversión. Estas formas se fijan directamente; no se inventa una muestra que las haya producido.")
        variable = "X = p es la tasa de conversión desconocida. Lo aleatorio aquí es nuestra incertidumbre sobre p, no la compra de un visitante."
        outcomes = "Tasas continuas de 0 a 1, equivalentes a 0 % a 100 %."
        noun, unit = "la tasa de conversión desconocida", "proporción"
        rationale = "Beta vive entre 0 y 1. Con una previa Beta y datos Bernoulli independientes de una tasa común, la posterior sigue siendo Beta."
        rows = [_row("α", _given(p['alpha']), "Forma hacia tasas altas; en la actualización Beta-Bernoulli se suma el número de éxitos a la forma previa."),
                _row("β", _given(p['beta']), "Forma hacia tasas bajas; se suma el número de fracasos a la forma previa.")]
        limitation = "La respuesta depende del modelo y de la previa. La probabilidad de que p supere un objetivo no es la proporción de visitantes que comprarán."
        experiment = "¿Qué pasaría si duplicás α y β? La tasa media α/(α + β) se conserva; observá qué ocurre con la incertidumbre alrededor de ella."
        repetition = "La respuesta expresa incertidumbre sobre la tasa desconocida según esta distribución; no una frecuencia de compras individuales ni una garantía del próximo grupo."

    elif distribution_id == "lognorm":
        title = "¿Un costo multiplicativo superará el presupuesto?"
        context = (f"El costo de un insumo resulta de factores multiplicativos. Su logaritmo natural se modela "
                   f"normal con media {_given(p['mu_log'])} y desviación {_given(p['sigma_log'])}.")
        variable = "X es el costo positivo, en unidades monetarias (u. m.); ln(X / 1 u. m.) es la variable normal."
        outcomes = "Cualquier costo positivo; nunca cero ni negativo."
        noun, unit = "el costo del insumo", "u. m."
        rationale = "Lognormal: una magnitud positiva cuyo logaritmo es normal. Los factores multiplicativos pueden producir una cola derecha larga."
        rows = [_row("μℓ", _given(p['mu_log']), "Media del logaritmo del costo expresado en la unidad elegida; no es el costo medio."),
                _row("σℓ", _given(p['sigma_log']), "Desviación del logaritmo; no es la desviación del costo.")]
        limitation = "No todo costo positivo es lognormal. La unidad monetaria debe mantenerse: cambiarla también cambia μℓ."
        experiment = "¿Qué pasaría si aumentás μℓ en ln(2)? Los cuantiles del costo se duplican, aunque σℓ no cambie."

    elif distribution_id == "t":
        title = "¿El llenado medio supera el objetivo?"
        if s:
            se = s['std'] / math.sqrt(s['n'])
            context = (f"Se miden {_given(s['n'])} envases: media {_given(s['mean'])} mL y desviación muestral "
                       f"{_given(s['std'])} mL. H₀ propone una media poblacional de {_given(s['null'])} mL; la varianza poblacional es desconocida.")
            rows = [_row("n", f"{_given(s['n'])} envases", "Tamaño de la muestra independiente."),
                    _row("x̄", f"{_given(s['mean'])} mL", "Media observada, un estadístico muestral."),
                    _row("s", f"{_given(s['std'])} mL", "Desviación muestral, calculada con n − 1."),
                    _row("μ₀", f"{_given(s['null'])} mL", "Media poblacional propuesta por H₀."),
                    _row("ν = n − 1", _given(p['df']), f"Grados de libertad; el error estándar observado es s/√n = {_n(se)} mL.")]
            repetition = (f"La diferencia observada es {_n(s['mean'] - s['null'])} mL, y su error estándar es {_n(se)} mL. "
                          "La probabilidad se refiere a estadísticos de muestras completas repetidas bajo H₀, recalculando s en cada muestra; no a envases individuales.")
        else:
            context = (f"En un control de llenado se explora la referencia t central con ν = {_given(p['df'])}. "
                       "El modo de parámetros estudia la distribución del desvío de una media respecto del objetivo, expresado en errores estándar.")
            rows = [_row("ν", _given(p['df']), "Grados de libertad de la referencia central. En el caso de una media normal, ν = n − 1; aquí no se inventa n para un ν arbitrario.")]
            repetition = "Es un área de la ley t central elegida, sin una muestra ni un estadístico observado. No se interpreta como un valor p de datos recogidos."
        variable = ("X = T = (x̄ − μ₀)/(s/√n): distancia de la media al objetivo en errores estándar; T no tiene unidades."
                    if s else "X es un estadístico de referencia t central sin diseño muestral concreto. La fórmula T = (x̄ − μ₀)/(s/√n) corresponde al modo Mediciones, con sus supuestos.")
        outcomes = "Cualquier real: negativo si la media está debajo del objetivo y positivo si está encima."
        noun, unit = ("el desvío estandarizado de la media", "errores estándar") if s else ("el estadístico t de referencia", "sin unidades")
        rationale = ("t central: bajo H₀, una muestra normal independiente y una varianza desconocida que se reemplaza por s² producen T con n − 1 grados de libertad."
                     if s else "La familia t central permite estudiar colas con cualquier ν positivo. Un ν arbitrario define una ley matemática; por sí solo no justifica una prueba sobre llenados.")
        limitation = ("La normalidad da la referencia exacta para muestras pequeñas. Una cola derecha responde a una alternativa mayor fijada antes de observar los datos; una prueba bilateral suma ambas colas. El resultado no es la probabilidad de que H₀ sea verdadera."
                      if s else "Para interpretar un valor p hacen falta datos, una hipótesis y un diseño compatible, como los del modo Mediciones. Un ν fraccionario no se presenta aquí como n − 1 de una muestra ficticia.")
        experiment = ("¿Qué pasaría si duplicás el tamaño muestral manteniendo x̄, s y μ₀? Anticipá qué ocurre con s/√n y con T. La hipótesis se elige antes de mirar los datos."
                      if s else "¿Qué pasaría si aumentás ν sin mover el umbral? Compará la cola t con el comportamiento de una normal estándar.")

    elif distribution_id == "chi2":
        title = "¿La variabilidad del diámetro supera la de referencia?"
        if s:
            context = (f"Se registran {_given(s['n'])} diámetros independientes: la varianza muestral es "
                       f"{_given(s['variance'])} mm². H₀ propone una varianza poblacional de {_given(s['null_variance'])} mm².")
            rows = [_row("n", f"{_given(s['n'])} piezas", "Tamaño muestral."),
                    _row("s²", f"{_given(s['variance'])} mm²", "Varianza muestral; usa divisor n − 1."),
                    _row("σ₀²", f"{_given(s['null_variance'])} mm²", "Varianza poblacional propuesta por H₀, no una desviación estándar."),
                    _row("ν = n − 1", _given(p['df']), "Grados de libertad del estadístico Q.")]
            repetition = (f"La dispersión observada es s² = {_given(s['variance'])} mm² frente a "
                          f"σ₀² = {_given(s['null_variance'])} mm². La cola se refiere a varianzas de muestras completas repetidas bajo H₀.")
        else:
            context = (f"En un control de variabilidad se explora la referencia chi-cuadrado central con "
                       f"ν = {_given(p['df'])}. El modo de parámetros estudia la magnitud relativa de una suma de cuadrados.")
            rows = [_row("ν", _given(p['df']), "Grados de libertad; para la prueba de una varianza normal son n − 1. Un ν real arbitrario no se presenta como tamaño de muestra.")]
            repetition = "Es un área de la ley chi-cuadrado central elegida; no se ha especificado una muestra ni un contraste observado de diámetros."
        variable = ("X = Q = (n − 1)s²/σ₀²; compara la variabilidad muestral con la de H₀ y no tiene unidades."
                    if s else "X es un estadístico de referencia chi-cuadrado central sin diseño muestral concreto. La fórmula Q = (n − 1)s²/σ₀² corresponde al modo Mediciones y requiere sus supuestos.")
        outcomes = "Valores continuos no negativos. Un Q mayor indica mayor varianza muestral para n y σ₀² fijos."
        noun = "el estadístico de variabilidad Q"
        rationale = ("Chi-cuadrado: para observaciones normales independientes, (n − 1)s²/σ₀² tiene esta distribución bajo la hipótesis nula H₀."
                     if s else "La familia chi-cuadrado central admite cualquier ν positivo; para ν entero puede construirse como suma de cuadrados de ν normales estándar independientes. Aquí no se atribuye un origen muestral a ν.")
        limitation = ("La normalidad es necesaria para esta prueba exacta de varianza. No confundir mm con mm², ni esta prueba con una prueba de independencia de una tabla. La cola no es la probabilidad de que H₀ sea verdadera."
                      if s else "Una ley matemática no basta para justificar una prueba de varianza. El modo Mediciones muestra el diseño normal, los datos y la hipótesis necesarios para esa interpretación.")
        experiment = ("¿Qué pasaría si duplicás s² y conservás n y σ₀²? Q se duplica; anticipá qué pasa con su cola derecha."
                      if s else "¿Qué pasaría si duplicás ν y mantenés el umbral? La media de la ley es ν; usala para anticipar hacia dónde se moverá la masa.")

    elif distribution_id == "f":
        title = "¿La primera máquina produce mayor variabilidad?"
        if s:
            context = (f"Dos máquinas producen muestras independientes de {_given(s['n1'])} y {_given(s['n2'])} piezas. "
                       f"Sus varianzas muestrales son {_given(s['variance1'])} y {_given(s['variance2'])} mm². H₀ propone varianzas poblacionales iguales.")
            rows = [_row("n₁; n₂", f"{_given(s['n1'])}; {_given(s['n2'])} piezas", "Tamaños de las dos muestras."),
                    _row("s₁²; s₂²", f"{_given(s['variance1'])}; {_given(s['variance2'])} mm²", "Varianzas muestrales, cada una calculada con su n − 1."),
                    _row("ν₁ = n₁ − 1", _given(p['dfn']), "Grados de libertad del numerador: primera máquina."),
                    _row("ν₂ = n₂ − 1", _given(p['dfd']), "Grados de libertad del denominador: segunda máquina.")]
            repetition = (f"El cociente observado es {_given(s['variance1'])}/{_given(s['variance2'])} = "
                          f"{_n(s['variance1'] / s['variance2'])}; las unidades mm² se cancelan. La probabilidad se refiere a pares de muestras completos bajo igualdad de varianzas, no a piezas individuales.")
        else:
            context = (f"Para comparar la variabilidad de dos máquinas se explora una F central con "
                       f"grados de libertad ({_given(p['dfn'])}, {_given(p['dfd'])}), bajo igualdad de varianzas poblacionales.")
            rows = [_row("ν₁", _given(p['dfn']), "Grados de libertad del numerador; en el caso normal, n₁ − 1."),
                    _row("ν₂", _given(p['dfd']), "Grados de libertad del denominador; en el caso normal, n₂ − 1.")]
            repetition = "Es un área de la ley F central elegida, sin dos muestras concretas ni un cociente observado de varianzas. No constituye un valor p de datos recogidos."
        variable = ("X = R = s₁²/s₂² bajo H₀ de varianzas iguales; es un cociente sin unidades. Valores grandes favorecen mayor variabilidad en la primera máquina."
                    if s else "X = (U/ν₁)/(V/ν₂), con U y V chi-cuadrado independientes de grados ν₁ y ν₂. Es un estadístico F de referencia; la fórmula s₁²/s₂² y su diseño corresponden al modo Mediciones.")
        outcomes = "Cocientes continuos no negativos, sin máximo fijo. El orden de las muestras importa."
        noun = "el cociente de varianzas muestrales" if s else "el estadístico F de referencia"
        rationale = ("F central: las dos muestras son normales e independientes, y la hipótesis nula H₀ hace iguales las varianzas poblacionales. Cada varianza aporta sus propios grados de libertad."
                     if s else "F central es la ley de un cociente de dos chi-cuadrado independientes divididas por sus grados de libertad. Los grados positivos arbitrarios no se convierten en tamaños muestrales ficticios.")
        limitation = ("La prueba exacta es sensible a falta de normalidad y dependencia. La alternativa mayor en la primera máquina se fija antes de mirar los datos. La cola no es la probabilidad de que H₀ sea verdadera."
                      if s else "Sin datos y un diseño compatible no se interpreta esta área como una prueba de dos máquinas. El modo Mediciones explicita normalidad, independencia e igualdad de varianzas bajo H₀.")
        experiment = ("¿Qué pasaría si intercambiás las máquinas? El cociente se vuelve 1/R y los grados se invierten; la misma evidencia pasa de la cola derecha a la izquierda."
                      if s else "¿Qué pasaría si invertís el cociente y cambiás también el orden de ν₁ y ν₂? La cola derecha original corresponde a la izquierda de la nueva referencia.")

    elif distribution_id == "weibull_min":
        title = "¿El equipo llegará al mantenimiento sin fallar?"
        behavior = "decreciente" if p['shape'] < 1 else "constante" if p['shape'] == 1 else "creciente"
        context = (f"La vida de un equipo en condiciones estables se modela con forma {_given(p['shape'])} "
                   f"y escala {_given(p['scale'])} horas. El riesgo instantáneo de falla es {behavior} con la edad.")
        variable = "X es el tiempo de operación desde la puesta en marcha hasta la falla, en horas."
        outcomes = "Tiempos continuos no negativos; no hay una duración garantizada."
        noun, unit = "el tiempo hasta la falla", "horas"
        rationale = "Weibull permite un riesgo de falla decreciente, constante o creciente según su forma; sirve para estudiar supervivencia y desgaste."
        rows = [_row("k", _given(p['shape']), f"Forma; aquí produce riesgo {behavior}. No es un número de fallas."),
                _row("η", f"{_given(p['scale'])} horas", "Tiempo hasta el que falla aproximadamente el 63,2 % del modelo; no es la vida media.")]
        limitation = "Una única Weibull no reproduce un riesgo completo con forma de bañera, mezclas de causas o cambios de carga."
        experiment = "¿Qué pasaría si fijás k = 1? La Weibull se vuelve exponencial: observá el riesgo constante y mantené la misma escala temporal."
        repetition = "El porcentaje se refiere a vidas completas de equipos comparables bajo esas condiciones de uso."

    elif distribution_id == "pareto":
        title = "Dentro de los reclamos grandes, ¿cuántos serán extremos?"
        context = (f"Se consideran solamente reclamos cuyo costo es al menos {_given(p['xmin'])} u. m. "
                   f"En esa población seleccionada se propone una cola Pareto de forma {_given(p['alpha'])}.")
        variable = "X es el costo de un reclamo perteneciente a esa población; no modela que ocurra un reclamo."
        outcomes = f"Costos desde {_given(p['xmin'])} u. m. en adelante, sin máximo."
        noun, unit = "el costo de un reclamo de esta población", "u. m."
        rationale = "Pareto tipo I representa una cola de potencia por encima de un umbral positivo. No se aplica a los costos que quedaron debajo del umbral."
        rows = [_row("α", _given(p['alpha']), "Forma de la cola: menor α implica mayor peso en costos muy grandes."),
                _row("xₘ", f"{_given(p['xmin'])} u. m.", "Mínimo de la población modelada; no es un desplazamiento adicional.")]
        limitation = "Un tope de cobertura contradice el soporte sin máximo. La media es infinita si α ≤ 1; con 1 < α ≤ 2 la media es finita y la varianza infinita."
        experiment = "¿Qué pasaría si bajás α y conservás el umbral? Observá cómo crece la cola de costos grandes, incluso cuando la mediana sigue existiendo."
        repetition = "La respuesta está condicionada a pertenecer a la población de reclamos seleccionada por el umbral."

    elif distribution_id == "cauchy":
        title = "Un cociente de señales que puede dispararse"
        context = (f"Dos señales U y V son normales estándar independientes. Se transforma su cociente como "
                   f"X = {_given(p['x0'])} + {_given(p['gamma'])} × U/V.")
        variable = "X es el cociente transformado de señales, sin unidades. Denominadores cercanos a cero generan valores extremos."
        outcomes = "Cualquier real, positivo o negativo; V = 0 tiene probabilidad cero."
        noun = "el cociente transformado"
        rationale = "El cociente U/V de dos normales estándar independientes es Cauchy estándar; ubicación y escala producen el modelo elegido."
        rows = [_row("x₀", _given(p['x0']), "Centro y mediana; no es una media porque la media no existe."),
                _row("γ", _given(p['gamma']), "Escala positiva del cociente; no es desviación estándar.")]
        limitation = "Correlación, medias no nulas o filtrar denominadores pequeños cambian el modelo. La media, la varianza y la desviación estándar de Cauchy no existen."
        experiment = "¿Qué pasaría si duplicás γ? Duplicá también las distancias de los límites respecto de x₀ y comprobá que se conserva el área."

    elif distribution_id == "laplace":
        title = "¿Cuánto se equivocará la estimación de tiempo?"
        context = f"El error de una estimación se modela con centro {_given(p['mu'])} horas y escala {_given(p['b'])} horas."
        variable = "X = tiempo real − tiempo estimado, en horas. Un error negativo indica que se terminó antes."
        outcomes = "Errores continuos de cualquier signo."
        noun, unit = "el error de la estimación", "horas"
        rationale = "Laplace permite errores simétricos con un pico agudo y colas exponenciales más pesadas que las normales."
        rows = [_row("μ", f"{_given(p['mu'])} horas", "Centro, media y mediana del error; distinto de cero representa sesgo central."),
                _row("b", f"{_given(p['b'])} horas", f"Escala; la desviación estándar es √2 b = {_n(math.sqrt(2) * p['b'])} horas.")]
        limitation = "Un sesgo asimétrico, colas de potencia o límites rígidos pueden exigir otro modelo. b no es la desviación estándar."
        experiment = "¿Qué pasaría si movés μ una hora? Anticipá cómo se desplaza la probabilidad de cumplir la misma tolerancia de error."

    elif distribution_id == "rayleigh":
        title = "¿La posición quedará fuera del radio permitido?"
        context = (f"Los errores horizontal U y vertical V son normales independientes, de media cero "
                   f"y desviación estándar {_given(p['sigma'])} mm en cada componente.")
        variable = "X = √(U² + V²) es la distancia al punto verdadero, en mm. Las componentes tienen signo; la distancia no."
        outcomes = "Distancias continuas no negativas."
        noun, unit = "la distancia al punto verdadero", "mm"
        rationale = "Rayleigh: magnitud de dos componentes normales centradas, independientes y con igual dispersión."
        rows = [_row("σ", f"{_given(p['sigma'])} mm", "Desviación de cada componente; no es la desviación de la distancia X.")]
        limitation = "Sesgo, correlación o escalas horizontal y vertical distintas invalidan esta Rayleigh simple."
        experiment = "¿Qué pasaría si reducís σ a la mitad? Compará el mismo radio con otro también reducido a la mitad."
        repetition = "El evento puede leerse como posiciones dentro o fuera de un círculo centrado en el punto verdadero."

    elif distribution_id == "gumbel_r":
        title = "¿El máximo anual de temperatura superará el umbral?"
        context = (f"Los máximos anuales de temperatura de una estación hipotética se modelan con ubicación "
                   f"{_given(p['mu'])} °C y escala {_given(p['beta'])} °C, en un régimen estable.")
        variable = "X es la temperatura más alta de un año completo, en °C; no una temperatura diaria elegida al azar."
        outcomes = "Máximos continuos. El modelo matemático tiene soporte real completo."
        noun, unit = "el máximo anual de temperatura", "°C"
        rationale = "Gumbel de máximos puede aproximar máximos de bloques comparables cuando el tipo de cola y la estabilidad del proceso lo justifican."
        rows = [_row("μ", f"{_given(p['mu'])} °C", "Ubicación y moda de los máximos, no su media."),
                _row("β", f"{_given(p['beta'])} °C", "Escala de dispersión de los máximos anuales.")]
        limitation = "No toda serie de máximos es Gumbel. Tendencias, dependencia o límites físicos relevantes exigen revisar esta aproximación. No es un pronóstico real."
        experiment = "¿Qué pasaría si movés μ dos grados y conservás β? Observá cuánto cambia la cola del mismo umbral anual."
        repetition = "La probabilidad se refiere a años completos con ese máximo, no a la fracción de días calurosos."

    elif distribution_id == "genextreme":
        title = "¿Habrá un día de lluvia extrema durante el año?"
        if p['xi'] > 0:
            support = f"X ≥ {_n(p['mu'] - p['sigma'] / p['xi'])} mm; cola derecha pesada."
            tail = "una cola derecha pesada"
        elif p['xi'] < 0:
            support = f"X ≤ {_n(p['mu'] - p['sigma'] / p['xi'])} mm; extremo superior finito."
            tail = "un extremo superior finito"
        else:
            support = "Cualquier real; se toma el límite Gumbel cuando ξ = 0."
            tail = "una cola de tipo Gumbel"
        context = (f"El máximo anual de lluvia acumulada en un día se modela con ξ = {_given(p['xi'])}, "
                   f"μ = {_given(p['mu'])} mm y σ = {_given(p['sigma'])} mm. La forma elegida implica {tail}.")
        variable = "X es la mayor lluvia diaria dentro de un año, en mm; no la lluvia total de todo el año."
        outcomes = support
        noun, unit = "el máximo anual de lluvia diaria", "mm"
        rationale = "GEV modela máximos de bloques y permite tres tipos de cola. Se usa la convención usual ξ; SciPy recibe c = −ξ."
        rows = [_row("ξ", _given(p['xi']), "Forma de la cola; su signo también determina si hay un extremo finito."),
                _row("μ", f"{_given(p['mu'])} mm", "Ubicación de la ley de máximos; no necesariamente su media."),
                _row("σ", f"{_given(p['sigma'])} mm", "Escala de los máximos; no necesariamente su desviación estándar.")]
        limitation = "Los máximos deben provenir de bloques comparables y un régimen estable. La aproximación puede admitir lluvias negativas con ciertos parámetros; revisá su plausibilidad física. No es un pronóstico real."
        experiment = "¿Qué pasaría si cambiás ξ de positivo a negativo con μ y σ fijos? Localizá el nuevo máximo posible antes de calcular una cola."
        repetition = "La probabilidad se refiere a años que contienen un día con ese máximo, no a la frecuencia de lluvia extrema entre días individuales."

    else:  # Registry additions must receive an intentional story.
        raise ValidationError("Falta un caso pedagógico para esta distribución.")

    if distribution_id == "multinomial":
        if event.get("kind") != "joint":
            raise ValidationError("El caso multinomial conjunto necesita un vector de conteos.")
        # Engine validates integer counts, dimensions and their sum before prose.
        result = _result(distribution_id, params, event, unit)
        names = p.get('names', [f"Canal {i + 1}" for i in range(len(p['probabilities']))])
        assignment = ", ".join(f"{name}: {_n(count)}" for name, count in zip(names, event['counts']))
        words = f"que los {_given(p['n'])} pedidos se repartan exactamente así: {assignment}"
        question = f"¿Cuál es la probabilidad de {words}?"
    else:
        words = _event_words(event, noun, unit, spec.kind == "discrete")
        result = _result(distribution_id, params, event, unit)
        question = (f"¿Qué valor de {noun} acumula al menos {_given(100 * event['q'])} % de probabilidad?" if event['kind'] == 'quantile'
                    else f"¿Cuál es la probabilidad de {words}?")
        if distribution_id == "bernoulli" and event['kind'] == 'eq' and event['x'] in {0, 1}:
            question = "¿Cuál es la probabilidad de que el visitante compre?" if event['x'] == 1 else "¿Cuál es la probabilidad de que el visitante no compre?"
        if distribution_id == "nbinom" and event['kind'] in {'le', 'gt', 'eq'}:
            threshold = math.floor(event['x']) if event['kind'] != 'eq' else event['x']
            relation = {'le': 'como máximo', 'gt': 'más de', 'eq': 'exactamente'}[event['kind']]
            words += f". Equivale a {relation} {_n(threshold + p['r'])} contactos totales"
        if distribution_id == "chi2" and s and event['kind'] in {'le', 'gt'}:
            bound = event['x'] * s['null_variance'] / (s['n'] - 1)
            relation = 'como máximo' if event['kind'] == 'le' else 'mayor que'
            words += f". Con estos n y σ₀², equivale a una varianza muestral {relation} {_n(bound)} mm²"
        if distribution_id == "chi2" and s and event['kind'] == 'interval':
            factor = s['null_variance'] / (s['n'] - 1)
            words += (f". Equivale a una varianza muestral entre {_n(event['a'] * factor)} "
                      f"y {_n(event['b'] * factor)} mm²")
        if distribution_id == "chi2" and s and event['kind'] == 'quantile':
            x = quantile(distribution_id, params, event['q'])
            if math.isfinite(x):
                result += f" En unidades de la medición, el umbral de varianza es {_n(x * s['null_variance'] / (s['n'] - 1))} mm²."

    if distribution_id in {'t', 'chi2', 'f'}:
        prefix = "Bajo H₀, " if s else "En esta referencia central, "
        question = prefix + question[0] + question[1].lower() + question[2:]
    return {
        "title": title,
        "context": "Datos ilustrativos. " + context,
        "question": question,
        "variable": variable,
        "outcomes": outcomes,
        "unit": unit,
        "rationale": rationale,
        "parameter_rows": rows,
        "event_words": words + ".",
        "interpretation": result + " " + repetition,
        "limitation": limitation,
        "experiment": experiment,
    }
