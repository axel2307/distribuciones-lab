"""Streamlit entry point. Run: python -m streamlit run app.py."""
from copy import deepcopy
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lab.catalog import REGISTRY, defaults
from lab.calculations import (ValidationError, validate_parameters, probability,
    quantile, moments, format_moment, multinomial_joint, validate_multinomial,
    log_probability)
from lab.content import LESSONS, FOUNDATIONS
from lab.plots import make_charts, category_chart
from lab.state import get_model, replace_model
from lab.case_studies import case_details
from lab.case_math import calculation_steps
from lab.case_inputs import SAMPLE_DEFAULTS, sample_defaults, sample_model

st.set_page_config(page_title="Distribuciones Lab", page_icon="📊", layout="wide")
st.markdown("""<style>
 .block-container {max-width: 1120px; padding-top: 2rem;}
 [data-testid="stSidebar"] {min-width: 310px;}
 [data-testid="stMetricValue"] {font-size: 1.7rem;}
 .katex-display {overflow-x:auto; overflow-y:hidden; padding: .4rem 0;}
 [data-testid="stPlotlyChart"] .main-svg text {fill: currentColor !important;}
 @media (max-width: 600px) {
   .block-container {padding: 1rem;}
   [data-testid="stSidebar"] {min-width: unset;}
 }
 </style>""", unsafe_allow_html=True)

EVENT_LABELS = {"le": "P(X ≤ x)", "gt": "P(X > x)",
    "interval": "P(a ≤ X ≤ b)", "eq": "P(X = k)", "quantile": "Cuantil"}


def number(value):
    if not math.isfinite(float(value)):
        return "Sin valor finito"
    return f"{float(value):.8g}".replace(".", ",")


def models():
    return st.session_state.setdefault("models", {})


def seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def install_model(distribution_id, params=None, event=None):
    """Button callback: update model and existing widgets before rerendering."""
    prefix = f"explore::{distribution_id}"
    for model_key in list(models()):
        if model_key.startswith(prefix + "::"):
            del models()[model_key]
    model = replace_model(models(), prefix, distribution_id, params, event)
    for key in list(st.session_state):
        if key.startswith(prefix + "::"):
            del st.session_state[key]
    st.session_state["notice"] = "Ejemplo cargado: abrí Explorar para cambiar parámetros y calcular el mismo evento." if params else "Valores predeterminados restablecidos."


def param_controls(distribution_id, prefix, model, container, compact=False):
    spec = REGISTRY[distribution_id]
    params = model["parameters"]
    for p in spec.parameters:
        key = f"{prefix}::param::{p.key}"
        cast = int if p.integer else float
        seed(key, cast(params[p.key]))
        params[p.key] = container.number_input(
            f"{p.name} ({p.symbol})", min_value=cast(p.minimum),
            max_value=cast(p.maximum), step=cast(p.step), key=key,
            format="%d" if p.integer else "%.4f",
            help=f"{p.explanation} Restricción matemática: {p.restriction}")
        if not compact:
            container.caption(f"{p.explanation} Matemática: {p.restriction}. "
                              f"Rango de exploración: {p.minimum} a {p.maximum}.")
    if distribution_id == "multinomial":
        count_key = f"{prefix}::category_count"
        seed(count_key, len(params["probabilities"]))
        count = container.number_input("Cantidad de categorías", min_value=2, max_value=5,
                                       step=1, key=count_key)
        container.caption("Categorías mutuamente excluyentes. Ingresá probabilidades que sumen 1; no se ajustan automáticamente.")
        old_p, old_names = params["probabilities"], params["names"]
        names, ps = [], []
        for i in range(count):
            nk, pk = f"{prefix}::name::{i}", f"{prefix}::prob::{i}"
            seed(nk, old_names[i] if i < len(old_names) else f"Categoría {i+1}")
            seed(pk, float(old_p[i]) if i < len(old_p) else 0.0)
            names.append(container.text_input(f"Nombre de categoría {i+1}", key=nk, max_chars=40))
            ps.append(container.number_input(f"Probabilidad p{i+1}", min_value=0.0, max_value=1.0,
                                             step=0.01, format="%.4f", key=pk,
                                             help="0 ≤ pᵢ ≤ 1; la suma de todas las categorías debe ser 1."))
        params["names"], params["probabilities"] = names, ps
        container.caption(f"Suma ingresada: {number(sum(ps))} (debe ser 1).")
    return deepcopy(params)


def event_controls(prefix, model, discrete):
    current = model["event"]
    options = ["le", "gt", "interval"] + (["eq"] if discrete else []) + ["quantile"]
    kind_key = f"{prefix}::event_kind"
    seed(kind_key, current.get("kind", "le") if current.get("kind") in options else "le")
    kind = st.selectbox("Evento o cuantil", options, format_func=EVENT_LABELS.get, key=kind_key)
    event = {"kind": kind}
    if kind == "quantile":
        seed(f"{prefix}::q", float(current.get("q", 0.9)))
        event["q"] = st.number_input("Probabilidad acumulada objetivo q", min_value=0.0,
            max_value=1.0, step=0.01, format="%.4f", key=f"{prefix}::q")
        st.caption("Q(q) es el menor valor con F(x) ≥ q. En q=0 y q=1 se usan los extremos del soporte efectivo, que pueden ser infinitos.")
    elif kind == "interval":
        for bound, label, fallback in [("a", "Límite inferior a (incluido)", 0.0),
                                       ("b", "Límite superior b (incluido)", 2.0)]:
            seed(f"{prefix}::{bound}", float(current.get(bound, fallback)))
            event[bound] = st.number_input(label, key=f"{prefix}::{bound}", step=1.0 if discrete else 0.1,
                                           help="Podés ingresar límites reales. En discretas se incluyen solamente los enteros entre ambos límites.")
    else:
        seed(f"{prefix}::x", float(current.get("x", 1.0)))
        event["x"] = st.number_input("Resultado entero k" if kind == "eq" else "Umbral x",
            key=f"{prefix}::x", step=1.0 if discrete else 0.1)
        if kind == "eq":
            st.caption("k debe ser entero. Un entero fuera del soporte tiene probabilidad cero.")
    model["event"] = deepcopy(event)
    return event


def event_latex(event, variable="X"):
    def exact(value):
        text = repr(float(value)).removesuffix(".0")
        if "e" in text:
            mantissa, exponent = text.split("e")
            return rf"{mantissa}\times10^{{{int(exponent)}}}"
        return text
    kind = event["kind"]
    if kind == "interval":
        return rf"P({exact(event['a'])}\le {variable}\le {exact(event['b'])})"
    if kind == "quantile":
        return rf"Q({exact(event['q'])})"
    return rf"P({variable}{ {'le':r'\le','gt':'>','eq':'='}[kind]}{exact(event['x'])})"


def show_probability(value, log_value=None):
    if value == 0 and log_value is not None and math.isfinite(log_value):
        st.warning("La probabilidad es positiva pero demasiado pequeña para representarla como un número de precisión doble. No es un cero exacto.")
        st.write(f"Logaritmo natural de la probabilidad: {number(log_value)}. "
                 f"Orden de magnitud: 10 elevado a {number(log_value / math.log(10))}.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad", number(value))
    c2.metric("Porcentaje", f"{number(100 * value)} %")
    if value == 0:
        st.write("Bajo estos supuestos, el evento tiene probabilidad cero.")
    else:
        st.write(f"Bajo estos supuestos, aproximadamente **{number(100 * value)} de cada 100 casos** cumplen el evento. Es una frecuencia esperada al repetir el modelo, no una garantía para una muestra de 100.")


def calculator(distribution_id, params, event):
    st.latex(event_latex(event))
    if event["kind"] == "quantile":
        qx = quantile(distribution_id, params, event["q"])
        if not math.isfinite(qx):
            st.info("Este cuantil es un extremo infinito del soporte. Elegí 0 < q < 1 para obtener un umbral finito y sombrear un evento.")
            return None
        st.metric("Cuantil", number(qx))
        plot_event = {"kind": "le", "x": qx}
        actual = probability(distribution_id, params, plot_event)
        st.write(f"El umbral es {number(qx)} y P(X ≤ {number(qx)}) = {number(actual)} ({number(100 * actual)} %).")
        if REGISTRY[distribution_id].kind == "continuous" and abs(actual-event["q"]) > 1e-8:
            st.warning("El cuantil está demasiado cerca de un extremo para representarlo con suficiente precisión. El umbral redondeado no logra exactamente la probabilidad objetivo; la acumulada y el sombreado corresponden al umbral representable mostrado.")
        if REGISTRY[distribution_id].kind == "discrete":
            st.caption("Por los saltos de la acumulada, la probabilidad lograda puede superar q; F(Q−1) < q ≤ F(Q), para 0 < q < 1.")
        return plot_event
    value = probability(distribution_id, params, event)
    show_probability(value, log_probability(distribution_id, params, event))
    return event


def show_moments(distribution_id, params):
    result = moments(distribution_id, params)
    labels = {"mean": "Media", "median": "Mediana", "variance": "Varianza", "std": "Desvío estándar"}
    st.subheader("Resumen del modelo")
    st.table([{"Cantidad": labels[k], "Valor": format_moment(v)} for k, v in result.items()])
    st.caption("La mediana se obtiene con el cuantil 0,5; puede no ser única en discretas. Usamos varianza central: requiere media finita. Si la media no existe o diverge, la varianza y el desvío no están definidos; si la media es finita pero el segundo momento diverge, son infinitos. Un error numérico se informa por separado.")


def show_charts(distribution_id, params, event, prefix, model):
    seed(f"{prefix}::coverage", float(model["coverage"]))
    coverage = st.select_slider("Amplitud del rango por cuantiles", options=[0.98, 0.99, 0.999, 0.9999],
        format_func=lambda v: f"{100*v:g} % central", key=f"{prefix}::coverage")
    model["coverage"] = coverage
    first, second, meta = make_charts(distribution_id, params, event, coverage=coverage)
    st.plotly_chart(first, width="stretch", key=f"{prefix}::mass", config={"scrollZoom": False})
    st.plotly_chart(second, width="stretch", key=f"{prefix}::cdf", config={"scrollZoom": False})
    st.caption(f"Rango mostrado: {number(meta['left'])} a {number(meta['right'])}. Probabilidad fuera del gráfico: {number(100*meta['excluded_probability'])} %. Las alturas y la acumulada conservan sus valores originales: no se renormaliza el fragmento visible.")
    for note in meta["notes"]:
        st.info(note)
    st.caption("Pasá el cursor para ver valores. Usá la barra del gráfico para acercar o alejar; el selector de amplitud vuelve a calcular el rango. Azul: modelo; naranja: evento seleccionado.")


def univariate_explore(distribution_id, params, prefix, model, title=None):
    st.subheader(title or "Calculadora de probabilidades")
    discrete = REGISTRY[distribution_id].kind == "discrete"
    if discrete:
        st.info("Cada barra representa P(X=k). La acumulada es escalonada; ≤ incluye el umbral y > lo excluye.")
    else:
        st.info("La altura de la densidad no es una probabilidad. La probabilidad es el área sombreada y P(X=x)=0 para cualquier punto exacto.")
    event = event_controls(prefix, model, discrete)
    try:
        plot_event = calculator(distribution_id, params, event)
        if plot_event is not None:
            show_charts(distribution_id, params, plot_event, prefix, model)
        show_moments(distribution_id, params)
    except ValidationError as exc:
        st.error(str(exc))


def count_controls(params, prefix, initial=None):
    n, ps, names = params["n"], params["probabilities"], params["names"]
    base = initial or [int(n*p) for p in ps]
    if initial is None:
        base[-1] += n - sum(base)
    counts = []
    for i, name in enumerate(names):
        key = f"{prefix}::count::{i}"
        seed(key, int(base[i]) if i < len(base) else 0)
        counts.append(st.number_input(f"Conteo de {name} (k{i+1})", min_value=0, step=1, key=key))
    st.caption(f"Suma de conteos: {sum(counts)}. Debe coincidir con n = {n}.")
    return counts


def multinomial_explore(params, prefix, model):
    n, ps, names = params["n"], params["probabilities"], params["names"]
    expected = [n*p for p in ps]
    st.subheader("Categorías y conteos esperados")
    st.table([{"Categoría": name, "Probabilidad": number(p), "Conteo esperado n·p": number(e)} for name,p,e in zip(names,ps,expected)])
    st.plotly_chart(category_chart(names, ps, expected), width="stretch", key=f"{prefix}::categories")
    st.info("Este gráfico muestra probabilidades de categorías en un ensayo. No representa toda la distribución conjunta de los conteos. Los conteos suman n y no son independientes.")
    st.subheader("Probabilidad conjunta de un vector de conteos")
    current = model["event"]
    counts = count_controls(params, prefix, current.get("counts"))
    model["event"] = {"kind": "joint", "counts": counts}
    try:
        joint = multinomial_joint(n, ps, counts)
        # logpmf only after explicit validation; never let SciPy repair inputs.
        from scipy.stats import multinomial
        log_joint = float(multinomial.logpmf(counts, n=n, p=ps))
        st.latex(rf"P(\mathbf{{K}}=({','.join(map(str,counts))}))")
        show_probability(joint, log_joint)
    except ValidationError as exc:
        st.error(str(exc))
    st.divider()
    st.subheader("Distribución marginal de una categoría")
    seed(f"{prefix}::category", min(model.get("category", 0), len(names)-1))
    if st.session_state[f"{prefix}::category"] >= len(names):
        st.session_state[f"{prefix}::category"] = 0
    index = st.selectbox("Categoría del conteo marginal", range(len(names)), format_func=lambda i:names[i],
                          key=f"{prefix}::category")
    model["category"] = index
    st.latex(rf"K_{{{index+1}}}\sim\operatorname{{Binomial}}(n={n},p={ps[index]:g})")
    st.write(f"X = conteo de «{names[index]}». Todos los gráficos, estadísticas y probabilidades siguientes corresponden solamente a esta marginal. No hay una CDF escalar conjunta de todo el vector.")
    marginal_prefix = f"{prefix}::marginal::{index}"
    marginal = get_model(models(), marginal_prefix, "binom", {"n":n,"p":ps[index]})
    univariate_explore("binom", {"n":n,"p":ps[index]}, marginal_prefix, marginal, "Calculadora de la marginal")


def understand(distribution_id):
    spec, lesson = REGISTRY[distribution_id], LESSONS[distribution_id]
    st.subheader("La variable y su soporte")
    st.write(lesson["variable"])
    st.latex(spec.support)
    st.subheader("Símbolos de la fórmula")
    for symbol, meaning in spec.symbols.items():
        st.markdown(f"- **${symbol}$**: {meaning}")
    st.subheader("Qué pasa al cambiar los parámetros")
    for p in spec.parameters:
        st.markdown(f"**{p.name} (${p.symbol}$).** {lesson['effects'].get(p.key, p.explanation)}")
        st.caption(f"Restricción matemática: {p.restriction}. Límites de la interfaz: {p.minimum} a {p.maximum}; son límites prácticos, no el dominio completo de la familia.")
    if distribution_id == "multinomial":
        st.write(lesson["effects"].get("probabilities", "Al aumentar pᵢ aumenta el conteo esperado n·pᵢ. Las probabilidades deben sumar 1."))
    st.subheader("Supuestos")
    for item in lesson["assumptions"]:
        st.markdown(f"- {item}")
    st.subheader("Aplicaciones habituales")
    for item in lesson["applications"]:
        st.markdown(f"- {item}")
    st.write("**Cuándo sería inadecuada.** " + lesson["inappropriate"])
    st.warning("Error frecuente: " + lesson["error"])
    with st.expander("Cuatro ideas para interpretar cualquier distribución", expanded=True):
        for title, text in FOUNDATIONS.items():
            st.markdown(f"**{title}.** {text}")
    st.markdown(f"Fuente técnica: [Documentación oficial de SciPy — {spec.scipy_name}]({spec.source_url}).")


def reset_case(distribution_id):
    prefix = f"case::{distribution_id}"
    for key in list(st.session_state):
        if key.startswith(prefix + "::"):
            del st.session_state[key]
    for key in list(models()):
        if key == prefix or key.startswith(prefix + "::"):
            del models()[key]


def sample_controls(distribution_id, prefix, model):
    """Edit measurements, then derive the reference law rather than invent df."""
    initial = model.get("samples", sample_defaults(distribution_id))
    fields = {
        "t": [("n", "Cantidad de mediciones n", 2, 1001, 1),
              ("mean", "Media muestral (mL)", None, None, 0.5),
              ("std", "Desvío muestral s (mL)", 0.0, None, 0.5),
              ("null", "Media de referencia μ₀ (mL)", None, None, 0.5)],
        "chi2": [("n", "Cantidad de mediciones n", 2, 1001, 1),
                 ("variance", "Varianza muestral s² (mm²)", 0.0, None, 0.1),
                 ("null_variance", "Varianza de referencia σ₀² (mm²)", 0.0, None, 0.1)],
        "f": [("n1", "Mediciones del proceso 1", 2, 1001, 1),
              ("n2", "Mediciones del proceso 2", 2, 1001, 1),
              ("variance1", "Varianza del proceso 1 (mm²)", 0.0, None, 0.5),
              ("variance2", "Varianza del proceso 2 (mm²)", 0.0, None, 0.5)],
    }
    columns = st.columns(2)
    samples = {}
    for i, (key, label, low, high, step) in enumerate(fields[distribution_id]):
        widget = f"{prefix}::sample::{key}"
        seed(widget, initial[key])
        samples[key] = columns[i % 2].number_input(label, min_value=low,
            max_value=high, step=step, key=widget)
    tail_key = f"{prefix}::sample_tail"
    seed(tail_key, model.get("sample_tail", "gt"))
    tail = st.selectbox("Dirección de la pregunta", ["gt", "le"], key=tail_key,
        format_func=lambda value: "Mayor que el estadístico observado" if value == "gt" else "Hasta el estadístico observado")
    st.caption("La dirección de una prueba se fija antes de observar los datos. Cambiarla acá permite comparar colas; no justifica elegir la más pequeña después de mirar una muestra.")
    model["samples"], model["sample_tail"] = deepcopy(samples), tail
    return samples, tail


def case_plot(distribution_id, params, event, details, prefix, model):
    if distribution_id == "multinomial":
        joint = probability(distribution_id, params, event)
        vector = ", ".join(f"{name}: {k}" for name, k in zip(params["names"], event["counts"]))
        figure = go.Figure(go.Bar(x=["Este reparto exacto", "Cualquier otro reparto"],
            y=[joint, 1 - joint], marker_color=["#F59E0B", "#3B82F6"],
            text=[f"{100 * joint:.4g} %", f"{100 * (1 - joint):.4g} %"],
            textposition="auto", customdata=[vector, "Todos los vectores distintos del seleccionado"],
            hovertemplate="%{x}<br>%{customdata}<br>Probabilidad: %{y:.8g}<extra></extra>"))
        figure.update_layout(title_text="El reparto completo frente a los demás repartos",
            yaxis_title="Probabilidad", yaxis_tickformat=".0%", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=35, r=20, t=55, b=40))
        st.plotly_chart(figure, width="stretch", key=f"{prefix}::event_plot")
        st.caption("Naranja: todos los órdenes de llegada que producen exactamente el vector pedido. Azul: todos los otros repartos. Las dos probabilidades suman 1.")
        names, counts = params["names"], event["counts"]
        st.table([{"Canal": name, "Conteo pedido por el evento": k,
                   "Conteo esperado n·p": number(params["n"] * p)}
                  for name, k, p in zip(names, counts, params["probabilities"])])
        st.caption("El evento conjunto fija todos los conteos de la tabla a la vez. No existe una CDF escalar para todo este vector.")
        with st.expander("Comparar con la marginal de un solo canal"):
            category_key = f"{prefix}::joint_category"
            seed(category_key, min(model.get("case_category", 0), len(names) - 1))
            if st.session_state[category_key] >= len(names):
                st.session_state[category_key] = 0
            category = st.selectbox("Ver el conteo de un canal", range(len(names)),
                format_func=lambda i: names[i], key=category_key)
            model["case_category"] = category
            marginal_params = {"n": params["n"], "p": params["probabilities"][category]}
            marginal_event = {"kind": "eq", "x": counts[category]}
            marginal, _, _ = make_charts("binom", marginal_params, marginal_event)
            marginal.update_xaxes(title_text=f"Pedidos por {names[category]}")
            marginal.update_layout(title_text=f"Una parte del reparto: {names[category]}")
            st.plotly_chart(marginal, width="stretch", key=f"{prefix}::marginal_plot")
            marginal_p = probability("binom", marginal_params, marginal_event)
            st.caption(f"La barra naranja muestra solamente P(K{category+1}={counts[category]}) = {number(marginal_p)}. Incluye todos los repartos de los otros canales; no es la probabilidad del vector conjunto.")
        return
    plot_event = event
    if event["kind"] == "quantile":
        threshold = quantile(distribution_id, params, event["q"])
        if not math.isfinite(threshold):
            st.info("Este cuantil es un extremo infinito: no hay un umbral finito que graficar.")
            return
        plot_event = {"kind": "le", "x": threshold}
    figure, cdf, meta = make_charts(distribution_id, params, plot_event,
                                   coverage=model["coverage"])
    axis = f"X ({details['unit']})" if details['unit'] else "Valor de X"
    figure.update_xaxes(title_text=axis)
    cdf.update_xaxes(title_text=axis)
    st.plotly_chart(figure, width="stretch", key=f"{prefix}::event_plot")
    if event["kind"] == "quantile":
        achieved = probability(distribution_id, params, plot_event)
        st.caption(f"Naranja: acumulada hasta el cuantil mostrado. Probabilidad lograda: {number(achieved)}; puede diferir del objetivo por saltos o precisión numérica.")
    else:
        st.caption("Naranja: exactamente el evento de la pregunta. En discretas se suman barras; en continuas se mide el área, no la altura.")
    st.caption(f"Ventana visible: {number(meta['left'])} a {number(meta['right'])}. Probabilidad fuera de la ventana: {number(100 * meta['excluded_probability'])} %. El resultado usa la distribución completa.")
    for note in meta["notes"]:
        if "no se renormaliza" not in note:
            st.info(note)
    with st.expander("Ver también la probabilidad acumulada"):
        st.plotly_chart(cdf, width="stretch", key=f"{prefix}::cdf")


def practical_case(distribution_id):
    case = LESSONS[distribution_id]["example"]
    prefix = f"case::{distribution_id}"
    model = get_model(models(), prefix, distribution_id, case["parameters"], case["event"])
    st.caption("CASO GUIADO · Pregunta → modelo → cálculo → gráfico → respuesta")
    st.caption("Datos y supuestos ilustrativos para aprender; no son una estimación real ni un pronóstico. Editá acá: toda la explicación se recalcula con estos datos.")
    question_area = st.container(border=True)
    samples = None
    valid = True
    with st.expander("Datos del caso · editá y compará", expanded=True):
        if distribution_id in SAMPLE_DEFAULTS:
            mode_key = f"{prefix}::input_mode"
            seed(mode_key, model.get("input_mode", "Mediciones de la muestra"))
            mode = st.radio("Cómo definir el caso", ["Mediciones de la muestra", "Parámetros del estadístico"],
                key=mode_key, horizontal=True)
            model["input_mode"] = mode
        else:
            mode = None
        if mode == "Mediciones de la muestra":
            samples, tail = sample_controls(distribution_id, prefix, model)
            try:
                params, event = sample_model(distribution_id, samples, tail)
                model["parameters"], model["event"] = deepcopy(params), deepcopy(event)
            except ValidationError as exc:
                valid = False
                st.error(str(exc))
                params, event = model["parameters"], model["event"]
        else:
            params = param_controls(distribution_id, prefix, model, st, compact=True)
            if distribution_id == "multinomial":
                counts = count_controls(params, prefix, model["event"].get("counts"))
                event = {"kind": "joint", "counts": counts}
                model["event"] = event
            else:
                event = event_controls(prefix, model, REGISTRY[distribution_id].kind == "discrete")
            if mode is not None:
                st.caption("Exploración de una distribución de referencia. Los grados de libertad pueden ser no enteros; en ese caso no representan el diseño muestral clásico de este caso.")
        st.button("Reiniciar este caso", key=f"{prefix}::reset", on_click=reset_case,
                  args=(distribution_id,))
    if valid:
        try:
            validate_parameters(distribution_id, params)
            details = case_details(distribution_id, params, event, samples=samples)
            steps = calculation_steps(distribution_id, params, event, samples=samples)
        except ValidationError as exc:
            valid = False
            st.error(str(exc))
    if not valid:
        with question_area:
            st.subheader("Revisemos los datos")
            st.write("Corregí los valores señalados para reconstruir la pregunta, el cálculo y el gráfico. No se conserva una respuesta de la variante anterior.")
        st.button("Cargar este ejemplo", key=f"{prefix}::load", disabled=True)
        return
    with question_area:
        st.subheader("1. " + details["title"])
        st.write(details["context"])
        st.markdown("**" + details["question"] + "**")
    st.subheader("2. De la situación al modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Qué puede pasar**")
        st.write(details["variable"])
        st.caption(details["outcomes"])
    with c2:
        st.markdown("**Por qué esta distribución**")
        st.write(details["rationale"])
    with st.expander("Qué significa cada parámetro", expanded=True):
        st.table(details["parameter_rows"])
    st.subheader("3. Resolvámoslo paso a paso")
    st.write(details["event_words"])
    if distribution_id == "multinomial":
        st.latex(rf"P(\mathbf{{K}}=({','.join(map(str,event['counts']))}))")
    else:
        st.latex(event_latex(event))
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}. {step['title']}**")
        st.write(step["explanation"])
        st.latex(step["latex"])
    if event["kind"] == "quantile":
        result = quantile(distribution_id, params, event["q"])
        st.metric("Umbral del caso", number(result), help=details["unit"])
    else:
        result = probability(distribution_id, params, event)
        log_result = log_probability(distribution_id, params, event)
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad del evento", number(result))
        c2.metric("Porcentaje", f"{number(100 * result)} %")
        if result == 0 and math.isfinite(log_result):
            st.warning(f"Probabilidad positiva demasiado pequeña para precisión doble; log(P) = {number(log_result)}. No es un cero exacto.")
    st.subheader("4. Veamos el mismo evento")
    try:
        case_plot(distribution_id, params, event, details, prefix, model)
    except ValidationError as exc:
        st.info("El cálculo es válido, pero este gráfico tiene una limitación numérica: " + str(exc))
    st.subheader("5. Qué nos dice la respuesta")
    st.success(details["interpretation"])
    st.caption(details["limitation"])
    st.info(details["experiment"] + " Cambiá los datos arriba y observá juntos el desarrollo, el gráfico y la respuesta.")
    st.button("Cargar este ejemplo", key=f"{prefix}::load", on_click=install_model,
        args=(distribution_id, deepcopy(params), deepcopy(event)),
        help="También podés copiar esta variante al explorador general.")


st.sidebar.title("Distribuciones Lab")
st.sidebar.caption("Elegí un modelo y explorá qué cambia.")
kind = st.sidebar.radio("Tipo de distribución", ["Continuas", "Discretas"], key="distribution_type")
ids = [k for k,v in REGISTRY.items() if (v.kind == "continuous") == (kind == "Continuas")]
distribution_id = st.sidebar.selectbox("Distribución", ids, format_func=lambda k:REGISTRY[k].name,
                                       key=f"distribution::{kind}")
spec = REGISTRY[distribution_id]
prefix = f"explore::{distribution_id}"
model = get_model(models(), prefix, distribution_id)
st.sidebar.divider()
st.sidebar.caption("Los datos del caso se editan en Caso práctico. Estos controles pertenecen a Explorar.")
with st.sidebar.expander("Parámetros del explorador"):
    st.button("Restablecer valores predeterminados", on_click=install_model, args=(distribution_id,), width="stretch")
    params = param_controls(distribution_id, prefix, model, st)
    st.caption("Los límites de los controles son prácticos. Las restricciones matemáticas se explican en la ayuda de cada parámetro.")

st.title("Distribuciones Lab")
st.caption("24 familias para aprender probabilidad · Modelos ilustrativos · Cálculos con SciPy")
st.header(spec.name)
st.write(LESSONS[distribution_id]["intuition"])
with st.expander("Fórmula de referencia"):
    st.latex(spec.formula)
explore, theory, case_tab = st.tabs(["Explorar", "Entender", "Caso práctico"],
    default="Caso práctico", key="lab_tabs")
if "notice" in st.session_state:
    st.success(st.session_state.pop("notice"))
with explore:
    try:
        validate_parameters(distribution_id,params)
        if distribution_id == "multinomial":
            multinomial_explore(params,prefix,model)
        else:
            univariate_explore(distribution_id,params,prefix,model)
    except ValidationError as exc:
        st.error(str(exc))
        st.info("Corregí los parámetros en la barra lateral para calcular. Podés seguir consultando Entender y Caso práctico.")
with theory:
    understand(distribution_id)
with case_tab:
    practical_case(distribution_id)
st.divider()
st.caption("Estas 24 familias son una selección educativa, no un catálogo exhaustivo. El resultado de un cálculo es válido bajo los supuestos del modelo; elegir la familia adecuada requiere conocer el fenómeno.")
