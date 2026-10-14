"""Bounded, faithful Plotly views of the active distribution.

The window is a *view* of the original law: neither PMFs nor PDFs are
renormalized. Probability outside the window is computed from CDF/SF, not
from a numerical integral of the displayed polyline.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .calculations import ValidationError, event_mask, frozen_dist, quantile, validate_event
from .catalog import REGISTRY

MAX_DISCRETE_POINTS = 1200
MAX_CONTINUOUS_POINTS = 1000
BLUE = "#3B82F6"
AMBER = "#F59E0B"
FILL = "rgba(245,158,11,0.30)"


def _evaluate(function, values):
    # Known endpoint limits (PDF infinity; geometric p=1 log(0)) can emit
    # NumPy warnings while returning the correct value. Every caller still
    # checks finiteness where a finite result is mathematically required.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return function(values)


def _layout(figure, title, ordinate, left, right, *, discrete=False):
    """Let Streamlit provide light/dark colors; traces stay high contrast."""
    figure.update_layout(
        title=dict(text=title, font=dict(size=19)),
        xaxis_title="Valor de X",
        yaxis_title=ordinate,
        height=390,
        margin=dict(l=45, r=20, t=55, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.20, x=0),
        hovermode="closest",
        dragmode="pan",
    )
    padding = 0.6 if discrete else 0.015 * (right - left)
    figure.update_xaxes(
        range=[left - padding, right + padding],
        gridcolor="rgba(128,128,128,0.16)",
        zerolinecolor="rgba(128,128,128,0.35)",
        nticks=9,
    )
    figure.update_yaxes(
        rangemode="tozero", gridcolor="rgba(128,128,128,0.16)",
        zerolinecolor="rgba(128,128,128,0.35)",
    )
    if ordinate.startswith("F("):
        figure.update_yaxes(range=[0, 1.025], tickformat=".0%")
    return figure


def _boundary_values(event):
    if event["kind"] == "interval":
        return [float(event["a"]), float(event["b"])]
    if event["kind"] in {"le", "gt", "eq"}:
        return [float(event["x"])]
    return []


def _boundaries(figures, event, left, right):
    for value in set(_boundary_values(event)):
        if left <= value <= right:
            for figure in figures:
                figure.add_vline(x=value, line=dict(color=AMBER, width=1, dash="dot"))


def _metadata(left, right, excluded, notes, max_points):
    # Only a roundoff correction, never a correction of conceptual inputs.
    if not math.isfinite(excluded) or excluded < -1e-12 or excluded > 1 + 1e-12:
        raise ValidationError("No se pudo calcular la probabilidad fuera del gráfico.")
    excluded = float(min(1.0, max(0.0, excluded)))
    if excluded > 1e-12:
        notes.append(
            "El gráfico muestra parte del soporte. La masa o el área visible "
            "no se renormaliza; el cálculo de probabilidades usa la distribución completa."
        )
    return dict(
        left=float(left), right=float(right), excluded_probability=excluded,
        notes=notes, truncated=excluded > 1e-12, max_points=max_points,
    )


def _discrete_charts(distribution_id, params, event, coverage):
    rv = frozen_dist(distribution_id, params)
    notes = []
    support_left = quantile(distribution_id, params, 0.0)
    support_right = quantile(distribution_id, params, 1.0)
    tail = (1 - coverage) / 2
    if (
        math.isfinite(support_left) and math.isfinite(support_right)
        and support_right - support_left + 1 <= MAX_DISCRETE_POINTS
    ):
        left, right = int(math.ceil(support_left)), int(math.floor(support_right))
    else:
        left = int(quantile(distribution_id, params, tail))
        right = int(quantile(distribution_id, params, 1 - tail))
    if right - left + 1 > MAX_DISCRETE_POINTS:
        median = int(quantile(distribution_id, params, 0.5))
        left = max(left, median - MAX_DISCRETE_POINTS // 2)
        left = min(left, right - MAX_DISCRETE_POINTS + 1)
        right = left + MAX_DISCRETE_POINTS - 1
        notes.append(
            f"Para mantener la interacción fluida se muestran como máximo "
            f"{MAX_DISCRETE_POINTS} valores enteros consecutivos, sin saltear barras."
        )
    x = np.arange(left, right + 1, dtype=np.int64)
    mass = np.asarray(_evaluate(rv.pmf, x), dtype=float)
    cdf = np.asarray(_evaluate(rv.cdf, x), dtype=float)
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(cdf)):
        raise ValidationError("No se pudo representar esta función de masa con precisión.")
    selected = np.asarray(event_mask(x, event), dtype=bool)
    figure = go.Figure()
    # Separate legend items also distinguish selection without relying only on color.
    for mask, name, color in (
        (~selected, "Fuera del evento", BLUE), (selected, "Evento seleccionado", AMBER)
    ):
        figure.add_trace(go.Bar(
            x=x[mask], y=mass[mask], name=name, marker_color=color,
            width=0.80, hovertemplate="X = %{x}<br>P(X = x) = %{y:.8g}<extra></extra>",
        ))
    figure.update_layout(barmode="overlay", bargap=0.15)
    _layout(figure, "Masa de probabilidad", "P(X = x)", left, right, discrete=True)

    accumulated = go.Figure()
    # hv is horizontal then vertical: F(k) holds on [k,k+1), with the
    # jump at the NEXT integer, exactly the right-continuous discrete CDF.
    step_x = np.concatenate(([left - 1], x, [right + 1]))
    step_y = np.concatenate(([_evaluate(rv.cdf, left - 1)], cdf, [_evaluate(rv.cdf, right + 1)]))
    accumulated.add_trace(go.Scatter(
        x=step_x, y=step_y, mode="lines", line=dict(color=BLUE, width=2, shape="hv"),
        name="Acumulada F(x)",
        hovertemplate="x = %{x}<br>P(X ≤ x) = %{y:.8g}<extra></extra>",
    ))
    accumulated.add_trace(go.Scatter(
        x=x[selected], y=cdf[selected], mode="markers",
        marker=dict(color=AMBER, size=7, symbol="diamond"),
        name="Valores del evento", hovertemplate="X = %{x}<br>F(x) = %{y:.8g}<extra></extra>",
    ))
    _layout(accumulated, "Distribución acumulada", "F(x) = P(X ≤ x)", left, right, discrete=True)
    _boundaries((figure, accumulated), event, left - 0.6, right + 0.6)
    excluded = float(_evaluate(rv.cdf, left - 1) + _evaluate(rv.sf, right))
    return figure, accumulated, _metadata(left, right, excluded, notes, MAX_DISCRETE_POINTS)


def _continuous_grid(rv, left, right, event):
    """A finite linear grid plus quantile points for unequal tail scales."""
    linear = np.linspace(left, right, 620)
    # Do not add arbitrarily close quantiles to a singular finite endpoint:
    # the unbounded density is explained rather than assigned a finite height.
    endpoint_pdf = np.asarray(_evaluate(rv.pdf, [left, right]), dtype=float)
    if np.all(np.isfinite(endpoint_pdf)):
        cdf_left, cdf_right = float(_evaluate(rv.cdf, left)), float(_evaluate(rv.cdf, right))
        probabilities = np.linspace(max(1e-6, cdf_left), min(1 - 1e-6, cdf_right), 340)
        extra = np.asarray(_evaluate(rv.ppf, probabilities), dtype=float)
        extra = extra[np.isfinite(extra) & (extra >= left) & (extra <= right)]
    else:
        extra = np.array([], dtype=float)
    boundaries = np.array([v for v in _boundary_values(event) if left <= v <= right])
    if event["kind"] == "le":
        event_left, event_right = left, min(right, event["x"])
    elif event["kind"] == "gt":
        event_left, event_right = max(left, event["x"]), right
    else:
        event_left, event_right = max(left, event["a"]), min(right, event["b"])
    # Even a narrow event next to a singular endpoint gets a visible polygon;
    # its probability still comes from CDF/SF, never the coarse filled area.
    event_grid = np.linspace(event_left, event_right, 30) if event_left < event_right else []
    return np.unique(np.concatenate((linear, extra, boundaries, event_grid)))


def _cdf_grid(rv, left, right, event):
    """Sample probability space even when a singular PDF prevents doing so."""
    lower = float(_evaluate(rv.cdf, left))
    upper = float(_evaluate(rv.cdf, right))
    probabilities = np.linspace(max(lower, 1e-8), min(upper, 1 - 1e-8), 600)
    x = np.asarray(_evaluate(rv.ppf, probabilities), dtype=float)
    x = x[np.isfinite(x) & (x >= left) & (x <= right)]
    boundaries = [v for v in _boundary_values(event) if left <= v <= right]
    return np.unique(np.concatenate((np.linspace(left, right, 350), x, boundaries)))


def _practical_window(distribution_id, params, rv, left, right, coverage, notes):
    """Keep enormous tails from collapsing the central curve into one pixel.

    This is an explicit visualization limit, never a change to the law. The
    four UI coverage settings progressively widen this limit too. Population
    variance is deliberately unused: it need not exist for these families.
    """
    native_scale = float(rv.kwds.get("scale", 1.0))
    if distribution_id == "chi2":
        native_scale *= math.sqrt(2 * params["df"])
    elif distribution_id == "gamma":
        native_scale *= math.sqrt(params["alpha"])
    max_width = 60 * native_scale * math.sqrt(0.01 / (1 - coverage))
    if right - left <= max_width:
        return left, right
    median = quantile(distribution_id, params, 0.5)
    support_left, support_right = rv.support()
    if math.isfinite(support_left) and median - left > max_width:
        # A heavy positive law's median can itself be extremely remote
        # (Pareto alpha=.1). Show its finite starting region rather than a
        # negligible sliver centered thousands of scales away.
        right = min(right, left + max_width)
    elif math.isfinite(support_right) and right - median > max_width:
        left = max(left, right - max_width)
    else:
        left = max(left, median - max_width / 2)
        right = min(right, left + max_width)
        left = max(float(support_left), left) if math.isfinite(support_left) else left
    notes.append(
        "Se limitó el ancho visible para que una cola muy pesada no comprima "
        "toda la curva en un píxel. La cobertura real puede ser menor que la "
        "solicitada y se informa con la probabilidad fuera del gráfico. "
        "Aumentá la cobertura para ampliar este límite práctico; no se "
        "modificó la distribución ni el cálculo del evento."
    )
    return left, right


def _continuous_charts(distribution_id, params, event, coverage):
    rv = frozen_dist(distribution_id, params)
    notes = []
    support_left, support_right = rv.support()
    tail = (1 - coverage) / 2
    lower_quantile = quantile(distribution_id, params, tail)
    upper_quantile = quantile(distribution_id, params, 1 - tail)
    left, right = lower_quantile, upper_quantile
    core_span = upper_quantile - lower_quantile
    # A finite endpoint can be remote too (e.g. GEV with xi close to zero).
    # Include nearby boundaries, but do not sacrifice all plot resolution to
    # an endpoint carrying negligible probability far from the central mass.
    both_finite = math.isfinite(support_left) and math.isfinite(support_right)
    if math.isfinite(support_left) and (both_finite or lower_quantile - support_left <= 10 * core_span):
        left = float(support_left)
    if math.isfinite(support_right) and (both_finite or support_right - upper_quantile <= 10 * core_span):
        right = float(support_right)
    left, right = _practical_window(distribution_id, params, rv, left, right, coverage, notes)
    if not math.isfinite(left) or not math.isfinite(right) or left >= right:
        raise ValidationError("No se pudo obtener un rango finito para estos parámetros.")
    x = _continuous_grid(rv, left, right, event)
    density = np.asarray(_evaluate(rv.pdf, x), dtype=float)
    cdf_x = _cdf_grid(rv, left, right, event)
    cumulative = np.asarray(_evaluate(rv.cdf, cdf_x), dtype=float)
    valid_density = np.isfinite(density)
    if not np.all(valid_density):
        notes.append(
            "La densidad diverge en un extremo del soporte. Ese punto no se dibuja "
            "con una altura ficticia; las probabilidades se calculan con la CDF/SF. "
            "La curva y el sombreado son una aproximación visual por puntos: "
            "puede haber mucha probabilidad cerca del extremo singular que "
            "la altura finita muestreada no permite apreciar. Consultá el "
            "resultado numérico y la acumulada para cuantificarla."
        )
    if not np.any(valid_density) or not np.all(np.isfinite(cumulative)):
        raise ValidationError("No se pudo representar esta densidad con precisión.")

    # Endpoints have zero probability for continuous laws. Include the selected
    # boundary in the fill so its polygon stops at the exact requested limit.
    if event["kind"] == "le":
        mask = x <= event["x"]
    elif event["kind"] == "gt":
        mask = x >= event["x"]
    else:
        mask = np.asarray(event_mask(x, event), dtype=bool)
    selected = mask & valid_density
    figure = go.Figure()
    if np.count_nonzero(selected) > 1:
        figure.add_trace(go.Scatter(
            x=x[selected], y=density[selected], mode="lines", fill="tozeroy",
            fillcolor=FILL, line=dict(color=AMBER, width=1), name="Área del evento",
            hovertemplate="x = %{x:.6g}<br>f(x) = %{y:.6g}<extra></extra>",
        ))
    figure.add_trace(go.Scatter(
        x=x[valid_density], y=density[valid_density], mode="lines",
        line=dict(color=BLUE, width=2), name="Densidad f(x)",
        hovertemplate="x = %{x:.6g}<br>Densidad f(x) = %{y:.6g}<extra></extra>",
    ))
    _layout(figure, "Función de densidad", "Densidad f(x), no probabilidad", left, right)

    accumulated = go.Figure()
    accumulated.add_trace(go.Scatter(
        x=cdf_x, y=cumulative, mode="lines", line=dict(color=BLUE, width=2), name="Acumulada F(x)",
        hovertemplate="x = %{x:.6g}<br>P(X ≤ x) = %{y:.8g}<extra></extra>",
    ))
    cdf_mask = np.asarray(event_mask(cdf_x, event), dtype=bool)
    if np.count_nonzero(cdf_mask) > 1:
        accumulated.add_trace(go.Scatter(
            x=cdf_x[cdf_mask], y=cumulative[cdf_mask], mode="lines", line=dict(color=AMBER, width=3),
            name="Valores del evento", hovertemplate="x = %{x:.6g}<br>F(x) = %{y:.8g}<extra></extra>",
        ))
    _layout(accumulated, "Distribución acumulada", "F(x) = P(X ≤ x)", left, right)
    _boundaries((figure, accumulated), event, left, right)
    if any(value < left or value > right for value in _boundary_values(event)):
        notes.append(
            "Algún límite del evento queda fuera de la ventana visible. "
            "El resultado numérico también incluye la probabilidad fuera de ella."
        )
    excluded = float(_evaluate(rv.cdf, left) + _evaluate(rv.sf, right))
    return figure, accumulated, _metadata(left, right, excluded, notes, MAX_CONTINUOUS_POINTS)


def make_charts(distribution_id, params, event, coverage=0.99):
    """Return PMF/PDF, CDF and honest window metadata for a univariate law.

    Multinomial callers must pass a selected binomial marginal instead of the
    joint law. A quantile query is plotted by the UI as a ``le`` event at its
    computed quantile. ``coverage`` controls the target central mass window;
    finite supports and integer granularity can yield greater coverage.
    """
    if not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValidationError("La cobertura del gráfico debe estar entre 0 y 1, sin incluir los extremos.")
    distribution = REGISTRY[distribution_id]
    if distribution.kind == "multivariate":
        raise ValidationError("La multinomial requiere una vista marginal binomial para estos gráficos.")
    discrete = distribution.kind == "discrete"
    validate_event(event, discrete=discrete)
    if event["kind"] not in {"le", "gt", "interval", "eq"}:
        raise ValidationError("Para representar un cuantil, usá el evento acumulado hasta ese valor.")
    if discrete:
        return _discrete_charts(distribution_id, params, event, float(coverage))
    return _continuous_charts(distribution_id, params, event, float(coverage))


def category_chart(names, probabilities, expected):
    """Category probabilities, explicitly distinct from the joint count law."""
    from .calculations import validate_multinomial

    # n is already validated by the caller; validate the categorical simplex
    # and its names here too, never normalize user-supplied probabilities.
    validate_multinomial(1, probabilities, names=names)
    expected = np.asarray(expected, dtype=float)
    if len(expected) != len(probabilities) or not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValidationError("Los conteos esperados deben ser finitos y corresponder a cada categoría.")
    figure = go.Figure(go.Bar(
        x=list(names), y=list(probabilities), marker_color=BLUE,
        customdata=expected,
        hovertemplate="%{x}<br>Probabilidad de categoría = %{y:.2%}<br>Conteo esperado = %{customdata:.4g}<extra></extra>",
        name="Probabilidad por categoría",
    ))
    figure.update_layout(
        title="Probabilidades por categoría<br>en un ensayo",
        xaxis_title="Categoría", yaxis_title="Probabilidad pᵢ",
        height=360, margin=dict(l=45, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    figure.update_yaxes(range=[0, 1.025], tickformat=".0%", gridcolor="rgba(128,128,128,0.16)")
    figure.update_xaxes(type="category")
    return figure
