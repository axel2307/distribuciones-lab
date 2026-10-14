"""Pedagogical distribution registry.

UI limits are practical exploration limits, not the mathematical domains.  The
calculation layer validates the mathematical domains independently.  Every
formula uses exactly the parameterisation named here (see docs/catalog-sources.md).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    key: str
    name: str
    symbol: str
    default: int | float
    minimum: int | float
    maximum: int | float
    step: int | float
    integer: bool
    explanation: str
    restriction: str


@dataclass(frozen=True)
class Distribution:
    id: str
    name: str
    kind: str
    scipy_name: str
    parameters: tuple[Parameter, ...]
    formula: str
    symbols: dict[str, str]
    support: str

    @property
    def source_url(self) -> str:
        return f"https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.{self.scipy_name}.html"


def P(key, name, symbol, default, minimum, maximum, step, explanation, restriction, integer=False):
    return Parameter(key, name, symbol, default, minimum, maximum, step, integer, explanation, restriction)


def S(kind, **symbols):
    """Shared mathematical notation, included in every lesson's symbol table."""
    if kind == "continuous":
        base = {"X": "Variable aleatoria continua.", "x": "Un valor posible de X.",
                "f_X(x)": "Densidad: las probabilidades son áreas, no alturas."}
    else:
        base = {"X": "Variable aleatoria de conteo o resultado discreto.",
                "k": "Un valor entero posible de X.",
                r"P(X=k)": "Probabilidad de un resultado exacto."}
    base.update(symbols)
    return base


REGISTRY: dict[str, Distribution] = {}


def D(id, name, kind, parameters, formula, symbols, support):
    REGISTRY[id] = Distribution(id, name, kind, id, tuple(parameters), formula, symbols, support)


D("bernoulli", "Bernoulli", "discrete", [
    P("p", "Probabilidad de éxito", "p", .3, 0., 1., .01,
      "Probabilidad de que el único ensayo termine en éxito (X = 1).", "0 ≤ p ≤ 1."),
], r"P(X=k)=\begin{cases}1-p,&k=0\\p,&k=1\\0,&\text{en otro caso}\end{cases}",
  S("discrete", p="Probabilidad poblacional de éxito; 1−p es la de fracaso."),
  r"k\in\{0,1\}")

D("binom", "Binomial", "discrete", [
    P("n", "Número de ensayos", "n", 20, 0, 500, 1, "Cantidad fija de ensayos independientes.", "n entero ≥ 0.", True),
    P("p", "Probabilidad de éxito", "p", .2, 0., 1., .01, "La misma probabilidad de éxito en cada ensayo.", "0 ≤ p ≤ 1."),
], r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}",
  S("discrete", n="Número fijo de ensayos.", p="Probabilidad poblacional de éxito en cada ensayo.",
    **{r"\binom{n}{k}": "Cantidad de maneras de elegir k éxitos entre n ensayos: n!/[k!(n−k)!].",
       "!": "Factorial de un entero no negativo; 0! = 1.", "0^0": "En esta función de masa se interpreta como 1; p = 0 o 1 produce un conteo determinista."}),
  r"k\in\{0,1,\ldots,n\}")

D("poisson", "Poisson", "discrete", [
    P("mu", "Número esperado de eventos", "μ", 5., 0., 200., .1,
      "Media de eventos en el intervalo fijo elegido; si la tasa es λ y la duración t, μ = λt.", "μ ≥ 0."),
], r"P(X=k)=e^{-\mu}\frac{\mu^k}{k!}",
  S("discrete", **{r"\mu": "Media de eventos en el intervalo, no necesariamente la tasa por unidad de tiempo.",
                   "e": "Base de los logaritmos naturales.", "!": "Factorial; 0! = 1.",
                   r"\mu=0": "Toda la probabilidad está en k = 0 (se usa 0^0 = 1 en la fórmula)."}),
  r"k\in\{0,1,2,\ldots\}")

D("geom", "Geométrica", "discrete", [
    P("p", "Probabilidad de éxito por intento", "p", .25, .001, 1., .01,
      "Probabilidad constante de éxito; X cuenta intentos e incluye el intento exitoso.", "0 < p ≤ 1."),
], r"P(X=k)=(1-p)^{k-1}p",
  S("discrete", p="Probabilidad de éxito por intento.",
    **{"k−1": "Número de fracasos anteriores al primer éxito.", "0^0": "Se interpreta como 1 para k = 1 y p = 1: éxito inmediato."}),
  r"k\in\{1,2,3,\ldots\}")

D("nbinom", "Binomial negativa", "discrete", [
    P("r", "Éxitos que queremos alcanzar", "r", 3, 1, 100, 1,
      "El proceso termina al obtener el éxito número r; X cuenta sólo los fracasos previos.", "r entero ≥ 1 en esta app; SciPy también permite r real positivo.", True),
    P("p", "Probabilidad de éxito por intento", "p", .4, .001, 1., .01,
      "Probabilidad constante de éxito en ensayos independientes.", "0 < p ≤ 1."),
], r"P(X=k)=\binom{k+r-1}{r-1}p^r(1-p)^k",
  S("discrete", r="Número de éxitos hasta detenerse (argumento n de SciPy).", p="Probabilidad de éxito por intento.",
    **{r"\binom{k+r-1}{r-1}": "Maneras de ubicar los primeros r−1 éxitos antes del éxito final.",
       "k": "Fracasos antes del éxito r; los intentos totales serían k+r.", "0^0": "Se interpreta como 1 cuando p = 1 y k = 0."}),
  r"k\in\{0,1,2,\ldots\}")

D("hypergeom", "Hipergeométrica", "discrete", [
    P("M", "Tamaño de la población", "M", 52, 1, 10000, 1, "Cantidad total de elementos disponibles.", "M entero ≥ 1.", True),
    P("K", "Éxitos en la población", "K", 4, 0, 10000, 1, "Elementos de la población que tienen la característica buscada.", "K entero, 0 ≤ K ≤ M.", True),
    P("n", "Tamaño de la extracción", "n", 5, 0, 1000, 1, "Cantidad de elementos extraídos sin reemplazo.", "n entero, 0 ≤ n ≤ M.", True),
], r"P(X=k)=\frac{\binom{K}{k}\binom{M-K}{n-k}}{\binom{M}{n}}",
  S("discrete", M="Tamaño de la población.", K="Éxitos en la población (argumento n de SciPy).", n="Elementos extraídos (argumento N de SciPy).",
    **{r"\binom{u}{v}": "Número de formas de elegir v elementos entre u: u!/[v!(u−v)!]."}),
  r"k\in\{\max(0,n-M+K),\ldots,\min(K,n)\}")

D("multinomial", "Multinomial", "multivariate", [
    P("n", "Número de ensayos", "n", 20, 0, 500, 1, "Cada ensayo pertenece exactamente a una categoría.", "n entero ≥ 0.", True),
], r"P(\mathbf X=\mathbf k)=\frac{n!}{\prod_{i=1}^{m}k_i!}\prod_{i=1}^{m}p_i^{k_i}",
  {r"\mathbf X": "Vector de conteos (X₁,…,Xₘ); no es una variable escalar.",
   r"\mathbf k": "Vector de conteos enteros no negativos cuya suma es n.", "n": "Número fijo de ensayos independientes.",
   "m": "Cantidad de categorías, entre 2 y 5 en esta interfaz.", "i": "Índice de categoría, de 1 a m.",
   "p_i": "Probabilidad de la categoría i: 0 ≤ pᵢ ≤ 1 y suma de pᵢ = 1.",
   "k_i": "Conteo de la categoría i en el vector elegido.", r"\prod": "Producto sobre todas las categorías.",
   "!": "Factorial; 0! = 1.", "0^0": "Se interpreta como 1 cuando una categoría imposible tiene conteo cero.",
   "E[X_i]=np_i": "Conteo esperado de la categoría i; su marginal es Binomial(n,pᵢ)."},
  r"\mathbf k\in\mathbb Z_{\ge0}^{m},\quad\sum_{i=1}^{m}k_i=n")

D("randint", "Uniforme discreta", "discrete", [
    P("low", "Menor entero incluido", "a", 1, -1000, 1000, 1, "Primer entero posible.", "a entero y a ≤ b.", True),
    P("high", "Mayor entero incluido", "b", 6, -1000, 1000, 1, "Último entero posible; este extremo se incluye.", "b entero y b ≥ a.", True),
], r"P(X=k)=\frac{1}{b-a+1}",
  S("discrete", a="Menor entero incluido.", b="Mayor entero incluido: se pasa b+1 como high a scipy.stats.randint.",
    **{"b−a+1": "Cantidad de resultados equiprobables; a = b permite un único resultado."}),
  r"k\in\{a,a+1,\ldots,b\}")

D("norm", "Normal o gaussiana", "continuous", [
    P("mu", "Media poblacional", "μ", 0., -10000., 10000., .1, "Centro y media de la distribución.", "μ real."),
    P("sigma", "Desviación estándar poblacional", "σ", 1., .05, 1000., .05, "Dispersión en las mismas unidades que X.", "σ > 0."),
], r"f_X(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]",
  S("continuous", **{r"\mu": "Media poblacional, distinta de la media de una muestra.", r"\sigma": "Desviación estándar poblacional; la varianza es σ².",
                      r"\pi": "Constante pi.", r"\exp(u)": "Función exponencial eᵘ."}), r"x\in\mathbb R")

D("uniform", "Uniforme continua", "continuous", [
    P("a", "Extremo inferior", "a", 0., -10000., 10000., .1, "Límite inferior del intervalo.", "a real y a < b."),
    P("b", "Extremo superior", "b", 10., -10000., 10000., .1, "Límite superior del intervalo.", "b real y b > a."),
], r"f_X(x)=\begin{cases}\frac{1}{b-a},&a\le x\le b\\0,&\text{en otro caso}\end{cases}",
  S("continuous", a="Extremo inferior (loc de SciPy).", b="Extremo superior; la amplitud b−a es scale de SciPy."),
  r"x\in[a,b]")

D("expon", "Exponencial", "continuous", [
    P("rate", "Tasa de eventos", "λ", .5, .001, 100., .05, "Eventos esperados por unidad de tiempo; escala = 1/λ es la espera media.", "λ > 0."),
], r"f_X(x)=\lambda e^{-\lambda x},\quad x\ge0",
  S("continuous", **{r"\lambda": "Tasa constante por unidad de tiempo; en SciPy scale = 1/λ y loc = 0.", "e": "Base de los logaritmos naturales."}),
  r"x\in[0,\infty)")

D("gamma", "Gamma", "continuous", [
    P("alpha", "Forma", "α", 3., .1, 50., .1, "Controla la forma; si es entera puede representar el número de llegadas hasta detener la espera.", "α > 0."),
    P("theta", "Escala de tiempo", "θ", 2., .05, 1000., .1, "Escala positiva, recíproca de la tasa; la media es αθ.", "θ > 0."),
], r"f_X(x)=\frac{x^{\alpha-1}e^{-x/\theta}}{\Gamma(\alpha)\theta^\alpha},\quad x>0",
  S("continuous", **{r"\alpha": "Forma (a de SciPy); no necesita ser entera.", r"\theta": "Escala (scale de SciPy), no tasa. La tasa sería 1/θ.",
                      r"\Gamma(\alpha)": "Función gamma: integral de t^(α−1)e^(−t) de 0 a infinito.", "e": "Base de los logaritmos naturales."}),
  r"x\in[0,\infty)\quad\text{(densidad en 0 por su límite)}")

D("beta", "Beta", "continuous", [
    P("alpha", "Forma hacia el éxito", "α", 2., .1, 100., .1, "Con β fija, aumentar α desplaza la masa hacia 1.", "α > 0."),
    P("beta", "Forma hacia el fracaso", "β", 5., .1, 100., .1, "Con α fija, aumentar β desplaza la masa hacia 0.", "β > 0."),
], r"f_X(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\quad 0<x<1",
  S("continuous", **{r"\alpha,\beta": "Formas positivas (a,b de SciPy); X vive entre 0 y 1 sin desplazamiento.",
                      r"B(\alpha,\beta)": "Función beta: Γ(α)Γ(β)/Γ(α+β). Γ es la función gamma.",
                      "0,1": "Extremos del soporte; la densidad puede divergir allí si la forma correspondiente es menor que 1."}),
  r"x\in[0,1]\quad\text{(densidad en extremos por sus límites)}")

D("lognorm", "Lognormal", "continuous", [
    P("mu_log", "Media del logaritmo natural", "μₗ", 0., -5., 10., .1, "Media de ln(X); no es la media de X. La mediana de X es exp(μₗ).", "μₗ real."),
    P("sigma_log", "Desviación del logaritmo natural", "σₗ", .5, .05, 3., .05, "Desviación estándar de ln(X), no de X.", "σₗ > 0."),
], r"f_X(x)=\frac{1}{x\sigma_\ell\sqrt{2\pi}}\exp\left[-\frac{(\ln x-\mu_\ell)^2}{2\sigma_\ell^2}\right],\quad x>0",
  S("continuous", **{r"\mu_\ell": "Media de ln(X); en SciPy scale = exp(μₗ).", r"\sigma_\ell": "Desviación estándar de ln(X); en SciPy s = σₗ, loc = 0.",
                      r"\ln": "Logaritmo natural.", r"\exp(u)": "Exponencial eᵘ.", r"\pi": "Constante pi."}),
  r"x\in(0,\infty)")

D("t", "t de Student", "continuous", [
    P("df", "Grados de libertad", "ν", 10., .1, 200., .5, "Controlan cuánto pesan las colas; en inferencia de una media suelen ser tamaño muestral menos uno.", "ν > 0; puede ser real, aunque muchos usos dan enteros."),
], r"f_X(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}",
  S("continuous", **{r"\nu": "Grados de libertad (df de SciPy); versión central, ubicación 0 y escala 1.",
                      r"\Gamma": "Función gamma, extensión del factorial: Γ(m) = (m−1)! para enteros positivos.", r"\pi": "Constante pi."}),
  r"x\in\mathbb R")

D("chi2", "Chi-cuadrado", "continuous", [
    P("df", "Grados de libertad", "ν", 5., .1, 200., .5, "Cantidad de componentes normales independientes elevados al cuadrado en el caso entero.", "ν > 0; el modelo también admite valores reales."),
], r"f_X(x)=\frac{x^{\nu/2-1}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)},\quad x>0",
  S("continuous", **{r"\nu": "Grados de libertad (df de SciPy); versión central, loc = 0 y scale = 1.",
                      r"\Gamma": "Función gamma; la distribución equivale a Gamma(α = ν/2, θ = 2).", "e": "Base de los logaritmos naturales."}),
  r"x\in[0,\infty)\quad\text{(densidad en 0 por su límite)}")

D("f", "F de Fisher", "continuous", [
    P("dfn", "Grados de libertad del numerador", "ν₁", 5., .1, 200., .5, "Grados de libertad de la primera chi-cuadrado dividida por ν₁.", "ν₁ > 0."),
    P("dfd", "Grados de libertad del denominador", "ν₂", 15., .1, 200., .5, "Grados de libertad de la segunda chi-cuadrado dividida por ν₂.", "ν₂ > 0."),
], r"f_X(x)=\frac{(\nu_1/\nu_2)^{\nu_1/2}x^{\nu_1/2-1}}{B(\nu_1/2,\nu_2/2)(1+\nu_1x/\nu_2)^{(\nu_1+\nu_2)/2}},\quad x>0",
  S("continuous", **{r"\nu_1": "Grados de libertad del numerador (dfn de SciPy).", r"\nu_2": "Grados de libertad del denominador (dfd de SciPy).",
                      "B(u,v)": "Función beta: Γ(u)Γ(v)/Γ(u+v), con Γ la función gamma.",
                      "X=(U/ν₁)/(V/ν₂)": "U y V son chi-cuadrado independientes; ésta es la F central, loc = 0 y scale = 1."}),
  r"x\in[0,\infty)\quad\text{(densidad en 0 por su límite)}")

D("weibull_min", "Weibull de mínimos", "continuous", [
    P("shape", "Forma de la tasa de falla", "k", 1.5, .1, 10., .1, "k < 1: tasa de falla decreciente; k = 1: constante; k > 1: creciente.", "k > 0."),
    P("scale", "Escala de vida útil", "η", 10., .05, 10000., .5, "Tiempo en el que la probabilidad acumulada es 1−exp(−1), aproximadamente 63,2%.", "η > 0."),
], r"f_X(x)=\frac{k}{\eta}\left(\frac{x}{\eta}\right)^{k-1}\exp\left[-\left(\frac{x}{\eta}\right)^k\right],\quad x>0",
  S("continuous", k="Forma (c de SciPy); no es un conteo.", **{r"\eta": "Escala (scale de SciPy), en unidades de X; loc = 0.", r"\exp(u)": "Exponencial eᵘ."}),
  r"x\in[0,\infty)\quad\text{(densidad en 0 por su límite)}")

D("pareto", "Pareto tipo I", "continuous", [
    P("alpha", "Forma de la cola", "α", 3., .1, 10., .1, "Un α menor produce una cola más pesada; la media es finita sólo para α > 1.", "α > 0."),
    P("xmin", "Umbral mínimo incluido", "xₘ", 1., .05, 10000., .5, "Valor mínimo positivo del modelo tipo I.", "xₘ > 0."),
], r"f_X(x)=\frac{\alpha x_m^{\alpha}}{x^{\alpha+1}},\quad x\ge x_m",
  S("continuous", **{r"\alpha": "Forma positiva (b de SciPy).", "x_m": "Umbral mínimo positivo: en SciPy scale = xₘ y loc = 0. No se usa loc = xₘ."}),
  r"x\in[x_m,\infty)")

D("cauchy", "Cauchy", "continuous", [
    P("x0", "Centro y mediana", "x₀", 0., -10000., 10000., .1, "Ubicación del centro; no es una media porque la media no existe.", "x₀ real."),
    P("gamma", "Escala de dispersión", "γ", 1., .05, 1000., .1, "Semiancho de la densidad a mitad de su altura máxima; no es desviación estándar.", "γ > 0."),
], r"f_X(x)=\frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}",
  S("continuous", **{"x_0": "Ubicación y mediana (loc de SciPy); la media no existe.", r"\gamma": "Escala (scale de SciPy); varianza y desviación estándar no existen.", r"\pi": "Constante pi."}),
  r"x\in\mathbb R")

D("laplace", "Laplace", "continuous", [
    P("mu", "Centro y media poblacional", "μ", 0., -10000., 10000., .1, "Media y mediana de esta distribución simétrica.", "μ real."),
    P("b", "Escala de dispersión", "b", 1., .05, 1000., .1, "Escala positiva; la desviación estándar es √2 b.", "b > 0."),
], r"f_X(x)=\frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)",
  S("continuous", **{r"\mu": "Ubicación y media (loc de SciPy).", "b": "Escala (scale de SciPy).", "|x−μ|": "Distancia absoluta al centro.", r"\exp(u)": "Exponencial eᵘ."}),
  r"x\in\mathbb R")

D("rayleigh", "Rayleigh", "continuous", [
    P("sigma", "Escala de las componentes", "σ", 2., .05, 1000., .1,
      "Desviación estándar de cada componente normal independiente cuya magnitud se modela; no de X.", "σ > 0."),
], r"f_X(x)=\frac{x}{\sigma^2}\exp\left(-\frac{x^2}{2\sigma^2}\right),\quad x\ge0",
  S("continuous", **{r"\sigma": "Escala (scale de SciPy), con loc = 0; para la interpretación geométrica es la desviación de cada componente, no de su magnitud X.", r"\exp(u)": "Exponencial eᵘ."}),
  r"x\in[0,\infty)")

D("gumbel_r", "Gumbel de máximos", "continuous", [
    P("mu", "Ubicación", "μ", 0., -10000., 10000., .1, "Desplaza la curva; no coincide con su media.", "μ real."),
    P("beta", "Escala", "β", 1., .05, 1000., .1, "Escala positiva que controla la dispersión.", "β > 0."),
], r"f_X(x)=\frac{1}{\beta}\exp\left[-z-e^{-z}\right],\quad z=\frac{x-\mu}{\beta}",
  S("continuous", **{r"\mu": "Ubicación (loc de SciPy); es la moda, no la media.", r"\beta": "Escala (scale de SciPy).", "z": "Valor estandarizado (x−μ)/β.", "e": "Base de los logaritmos naturales.", r"\exp(u)": "Exponencial eᵘ."}),
  r"x\in\mathbb R")

D("genextreme", "Generalizada de valores extremos (GEV)", "continuous", [
    P("xi", "Forma de la cola", "ξ", .1, -1., 1.5, .05, "ξ > 0: cola derecha pesada; ξ < 0: extremo superior finito; ξ = 0: Gumbel de máximos.", "ξ real; en SciPy c = −ξ."),
    P("mu", "Ubicación", "μ", 0., -10000., 10000., .1, "Desplaza la distribución de máximos; no es necesariamente la media.", "μ real."),
    P("sigma", "Escala", "σ", 1., .05, 10000., .1, "Escala positiva; no es necesariamente la desviación estándar.", "σ > 0."),
], r"\begin{aligned}z&=\frac{x-\mu}{\sigma},\quad t=1+\xi z\\f_X(x)&=\begin{cases}\sigma^{-1}t^{-1/\xi-1}\exp(-t^{-1/\xi}),&\xi\ne0,\ t>0\\\sigma^{-1}\exp(-z-e^{-z}),&\xi=0\end{cases}\\F_X(x)&=\begin{cases}\exp(-t^{-1/\xi}),&\xi\ne0,\ t>0\\\exp(-e^{-z}),&\xi=0\end{cases}\end{aligned}",
  S("continuous", **{r"\xi": "Forma en la convención usual; se pasa c = −ξ a scipy.stats.genextreme.", r"\mu": "Ubicación (loc de SciPy), no necesariamente media.", r"\sigma": "Escala (scale de SciPy), no necesariamente desviación estándar.",
                      "z": "Valor estandarizado (x−μ)/σ.", "t": "1+ξz; debe ser positivo en el interior del soporte.", "F_X(x)": "CDF: P(X ≤ x). Se completa con 0 o 1 fuera del soporte.",
                      "e": "Base de los logaritmos naturales.", r"\exp(u)": "Exponencial eᵘ."}),
  r"1+\xi(x-\mu)/\sigma\ge0\ (\xi\ne0);\quad x\in\mathbb R\ (\xi=0)\quad\text{(densidad en borde por su límite)}")


def defaults(distribution_id: str) -> dict:
    """Return independent defaults, including the multinomial vector controls."""
    result = {p.key: p.default for p in REGISTRY[distribution_id].parameters}
    if distribution_id == "multinomial":
        result.update(probabilities=[.5, .3, .2], names=["Compra", "Consulta", "Sin acción"])
    return result
