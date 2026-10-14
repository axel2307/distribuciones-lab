"""Educational content, kept separate from the numerical engine."""
from .lessons_continuous import LESSONS as CONTINUOUS
from .lessons_discrete import LESSONS as DISCRETE

LESSONS = {**DISCRETE, **CONTINUOUS}

FOUNDATIONS = {
    "Masa": "En una variable discreta, P(X=k) es la probabilidad de un resultado y las masas suman 1.",
    "Densidad": "En una variable continua, f(x) es una altura por unidad de X. Puede superar 1: la probabilidad es un área, no la altura. P(X=x)=0.",
    "Acumulada": "F(x)=P(X≤x) siempre está entre 0 y 1 y nunca disminuye. En discretas tiene saltos; en estas continuas es continua.",
    "Parámetro y estadístico": "Un parámetro describe la población o el modelo (por ejemplo μ). Un estadístico se calcula con una muestra (por ejemplo la media muestral). Cambiar un parámetro acá no estima ese valor a partir de datos.",
}
