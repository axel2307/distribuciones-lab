"""Spanish lessons and live example inputs for the eight discrete families.

The application computes every example result; prose deliberately contains no
hard-coded probability. Conventions were checked against the official SciPy
reference pages for bernoulli, binom, poisson, geom, nbinom, hypergeom,
multinomial and randint. Source links are built from the registry's scipy_name.
"""

LESSONS = {
    "bernoulli": {
        "intuition": (
            "Modela un único ensayo con dos resultados: éxito (1) o fracaso (0). "
            "Éxito es una etiqueta del evento que te interesa, no un juicio de valor."
        ),
        "variable": "X vale 1 si ocurre el evento de interés y 0 si no ocurre. Su soporte es {0, 1}.",
        "assumptions": [
            "Los dos resultados son mutuamente excluyentes y cubren todas las posibilidades.",
            "La probabilidad poblacional p está fijada para el ensayo modelado; 0 ≤ p ≤ 1.",
            "Para un único ensayo no hace falta suponer independencia; sí para combinar ensayos en una binomial.",
        ],
        "applications": [
            "Compra o no compra de un visitante.",
            "Una pieza cumple o no cumple un criterio de calidad.",
            "Una respuesta es correcta o incorrecta.",
        ],
        "inappropriate": (
            "No describe directamente un conteo de varios ensayos ni una respuesta con más de dos "
            "categorías. Una p común puede ser inadecuada si se mezclan perfiles muy distintos."
        ),
        "error": (
            "Confundir p, una probabilidad del modelo, con la proporción observada en una muestra. "
            "La proporción muestral estima p y puede variar de una muestra a otra."
        ),
        "effects": {
            "p": "Al aumentar p, la masa pasa de 0 a 1. La varianza p(1−p) es máxima en p=0,5; en p=0 o p=1 el resultado es determinista.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: para un visitante de una tienda se propone una probabilidad de compra de 0,30.",
            "question": "¿Cuál es la probabilidad de que ese visitante compre?",
            "variable": "X=1 si el visitante compra; X=0 si no compra.",
            "justification": "Hay dos resultados excluyentes y exhaustivos. Se supone que p=0,30 es adecuada para ese visitante y ese contexto.",
            "parameters": {"p": 0.30},
            "event": {"kind": "eq", "x": 1},
            "interpretation": "La probabilidad calculada corresponde a una compra en un único ensayo; no garantiza el comportamiento de una persona concreta.",
            "limitations": "El valor de p es inventado para aprender. En una aplicación real habría que estimarlo con datos pertinentes y considerar diferencias entre visitantes.",
        },
    },
    "binom": {
        "intuition": "Cuenta cuántos éxitos aparecen en un número fijo de ensayos con la misma probabilidad de éxito.",
        "variable": "X es el número de éxitos entre n ensayos. Sus valores son los enteros 0, 1, …, n.",
        "assumptions": [
            "n es un entero no negativo y se fija antes de observar los resultados.",
            "Cada ensayo tiene dos resultados y probabilidad de éxito p constante, con 0 ≤ p ≤ 1.",
            "Los ensayos son independientes: conocer un resultado no cambia la probabilidad de los demás.",
        ],
        "applications": [
            "Conversiones entre un número fijo de visitantes comparables.",
            "Piezas defectuosas en una muestra cuando la independencia sea razonable.",
            "Aciertos en preguntas independientes de igual dificultad.",
        ],
        "inappropriate": (
            "No es exacta para extraer sin reemplazo de una población pequeña: las probabilidades cambian. "
            "Tampoco si hay dependencia o probabilidades distintas entre ensayos."
        ),
        "error": "Usar n como número de éxitos deseados: aquí n cuenta los ensayos, y X cuenta los éxitos observados.",
        "effects": {
            "n": "Con p fija, más ensayos aumentan la media np y la varianza np(1−p). La proporción X/n suele fluctuar menos cuando n crece y n>0.",
            "p": "Con n fijo, subir p desplaza la masa hacia más éxitos. La dispersión es mayor cerca de p=0,5 y desaparece en los extremos.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: una campaña recibe 20 visitantes comparables; cada uno convierte con probabilidad 0,15.",
            "question": "¿Cuál es la probabilidad de obtener al menos 5 conversiones?",
            "variable": "X es la cantidad de conversiones entre los 20 visitantes, de 0 a 20.",
            "justification": "Se fija n=20 y se asumen visitantes independientes con una misma p=0,15. Para un conteo entero, al menos 5 equivale a X>4.",
            "parameters": {"n": 20, "p": 0.15},
            "event": {"kind": "gt", "x": 4},
            "interpretation": "El resultado es la probabilidad de que un grupo de 20 visitantes produzca 5 o más conversiones bajo esos supuestos.",
            "limitations": "Los valores son sintéticos. Sesiones repetidas, contagio social o canales distintos pueden romper la independencia y la p común.",
        },
    },
    "poisson": {
        "intuition": "Cuenta eventos en una ventana fija cuando llegan de manera independiente y a una intensidad aproximadamente constante.",
        "variable": "X es el número de eventos en la ventana elegida. Toma valores enteros 0, 1, 2, …, sin máximo teórico.",
        "assumptions": [
            "La intensidad del proceso es constante dentro de la ventana; μ es el número esperado de eventos en esa ventana.",
            "Los conteos en intervalos disjuntos son independientes.",
            "En intervalos muy pequeños, la probabilidad de dos o más eventos es despreciable frente a la de uno.",
            "μ ≥ 0; con μ=0 el conteo es siempre cero.",
        ],
        "applications": [
            "Consultas recibidas durante una hora estable.",
            "Defectos por unidad de longitud o superficie, con condiciones homogéneas.",
            "Llegadas de partículas a un detector en un intervalo fijo.",
        ],
        "inappropriate": (
            "Las ráfagas, la estacionalidad o la dependencia entre llegadas suelen invalidar el modelo. "
            "También puede fallar si la varianza de los conteos es muy diferente de su media."
        ),
        "error": (
            "Confundir intensidad por unidad de tiempo con μ. Si la intensidad es λ eventos/hora y la "
            "ventana dura t horas, el parámetro del conteo es μ=λt."
        ),
        "effects": {
            "mu": "Al subir μ aumentan tanto la media como la varianza, ambas iguales a μ. El conteo se concentra relativamente más alrededor de su media cuando μ es grande.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: una mesa de ayuda recibe consultas con una intensidad constante de 4 por hora.",
            "question": "¿Cuál es la probabilidad de recibir como máximo 6 consultas en la próxima hora?",
            "variable": "X es la cantidad de consultas en una ventana de una hora.",
            "justification": "Se asumen llegadas independientes y una intensidad homogénea. Para una hora, μ=4×1=4 consultas esperadas.",
            "parameters": {"mu": 4.0},
            "event": {"kind": "le", "x": 6},
            "interpretation": "El resultado corresponde a ventanas de una hora con 0, 1, …, 6 consultas, no a una consulta individual.",
            "limitations": "La intensidad es inventada. Un pico luego de un incidente o diferencias entre horarios exigirían revisar el modelo.",
        },
    },
    "geom": {
        "intuition": "Mide cuántos intentos hacen falta hasta el primer éxito, incluyendo el intento exitoso.",
        "variable": "X es el número total de intentos hasta el primer éxito. Su soporte es 1, 2, 3, …; nunca cero.",
        "assumptions": [
            "Los intentos son independientes y cada uno tiene dos resultados.",
            "La probabilidad p de éxito no cambia entre intentos; 0<p≤1.",
            "Se continúa hasta el primer éxito; no se impone un límite de intentos en el modelo.",
        ],
        "applications": [
            "Intentos de contacto hasta la primera respuesta en un modelo simplificado.",
            "Ensayos repetidos de laboratorio hasta un resultado positivo.",
            "Lanzamientos independientes hasta que aparece una cara elegida.",
        ],
        "inappropriate": "No es adecuada si el aprendizaje, el desgaste o la selección de candidatos hace que p cambie con el número de intentos.",
        "error": (
            "Confundir intentos con fracasos. SciPy geom y este laboratorio cuentan intentos desde 1. "
            "Si Y cuenta fracasos antes del éxito, Y=X−1 y su soporte empieza en 0."
        ),
        "effects": {
            "p": "Subir p concentra la masa cerca de 1 y reduce la media 1/p y la varianza (1−p)/p². Con p=1, X=1 con certeza.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: cada intento independiente de contactar a un cliente logra respuesta con probabilidad 0,20.",
            "question": "¿Cuál es la probabilidad de conseguir la primera respuesta dentro de los primeros 5 intentos?",
            "variable": "X es el número de intentos hasta la primera respuesta, incluida la respuesta.",
            "justification": "Se suponen intentos independientes con p=0,20 constante. El evento X≤5 incluye los intentos 1 a 5.",
            "parameters": {"p": 0.20},
            "event": {"kind": "le", "x": 5},
            "interpretation": "El resultado expresa la probabilidad de alcanzar una primera respuesta a más tardar en el quinto intento.",
            "limitations": "p es ilustrativa. En contactos reales, insistir puede cambiar la probabilidad de respuesta y los intentos suelen tener un límite.",
        },
    },
    "nbinom": {
        "intuition": "Cuenta los fracasos acumulados antes de alcanzar un número fijo de éxitos en ensayos repetidos.",
        "variable": "X es el número de fracasos antes del éxito número r. Toma valores enteros 0, 1, 2, …; los intentos totales son T=X+r.",
        "assumptions": [
            "Para esta interpretación, r es un entero positivo y se fija antes de empezar.",
            "Los ensayos son independientes y tienen probabilidad de éxito p constante, con 0<p≤1.",
            "El proceso termina inmediatamente cuando se obtiene el éxito número r.",
        ],
        "applications": [
            "Intentos fallidos hasta completar un objetivo de ventas, en un modelo simplificado.",
            "Resultados negativos antes de obtener r positivos en ensayos independientes.",
            "Como familia de conteos, también puede representar sobredispersión respecto de Poisson con otra interpretación de sus parámetros.",
        ],
        "inappropriate": "No describe un número fijo de ensayos: ese problema corresponde a la binomial. La interpretación de espera falla si p cambia o los ensayos dependen entre sí.",
        "error": (
            "Interpretar X como el total de intentos. SciPy nbinom cuenta fracasos: usa n=r y p. "
            "Para calcular P(T≤t), hay que evaluar P(X≤t−r)."
        ),
        "effects": {
            "r": "Con p fija, exigir más éxitos aumenta la media de fracasos r(1−p)/p y la varianza r(1−p)/p².",
            "p": "Con r fijo, subir p reduce los fracasos esperados y la dispersión. Con p=1 se alcanzan r éxitos sin ningún fracaso.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: una operación comercial busca concretar 3 ventas; cada contacto independiente convierte con probabilidad 0,25.",
            "question": "¿Cuál es la probabilidad de alcanzar 3 ventas con como máximo 8 contactos fallidos, es decir, dentro de 11 contactos totales?",
            "variable": "X cuenta los contactos fallidos antes de la tercera venta. El total de contactos es T=X+3.",
            "justification": "La meta r=3 es fija y p=0,25 es constante. X≤8 equivale exactamente a T≤11; el último contacto contado es la tercera venta.",
            "parameters": {"r": 3, "p": 0.25},
            "event": {"kind": "le", "x": 8},
            "interpretation": "La probabilidad obtenida corresponde a completar las 3 ventas a más tardar en el contacto número 11.",
            "limitations": "Son supuestos inventados. Contactos con distinta calidad, aprendizaje del vendedor o dependencia entre compradores pueden alterar el resultado.",
        },
    },
    "hypergeom": {
        "intuition": "Cuenta elementos de una clase al extraer una muestra sin reemplazo de una población finita.",
        "variable": "X cuenta los elementos de interés en n extracciones de una población de M, que contiene K elementos de interés. Su soporte entero va de max(0,n−M+K) a min(K,n).",
        "assumptions": [
            "La población tiene M≥1 elementos; K y n son enteros entre 0 y M.",
            "Se extraen n elementos sin reemplazo y todos los subconjuntos de ese tamaño son equiprobables.",
            "La pertenencia a la clase de interés está fijada antes de extraer la muestra.",
        ],
        "applications": [
            "Ases en una mano de cartas de una baraja bien mezclada.",
            "Piezas defectuosas en una inspección sin reemplazo de un lote conocido.",
            "Individuos de una clase en una muestra aleatoria de una población finita.",
        ],
        "inappropriate": "No corresponde a extracciones con reemplazo ni a muestreo sesgado. Si K es desconocido, este cálculo requiere primero un modelo o una estimación de ese parámetro.",
        "error": (
            "Suponer que las extracciones son independientes. Sin reemplazo cambian las probabilidades. "
            "En este laboratorio los parámetros son M,K,n; SciPy los recibe como hypergeom(M,K,n), aunque llama n al segundo y N al tercero."
        ),
        "effects": {
            "M": "Con K y n fijos, aumentar M diluye la proporción K/M y reduce la media nK/M. Los demás parámetros deben seguir siendo válidos.",
            "K": "Con M y n fijos, más elementos de interés desplazan la masa hacia conteos mayores.",
            "n": "Con M y K fijos, una muestra mayor aumenta el conteo esperado. Si n=M, X=K con certeza.",
        },
        "example": {
            "context": "Ejemplo ilustrativo de póker: se extraen 5 cartas de una baraja estándar de 52 cartas bien mezclada, con 4 ases.",
            "question": "¿Cuál es la probabilidad de recibir al menos un as?",
            "variable": "X es el número de ases en las 5 cartas; en este caso puede valer 0, 1, 2, 3 o 4.",
            "justification": "Las cartas se extraen sin reemplazo y las manos de 5 cartas son equiprobables. M=52, K=4 y n=5; al menos un as equivale a X>0.",
            "parameters": {"M": 52, "K": 4, "n": 5},
            "event": {"kind": "gt", "x": 0},
            "interpretation": "El resultado representa manos iniciales de 5 cartas que contienen uno o más ases, bajo un reparto aleatorio.",
            "limitations": "Es un ejercicio ilustrativo, no una probabilidad de ganar una mano. Cartas conocidas, reglas distintas o un mazo incompleto cambian la población relevante.",
        },
    },
    "multinomial": {
        "intuition": "Reparte un número fijo de ensayos entre varias categorías y modela conjuntamente todos sus conteos.",
        "variable": "X=(X₁,…,Xₘ) es el vector de conteos de m categorías. Cada Xᵢ es un entero no negativo y la suma de los conteos es n.",
        "assumptions": [
            "n es un entero no negativo fijado de antemano; este laboratorio permite entre 2 y 5 categorías.",
            "Cada ensayo cae en exactamente una categoría: son excluyentes y exhaustivas.",
            "Los ensayos son independientes y conservan las mismas probabilidades pᵢ entre 0 y 1, cuya suma debe ser 1.",
            "Los conteos de distintas categorías no son independientes: deben sumar n.",
        ],
        "applications": [
            "Reparto de pedidos entre canales excluyentes.",
            "Respuestas de una encuesta con una única opción por persona.",
            "Resultados de un experimento con más de dos categorías.",
        ],
        "inappropriate": "No es adecuada para respuestas múltiples, categorías superpuestas o muestreo sin reemplazo de una población finita pequeña. Las probabilidades variables entre ensayos exigen otro modelo.",
        "error": (
            "Creer que las barras de pᵢ muestran toda la distribución conjunta. Muestran probabilidades "
            "de categorías en un ensayo; la conjunta asigna probabilidades a vectores de conteos. "
            "La marginal Xᵢ es binomial(n,pᵢ) y su CDF pertenece sólo a ese conteo."
        ),
        "effects": {
            "n": "Aumentar n eleva los conteos esperados E[Xᵢ]=npᵢ. La varianza marginal es npᵢ(1−pᵢ) y Cov(Xᵢ,Xⱼ)=−npᵢpⱼ para i≠j.",
            "probabilities": "Subir pᵢ eleva el conteo esperado de esa categoría; para mantener la suma en 1 hay que reducir explícitamente otras probabilidades. La aplicación no las normaliza por vos.",
            "names": "Cambiar un nombre sólo modifica la etiqueta de la categoría, no sus probabilidades ni sus conteos.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: 10 pedidos llegan por tres canales excluyentes: local, teléfono y web, con probabilidades 0,20; 0,30; 0,50.",
            "question": "¿Cuál es la probabilidad de que los 10 pedidos se repartan exactamente en 2 de local, 3 de teléfono y 5 de web?",
            "variable": "X=(X_local,X_teléfono,X_web) cuenta los pedidos por canal; los tres conteos suman 10.",
            "justification": "El total n=10 es fijo; cada pedido usa un único canal. Se suponen pedidos independientes y probabilidades constantes que suman 1.",
            "parameters": {"n": 10, "probabilities": [0.20, 0.30, 0.50], "names": ["Local", "Teléfono", "Web"]},
            "event": {"kind": "joint", "counts": [2, 3, 5]},
            "interpretation": "El resultado corresponde al vector exacto (2,3,5), sumando todos los órdenes de llegada que producen ese reparto. No es el producto de tres probabilidades marginales.",
            "limitations": "Las probabilidades son inventadas. Promociones, horarios y clientes repetidos pueden generar dependencia o cambiar las probabilidades de canal.",
        },
    },
    "randint": {
        "intuition": "Asigna la misma probabilidad a cada entero de un intervalo finito, incluyendo ambos extremos.",
        "variable": "X es un entero entre L y U, ambos incluidos. Hay U−L+1 valores posibles, cada uno con probabilidad 1/(U−L+1).",
        "assumptions": [
            "L y U son enteros y L≤U; L=U representa un único valor determinista.",
            "Cada entero del intervalo es equiprobable.",
            "La uniformidad es una hipótesis del mecanismo, no una consecuencia de desconocer sus probabilidades.",
        ],
        "applications": [
            "Resultado de un dado equilibrado.",
            "Selección aleatoria de un índice entero en simulación.",
            "Sorteo entre números consecutivos con igual probabilidad.",
        ],
        "inappropriate": "No corresponde si algunos enteros no pueden aparecer, si el mecanismo favorece ciertos valores o si la variable puede tomar cualquier número real del intervalo.",
        "error": (
            "Olvidar el extremo superior. La interfaz usa U incluido; SciPy randint excluye su argumento high. "
            "Por eso el motor llama randint(L,U+1), y el denominador correcto es U−L+1."
        ),
        "effects": {
            "low": "Con U fijo, subir L elimina valores bajos, aumenta la media (L+U)/2 y reduce la cantidad de resultados posibles.",
            "high": "Con L fijo, subir U agrega valores altos, aumenta la media y la dispersión. Trasladar ambos extremos por igual sólo mueve la distribución.",
        },
        "example": {
            "context": "Ejemplo ilustrativo: se modela el lanzamiento de un dado ideal de seis caras equiprobables.",
            "question": "¿Cuál es la probabilidad de obtener un resultado de 3 o menos?",
            "variable": "X es el número que muestra el dado, uno de los enteros 1, 2, 3, 4, 5 o 6.",
            "justification": "Se asume un dado equilibrado. L=1 y U=6, ambos incluidos; el evento reúne los valores 1, 2 y 3.",
            "parameters": {"low": 1, "high": 6},
            "event": {"kind": "le", "x": 3},
            "interpretation": "El resultado corresponde a los tres resultados favorables entre seis resultados igualmente probables.",
            "limitations": "El dado ideal es un supuesto ilustrativo. Un dado físico sesgado o un procedimiento de lanzamiento no aleatorio puede no ser uniforme.",
        },
    },
}
