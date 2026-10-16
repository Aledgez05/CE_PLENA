# PLENA — Plan de implementación para el Avance 2 (meta: 60%)

> **Documento para el agente de Cursor.** Léelo completo antes de escribir código. Trabaja **un paso a la vez**, en el orden indicado, y no avances si el criterio de aceptación del paso actual no se cumple.

**Materia:** Cómputo Evolutivo
**Proyecto:** PLENA — selección y ponderación de características con algoritmo genético para agrupar y analizar datos de depresión.
**Enfoque de este avance:** depresión en población general (no limitar el alcance a posparto). El pivote desde posparto se justifica en el informe como etapa de desarrollo del método.
**Meta:** una prueba experimental **reproducible** donde un GA de dominio real optimiza la selección y ponderación de variables para mejorar un agrupamiento, comparado con baselines justos y medido con métricas internas y externas.

**Aviso:** este es un prototipo académico de análisis de datos. No diagnostica depresión, y los clusters no equivalen a niveles clínicos de riesgo. Nómbralos `cluster_0, cluster_1, …`.

---

## 0. Reglas de trabajo para el agente

1. **No inventes el significado de columnas.** Si un tipo, codificación o significado no está documentado, márcalo como `UNCONFIRMED` y pregunta al usuario (ver sección 4, "Protocolo de preguntas").
2. **Nunca uses la etiqueta `depressed` como característica de entrada del clustering.** Solo se usa para evaluación externa (y en la variante V2, solo sobre el conjunto de entrenamiento; ver paso 4.4).
3. **Ajusta todo preprocesamiento solo con train.** Validación y test solo reciben transformaciones ya aprendidas.
4. **El conjunto test se toca una sola vez**, al final del paso 5.3. Nunca lo uses para elegir k, pesos, λ, α ni hiperparámetros.
5. **Cero valores fijos de dataset en el código.** Nombres de columnas, tipos y rutas viven en `config/schema.yaml`. Las dos filas de ejemplo del dataset son solo ilustración de esquema; **no entrenes ni reportes resultados con ellas**.
6. **Reproducibilidad:** toda aleatoriedad pasa por una semilla de la configuración (`numpy.random.Generator`, `random_state` en scikit-learn). Mismo cromosoma + misma semilla ⇒ mismo fitness.
7. **Tests primero para lo crítico:** pesos, fitness, anti-fuga, operadores. Un paso no está terminado si `pytest` falla.
8. **Un commit por paso**, con mensaje `step X.Y: descripción`.
9. **No declares que algo "mejora" sin evidencia:** reporta media ± desviación entre semillas y la comparación contra baselines. Es válido que el GA empate o pierda; repórtalo así.
10. **No subas datos a ningún repositorio público** ni guardes información personal en `results/`. Añade el CSV original a `.gitignore`.
11. Si una librería falla por compatibilidad (por ejemplo K-Medoids o FAISS), documenta el error, usa el plan B indicado y continúa; no bloquees todo el avance.

---

## 1. Decisiones de diseño (ya tomadas)

| # | Decisión | Motivo |
|---|---|---|
| D1 | **Cromosoma real en [-1, 1]**, de longitud D (una posición por **variable original**, no por columna one-hot). | Recomendación del equipo; permite selección + ponderación en un solo vector. |
| D2 | Decodificación: `g_j ≤ τ` ⇒ variable **excluida** (τ=0 por defecto); `g_j > τ` ⇒ peso `w_j = g_j`. Los pesos activos se **normalizan a suma 1**. | Cubre "reducir dimensionalidad" sin codificación híbrida. |
| D3 | El peso se aplica como `sqrt(w_j)` al **bloque de columnas** de la variable j: `d² = Σ w_j (x_j − y_j)²`. En one-hot, cada bloque se escala por `sqrt(w_j / n_cols_j)`. | Evita que variables con muchos niveles dominen y hace correctos los pesos. |
| D4 | **Ruta A (principal):** representación numérica densa (imputación + escalado + one-hot) con distancia euclidiana ponderada. **Ruta B (opcional ○):** Gower ponderada con K-Medoids. | Ruta A es compatible con FAISS e inferencia rápida (objetivo original). Ruta B se usa como comparación de representación si hay tiempo. |
| D5 | Clustering principal: **K-Means**. `k` se fija por corrida; se corre **k ∈ {2,3,4,5}** y **k=3 siempre se reporta** (la propuesta v1 hablaba de riesgo bajo/medio/alto). | Silhouette no es comparable entre k distintos. |
| D6 | **Dos variantes de GA:** **V1 no supervisada** (fitness = Silhouette con penalizaciones; RI/ARI solo se evalúa después) y **V2 semisupervisada** (fitness mezcla Silhouette y ARI, calculado solo sobre train). | Interpreta "Silhouette no supervisado + Rand supervisado" sin romper la validez. V2 se etiqueta siempre como variante separada. |
| D7 | Métricas externas: **RI y ARI** (ARI es la principal), AMI/NMI, prevalencia de `depressed` por cluster y AUC usando la pertenencia al cluster. | RI se infla con clases desbalanceadas; ARI corrige el azar. |
| D8 | "Precisión" = asignar a cada cluster la etiqueta mayoritaria **de train** y medir accuracy balanceada / F1 en validación y test; además, **KNN sobre FAISS** como inferencia alternativa. | Un clustering no tiene precisión por sí solo; esto une KNN (marco teórico v1), FAISS y GA. |
| D9 | **Particiones train / validación / test**, con agrupación por persona/encuesta si hay repetidos. | El GA selecciona en train, se eligen configuraciones en validación, se reporta en test. |
| D10 | **Baselines obligatorios:** pesos iguales; búsqueda aleatoria con el **mismo número de evaluaciones de fitness** que el GA; referencia por permutación. **Sin criterio de parada "hasta superar el baseline".** | Comparación justa. Detener el GA al superar el azar no tiene sentido. |
| D11 | **Semillas:** 10 para el GA (mínimo 5 si el cómputo no alcanza); 30 repeticiones para la búsqueda aleatoria. Comparación con prueba pareada (Wilcoxon). | Estimar dispersión y significancia. |
| D12 | **NSGA-II (multiobjetivo) = extensión ○** de la Fase 5B, no método principal. Objetivos: maximizar Silhouette, maximizar ARI en train y minimizar el número de variables activas. **CMA-ES** como comparador ○. | Alcance controlado para el 60%; se activan solo si 5.1–5.4 ya están completos. |
| D13 | FAISS entra en versión mínima en este avance (índice sobre vectores ponderados, no normalizados). FastAPI queda para el cierre. | Objetivo de baja latencia, pero con alcance acotado. |
| D14 | Texto (BoW vs embeddings), sueño y actividad física de la presentación quedan fuera de este avance (opcional 9.2). Debe decirse explícitamente en el informe. | Este avance trabaja con datos tabulares. |

### Protección contra pesos degenerados

El GA, maximizando Silhouette, tiende a concentrar todo el peso en una o dos variables (una binaria produce clusters "perfectos"). Implementa **siempre**:
- tope por variable tras normalizar (por defecto `w_max = 0.3`), recortando y renormalizando;
- mínimo de variables activas (por defecto `min_active = 3`, configurable);
- penalización por concentración usando el **número efectivo de variables** `N_eff = 1 / Σ w_j²`;
- reportar `N_eff` y variables activas en todos los resultados.

---

## 2. Estructura del repositorio

```
plena/
├── config/
│   ├── schema.yaml          # columnas, tipos, exclusiones, rutas (generado en 1.7, confirmado por humano)
│   └── experiment.yaml      # semillas, GA, clustering, presupuesto
├── data/                    # CSV originales (en .gitignore)
├── src/plena/
│   ├── config.py
│   ├── data.py              # carga, validación, particiones
│   ├── audit.py             # auditoría del dataset (Fase 1)
│   ├── synthetic.py         # generador con señal plantada
│   ├── preprocessing.py     # Ruta A + mapeo columna→variable
│   ├── distance.py          # euclidiana ponderada; Gower (○)
│   ├── clustering.py        # K-Means, predict, (K-Medoids ○)
│   ├── fitness.py           # V1, V2, penalizaciones, caché
│   ├── genetic_algorithm.py # población, operadores, ciclo
│   ├── evaluation.py        # métricas internas/externas, estabilidad
│   ├── baselines.py         # iguales, aleatorio, permutación
│   ├── faiss_index.py       # índice, KNN, benchmark
│   └── nsga2.py             # ○ extensión
├── scripts/run_experiments.py
├── tests/
├── results/                 # salidas (en .gitignore si hay datos personales)
├── pyproject.toml
└── README.md
```

**Dependencias:** Python ≥ 3.10, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`, `joblib`, `matplotlib`, `faiss-cpu`, `pytest`. Opcionales: `pymoo` (NSGA-II), `cma`, `kmedoids`. Fija versiones en un lockfile y guarda `pip freeze` en `results/`.

---

## 3. Fases y pasos

Leyenda: ★ obligatorio para el 60% · ○ opcional · Esf.: S ≤ ½ día, M ≈ 1 día, L ≥ 2 días.

### Fase 0 — Preparación

**0.1 ★ Repo, entorno y utilidades (S).** Crea la estructura, `pyproject.toml`, lockfile, `set_seed()` y logging.
*Aceptación:* el entorno se reconstruye desde cero con un comando y `pytest` corre.

**0.2 ★ Configuración declarativa (S).** Implementa `config.py` que lee `schema.yaml` y `experiment.yaml` y **valida**: columnas existentes, tipos válidos, sin solapes entre columnas de entrada/etiqueta/ID. Ejemplo mínimo:

```yaml
# schema.yaml (DRAFT hasta confirmación humana)
dataset_path: data/dataset.csv
target: depressed            # solo evaluación
id_columns: [survey_id]
group_column: null           # persona/encuesta si hay repetidos
features:
  age:      {type: continuous}
  married:  {type: binary}
  # ...
exclusions:
  - {column: survey_id, reason: identificador}
```
```yaml
# experiment.yaml
seed: 42
n_seeds_ga: 10
n_random_runs: 30
k_values: [2, 3, 4, 5]
force_k: 3
fitness_sample_size: 3000
split: {train: 0.6, val: 0.2, test: 0.2}
ga: {population: 40, generations: 40, elitism: 2, tournament: 3,
     crossover: blend, mutation: gaussian, mutation_sigma: 0.2,
     tau: 0.0, w_max: 0.3, min_active: 3, lambda_active: 0.02,
     stagnation_patience: 10, seed_equal_weights: false}
variants: {v2_alpha: 0.5}
penalties: {min_cluster_frac: 0.05, neff_min: 3}
```
*Aceptación:* cambiar de dataset implica solo editar estos YAML (se demuestra con el sintético).

**0.3 ★ Generador sintético con señal plantada (M).** `synthetic.py`: datos mixtos (continuas, binarias, nominales de alta cardinalidad, una variable con escala enorme), con k clusters definidos **solo por un subconjunto conocido de variables**; el resto es ruido. Guarda la lista de variables informativas verdaderas.
*Aceptación:* reproducible por semilla; hay un CSV sintético y su `schema.yaml` de ejemplo.

### Fase 1 — Auditoría del dataset

**Esta fase es obligatoria y bloqueante.** Hasta que el humano confirme `schema.yaml`, no se debe hacer ningún análisis posterior sobre datos reales.

**1.1 ★ Carga y normalización de nombres (S).** Lee el CSV real. Normaliza nombres de columna con `html.unescape` (p. ej. `*&#x4D;arried` es `*Married`), elimina `*`, espacios y mayúsculas, y guarda el mapeo `nombre_original → nombre_normalizado` en `results/audit/column_map.json`.
*Aceptación:* test con los nombres alterados `*Age` y `*&#x4D;arried`.

**1.2 ★ Auditoría automática (M).** Implementa `audit.py` y ejecútalo con `python -m plena.audit --config config/schema.yaml`. Debe generar `results/audit/data_audit.md` y `data_audit.json` con **todo lo siguiente**:

*Estructura*
- filas, columnas, memoria, tipos de pandas vs tipos inferidos (continua, discreta, binaria, categórica, texto, fecha);
- columnas **ID-like** (relación de valores únicos ≈ 1) y candidatas a identificador;
- duplicados de filas completas y de ID.

*Calidad*
- faltantes por columna y por fila; patrón de faltantes (¿se concentran en grupos de columnas?);
- **códigos centinela** sospechosos (`-1`, `-9`, `99`, `999`, `9999`, `""`, `"NA"`, `"?"`) tratados como posibles faltantes (solo reportar, no corregir sin confirmación);
- columnas constantes y casi constantes (>99% un valor);
- cardinalidad de categóricas; niveles raros (<1%);
- valores imposibles o fuera de rango esperado;
- columnas casi duplicadas (|ρ| > 0.95 o categóricas idénticas).

*Distribuciones*
- por variable continua: media, mediana, desviación, asimetría, curtosis, % de outliers (regla IQR), mínimo y máximo;
- **variables monetarias/escala enorme** (rango o asimetría extremos): sugerir transformación log1p o `RobustScaler` (solo como sugerencia).

*Etiqueta `depressed`*
- valores únicos, codificación observada, prevalencia, faltantes;
- **no inferir la semántica** (qué es 0 y qué es 1) solo por el nombre: marcar `UNCONFIRMED`.

*Fuga de información*
- para cada variable, su asociación con la etiqueta (AUC univariado, información mutua, V de Cramér o correlación punto-biserial según el tipo) y **alertar** si el AUC univariado > 0.85 o la asociación es sospechosamente alta;
- buscar columnas que parezcan **derivadas** de la etiqueta (por ejemplo, una puntuación de cuestionario de depresión o un indicador "diagnosticado");
- reportar cuáles variables fueron el resultado de la misma encuesta que genera la etiqueta.

*Estructura de grupos y partición*
- ¿hay repetidos por persona/hogar/ciudad que obliguen a particionar por grupo? (revisar `survey_id`, `ville_id` y similares);
- variables geográficas (`ville_id`): cardinalidad y concentración de la etiqueta por valor.

*Viabilidad computacional*
- n, D (variables) y D tras one-hot; memoria de una matriz n×n (`n² · 8 bytes`); recomendación de `fitness_sample_size`;
- tiempo medido de **una** evaluación de fitness (K-Means + Silhouette) en una submuestra.

*Sesgo y sensibilidad*
- variables sensibles (edad, estado civil, género, ubicación): tamaños de subgrupo y prevalencia de la etiqueta por subgrupo.

*Suficiencia de datos*
- razón n/D y tamaño de la clase minoritaria en cada partición.

*Aceptación:* el informe se genera sin intervención manual, incluye una sección **"Hallazgos y decisiones pendientes"** y gráficos básicos en `results/audit/figures/` (faltantes, distribuciones, prevalencia).

**1.3 ★ Borrador de `schema.yaml` (S).** A partir de la auditoría, genera `config/schema.draft.yaml` con tipo propuesto por columna, exclusiones propuestas con motivo y todas las ambigüedades marcadas `UNCONFIRMED`. **No** lo copies a `schema.yaml`.
*Aceptación:* cada columna aparece una vez, con tipo propuesto y una justificación.

**1.4 ★ PARADA: confirmación humana (S).** Presenta al usuario un resumen corto con: (a) hallazgos críticos de la auditoría, (b) lista de decisiones pendientes (ver sección 4), (c) el borrador de esquema. **Espera respuesta.** Con las respuestas, genera `config/schema.yaml`.
*Aceptación:* `schema.yaml` sin ningún `UNCONFIRMED`.

**1.5 ★ Exclusiones y test anti-fuga (M).** Aplica el esquema: excluye ID y etiqueta de `X` por defecto, y las columnas derivadas de la etiqueta que el humano haya confirmado. Test que verifica que ninguna columna prohibida entra en la matriz de características.
*Aceptación:* `pytest tests/test_no_leakage.py` pasa.

**1.6 ★ Particiones train / validación / test (S).** Reproducibles, agrupadas por `group_column` si existe; estratificar por etiqueta solo para balancear particiones. Guarda los índices en `results/splits.json`.
*Aceptación:* sin solapamiento; mismos índices con la misma semilla; test de prevalencia por partición.

**1.7 ★ Documentación del diccionario de datos (S).** `results/audit/data_dictionary.md`: columna, significado, tipo, tratamiento de faltantes, incluida/excluida, motivo.

### Fase 2 — Representación

**2.1 ★ Preprocesamiento Ruta A (M).** `ColumnTransformer`: imputación (mediana/moda; indicador de faltante si el humano lo pidió), `RobustScaler` o `StandardScaler` según la auditoría, one-hot para nominales (`handle_unknown="ignore"`), ordinales solo con orden documentado. Devuelve además el **mapeo columna transformada → variable original** (necesario para D3).
*Aceptación:* se ajusta solo con train; el mapeo cubre el 100% de las columnas.

**2.2 ★ Pruebas del preprocesamiento (S).** Sin NaN ni inf; sin uso de estadísticas de val/test; mismas columnas en las tres particiones; escala enorme manejada.

**2.3 ★ Submuestra para fitness y estimación de coste (S).** Fija una submuestra aleatoria reproducible de train de `fitness_sample_size` filas. Mide el tiempo de una evaluación y estima el coste total del GA (`pob × gens × tiempo`). Si supera el presupuesto del usuario, reduce la submuestra, población o generaciones y regístralo.
*Aceptación:* tabla `results/cost_estimate.csv`.

**2.4 ○ Ruta B: Gower (L).** Matrices `D_j` por variable precalculadas (rango para continuas, coincidencia para nominales/binarias) y `D(w) = Σ w_j D_j / Σ w_j`. K-Medoids sobre matriz precalculada (verifica compatibilidad de `kmedoids`; plan B: implementación propia de PAM). Compara con una implementación de referencia en un caso pequeño.
*No afirmes que FAISS acelera Gower.*

### Fase 3 — Baseline de clustering

**3.1 ★ K-Means, k ∈ {2,3,4,5} (S).** Con `random_state` y `n_init` fijos. Registra Silhouette, tamaños, tiempo. Invalida k=1 y k=n.

**3.2 ★ Asignación de muestras nuevas (M).** `predict()`: centroide más cercano usando el mismo preprocesamiento y los mismos pesos. Añade el modo **KNN sobre vecinos de train** (se completará con FAISS en 7.3).
*Aceptación:* predecir train reproduce las etiquetas del ajuste.

**3.3 ★ Módulo de evaluación (M).** Internas: Silhouette (misma métrica/espacio que el clustering), Davies-Bouldin, Calinski-Harabasz. Externas (si hay etiqueta): RI, ARI, AMI, NMI, prevalencia por cluster, chi-cuadrado, AUC de pertenencia. Clasificación por mapeo cluster→etiqueta mayoritaria: balanced accuracy, F1, ROC-AUC. Estabilidad: ARI entre semillas/bootstrap.
*Aceptación:* tests con valores conocidos; las métricas internas funcionan **sin etiqueta**.

**3.4 ★ Tabla baseline (S).** Resultados de 3.1 con pesos iguales en train y validación. Referencia por azar: Silhouette/ARI con etiquetas permutadas (≥100 permutaciones).
*Aceptación:* `results/baseline.csv` reproducible.

### Fase 4 — Algoritmo genético

**4.1 ★ Decodificación del cromosoma (S).** `decode(g, tau, w_max, min_active)` → `(mask, weights)`. Garantiza: longitud D, valores finitos, pesos ≥ 0 suma 1, `min_active` respetado (si hay menos, activa las de mayor gen), tope `w_max`. Y `transform_weighted(X, w, col_map)` con `sqrt(w_j / n_cols_j)` por bloque.
*Aceptación:* tests de longitud, finitud, suma 1, tope, mínimo de activas y equivalencia con la distancia ponderada calculada a mano.

**4.2 ★ Operadores (M).** Selección por torneo (3), cruce blend (BLX-α) o aritmético, mutación gaussiana con recorte a [-1,1], elitismo (1–3). Todos usan un `Generator` con semilla.
*Aceptación:* los hijos siempre están en [-1,1]; las élites pasan intactas.

**4.3 ★ Ciclo evolutivo (M).** Población 30–60, generaciones 30–50. Parada por máximo de generaciones, tiempo o estancamiento. **No** se detiene por "superar el baseline". Registra por generación: mejor, media, diversidad, N_eff, número de variables activas y evaluaciones totales. Opción `seed_equal_weights` (un individuo con pesos iguales); por defecto desactivada, y reporta ambas variantes si se usa.
*Aceptación:* corre de extremo a extremo; resultados idénticos con la misma semilla.

**4.4 ★ Función de aptitud V1 y V2 (M).**
- **V1 (no supervisada):** `fitness = S − λ·(n_activas / D) − penalizaciones`. S es el Silhouette sobre el **mismo espacio ponderado** que produjo los clusters, calculado en la submuestra de train.
- **V2 (semisupervisada):** `fitness = α·S + (1−α)·ARI_train − λ·(n_activas / D) − penalizaciones`, con `ARI_train` calculado **solo en train** contra `depressed`. Se etiqueta siempre como V2.
- Penalizaciones: clusters inválidos (1 solo o más de lo permitido), cluster menor a `min_cluster_frac`, `N_eff < neff_min`.
- Fitness **determinista** (K-Means con `random_state` fijo dentro del fitness) y **caché** por cromosoma redondeado.
- Devuelve puntuación **y** desglose de penalizaciones. Una solución inválida recibe un fitness muy bajo, nunca una excepción.
*Aceptación:* tests de cada penalización; mismo cromosoma ⇒ mismo valor.

**4.5 ★ Prueba de señal plantada (S).** Ejecuta el GA (V1) sobre el dataset sintético de 0.3. Criterio definido **antes** de correr: en la mayoría de semillas, ≥70% del peso total cae en las variables informativas verdaderas.
*Aceptación:* el criterio se cumple. Si no, revisa el GA antes de seguir (no uses datos reales hasta que pase).

**4.6 ★ Reproducibilidad (S).** Dos corridas con la misma semilla dan el mismo historial; semillas distintas dan historiales distintos.

### Fase 5 — Experimentos principales

**5.1 ★ Baselines de optimización (M).** (a) pesos iguales con todas las variables; (b) **búsqueda aleatoria** con el mismo número de evaluaciones que el GA, 30 repeticiones; (c) ○ filtro de varianza o Laplacian score como selección no evolutiva.

**5.2 ★ GA de selección + ponderación (M).** Para cada k de `k_values` y para V1 y V2, 10 semillas. Guarda pesos, máscara, historial, tiempos y N_eff.

**5.3 ★ Evaluación en validación y test (M).** En validación: elegir configuración (k, α, λ, mejor individuo por semilla). En **test, una sola vez**, evaluar con la configuración elegida. Métricas internas, externas y de clasificación por mapeo.
*Aceptación:* se registra explícitamente que test no se usó para ninguna decisión.

**5.4 ★ Tabla comparativa y estadística (M).** Método, variante, representación, k, Silhouette, ARI, RI, AMI, balanced accuracy, variables activas, N_eff, tiempo de optimización, tiempo de inferencia, semilla. Media ± desv., prueba de Wilcoxon pareada GA vs. cada baseline. Declara si **mejora, empata o empeora**. ARI cercano a 0 es un resultado legítimo: no lo presentes como fallo ni como éxito.
*Aceptación:* `results/comparison.csv` y `results/comparison.md`.

**5.5 ★ Gráfica Pareto y convergencia (S).** Silhouette vs. número de variables activas; convergencia media ± desv. del GA; barras de pesos medios.

### Fase 5B — Extensiones (solo si 5.1–5.4 están completas)

**5B.1 ○ NSGA-II (M).** Con `pymoo` o `DEAP`. Objetivos: maximizar S, maximizar ARI en train y minimizar variables activas. Evalúa el frente de Pareto en validación y selecciona una solución con criterio declarado de antemano.
**5B.2 ○ CMA-ES (S).** Mismo presupuesto de evaluaciones que el GA, como optimizador comparador.
**5B.3 ○ MLP pequeño (M).** Referencia de aprendizaje profundo de la propuesta v1 (ver 8.1).

### Fase 6 — Robustez e interpretabilidad

**6.1 ★ Estabilidad (S).** Spearman y Jaccard entre pesos/variables de distintas semillas; ARI entre asignaciones de clusters.
**6.2 ★ Ablaciones (M).** Sin `ville_id` (o variable geográfica), sin variables financieras, por tipo de variable.
**6.3 ★ Perfil de clusters (S).** Medianas/proporciones por cluster, tamaños, métricas por cluster. Revisa si el resultado lo domina una sola variable o un proxy demográfico. Sin inferir causalidad.
**6.4 ○ Sensibilidad de hiperparámetros (M).** σ de mutación, tasa de cruce, población, τ, `w_max`.
**6.5 ★ Revisión por subgrupos (S).** Métricas por edad / estado civil / ubicación si la auditoría los marcó.

### Fase 7 — FAISS e inferencia

**7.1 ★ Vectores ponderados (S).** Construye `X_train · sqrt(w)` (con el escalado por bloque). **No normalices** a norma unitaria. Verifica que L2 sobre ellos reproduce la distancia ponderada.
**7.2 ★ Índice y correctitud (S).** `IndexFlatL2` (devuelve distancia **al cuadrado**). Compara vecinos con fuerza bruta (NumPy/scikit-learn) sobre consultas de validación.
*Aceptación:* coincidencia exacta de los k vecinos (salvo empates).
**7.3 ★ Inferencia KNN sobre FAISS (M).** Asigna cluster a un punto nuevo por mayoría de sus k vecinos de train; compara con la asignación por centroide (acuerdo ARI y métricas de clasificación).
**7.4 ★ Benchmark de latencia (M).** Media y p95 por consulta, memoria aproximada; FAISS vs. fuerza bruta. Si el dataset es pequeño, documenta que el beneficio puede ser nulo o negativo; no lo ocultes.
**7.5 ★ Persistencia (S).** Guarda el índice, el mapeo id interno → fila original, los pesos y los transformadores ajustados (`joblib`).

### Fase 8 — Modelos de referencia

**8.1 ○ Supervisados (M).** Regresión logística y árbol de decisión sobre `depressed`, mismas particiones; F1, balanced accuracy, ROC-AUC. Opcional: MLP pequeño. Aclara que estas métricas no son comparables con Silhouette y que la etiqueta puede ser un indicador de encuesta, no un diagnóstico.

### Fase 9 — Cierre

**9.1 ★ Artefactos (S).** En `results/`: configuración exacta, semillas, versiones, mejores pesos y máscara, historial de fitness, CSV de todos los experimentos, transformadores, índice FAISS y mapeo.
**9.2 ○ Texto: BoW vs embeddings (L).** Solo si el equipo decide cubrir lo prometido en la presentación del Avance 1.
**9.3 ★ README y reporte técnico (M).** Comando de ejecución, decisiones (sección 1), resultados, amenazas a la validez y limitaciones clínicas. Incluye la justificación del pivote de posparto a depresión general y de no fijar k=3 como único valor.
**9.4 ★ Verificación independiente (S).** Otra persona ejecuta desde cero con el mismo YAML y compara resultados.

---

## 4. Protocolo de preguntas al usuario

Tras la auditoría (paso 1.4), pregunta **de una vez** lo siguiente, con las opciones que haya detectado:

1. ¿Qué significan los valores de `depressed` (0/1) y cómo se obtuvo la etiqueta?
2. ¿Cuáles columnas son nominales, ordinales (con qué orden), binarias, continuas y cuáles son identificadores?
3. ¿Hay columnas derivadas de la etiqueta o de la misma fuente que la etiqueta (excluirlas)?
4. ¿Los códigos sospechosos (`-1`, `99`, `999`, …) son faltantes?
5. ¿`ville_id` se trata como categórica, se excluye o se usa en ablación?
6. ¿Hay observaciones repetidas por persona/hogar que obliguen a particionar por grupo?
7. ¿Cuántas semillas y qué presupuesto de cómputo hay (tiempo máximo por corrida)?
8. ¿Se desea que NSGA-II (5B) se active si sobra tiempo?
9. ¿Este es el dataset definitivo o seguirá cambiando? (si cambia, mantener todo parametrizado)

No continúes con la Fase 2 hasta recibir respuesta o decisión explícita de seguir con los supuestos del borrador.

---

## 5. Orden de trabajo y definición del 60%

**Ruta crítica:** 0.1 → 0.2 → 0.3 → 1.1–1.7 → 2.1–2.3 → 3.1–3.4 → 4.1–4.6 → 5.1–5.5 → 6.1–6.3, 6.5 → 7.1–7.5 → 9.1, 9.3, 9.4.
**Paralelizable:** 0.3 con 1.1–1.2; 7.x con 6.x.

**Hitos:**
- **A — Datos listos:** 1.1–1.7 y 2.1–2.2, con test anti-fuga.
- **B — Baseline listo:** 3.1–3.4, con `predict()`.
- **C — Optimización lista:** 4.1–4.6, incluida la prueba de señal plantada.
- **D — Evidencia:** 5.1–5.5 con 10 semillas y prueba estadística, V1 y V2.
- **E — Demo técnica:** 7.1–7.5.

**El 60% se declara al cumplir A–E más 9.1 y 9.3.** Si el tiempo aprieta, recorta en este orden: 5B.x, 6.4, 8.1, 2.4, 9.2.

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Dataset final distinto del ilustrativo | Configuración declarativa, auditoría automática y prueba con sintético. |
| Pesos colapsan a 1–2 variables | Tope `w_max`, `min_active`, penalización por `N_eff`. |
| Fuga de la etiqueta | `depressed` fuera de `X`; auditoría de columnas derivadas; V2 con ARI solo en train. |
| Resultado = ruido (no hay estructura) | Referencia por permutación y prueba de señal plantada. |
| Fitness lento | Submuestra, caché, paralelización con `joblib`, medición previa (2.3). |
| Selección sobre el conjunto reportado | Train/val/test; test se toca una vez. |
| Clusters interpretados como niveles clínicos | Nombres `cluster_i`; perfiles y evidencia externa antes de etiquetar. |
| Clusters que reflejan solo demografía o ciudad | Ablaciones y revisión por subgrupos. |
| Incompatibilidad de librerías | Verificar en 0.1; plan B documentado. |
| FAISS mal aplicado | Solo vectores densos ponderados; no afirmar aceleración de Gower. |
| Datos de salud mental sensibles | No subir datos; no guardar información personal en `results/`. |

---

## 7. Fuentes

**Del proyecto:** `PresentacionAvance1.txt` (motivación, GA-based Automated Feature Weighting, inferencia rápida), `DocumentoPropuestaAvance1.txt` (cromosoma, fitness, Silhouette, baseline, validación, FastAPI/FAISS), recomendaciones del equipo para el Avance 2 (dominio [-1,1], FAISS, Silhouette y Rand Index).

**Técnicas (verificar enlaces antes de citarlas):**
- Rousseeuw (1987), *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*.
- Rand (1971) y Hubert & Arabie (1985) (Rand Index y Adjusted Rand Index).
- Gower (1971) (coeficiente general de similitud para datos mixtos).
- Documentación de scikit-learn: `silhouette_score`, `rand_score`, `adjusted_rand_score`.
- Documentación de FAISS: https://faiss.ai/
- Johnson, Douze & Jégou (2021), *Billion-scale similarity search with GPUs*, IEEE Trans. Big Data 7(3).
- Deb et al. (2002), *A fast and elitist multiobjective genetic algorithm: NSGA-II*, IEEE Trans. Evol. Comput. (para 5B).
- Hansen (2016), *The CMA Evolution Strategy: A Tutorial* (para 5B).
- Liu et al. (2024), distancia Gower ponderada para datos mixtos: https://doi.org/10.1186/s12874-024-02427-8
- Antecedente de GA para selección de variables optimizando Silhouette en K-Means: https://doi.org/10.1016/j.aca.2003.12.020
- Mitchell (1998); Goldberg (1989); Cover & Hart (1967) (de la propuesta v1).

---

## 8. Entrega esperada al cerrar este avance

El experimento debe responder con evidencia:
1. ¿El GA mejora el Silhouette frente a pesos iguales y a la búsqueda aleatoria con el mismo presupuesto?
2. ¿La mejora se mantiene entre semillas y en validación/test?
3. ¿Cuántas variables bastan para un resultado comparable (reducción de dimensionalidad)?
4. ¿Los clusters guardan asociación con `depressed` (ARI, RI, AMI, prevalencia por cluster) y cuánto aporta la variante V2 frente a V1?
5. ¿Qué precisión (balanced accuracy / F1) se logra por mapeo cluster→etiqueta y por KNN sobre FAISS?
6. ¿Cuánto cuesta la optimización y cuánto la inferencia?
7. ¿Qué habría que cambiar para usar el dataset definitivo?

**Conclusión esperada:** la contribución principal es un GA de dominio real bien definido, evaluado y comparado de forma justa. FAISS y la API apoyan la eficiencia del sistema, pero no sustituyen la validación del método evolutivo. Los clusters son patrones estadísticos exploratorios, no diagnósticos de depresión.
