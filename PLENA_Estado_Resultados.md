# PLENA — estado, resultados y siguiente paso

Auditoría del 21 de octubre de 2026 sobre el código de `main` (`a35e50f`), la corrida guardada en `results/real/` y el documento de lectura que ya estaba en este archivo. La corrida real quedó versionada el 15 y 16 de octubre de 2026; el cierre documental anterior es del 20 de octubre. Este texto conserva esas cifras, las vuelve a contrastar con el código y los CSV, y separa lo implementado de lo demostrado.

PLENA agrupa perfiles de una encuesta. No diagnostica depresión, no la predice y no asigna niveles clínicos. Los grupos se llaman `cluster_0`, `cluster_1`, … 

## Cómo se usan las palabras de estado

| Palabra | Significado en este documento |
|---|---|
| Implementado | El comportamiento existe en el código. |
| Ejecutado | Hay un artefacto de una corrida. |
| Evaluado | Esa corrida dejó una métrica comparable. |
| Demostrado | La evidencia basta para sostener la frase en cuestión. |
| Propuesto | Trabajo futuro. No se ha corrido. |

## 1. Resumen ejecutivo

PLENA, en el estado versionado, hace esto: un algoritmo genético de cromosoma real elige un subconjunto de variables y les asigna pesos no negativos; K-Means agrupa la representación ponderada; la validación elige `k` y el individuo; la prueba se lee una vez.

La pregunta de este avance es si esa ponderación produce una representación más útil para agrupar perfiles relacionados con la depresión que las mismas variables sin ponderar.

Hay dos respuestas, y no dicen lo mismo.

**Demostrado para la geometría.** En validación, el genético mejora el Silhouette euclidiano frente a pesos iguales y frente a la búsqueda aleatoria, en V1 y en V2, para `k ∈ {2, 3, 4, 5}`. El contraste es Wilcoxon bilateral, 10 parejas, p < 0.05. La configuración elegida llega a Silhouette 1.0 también en la prueba. Fuente: `results/real/comparison.md`, `results/real/comparison.csv`, `results/real/chosen.json`.

**No demostrado para la depresión.** Esa mejora no se traslada a la etiqueta `depressed`. El ARI de prueba de las dos soluciones elegidas es −0.0167. El intervalo bootstrap por aldea, 400 réplicas, va de −0.0436 a 0.0095 e incluye el cero. En varios `k` el ARI de validación del genético empata o empeora frente a pesos iguales. Las cinco combinaciones de las variables elegidas tienen prevalencias entre 12.5 % y 17.5 %, con una tasa global de 16.7 % (238/1429). Una logística y un árbol, ajustados en el espacio ponderado de V1 y puntuados en validación, quedan en AUC 0.483 y 0.485.

La causa de la diferencia está en la función objetivo. Con `k = 5`, cuatro binarias de ingreso y trabajo solo ocupan cinco celdas. K-Means coloca un cluster en cada celda, la distancia interna es 0 y el Silhouette es 1 para cualquier peso positivo que mantenga esas celdas separadas. El vector concreto de la semilla 50 es un ejemplo de ese soporte, no un peso identificado. Maximizar Silhouette, en este CSV, enumera un catálogo de formas de ingreso y trabajo.

Esa lectura es la de V1 y V2 en `results/real/`. No describe la sonda posterior.

La sonda de `results/collapse_probe/` resta una penalización cuando un cluster tiene radio casi nulo. Las 20 corridas evitaron el Silhouette 1: el máximo fue 0.756 y ninguna enumeró patrones (`enumerated_fraction` 0). La regla, escrita antes de optimizar, eligió `k = 5` por el mayor ARI medio de validación. Esa media es 0.003, y pesos iguales en el mismo `k` dan 0.018. La hipótesis no recibió apoyo. Cinco semillas no autorizan una mejora general. El detalle está en la sección 11.

## 2. Alcance actual

El fenómeno de interés de este avance es depresión en general. El plan (`PLENA_Avance2_Plan_Cursor.md`, primera página) deja el posparto como etapa anterior del método, no como el requisito vigente.

Eso no convierte a la muestra en una muestra de población general. `config/schema.yaml` describe el archivo como una encuesta de condiciones de vida en zonas rurales con un indicador binario cuya semántica no está documentada, y dice de forma explícita que no es un estudio de posparto ni una muestra descrita aquí como población general.

| Pieza | Qué es en este repositorio |
|---|---|
| Pregunta vigente | Si ponderar y optimizar variables mejora la representación usada para agrupar perfiles relacionados con depresión, frente a baselines reproducibles. |
| Dato usado | `data/b_depressed.csv`. Encuesta rural, 1 429 filas, 23 columnas. |
| Etiqueta | `depressed`: 0 en 1 191 filas, 1 en 238. Según la ficha pública de OpenML 43419, 0 es no deprimido y 1 es deprimido. Cómo se construyó el indicador no está documentado. |
| Propuesta inicial | `PresentacionAvance1.txt`, `DocumentoPropuestaAvance1.txt` y `PlanPropuestaParaAvance2.txt` no están en el repositorio ni en `c:\Users\aledo\source`. El plan los cita en su sección 7. No se reconstruye su texto. |
| Fuera de este avance, según el plan | Texto, sueño, actividad física (D14) y FastAPI (D13, queda para un cierre posterior). |

`k = 3` se reporta siempre porque la propuesta antigua hablaba de riesgo bajo, medio y alto. En el código y en los resultados, `k = 3` es un valor más del recorrido `{2, 3, 4, 5}`. No es una escala clínica, y no fue el `k` elegido.

## 3. Estado de la implementación

| Componente | Estado | Dónde |
|---|---|---|
| Carga, normalización de nombres, partición por aldea, anti-fuga | Implementado y ejecutado | `src/plena/data.py`, `tests/test_no_leakage.py`, `results/real/splits.json` |
| Ruta A: imputación, `log1p`, escalado, distancia euclidiana ponderada, K-Means | Implementado y evaluado | `preprocessing.py`, `distance.py`, `clustering.py` |
| Genético, aptitudes V1 y V2, pesos iguales, búsqueda aleatoria, permutación | Implementado y evaluado en `results/real/` | `genetic_algorithm.py`, `fitness.py`, `baselines.py` |
| Penalización de colapso sobre V1, sonda de 5 semillas | Implementado y evaluado solo en validación. No mejora el ARI medio frente a pesos iguales | `collapse.py`, `collapse_probe.py`, `results/collapse_probe/` |
| CMA-ES | Implementado y ejecutado como comparador de train, una semilla, `k = 5`. No elige el modelo | `cma_es.py`, `results/real/cma_es.json` |
| NSGA-II | Implementado y ejecutado en train, `k = 3`, frente de 6 soluciones. No evaluado en validación | `nsga2.py`, `results/real/nsga2.json` |
| Ruta B, Gower y K-Medoids | Implementado y ejecutado en 180 filas de train, `k = 3`, backend `kmedoids.fasterpam`. Las etiquetas se calculan y se descartan. El CSV no trae Silhouette | `distance.py`, `experiments.py` (`_route_b`), `results/real/route_b.csv` |
| FAISS `IndexFlatL2` | Implementado y evaluado sobre los vectores ponderados de train | `faiss_index.py`, `results/real/faiss_report.json` |
| Logística y árbol | Implementado y evaluado en validación, espacio de V1. Referencia auxiliar | `supervised.py`, `results/real/supervised_reference.json` |
| API FastAPI | No implementada | No hay módulo ni dependencia de FastAPI |
| Filtro no evolutivo, sensibilidad de hiperparámetros, MLP, texto | No implementados como estudio. El plan los marca opcionales | Pasos 5.1c, 6.4, 5B.3, 9.2 |
| Señal plantada, criterio de mayoría de semillas | La función `planted_signal_report` está implementada. La corrida guardada en `results/synthetic/` es un pipeline con presupuesto reducido. No hay un informe que reconte el criterio «mayoría de semillas con al menos 70 % del peso en las variables informativas» | `experiments.py`, `results/synthetic/chosen.json` |

`pytest` el 21 de octubre de 2026, con la penalización de colapso incluida: 40 pruebas pasadas en 25.30 s. Entre ellas están `tests/test_collapse.py` y `tests/test_no_leakage.py`.

## 4. Flujo real del sistema

El flujo que ejecuta `scripts/run_experiments.py` es el siguiente.

1. Lee `config/schema.yaml` y `config/experiment.yaml`. Los nombres de columna se normalizan (`Survey_id` pasa a `survey_id`, `Age` a `age`). El mapeo de la auditoría está en `results/audit/column_map.json`.
2. Quita filas sin etiqueta. En este CSV no se quitó ninguna: `depressed` no tiene faltantes.
3. Construye `X` solo con las 20 variables de `features`. `banned_columns` excluye la etiqueta, `survey_id` y las exclusiones. `ville_id` no está en `features`: es `group_column`.
4. Parte por aldea con `StratifiedGroupKFold` de 5 pliegues, semilla 42. El pliegue 0 es prueba, el 1 es validación y los pliegues 2–4 son train. Ninguna aldea cae en dos partes. La etiqueta solo equilibra las partes.
5. Ajusta el preprocesamiento solo con train y lo aplica a validación y prueba.
6. El genético y la búsqueda aleatoria optimizan la aptitud en train. Con este archivo la submuestra de aptitud es train completo, porque `fitness_sample_size` es 1 500 y train tiene 856 filas.
7. Cada candidato se vuelve a puntuar en validación: K-Means se ajusta en train completo y los puntos de validación se asignan al centroide más cercano.
8. Se elige un individuo por variante. El criterio quedó escrito en `results/real/selection_criterion.txt` antes de usar la prueba. `run_manifest.json` registra `test_used_for_selection: false`.
9. La prueba se puntúa una vez, para el individuo elegido, para pesos iguales y para un candidato de la búsqueda aleatoria.

### Datos

| Hecho | Valor | Fuente |
|---|---|---|
| Archivo | `data/b_depressed.csv` | `config/schema.yaml` |
| Tamaño | 1 429 filas, 23 columnas, 0 filas duplicadas | `results/audit/data_audit.md`; recontado en esta auditoría |
| Positivos | 238 (16.7 %) | El mismo informe; 238/1429 = 0.1666 |
| Faltantes | 20 celdas, solo en `no_lasting_investmen` | Auditoría |
| Centinelas 99 y 999 | Identificadores en `survey_id` y `ville_id`. No hay −1 ni −9. No se tratan como faltantes | `PREGUNTAS_PENDIENTES.md`, pregunta 4 |
| Duplicados | 0 filas completas. `survey_id` es único | Auditoría |
| Atípicos | La auditoría puede contarlos por IQR. El preprocesamiento no elimina filas ni recorta valores | `preprocessing.py` |
| AUC univariado máximo | 0.570 en `age`. Ninguna alerta por encima de 0.85 | `results/audit/data_audit.json` |

Variables de `X`, en el orden del cromosoma:

| Índice | Variable | Tratamiento real |
|---|---|---|
| 0 | `sex` | Binaria. Moda y `StandardScaler` |
| 1 | `age` | Continua, 17–91. Mediana y `StandardScaler` |
| 2 | `married` | Binaria. Moda y `StandardScaler` |
| 3 | `number_children` | Discreta numérica, no nominal. Mediana y `StandardScaler` |
| 4 | `education_level` | Ordinal 1–19. El 15 no aparece y no se imputa. `OrdinalEncoder` y `StandardScaler` |
| 5 | `total_members` | Discreta numérica. Mediana y `StandardScaler` |
| 6–10 | activos y gastos: `gained_asset`, `durable_asset`, `save_asset`, `living_expenses`, `other_expenses` | `log1p` y `StandardScaler`. No había negativos, así que `log1p` se aplicó |
| 11–14 | `incoming_salary`, `incoming_own_farm`, `incoming_business`, `incoming_no_business` | Binarias. Moda y `StandardScaler` |
| 15–16 | `incoming_agricultural`, `farm_expenses` | `log1p` y `StandardScaler` |
| 17 | `labor_primary` | Binaria. Moda y `StandardScaler` |
| 18–19 | `lasting_investment`, `no_lasting_investmen` | `log1p` y `StandardScaler`. Los 20 vacíos de la última se imputan con la mediana de train |

Fuera de `X`: `survey_id`, `ville_id`, `depressed`.

No hay variables nominales en la ruta principal. Una comparación aparte codifica `education_level` como nominal y otra añade `ville_id` como nominal de 255 niveles, ambas con pesos iguales y sin una segunda búsqueda (`results/real/representation_ablation.csv`). Al partir por aldea, los niveles de `ville_id` de validación no están en train.

Las binarias se estandarizan. Dos filas con el mismo patrón 0/1 siguen siendo el mismo punto después de una transformación afín por columna. Por eso un catálogo de cinco patrones puede tener distancia interna cero aunque las columnas ya no estén en la escala 0/1.

### Partición real

Semilla 42. Train 856 filas y 143 positivos. Validación 286 y 47. Prueba 287 y 48. El estudio se detendría si alguna parte tuviera menos de 20 positivos. No se detuvo. Índices: `results/real/splits.json`.

### Agrupamiento

K-Means de scikit-learn sobre la matriz ya ponderada, distancia euclidiana, `n_init = 10`, `random_state = 42`. `k` se fija por corrida y recorre `{2, 3, 4, 5}`. Se rechazan `k = 1` y `k = n`. La asignación de un punto nuevo es al centroide de menor distancia euclidiana al cuadrado. Hay también un KNN de 15 vecinos sobre train; en la solución elegida coincide con el centroide porque cada cluster es un solo punto repetido.

El agrupamiento no se hace sobre el CSV crudo. Se hace sobre la ruta A ponderada. Los pesos iguales usan las 20 variables y no pasan por el tope `w_max`.

### Fuga

`depressed` no entra en `X`. En V1 tampoco entra en la aptitud. En V2 entra solo como ARI de train dentro de la aptitud. La selección de V2 usa además el ARI de validación: la etiqueta de validación influye en qué individuo se elige, y por eso el ARI de validación del elegido no es una confirmación independiente. La lectura externa que no participó en la elección es la de prueba.

No hay una columna con nombre de instrumento de depresión. El AUC univariado máximo es 0.570. Eso no demuestra que la etiqueta sea independiente de las covariables: la construcción del indicador no está documentada y una fuga no registrada seguiría siendo posible. Queda como limitación, no como una exclusión adicional.

## 5. Método de optimización

El optimizador principal es un algoritmo genético de dominio real. No es CMA-ES ni NSGA-II. Esas dos piezas existen como comparadores y no eligieron el modelo reportado.

Cada individuo es un vector de 20 genes, uno por variable original, no por columna transformada. El gen vive en `[−1, 1]`. El plan pedía ese dominio, y el código lo cumple para el gen.

La traducción a pesos es otra restricción, y es la que entra en la distancia:

- Si `g_j ≤ τ` con `τ = 0`, la variable queda fuera.
- Si `g_j > 0`, el valor positivo propone un peso.
- Los pesos activos se normalizan para sumar 1.
- Después se proyectan al tope `w_max = 0.3` y se renormalizan.
- Un peso decodificado es cero o positivo. Un gen negativo apaga la variable. No invierte la dirección de la característica en la distancia.
- Si no hay suficientes activas para que el tope permita sumar 1, se encienden más genes en orden decreciente. Con `w_max = 0.3` hacen falta al menos 4 variables. `min_active = 3` no es el límite que manda.
- El piso de `N_eff = 1 / Σ w²` en el vector más concentrado que el tope permite, `(0.3, 0.3, 0.3, 0.1)`, es 3.571.

El peso de la variable `j` entra como `sqrt(w_j / n_columnas_j)` en su bloque. En la solución elegida las cuatro variables son binarias de un solo bloque, así que el factor es `sqrt(w_j)`.

### Operadores y parada

Implementado en `src/plena/genetic_algorithm.py` y `config/experiment.yaml`:

| Pieza | Valor real |
|---|---|
| Inicialización | Uniforme en `[−1, 1]`. `seed_equal_weights` está en falso: la población no arranca con un individuo de pesos iguales |
| Población y generaciones | 40 y 40 |
| Selección | Torneo de 3 |
| Cruce | BLX-α con `α = 0.5` y probabilidad 0.8. El cruce aritmético existe y no es el configurado |
| Mutación | Gaussiana, `σ = 0.2`, probabilidad 0.2 por gen, recortada a `[−1, 1]` |
| Elitismo | 2 individuos |
| Parada | Máximo de generaciones, o 10 generaciones sin mejorar el mejor fitness en más de `1e-9` |
| Lo que no está en el ciclo | No se detiene al superar un baseline. No hay un reloj por corrida dentro de `run_ga`. El tope de 10 minutos era una meta de presupuesto, no un aborto codificado |
| Semilla | `numpy.random.Generator` por semilla de la corrida. K-Means dentro de la aptitud usa `random_state = 42` |
| Caché | Por variante, `k` y gen redondeado a 6 decimales |
| Mejor solución | El mejor fitness visto durante la corrida, no el individuo de la última generación |

El genético de la solución V1 hizo 920 evaluaciones y el de V2 hizo 840. El presupuesto configurado es 1 600 (40 × 40). La diferencia es la parada por estancamiento. La búsqueda aleatoria recibe las 1 600 evaluaciones completas, con la misma decodificación y la misma aptitud, y con cromosomas distintos en V1 y en V2 (`random_search_seed` depende de la variante y de `k`). En Silhouette, el genético ganó aun con ese presupuesto de evaluaciones más corto. Las medias de tiempo observado están en la sección 6.

### Aptitud

V1, no supervisada:

`Silhouette − λ · (n_activas / 20) − penalización por cluster pequeño − penalización por N_eff`

`λ = 0.02`. Con 4 activas el término `λ` vale 0.004. El Silhouette se calcula en el mismo espacio ponderado que produjo los clusters, con distancia euclidiana. La etiqueta no entra.

V2, semisupervisada:

`α · Silhouette + (1 − α) · ARI_train − las mismas penalizaciones`

`α = 0.5`. El ARI de train usa `depressed` solo en train. Cuando el Silhouette llega a 1 y el ARI de train está junto a cero, la mezcla vale cerca de 0.5. Una solución con Silhouette 0.3 y ARI 0.2 vale cerca de 0.25. Con este `α`, un Silhouette perfecto domina cualquier ARI plausible de este CSV. V2 no es, en la práctica, un criterio que pueda preferir la etiqueta sobre el catálogo de cinco celdas.

Penalizaciones: solución inválida (un solo cluster, o tantos clusters como filas) recibe puntuación −1. Un cluster por debajo del 5 % de las filas y un `N_eff` menor que 3 restan. La dirección de las dos aptitudes es maximizar.

La selección entre semillas no es el fitness de train. V1 elige el mayor Silhouette de validación; a igualdad, mayor `N_eff`, luego menos activas, luego la semilla menor. V2 elige `α · Silhouette + (1 − α) · ARI` de validación, menos `λ · (activas / 20)`, y después los mismos desempates. `k = 3` se guarda siempre, aunque no gane.

Riesgo que esta corrida materializó: el Silhouette de un espacio discreto con tantas celdas ocupadas como `k` llega a 1 en cuanto cada celda es su propio cluster. El tope evitó concentrar el peso en una sola variable. No evitó enumerar cinco patrones binarios. El criterio de desempate por `N_eff` no llegó a actuar entre las semillas 47 y 50, porque el Silhouette de la 47 es 0.9999999998827571 y el de la 50 es 1.0. El código compara el flotante antes del desempate. El hallazgo estable es el soporte de cuatro variables, no el vector de la semilla 50.

`collapse_coef` vale 0 en `config/experiment.yaml`. Con ese valor la aptitud de V1 y de V2 no cambia. La sonda usa `config/experiment.collapse.yaml`, donde el coeficiente es 1, y no reescribe `results/real/`.

## 6. Resultados iniciales

Las cifras de esta sección, hasta la sección 9, son el estudio histórico V1/V2 de `results/real/`. La sonda que penaliza el colapso está aparte, en la sección 11, y no se mezcla con estas tablas.

Todas las filas de esta sección salen de archivos guardados. No se volvió a optimizar. El recuento de las cinco combinaciones se hizo en esta auditoría sobre `data/b_depressed.csv`, agrupando las cuatro columnas y contando `depressed`. Las 1 429 filas caen en esas cinco filas; no hay otras combinaciones.

### Catálogo que explica el Silhouette 1

| Salario | Finca propia | Negocio | Trabajo primario | Hogares | Positivos | Tasa |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 610 | 106 | 17.4 % |
| 0 | 1 | 0 | 0 | 360 | 63 | 17.5 % |
| 1 | 0 | 0 | 1 | 257 | 42 | 16.3 % |
| 0 | 0 | 1 | 0 | 154 | 21 | 13.6 % |
| 0 | 0 | 0 | 1 | 48 | 6 | 12.5 % |

Tasa global: 16.7 %. Rango entre patrones: 12.5–17.5 %.

`results/real/cluster_profiles.csv` es el ajuste de train del V1 elegido (856 filas), no la tabla anterior. Los cinco clusters de train tienen tamaños 361, 157, 98, 211 y 29, y reproducen los mismos cinco patrones. Las prevalencias de `depressed` en ese ajuste van de 0.138 a 0.177.

### Validación: media entre corridas

Fuente: `results/real/comparison.csv` (334 filas: 328 de validación y 6 de prueba) y el resumen publicado en `results/real/comparison.md`. Pesos iguales es una sola corrida por `k` y variante, por eso no tiene desviación. El genético tiene 10 semillas (42–51). La búsqueda aleatoria tiene 30.

Silhouette de validación, media. Mayor es mejor geometría en ese espacio. No es comparable entre valores distintos de `k`.

| Método | V1, k=2 | V1, k=3 | V1, k=4 | V1, k=5 | V2, k=5 |
|---|---:|---:|---:|---:|---:|
| Pesos iguales | 0.165 | 0.136 | 0.155 | 0.127 | 0.127 |
| Búsqueda aleatoria | 0.591 ± 0.033 | 0.631 ± 0.049 | 0.684 ± 0.068 | 0.700 ± 0.068 | 0.736 ± 0.106 |
| Genético | 0.660 ± 0.056 | 0.839 ± 0.008 | 0.916 ± 0.050 | 0.963 ± 0.035 | 0.940 ± 0.038 |

ARI de validación contra `depressed`. Cero es concordancia de azar. Estas medias están junto a cero en los tres métodos. El detalle del contraste está en la sección 7.

| Método | V1, k=5 | V2, k=5 |
|---|---:|---:|
| Pesos iguales | 0.018 | 0.018 |
| Búsqueda aleatoria | 0.007 ± 0.012 | 0.013 ± 0.018 |
| Genético | 0.003 ± 0.013 | 0.005 ± 0.011 |

Tiempo medio de optimización observado en V1: el genético va de 66.8 s (`k = 2`) a 82.3 s (`k = 3`); la búsqueda aleatoria, de 130.0 s a 149.6 s. Pesos iguales no optimiza. La sonda previa a la malla (`results/real/cost_estimate.csv`) estimó 0.0324 s por evaluación, 0.865 min por corrida de GA y 0.995 h para el estudio, sin recortes, con tope de 6 h. No hay en los artefactos un reloj de pared del estudio completo. Las medias por corrida sí están medidas y quedan por debajo de los 10 minutos que el presupuesto se había fijado.

El genético activó 4 variables en las 80 corridas de validación. `N_eff` medio en V1, `k = 5`: 3.63. La búsqueda aleatoria activó en media unas 4.6–5.1 variables.

### Soluciones elegidas

Las dos variantes eligieron `k = 5`. Las dos usan las mismas cuatro variables: `incoming_salary`, `incoming_own_farm`, `incoming_business` y `labor_primary`. Los pesos no son los mismos. Fuente: `results/real/chosen.json`.

V1, semilla 50. Silhouette de validación 1, ARI de validación 0.000264, `N_eff` 3.684, 920 evaluaciones.

| Variable | Antes del tope | Después |
|---|---:|---:|
| `incoming_salary` | 0.298 | 0.300 |
| `incoming_business` | 0.306 | 0.300 |
| `labor_primary` | 0.273 | 0.276 |
| `incoming_own_farm` | 0.123 | 0.124 |

V2, semilla 43. Mismo Silhouette y mismo ARI de validación, `N_eff` 3.672, 840 evaluaciones.

| Variable | Antes del tope | Después |
|---|---:|---:|
| `labor_primary` | 0.331 | 0.300 |
| `incoming_own_farm` | 0.279 | 0.291 |
| `incoming_business` | 0.275 | 0.288 |
| `incoming_salary` | 0.116 | 0.121 |

En V2 el tope sí actúa: `labor_primary` baja de 0.331 a 0.300.

Tres semillas de V1 en `k = 5` quedan en el mismo ARI de validación, 0.000264: la 47, la 49 y la 50. La 49 y la 50 tienen Silhouette 1. La 47 tiene Silhouette 0.9999999998827571 y `N_eff` 3.919, con las mismas cuatro variables y pesos más parejos (0.205, 0.265, 0.300 y 0.230). Fuente del flotante: `results/real/checkpoints/main__ga__v1__k5__s47.json`. A igualdad estricta de Silhouette, el criterio escrito habría elegido la 47 por su `N_eff`. El código no declara esa igualdad.

`k = 3`, que se reporta siempre. El tope fabrica una cuarta variable a partir de un gen casi nulo.

| Variante | Semilla | Pesos finales | Silhouette val. | ARI val. |
|---|---:|---|---:|---:|
| V1 | 42 | `sex` 0.10, `incoming_salary` 0.30, `incoming_business` 0.30, `labor_primary` 0.30 | 0.843 | −0.012 |
| V2 | 43 | `sex` 0.10, `incoming_salary` 0.30, `incoming_own_farm` 0.30, `labor_primary` 0.30 | 0.814 | 0.029 |

Antes del tope, en V1 `k = 3`, `sex` era del orden de `1e-6` y `incoming_business` se llevaba 0.811. La cuarta variable de ese reporte no es `farm_expenses`.

### Prueba, leída una vez

Fuente: las seis filas `split = test` de `results/real/comparison.csv`. `k = 5`.

| Método | Variante | Semilla | Silhouette | ARI | RI | AMI | Accuracy balanceada | F1 macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Genético | V1 | 50 | 1.000 | −0.0167 | 0.409 | −0.0056 | 0.5 | 0.454 |
| Pesos iguales | V1 | 42 | 0.137 | −0.0025 | 0.393 | −0.0026 | 0.5 | 0.454 |
| Búsqueda aleatoria | V1 | 65 | 0.865 | −0.0112 | 0.399 | −0.0025 | 0.5 | 0.454 |
| Genético | V2 | 43 | 1.000 | −0.0167 | 0.409 | −0.0056 | 0.5 | 0.454 |
| Pesos iguales | V2 | 42 | 0.137 | −0.0025 | 0.393 | −0.0026 | 0.5 | 0.454 |
| Búsqueda aleatoria | V2 | 56 | 0.703 | −0.0255 | 0.428 | −0.0043 | 0.5 | 0.454 |

V1 y V2 tienen el mismo ARI y el mismo RI de prueba. Cambian índices de cluster, no la partición de los hogares. El intervalo bootstrap del ARI, por aldea y con 400 réplicas, es el mismo en las dos variantes: de −0.0436 a 0.0095, media −0.0164 (`results/real/bootstrap_test.json`). El Silhouette bootstrap se mantiene entre 0.9999 y 1.

La fila de búsqueda aleatoria en prueba no usa la misma regla de elección que el genético de V2. Para V2, el código se queda con el candidato de mayor ARI de validación, no con la mezcla `α · Silhouette + (1 − α) · ARI`. El contraste de validación, que empareja las 10 semillas, no depende de esa fila.

Ningún cluster tiene mayoría deprimida en train. El mapeo a la etiqueta mayoritaria predice siempre «no». Por eso la accuracy balanceada es 0.5 en los tres métodos. Ese 0.5 no es una precisión diagnóstica.

### Otros resultados guardados

| Pieza | Qué se corrió | Qué quedó | Fuente |
|---|---|---|---|
| Wilcoxon | Genético contra cada baseline, validación, Silhouette y ARI | Publicado en la sección 7 | `comparison.md` |
| Permutación | 100 repeticiones en el espacio del V1 elegido | ARI nulo medio −0.0006. Silhouette de particiones aleatorias medio −0.053 | `permutation_reference.json` |
| Estabilidad | 10 semillas de V1 en el `k` elegido, 45 parejas | Spearman de pesos 0.595, Jaccard de máscaras 0.548, ARI de asignaciones 0.624 | `stability.json` |
| Robustez | 5 particiones × 5 semillas, sin reoptimizar la malla, `k = 3` y `k = 5`, solo validación | En `k = 5`, 25 filas por variante: Silhouette medio 0.976 (V1) y 0.958 (V2); ARI medio −0.0085 y −0.0066 | `robustness.csv` |
| Ablación financiera | Quitar el bloque de gastos sobre los pesos ya elegidos | Silhouette 1 y ARI 0.000264, idénticos al elegido. Esas columnas ya pesaban 0 | `ablation.csv` |
| Codificación | Educación nominal y aldea nominal, pesos iguales | Silhouette de validación entre 0.127 y 0.185 | `representation_ablation.csv` |
| CMA-ES | Misma aptitud de train, `k = 5`, semilla 42, tope de 1 600 evaluaciones | Puntuación 0.915 en V1 y 0.479 en V2, `N_eff` 3.571. No se puntuó en validación para elegir | `cma_es.json` |
| NSGA-II | Train, `k = 3`, población 20, 15 generaciones, semilla 42 | 6 soluciones. Silhouette de train entre 0.403 y 0.733. ARI de train hasta 0.085. Ninguna se puntuó en validación | `nsga2.json` |
| Supervisados | Logística y árbol en el espacio ponderado de V1, ajuste en train, puntuación en validación | Accuracy balanceada 0.5. AUC 0.483 y 0.485. F1 macro 0.455 | `supervised_reference.json` |
| FAISS | Índice plano L2, 286 consultas de validación, `k = 15` vecinos | Acuerdo por fila 1.0, 0 desacuerdos, 286 empates de frontera en float32, 0 coincidencias exactas. NumPy medio 1.75e-5 s, FAISS 2.23e-5 s. ARI centroide contra KNN: 1.0 | `faiss_report.json` |
| Sintético | Presupuesto reducido: 5 semillas, población 16, 8 generaciones. Esquema usado de 8 variables | El V1 elegido pone 0.30 en `signal_x`, `signal_y` y `signal_flag`, y 0.10 en `noise_nominal`. El generador actual tiene 21 variables de entrada; esa corrida no usó ese esquema completo. No hay informe del criterio de mayoría | `results/synthetic/chosen.json`, `schema.used.yaml`, `data/synthetic/informative_variables.json` |

El procedimiento de las medias de robustez fue agrupar `results/real/robustness.csv` por variante con `k = 5` (25 filas cada una) y promediar Silhouette y ARI. No es el mismo número que la media de las 10 semillas de la partición principal.

## 7. Análisis comparativo

La comparación justa, ya ejecutada y evaluada, es genético contra pesos iguales y contra búsqueda aleatoria, misma representación, mismo `k`, misma partición, Silhouette y ARI de validación. El Wilcoxon empareja por la semilla etiquetada. Pesos iguales tiene una sola fila, así que las 10 semillas del genético se comparan contra ese valor constante. La búsqueda aleatoria tiene 30 corridas; el contraste usa las 10 cuya semilla etiquetada coincide con la del genético (42–51). Las medias de la tabla anterior sí usan las 30.

Fuente de cada frase de esta lista: `results/real/comparison.md`. Umbral publicado: p < 0.05, Wilcoxon bilateral.

**Silhouette.** El genético mejora frente a los dos baselines, en V1 y en V2, en los cuatro `k`. Las diferencias medias contra pesos iguales van de 0.495 (`k = 2`, V1) a 0.836 (`k = 5`, V1). Contra la búsqueda aleatoria van de 0.061 a 0.269. El p de esos contrastes es 0.0020, salvo V1 `k = 2` contra búsqueda aleatoria (0.014) y V2 `k = 2` contra búsqueda aleatoria (0.0039).

**ARI contra pesos iguales.**

| Contraste | Resultado publicado |
|---|---|
| V1, k = 2 | Empata (diferencia media −0.0018, p = 0.65) |
| V1, k = 3 | Empeora (−0.017, p = 0.002) |
| V1, k = 4 | Mejora (+0.007, p = 0.043) |
| V1, k = 5 | Empeora (−0.015, p = 0.010) |
| V2, k = 2, 3 y 4 | Empata. En k = 3 y k = 4 el p queda en 0.078 y 0.082 |
| V2, k = 5 | Empeora (−0.013, p = 0.014) |

**ARI contra búsqueda aleatoria.** Empata en todos los `k` y en las dos variantes.

`comparison.md` arrastra una frase de plantilla: con cinco parejas, el p mínimo si todas las diferencias tienen el mismo signo es 0.0625. El contraste publicado empareja 10 semillas. Con 10 parejas del mismo signo el p bilateral llega a 0.002, y eso es lo que ocurrió en el Silhouette. La frase de las cinco parejas sí aplica a los empates de ARI cuyo p quedó entre 0.06 y 0.08: ahí una diferencia pequeña se declara empate. No explica el Silhouette.

La referencia por permutación deja el ARI de validación del elegido (0.000264) dentro del ruido nulo (media −0.0006, desviación 0.0068). El Silhouette 1 queda lejos del Silhouette de particiones aleatorias en el mismo espacio (media −0.053). Está cerca de lo que produce separar cinco celdas ocupadas.

En la prueba, el Silhouette del genético (1.0) sigue por encima del de pesos iguales (0.137) y del de la búsqueda aleatoria de V1 (0.865). El ARI del genético (−0.0167) es más negativo que el de pesos iguales (−0.0025). Es una sola lectura, no un segundo Wilcoxon. Alcanza para no afirmar que la ponderación haya mejorado la concordancia en datos no usados para elegir.

CMA-ES y NSGA-II no entran en esta comparación como métodos evaluados en las mismas condiciones. Sus cifras son de train, con otra `k` en el caso de NSGA-II, y no pasaron por la regla de selección.

### Qué se puede concluir

- Optimizar con el genético mejora el Silhouette de validación frente a los dos baselines, y esa geometría se sostiene en la prueba. Demostrado para este CSV, esta partición y esta aptitud.
- La solución describe cinco patrones de ingreso y trabajo. La prevalencia de `depressed` dentro de ellos permanece cerca de la tasa global. Eso está contado en la tabla de la sección 6.
- V2 no aportó otra partición en la prueba. Eligió el mismo soporte.
- La ponderación no ha demostrado ser más útil que los pesos iguales para concordar con `depressed`. En el `k` elegido, el ARI de validación empeora de forma estadística y el de prueba es compatible con el cero.
- No se puede ordenar CMA-ES, NSGA-II ni Gower respecto del genético con los archivos actuales. Gower se ejecutó y no se evaluó. Los otros dos se evaluaron en train y no sustituyen al individuo elegido.

## 8. Interpretación de las métricas

El código calcula, en `src/plena/evaluation.py`, Silhouette, Davies–Bouldin, Calinski–Harabasz, Rand Index (`rand_score`), Adjusted Rand Index (`adjusted_rand_score`), AMI, NMI, chi-cuadrado, AUC de la pertenencia al cluster, accuracy balanceada y F1 macro. `comparison.csv` guarda Silhouette, ARI, RI, AMI, accuracy balanceada y F1. Davies–Bouldin, Calinski–Harabasz, NMI y chi-cuadrado se calculan y no se escriben en esa tabla. RI y ARI no son intercambiables: los dos se calculan, y el que decide la lectura externa es el ARI.

| Métrica | Qué mide aquí | Límite de esta corrida |
|---|---|---|
| Silhouette | Separación de los grupos en el espacio ponderado y con la distancia euclidiana que usó K-Means. Se maximiza | Un valor más alto no identifica mejor la depresión. Aquí llega a 1 porque las filas de una misma celda binaria son el mismo punto |
| ARI | Concordancia con `depressed`, corregida por azar. Cero es el acuerdo esperado por azar | No es causalidad, ni capacidad diagnóstica, ni validación clínica. La construcción de la etiqueta no está documentada |
| RI | Concordancia cruda de pares, sin corregir el azar | En prueba vale 0.409 para el genético y 0.393 para pesos iguales. Con una clase minoritaria del 16.7 %, el RI se infla. No se presenta como éxito |
| AMI | Información mutua normalizada con la etiqueta | En prueba es −0.0056 para el genético. Junto a cero, igual que el ARI |
| Estabilidad | Spearman y Jaccard entre semillas; ARI entre asignaciones | 0.595, 0.548 y 0.624 en V1, `k = 5`. El soporte de cuatro variables se repite; el vector de pesos no es único |
| Robustez | Misma búsqueda repetida en otras particiones, sin rehacer toda la malla de métodos | El Silhouette de `k = 5` sigue alto y el ARI sigue junto a cero |
| Accuracy balanceada y F1 | Mapeo de cada cluster a la etiqueta mayoritaria de train | 0.5 y 0.454 en la prueba, en los tres métodos, porque ningún cluster de train tiene mayoría deprimida |
| AUC supervisado | Logística o árbol sobre el espacio ya ponderado | 0.48 en validación. No se compara con el Silhouette. No es el conjunto de prueba del clustering |
| Latencia | Tiempo de una evaluación, de una corrida y de una consulta de vecinos | La inferencia por centroide está en microsegundos. FAISS, con 856 vectores de train, es más lento que NumPy |

La evaluación externa es válida como concordancia exploratoria porque la etiqueta no entra en `X` y la prueba no eligió el modelo. No es válida como evidencia de que los grupos sean perfiles clínicos de depresión.

## 9. Limitaciones y riesgos

- La etiqueta es un indicador de la fuente. ARI, RI, AMI y AUC no son validación clínica.
- Maximizar Silhouette describió una partición geométrica. En este CSV ocurrió, y el plan ya advertía que una binaria puede fabricar clusters perfectos. El tope desplazó el colapso de una variable a cuatro binarias con cinco celdas.
- El vector de pesos no está identificado. Tres semillas de V1 en `k = 5` comparten soporte y ARI. El desempate flotante eligió la semilla 50 sin usar la prueba.
- Elegir con el mismo conjunto que se reporta inflaría el resultado. Para `k` y para los pesos se evitó: train busca, validación elige, prueba se lee una vez. El ARI de validación del elegido sí participó en la selección de V2, así que la cifra externa de referencia es la de prueba.
- V2 mezcla Silhouette y ARI con `α = 0.5`. Con Silhouette igual a 1, la mezcla no puede preferir la etiqueta.
- La ablación financiera no puso a prueba al modelo elegido. Esas columnas ya tenían peso cero.
- La edad en `subgroups.json` se corta a 12 niveles, en orden de aparición (`max_levels` en `stability.py`). Sexo y estado civil están completos en ese archivo.
- Los datos son mixtos y la ruta principal los mete todos en una euclidiana después de escalar. La ruta de Gower se ejecutó sin métrica guardada, así que no hay una comparación de representación.
- La estabilidad de los pesos es moderada (Spearman 0.595). Repetir la corrida con otra semilla cambia el vector dentro del mismo tipo de solución.
- `results/synthetic/` no responde la pregunta del CSV real. Además, el esquema que esa corrida usó tiene 8 variables, y el generador versionado tiene 21.
- No se midió el tiempo de pared del estudio completo. La estimación previa es 0.995 h. Las medias por corrida sí están en `comparison.md`.
- Los documentos de la propuesta inicial no están en el repositorio. Las afirmaciones sobre esa propuesta se limitan a lo que el plan todavía dice de ella.
- Versionar `data/b_depressed.csv` contradice la regla antigua del plan que pedía ignorarlo. El `.gitignore` actual lo deja versionado a propósito, para que un clon traiga los datos y las cifras. Esta auditoría no cambia esa decisión.

## 10. Recomendación técnica

No defender el genético como contribuyente a los perfiles de depresión. En la métrica que habla de la etiqueta, V1 y V2 no mejoraron el baseline, y la sonda de la sección 11 tampoco.

El problema concreto es que la aptitud premia una partición cuya distancia interna es cero. El genético la encuentra mejor que la búsqueda aleatoria, y CMA-ES, con la misma aptitud, también llega al piso de `N_eff` del tope. Cambiar de optimizador sin cambiar el objetivo repite el catálogo de cinco celdas. Gower queda como comparación de representación para más adelante: hoy no tiene métricas guardadas, y una distancia distinta seguiría premiando celdas binarias si el objetivo sigue siendo el Silhouette.

Esa hipótesis ya se contrastó en la sección 11. La penalización evitó el catálogo de celdas. El ARI medio de validación no superó a pesos iguales. No corresponde repetir el genético ni cambiar de optimizador.

El siguiente paso, todavía no ejecutado, es un filtro de una sola pasada. En train, y solo en train, se calcula la correlación punto-biserial de cada variable con `depressed`. Se conservan las de mayor valor absoluto, al menos las cuatro que exige `w_max`, con un tope de seis. Los pesos son proporcionales a ese valor absoluto y se proyectan con el mismo tope. K-Means usa la partición que ya existe. La métrica es el ARI de validación, el baseline son los pesos iguales de `results/collapse_probe/equal_weights.csv`, y el margen sigue siendo 0.02. La prueba no se lee. Si ese filtro tampoco supera el margen, no queda evidencia para seguir optimizando pesos con el fin de concordar con esta etiqueta en esta representación.

## 11. Sonda de colapso

Ejecutada el 21 de octubre de 2026. Directorio nuevo: `results/collapse_probe/`. `results/real/` no se modificó. La prueba no se puntuó. El protocolo se escribió en `protocol.txt` antes de lanzar el genético.

### Regla congelada

La aptitud es la de V1 más `collapse_coef * near_collapse`, con coeficiente 1 y `tau = 0.15`. `depressed` no entra en `X` ni en esa aptitud. El ARI se calcula después, en validación, con `adjusted_rand_score` sobre las etiquetas que K-Means asigna al centroide más cercano. Usarlo para elegir el `k` hace que ese ARI de validación no sea una confirmación independiente.

`n_unique` cuenta filas distintas de la matriz ponderada tras redondear a 6 decimales. Eso solo describe el soporte. La penalización es otra cosa: para cada punto, `a` es la distancia a su centroide y `b` la de ese centroide al más cercano. El término es `exp(−a / (tau · b))`. Vale 1 con radio cero y baja cuando el radio es una fracción real del hueco entre clusters. Un polvo de `1e-8` no lo apaga: las pruebas de `tests/test_collapse.py` lo fijan. Tampoco convierte la búsqueda en una constante. El término está entre 0 y 1, la misma escala que el Silhouette, y un cluster cuyo radio ya es comparable al hueco casi no paga. Hace falta dispersión dentro del grupo, no un truco de redondeo.

Una corrida es elegible si, en validación, `n_unique > k`, `near_collapse < 0.5` y Silhouette `< 0.99`. Un `k` es viable si al menos 3 de las 5 semillas lo son. Se toma el `k` viable de mayor ARI medio. El apoyo descriptivo exigía que esa media superara a pesos iguales en al menos 0.02. Con 5 semillas no hay Wilcoxon ni mejora general. El margen de 0.02 está por encima de la desviación del ARI nulo histórico (0.0068).

La partición de train y de validación coincide con `results/real/splits.json` (`split_check.json`: 856 y 286 filas). Configuración: semillas 42–46, población 20, 20 generaciones, `k ∈ {2, 3, 4, 5}`, parada por estancamiento de 8 generaciones. Cada corrida hizo 360 o 400 evaluaciones y tardó entre 20.9 y 32.3 s. El reloj de pared del script fue 116 s.

### Resultado

Las 20 corridas fueron elegibles. La fracción de filas en clusters de un solo patrón fue 0. El Silhouette máximo fue 0.756. El control histórico V1, en `k = 5`, tenía media 0.963 y soluciones en 1.0. Fuente de la sonda: `validation.csv`. Control: `historical_v1_validation.csv`.

| k | ARI medio (5 semillas) | ARI pesos iguales | Diferencia | Silhouette medio | near_collapse medio | n_unique mediano |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | −0.0196 | −0.0095 | −0.0100 | 0.592 | 0.127 | 12 |
| 3 | −0.0020 | 0.0058 | −0.0078 | 0.656 | 0.181 | 11 |
| 4 | 0.0019 | −0.0100 | +0.0119 | 0.701 | 0.220 | 11 |
| 5 | 0.0031 | 0.0179 | −0.0148 | 0.681 | 0.218 | 15 |

La regla eligió `k = 5` porque su media, 0.0031, es la más alta. Pesos iguales en ese `k` dan 0.0179. La diferencia es −0.0148. Estado en `decision.json`: `rechazado`. La hipótesis no recibió apoyo.

El individuo de mayor ARI dentro de ese `k` es la semilla 43: ARI 0.0246, Silhouette 0.693, `n_unique` 14, `near_collapse` 0.210, cinco variables (`married` 0.137, `other_expenses` 0.289, `incoming_salary` 0.237, `incoming_own_farm` 0.079, `incoming_business` 0.258). Esa semilla se eligió con el ARI de validación. No confirma el resultado. La media de las cinco es la cifra de decisión, y queda por debajo del baseline.

En `k = 4` la media supera a pesos iguales por 0.012. Está por debajo del margen de 0.02 y no es la comparación que la regla congeló. No se reinterpreta como apoyo.

El V1 histórico sigue siendo el control de la aptitud vieja: mejora el Silhouette y no el ARI. La sonda baja el Silhouette respecto de ese control y tampoco mejora el ARI. Penalizar el colapso resuelve la geometría degenerada. No produce clusters con mayor concordancia exploratoria con `depressed`.

Quedan huecos opcionales que no son el siguiente paso: volcar Davies–Bouldin, Calinski–Harabasz, NMI y chi-cuadrado a la tabla histórica; puntuar en validación una solución del frente NSGA-II; guardar el Silhouette de la ruta B; quitar el corte de 12 niveles en la edad.

## 12. Estado de los componentes auxiliares

**FAISS.** Implementado, ejecutado y evaluado. `IndexFlatL2` sobre los vectores de train ya ponderados, sin normalizar a norma 1. No optimiza variables y no acelera la distancia de Gower. En esta corrida el acuerdo con la fuerza bruta en float32 es 1.0 porque los 286 casos de validación caen en empate de frontera, y NumPy fue más rápido (1.75e-5 s frente a 2.23e-5 s). El KNN y el centroide coinciden (ARI 1.0) porque cada cluster es un punto repetido. Con 856 filas de train, FAISS no aporta al objetivo de ponderar variables. No merece más esfuerzo hasta que haya una representación que valga la pena consultar.

**API.** No hay FastAPI ni otro servicio. El plan la deja fuera de este avance. Sigue fuera.

**Modelos de referencia.** La logística y el árbol están evaluados en validación, en el espacio de V1. Su AUC cerca de 0.48 dice que ese espacio no ordena la etiqueta. Son una comparación auxiliar, no un modelo de depresión y no un equivalente del clustering.

**CMA-ES y NSGA-II.** Ejecutados como extensiones después del estudio principal. No forman parte del modelo elegido. NSGA-II tiene en el código una función `select_from_front` que el estudio no llamó.

## 13. Reproducibilidad

Python 3.10 o superior. El cierre de la máquina que produjo `results/real/` está en `results/real/environment.txt`: Python 3.14.3, plena 0.2.0, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, faiss-cpu 1.15.1. El comando de entorno está en el README.

Semillas del genético: 42 a 51. Búsqueda aleatoria: 30 corridas, de las cuales el Wilcoxon usa las semillas etiquetadas 42 a 51. Partición: semilla 42. K-Means: `random_state` 42, `n_init` 10. Bootstrap: 400 réplicas, semilla del experimento. CMA-ES y NSGA-II: semilla 42.

Volver a lanzar `scripts/run_experiments.py` sobre `results/real` se salta los checkpoints que ya existen. Para verificar las cifras no hace falta borrar nada: `chosen.json`, `comparison.csv` y este documento se pueden leer en un clon. Reejecutar el estudio solo tendría sentido en otra máquina, o después de apartar los checkpoints a propósito. Esta auditoría no los tocó.

`results/_smoke/` sigue ignorado. No forma parte del estudio. Los CSV, `results/real/` y `results/synthetic/` están versionados. No se añadió ninguna regla de `.gitignore`.

### Mapa de `results/real/`

| Archivo | Contenido |
|---|---|
| `selection_criterion.txt` | Criterio escrito antes de la prueba |
| `chosen.json` | V1, V2 y el `k = 3` de reporte, con pesos antes y después del tope |
| `comparison.csv` | Validación de todas las corridas y las seis filas de prueba. Incluye RI y AMI |
| `comparison.md` | Medias y Wilcoxon |
| `splits.json` | Índices y conteo de positivos |
| `bootstrap_test.json` | Intervalo por aldea en la prueba |
| `stability.json` | Spearman, Jaccard y ARI de asignaciones, V1, `k = 5` |
| `robustness.csv` | 5 particiones × 5 semillas, `k = 3` y `k = 5`, solo validación |
| `cluster_profiles.csv` | Perfil de train del V1 elegido |
| `cost_estimate.csv` | Sonda de tiempo y presupuesto estimado |
| `faiss_report.json` | Vecinos, empates y latencia |
| `cma_es.json`, `nsga2.json` | Comparadores. No eligen el modelo |
| `representation_ablation.csv` | Educación nominal y aldea nominal, pesos iguales |
| `ablation.csv` | Bloque financiero quitado sobre los pesos ya elegidos |
| `supervised_reference.json` | Logística y árbol, puntuados en validación |
| `route_b.csv` | Backend y tamaño de la corrida Gower. Sin métrica |
| `checkpoints/` | Una corrida por archivo. La segunda ejecución se la salta |
| `artifacts/` | Transformador, pesos, centroides, índice FAISS, filas de train |
| `figures/` | Convergencia V1 en `k = 3`, que es el `force_k`, no el `k` elegido; Pareto; barras del V1 elegido |

## 14. Matriz de cumplimiento del plan

El plan de implementación sigue en `PLENA_Avance2_Plan_Cursor.md`. Esta matriz no lo da por cerrado en los puntos que la evidencia deja a medias. Los requisitos vigentes son los del avance de depresión general y de la ruta crítica del plan. Debajo van las metas de la propuesta inicial que este avance ya no trata como entrega.

### Requisitos vigentes

| Requisito | Fuente | Evidencia encontrada | Estado | Acción recomendada |
|---|---|---|---|---|
| Depresión general, no solo posparto | Plan, apertura; `schema.yaml` | El esquema y el plan lo dicen. La muestra sigue siendo una encuesta rural, no una muestra de población general | COMPLETO | Decirlo así en el reporte, con las dos frases |
| `depressed` fuera de `X` | Plan, regla 2; D6 | `banned_columns`, `tests/test_no_leakage.py`, 40 pruebas pasadas | COMPLETO | Mantener el test |
| Preprocesamiento ajustado solo con train | Plan, regla 3 | `preprocessing.py`; imputación de mediana de train | COMPLETO | Ninguna sobre el código actual |
| Prueba leída una vez | Plan, regla 4; paso 5.3 | `selection_criterion.txt`, `run_manifest.json` con `test_used_for_selection: false`, seis filas de prueba | COMPLETO | No volver a elegir con la prueba |
| Cromosoma real, selección y ponderación, tope y `N_eff` | D1, D2, D3 | `chromosome.py`, `chosen.json` | COMPLETO | Citar el soporte, no un único vector |
| K-Means, `k` en {2, 3, 4, 5}, `k = 3` reportado | D5 | `comparison.csv`, `chosen.json` (`reported_k3`) | COMPLETO | Seguir sin llamar a `k = 3` riesgo clínico |
| V1 sin etiqueta y V2 con ARI solo de train | D6, paso 4.4 | `fitness.py` | COMPLETO | No interpretar V2 como éxito sobre la etiqueta |
| Baselines de pesos iguales, búsqueda aleatoria y permutación | D10, pasos 3.4 y 5.1 | `comparison.csv`, `permutation_reference.json` | COMPLETO | Usarlos como referencia de la aptitud actual |
| 10 semillas, Wilcoxon, Silhouette y ARI | D11, paso 5.4 | `comparison.md` | COMPLETO | Reportar las dos métricas, no solo el Silhouette |
| Partición 60/20/20 por aldea | Pregunta 6, `data.py` | `splits.json`: 856 / 286 / 287 | COMPLETO | Conservar la semilla 42 |
| Estabilidad y perfiles | Pasos 6.1 y 6.3 | `stability.json`, `cluster_profiles.csv` | COMPLETO | Decir que el vector no es único |
| FAISS mínimo, sin afirmar que acelera Gower | D13, fase 7 | `faiss_report.json`: NumPy más rápido | COMPLETO | No seguir invirtiendo ahí |
| Artefactos y README | Pasos 9.1 y 9.3 | `results/real/`, README, este archivo | COMPLETO | Este documento es la lectura del avance |
| El genético mejora el Silhouette frente a los dos baselines | Pregunta 1 de la entrega | Wilcoxon, p = 0.002 en la mayoría de celdas | COMPLETO | Limitar la frase a la geometría |
| La mejora de Silhouette se sostiene en prueba | Pregunta 2 | Prueba: Silhouette 1.0. Bootstrap de Silhouette por encima de 0.9999 | COMPLETO | Añadir que es el catálogo de cinco celdas |
| Asociación con `depressed` y aporte de V2 | Pregunta 4 | ARI de prueba −0.0167, intervalo que incluye el cero. V2, misma partición | COMPLETO | La asociación no quedó demostrada. V2 no cambió el hallazgo |
| Accuracy y acuerdo del KNN | Pregunta 5 | Accuracy balanceada 0.5, F1 0.454, ARI centroide–KNN 1.0 | COMPLETO | Explicar el 0.5 por la clase mayoritaria |
| Ruta B, Gower | Paso 2.4, opcional | `route_b.csv` sin métrica. Etiquetas descartadas en el código | PARCIAL | No citar un Silhouette de Gower. Si se retoma, guardar la métrica |
| Ablaciones | Paso 6.2 | Financiera nula porque el peso ya era cero. Codificaciones alternativas con pesos iguales | PARCIAL | No presentarlas como prueba del modelo elegido |
| Subgrupos | Paso 6.5 | Sexo y estado civil completos. Edad cortada a 12 niveles | PARCIAL | Quitar el corte solo si el reporte lo exige |
| CMA-ES | Paso 5B.2 | `cma_es.json`, train, una semilla, `k = 5` | PARCIAL | No usarlo como modelo reportado |
| NSGA-II | Paso 5B.1 | Frente de train, `k = 3`, sin validación | PARCIAL | Puntuar una solución en validación solo si hace falta para el texto |
| Señal plantada, criterio de mayoría | Paso 4.5 | Un individuo V1 con 0.90 del peso en las tres señales y 0.10 en `noise_nominal`. Sin recuento de semillas | PARCIAL | No declarar el criterio cumplido |
| Coste medido del estudio completo | Pregunta 6 | Estimación 0.995 h. Medias por corrida en `comparison.md`. Tiempo de pared total no encontrado | PARCIAL | Citar la estimación y las medias, no un reloj que no está |
| Verificación por otra persona | Paso 9.4 | `pytest` local, 40 pasadas. No hay una segunda máquina | PENDIENTE | Lectura de `chosen.json` contra este documento. Reejecutar solo si se apartan los checkpoints |
| Filtro no evolutivo, sensibilidad, MLP | 5.1c, 6.4, 5B.3 | No hay archivos de resultado | PENDIENTE | Dejarlos fuera del experimento mínimo |
| Aptitud que penalice el colapso y decida por ARI de validación | Sección 11, `results/collapse_probe/` | 20/20 corridas fuera del colapso. ARI medio en el k elegido, 0.003, por debajo de pesos iguales, 0.018 | COMPLETO | No repetir la sonda. El resultado es negativo para la concordancia |

### Metas de la propuesta inicial que ya no son el alcance

Los tres txt de la propuesta no están en el repositorio. La fuente de esta tabla es la sección 7 del plan, que los cita, y las decisiones D13 y D14.

| Requisito | Fuente | Evidencia encontrada | Estado | Acción recomendada |
|---|---|---|---|---|
| Alcance limitado a depresión posparto | Plan, sección 7, como antecedente | El plan y el esquema lo sustituyen por depresión general, sobre una encuesta rural | SUSTITUIDO POR CAMBIO DE ALCANCE | Contarlo como etapa del método, no como el estudio actual |
| Tres clusters de riesgo bajo, medio y alto | Plan, D5, «la propuesta v1» | `k = 3` se reporta y no se eligió. Los clusters no tienen nombre clínico | SUSTITUIDO POR CAMBIO DE ALCANCE | Mantener los nombres `cluster_i` |
| FastAPI como entrega de este avance | Plan, D13 y sección 7 | No hay código de API. El plan la mueve al cierre | SUSTITUIDO POR CAMBIO DE ALCANCE | No implementarla hasta tener evidencia de la ponderación |
| Texto, sueño y actividad física | Plan, D14 y paso 9.2 | No hay datos ni código de esa ruta | SUSTITUIDO POR CAMBIO DE ALCANCE | El avance es tabular |
| Ignorar el CSV en Git | Plan, regla 10 | `.gitignore` versiona `data/b_depressed.csv` y `results/` a propósito | SUSTITUIDO POR CAMBIO DE ALCANCE | No volver a ignorar datos ni resultados |

## 15. Conclusión para presentar al profesor

Se implementó y se ejecutó un estudio reproducible: partición por aldea, preprocesamiento ajustado en train, algoritmo genético que selecciona y pondera variables, K-Means, dos aptitudes declaradas antes de correr, pesos iguales, búsqueda aleatoria y una lectura única de la prueba.

Se evaluó si esa ponderación mejora el agrupamiento. Mejora el Silhouette. En validación la diferencia frente a pesos iguales llega a 0.84 en `k = 5`, con p = 0.002. La configuración elegida tiene Silhouette 1.0 en la prueba.

Ese resultado es un catálogo de cinco patrones de ingreso y trabajo. La tasa de `depressed` dentro de ellos va de 12.5 % a 17.5 %, frente a 16.7 % global. El ARI de prueba es −0.0167 y su intervalo por aldea incluye el cero. V2, que mete el ARI en la aptitud de train, eligió el mismo soporte. Los modelos auxiliares en ese espacio tienen AUC de aproximadamente 0.48.

No está demostrado que optimizar las variables produzca grupos más útiles para la depresión que usar las variables sin ponderar. Está demostrado que optimizan una geometría que, en este archivo, se resuelve enumerando cuatro columnas binarias.

Después de ese estudio se corrió una sonda aparte, sin leer la prueba. Penalizar el radio nulo evitó el Silhouette 1 en las 20 corridas. El ARI medio de validación no superó a pesos iguales. Eso no es una mejora general: son 5 semillas y la regla ya rechazó la hipótesis. El reporte puede presentar V1/V2 como evidencia geométrica y la sonda como un resultado negativo sobre la concordancia, sin convertir los clusters en un hallazgo clínico.

### Qué se puede decir

- El objetivo ejecutado es ponderar variables para un agrupamiento, con dos aptitudes declaradas antes de correr.
- Train busca, validación elige, prueba se lee una vez. La partición es por aldea.
- El genético mejora el Silhouette de validación frente a los dos baselines, y el Silhouette elegido se sostiene en la prueba.
- El modelo elegido describe cinco patrones de ingreso y trabajo. La prevalencia del indicador permanece cerca de la tasa global.
- El tope `w_max` evitó el peso en una sola variable. No evitó el Silhouette 1.

### Qué no se sostiene

- Que PLENA detecte, diagnostique o prediga depresión.
- Que Silhouette 1.0 sea un ajuste del indicador.
- Que la cuarta variable del reporte `k = 3` sea `farm_expenses`. Es `labor_primary` en V1, junto con `sex`, `incoming_salary` e `incoming_business`.
- Que haya un solo vector de pesos.
- Que el estudio no tuviera potencia para el Silhouette. El p fue 0.002.
- Que la ablación financiera haya puesto a prueba al modelo elegido.
- Que CMA-ES, NSGA-II o FAISS sean el método que optimiza las variables en el resultado principal.
