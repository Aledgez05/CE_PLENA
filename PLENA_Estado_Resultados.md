# PLENA — estado, resultados y siguiente paso

Documento de lectura del avance. Cruza el código en `main`, `PLENA_Avance2_Plan_Cursor.md`, `PREGUNTAS_PENDIENTES.md` y la corrida guardada en `results/real/` (15–16 de octubre de 2026). Sirve para el reporte IEEE, la documentación técnica y para decidir qué no hace falta volver a correr.

PLENA no diagnostica depresión. Los grupos se llaman `cluster_0`, `cluster_1`, … y no son niveles clínicos.

## 1. Qué se clona

Un clon de este repositorio trae:

| Qué | Dónde |
|---|---|
| Código, pruebas y configuración confirmada | `src/plena/`, `tests/`, `config/` |
| Encuesta de este avance | `data/b_depressed.csv` (1 429 filas, 23 columnas) |
| Respuestas ya cerradas del esquema | `PREGUNTAS_PENDIENTES.md`, `config/schema.yaml` |
| Plan de implementación | `PLENA_Avance2_Plan_Cursor.md` |
| Auditoría automática del CSV | `results/audit/` |
| Estudio real cerrado | `results/real/` |
| Corrida sintética de pipeline | `results/synthetic/` |
| Este documento | `PLENA_Estado_Resultados.md` |

`results/_smoke/` es una corrida de humo local y sigue ignorada. No hace falta para leer ni para reproducir el avance.

El plan pedía no subir información personal en `results/`. La encuesta ya viajaba en el repositorio para que el equipo pudiera trabajar. Los artefactos de `results/` son transformaciones y métricas de ese mismo archivo (índices de fila, pesos, clusters). Se versionan para que el clon tenga las cifras. No añaden nombres ni expedientes.

Python 3.10 o superior. El cierre de esta máquina quedó en `results/real/environment.txt`: Python 3.14.3, plena 0.2.0, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, faiss-cpu 1.15.1. El comando de entorno está en el README. Volver a lanzar `scripts/run_experiments.py` sobre `results/real` se salta los checkpoints que ya existen.

## 2. Qué es el proyecto

Un algoritmo genético de dominio real elige y pondera variables para agrupar la encuesta. No entrena un clasificador de depresión.

La etiqueta `depressed` no entra en `X`. En la variante V1 tampoco entra en la aptitud. En V2 entra solo como ARI de train. La validación elige `k` y el individuo. La prueba se lee una vez, después de congelar esa elección. El criterio quedó escrito en `results/real/selection_criterion.txt` antes de usar la prueba.

Hay dos aptitudes, las dos del plan (paso 4.4):

- **V1.** Silhouette en la submuestra de train, menos una penalización por variables activas, clusters pequeños y `N_eff` bajo. Sin etiqueta.
- **V2.** `alpha` · Silhouette + `(1 − alpha)` · ARI de train, con las mismas penalizaciones. `alpha` = 0.5. El ARI usa `depressed` solo en train.

El cromosoma es real, en `[−1, 1]`, con un gen por variable original. Si el gen no supera `tau` = 0, la variable sale. Los pesos activos suman 1 y no pasan de `w_max` = 0.3. Con ese tope hacen falta al menos cuatro variables. El piso de `N_eff = 1 / Σ w²` en el vector más concentrado que el tope permite (0.3, 0.3, 0.3, 0.1) es 3.57. `min_active` = 3 no es el límite que manda: manda el tope.

El peso de la variable `j` entra como `sqrt(w_j / n_columnas_j)` en su bloque. Así una nominal de muchos niveles no domina. En la solución elegida las cuatro variables son binarias, así que cada una es un solo bloque.

## 3. Decisiones cerradas

Salen de la sección 1 del plan y de las nueve respuestas en `PREGUNTAS_PENDIENTES.md`. El esquema confirmado no tiene campos `UNCONFIRMED`.

| Tema | Decisión | Dónde quedó |
|---|---|---|
| Etiqueta | 0 = no deprimido, 1 = deprimido, según la ficha pública de Kaggle `diegobabativa/depression` y la cita del repo a OpenML 43419. Cómo se construyó el indicador no está documentado | `data/README.md`, pregunta 1 |
| Tipos | 20 variables de entrada confirmadas. Binarias, continuas, discretas y `education_level` ordinal 1–19. El nivel 15 no aparece y no se rellena | `config/schema.yaml`, pregunta 2 |
| Fuga por nombre | Ningún nombre parece un instrumento de depresión. El AUC univariado más alto de la auditoría ronda 0.57. No se excluyó ninguna columna por este motivo, y no se puede descartar una fuga no documentada | pregunta 3 |
| Códigos 99 y 999 | Identificadores reales en `survey_id` y `ville_id`, no faltantes. No hay −1 ni −9. Los 20 vacíos de `no_lasting_investmen` sí se imputan con la mediana de train | pregunta 4 |
| `survey_id` | Fuera de `X` | pregunta 6 |
| `ville_id` | Fuera de `X`. Es `group_column`: ninguna aldea cae en dos particiones. Hay una comparación aparte como nominal de 255 niveles, con pesos iguales y sin una segunda búsqueda | pregunta 5 |
| Partición | Train 60 % / validación 20 % / prueba 20 %, semilla 42, por aldea con `StratifiedGroupKFold` de 5 pliegues (0 prueba, 1 validación, 2–4 train) | pregunta 6, `data.py` |
| Presupuesto | 10 semillas del genético, 30 búsquedas aleatorias, población 40, 40 generaciones. Tope de estudio 6 h y de una corrida 10 min. No se recortó nada | pregunta 7, `cost_estimate.csv` |
| Escala grande | `log1p` y después `StandardScaler`. No había negativos, así que `log1p` se aplicó. El código puede usar `RobustScaler` si el esquema lo pide; esta corrida no lo pidió. La frase de que el rango intercuartílico quedaría en cero por los ceros no está escrita en el repo | pregunta 2, `preprocessing.py` |
| NSGA-II | Sí, solo si el estudio principal cerró y quedó tiempo. Antes, CMA-ES | pregunta 8 |
| Dataset | `b_depressed.csv` es el definitivo de este avance | pregunta 9 |
| k | Se recorre `{2, 3, 4, 5}`. Silhouette no se compara entre k como si fuera la misma escala. k = 3 se reporta siempre, aunque no gane. La propuesta vieja de riesgo bajo/medio/alto no fija el k | plan D5 |
| Nombres de cluster | `cluster_i`. No se etiquetan como riesgo clínico | plan, aviso inicial |
| Baselines | Pesos iguales, sin pasar por `w_max`. Búsqueda aleatoria con el mismo número de evaluaciones y con cromosomas distintos en V1 y en V2. Referencia por permutación. Sin parada del tipo «hasta superar el baseline» | plan D10, `baselines.py` |
| FAISS | Índice plano L2 sobre vectores ponderados, sin normalizar a norma 1. No se afirma que acelere la distancia de Gower. FastAPI queda fuera de este avance | plan D13 |
| Texto, sueño, actividad | Fuera de este avance | plan D14 |

La ficha de Kaggle declara licencia GPL-2 y 1 432 filas, y dice que las columnas originales del reto no estaban explicadas. `data/README.md` todavía dice que la licencia no está verificada. El archivo del repo tiene 1 429 filas y 238 positivos (16.7 %). Esa diferencia de filas hay que decirla en el reporte, no resolverla inventando un filtro.

Con semilla 42 la partición real queda así: train 856 filas y 143 positivos, validación 286 y 47, prueba 287 y 48. El estudio se detendría si alguna parte tuviera menos de 20 positivos. No se detuvo.

`fitness_sample_size` es 1 500 y train tiene 856 filas, así que la aptitud usa train completo. La submuestra del plan no llegó a recortar.

## 4. El plan contra lo que existe

La ruta crítica del 60 % (hitos A–E, más 9.1 y 9.3) está cubierta en código y en `results/real/`. Estos son los huecos reales, no una lista de pasos sin empezar.

| Paso del plan | Estado | Nota para el reporte |
|---|---|---|
| 0–1. Auditoría, esquema, anti-fuga, partición, diccionario | Hecho | `results/audit/`, `tests/test_no_leakage.py`. Los índices están en `results/real/splits.json`, no en `results/splits.json` |
| 2.1–2.3 Ruta A, costo | Hecho | 0.032 s por evaluación, 7 procesos de 8 núcleos, 0.87 min por corrida de GA, estudio estimado en 1.0 h, recortes: ninguno |
| 2.4 Ruta B, Gower y K-Medoids | Hecho, acotado | 180 filas de train, k = 3, backend `kmedoids.fasterpam`. Su Silhouette no se compara con la ruta A. `results/real/route_b.csv` |
| 3.1–3.4 K-Means, predict, baselines, permutación | Hecho | Davies–Bouldin, Calinski–Harabasz, NMI y chi-cuadrado se calculan en `evaluation.py` y no se escriben en `comparison.csv` |
| 4.1–4.4 Cromosoma, operadores, V1/V2 | Hecho | La selección de V2 en código resta además `lambda · (activas / 20)`. En esta corrida todos los genéticos tienen 4 activas, así que no cambió al ganador |
| 4.5 Señal plantada | Pipeline guardado, criterio de mayoría no recontado | `results/synthetic/` usa presupuesto reducido y `schema.used.yaml` (8 variables). El V1 elegido pone 0.30 en `signal_x`, `signal_y` y `signal_flag`. No hay checkpoints de esa corrida para recontar las semillas |
| 4.6 Misma semilla, mismo historial | Cubierto por pruebas | `tests/test_search.py` |
| 5.1–5.4 Estudio, Wilcoxon, prueba una vez | Hecho | `comparison.csv` (334 filas), `comparison.md`, `run_manifest.json` con `test_used_for_selection: false` |
| 5.5 Figuras | Hecho, con alcance | `results/real/figures/`: Pareto, pesos del V1 elegido, convergencia de V1 en **k = 3** (el `force_k`), no en el k elegido |
| 5B.2 CMA-ES | Hecho como comparador de train | Mismo presupuesto de evaluaciones, k = 5, una semilla. No se usó para elegir. `cma_es.json` |
| 5B.1 NSGA-II | Hecho a medias | Frente guardado, k = 3, población 20, 15 generaciones, aptitud de train. No hay una solución del frente evaluada en validación |
| 5.1c filtro no evolutivo, 5B.3 MLP, 6.4 sensibilidad, 9.2 texto | No hechos | El plan los marca opcionales y los pone primero en la lista de recorte |
| 6.1 Estabilidad | Hecho para V1 y k = 5 | Spearman de pesos 0.595, Jaccard de máscaras 0.548, ARI de asignaciones 0.624, 45 parejas |
| 6.2 Ablaciones | Parcial | Quitar el bloque financiero no cambia nada: esas columnas ya tenían peso 0. Educación nominal y `ville_id` nominal se compararon con pesos iguales, sin segunda búsqueda. Al partir por aldea, los niveles de `ville_id` de validación no están en train |
| 6.3 Perfiles | Hecho | `cluster_profiles.csv` es el ajuste de train del V1 elegido |
| 6.5 Subgrupos | Parcial | Sexo y estado civil completos. Edad se corta a 12 niveles (`max_levels` en `stability.py`), en orden de aparición, no en todos los años |
| 7 FAISS | Hecho | 286 consultas de validación: 0 coincidencias exactas de vecinos, 0 desacuerdos, 286 empates de frontera en float32. Acuerdo por fila 1.0 porque el empate cuenta como acuerdo. NumPy fue más rápido (media 1.8e-5 s frente a 2.2e-5 s). Centroide y KNN coinciden (ARI 1.0) |
| 8.1 Logística y árbol | Hecho | Sobre el espacio ponderado de V1. No son comparables con Silhouette. AUC cerca de 0.48 |
| 9.1 Artefactos | Hecho | Configuración usada, entorno, pesos, historial, índice, `model.joblib`, mapa de filas |
| 9.3 README | Hecho | Este archivo es la lectura que el README apunta |
| 9.4 Otra persona corre desde cero | Pendiente | Los checkpoints permiten saltarse la búsqueda. Para verificar, hace falta otra máquina o borrar checkpoints a propósito |

## 5. Resultados, en lectura

### Lo que se optimizó y lo que salió

La búsqueda mejora una geometría. La geometría ganadora es un catálogo de formas de ingreso y trabajo que ya está en la tabla. No es un indicador de depresión.

En las 1 429 filas, estas cuatro columnas binarias solo forman cinco combinaciones. No aparece, por ejemplo, salario junto con negocio, ni finca propia junto con salario.

| Salario | Finca propia | Negocio | Trabajo primario | Hogares | Positivos | Tasa |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 610 | 106 | 17.4 % |
| 0 | 1 | 0 | 0 | 360 | 63 | 17.5 % |
| 1 | 0 | 0 | 1 | 257 | 42 | 16.3 % |
| 0 | 0 | 1 | 0 | 154 | 21 | 13.6 % |
| 0 | 0 | 0 | 1 | 48 | 6 | 12.5 % |

La tasa global es 16.7 %. El rango entre patrones es 12.5–17.5 %.

Con k = 5, K-Means pone un cluster en cada combinación. Dentro del cluster todos los puntos ponderados son iguales, la distancia interna es 0 y el Silhouette euclidiano es exactamente 1. Eso pasa en train, en validación y en prueba. El Silhouette medio de validación del genético sube con k (V1: 0.66, 0.84, 0.92, 0.96) porque cada k adicional separa otra celda de ese catálogo. Pesos iguales, usando las 20 variables, se quedan entre 0.13 y 0.17.

### Quién ganó

Las dos variantes eligieron **k = 5** por validación. Las dos usan las mismas cuatro variables. Los pesos no son los mismos.

V1, semilla 50. Silhouette de validación 1, ARI de validación 0.00026, `N_eff` 3.68, 920 evaluaciones.

| Variable | Antes del tope | Después |
|---|---:|---:|
| `incoming_salary` | 0.298 | 0.300 |
| `incoming_business` | 0.306 | 0.300 |
| `labor_primary` | 0.273 | 0.276 |
| `incoming_own_farm` | 0.123 | 0.124 |

V2, semilla 43. Mismas métricas de validación en Silhouette y ARI, `N_eff` 3.67, 840 evaluaciones.

| Variable | Antes del tope | Después |
|---|---:|---:|
| `labor_primary` | 0.331 | 0.300 |
| `incoming_own_farm` | 0.279 | 0.291 |
| `incoming_business` | 0.275 | 0.288 |
| `incoming_salary` | 0.116 | 0.121 |

En la prueba, leída una vez, las dos variantes dan Silhouette **1.0** y ARI **−0.0167**. El intervalo bootstrap por aldea, 400 réplicas, va de **−0.0436 a 0.0095** e incluye el cero. La partición de prueba es la misma en V1 y en V2; cambian los índices de cluster, no los grupos.

La accuracy balanceada es 0.5 y el F1 macro es 0.454 en el genético, en pesos iguales y en la búsqueda aleatoria. Ningún cluster tiene mayoría deprimida, así que la etiqueta mayoritaria de train predice siempre «no». El KNN de FAISS y el centroide coinciden porque cada cluster es un solo punto.

Tres semillas de V1 en k = 5 (47, 49 y 50) tienen Silhouette de validación indistinguible de 1 y el mismo ARI 0.00026. La 47 usa las mismas cuatro variables con pesos más parejos (`N_eff` 3.92) y un Silhouette de 0.99999999988. El criterio escrito, a igualdad de Silhouette, se queda con el mayor `N_eff` y habría elegido la 47. El código compara el flotante antes del desempate, así que gana la 50. Para el reporte el hallazgo estable es el soporte de cuatro variables y la tabla de cinco patrones. El vector de la semilla 50 es un ejemplo, no el peso identificado.

En el k = 3 que se reporta siempre, el tope fabrica una cuarta variable. En V1 los pesos finales son `sex` 0.10, `incoming_salary` 0.30, `incoming_business` 0.30 y `labor_primary` 0.30. Antes del tope, `sex` era esencialmente cero y `incoming_business` se llevaba 0.81. Silhouette de validación 0.84, ARI −0.012. En V2 el cuarto peso también cae en `sex` y entra `incoming_own_farm` en lugar de `incoming_business`. Silhouette 0.81, ARI 0.029.

### Comparación que sí se puede afirmar

Wilcoxon bilateral, p < 0.05, validación, genético contra cada baseline. Fuente: `results/real/comparison.md`.

El genético **mejora el Silhouette** frente a pesos iguales y frente a la búsqueda aleatoria, en V1 y en V2, en los cuatro k. Las diferencias medias contra pesos iguales van de 0.49 (k = 2) a 0.84 (k = 5). Contra la búsqueda aleatoria van de 0.06 a 0.27. El p de esos contrastes es 0.002, salvo dos de k = 2 que quedan en 0.014 y 0.004.

El ARI no acompaña esa mejora.

| Contraste de ARI contra pesos iguales | Resultado |
|---|---|
| V1, k = 2 | Empata (p = 0.65) |
| V1, k = 3 | Empeora, diferencia media −0.017, p = 0.002 |
| V1, k = 4 | Mejora, diferencia media +0.007, p = 0.043 |
| V1, k = 5 | Empeora, diferencia media −0.015, p = 0.010 |
| V2, k = 2, 3 y 4 | Empata. En k = 3 y k = 4 el p queda en 0.078 y 0.082 |
| V2, k = 5 | Empeora, diferencia media −0.013, p = 0.014 |

Contra la búsqueda aleatoria, el ARI empata en todos los k.

`comparison.md` dice que con cinco parejas el p mínimo, si todas las diferencias tienen el mismo signo, es 0.0625. Esa frase es fija en `reporting.py`. El contraste publicado empareja 10 semillas del genético. Con 10 parejas del mismo signo el p bilateral llega a 0.002, y eso es lo que ocurrió en el Silhouette. El aviso sí aplica a los empates de ARI cuyo p quedó entre 0.06 y 0.08: ahí una diferencia pequeña se declara empate. No aplica como explicación de que el Silhouette «no alcanzó significancia».

La referencia por permutación, 100 repeticiones, deja un ARI nulo de media −0.0006. El ARI observado en validación (0.00026) cabe en ese ruido. El Silhouette de particiones aleatorias en el mismo espacio tiene media −0.05. El Silhouette 1 está lejos de ese nulo y cerca de lo que produce enumerar cinco celdas.

Entre las cinco particiones de robustez y cinco semillas, sin reoptimizar la malla, el Silhouette de validación en k = 5 sigue alto (media 0.98 en V1 y 0.96 en V2) y el ARI sigue junto a cero (medias −0.009 y −0.007).

CMA-ES, en aptitud de train y k = 5, llega a puntuación 0.92 en V1 y 0.48 en V2, con `N_eff` 3.57. El frente de NSGA-II, en train y k = 3, tiene Silhouette entre 0.40 y 0.73 y ARI de train hasta 0.085. Ninguno sustituye al individuo elegido en validación.

La logística y el árbol, en el espacio ponderado de V1, quedan en accuracy balanceada 0.5 y AUC 0.48 y 0.48. El espacio que maximiza el Silhouette no ordena la etiqueta.

### Respuestas a las siete preguntas de la entrega del plan

1. **¿El genético mejora el Silhouette frente a pesos iguales y a la búsqueda aleatoria, con el mismo presupuesto?** Sí, en validación, en los cuatro k y en las dos variantes.
2. **¿Se mantiene entre semillas y en prueba?** El Silhouette de la configuración elegida es 1.0 también en prueba, y la robustez lo mantiene alto. Es la misma geometría de cinco celdas. El ARI no se mantiene como mejora porque no la hubo.
3. **¿Cuántas variables bastan?** Cuatro de veinte. `N_eff` de los elegidos: 3.67 y 3.68. Son cuatro indicadores binarios de ingreso y trabajo, no un subconjunto continuo de gastos.
4. **¿Hay asociación con `depressed`, y qué aporta V2?** El ARI de prueba es −0.017 y el intervalo por aldea cruza el cero. V2 eligió el mismo soporte y, en prueba, la misma partición que V1.
5. **¿Qué accuracy y qué acuerdo tiene el KNN?** Accuracy balanceada 0.5 y F1 macro 0.45 por mapeo de mayoría. El KNN y el centroide coinciden (ARI 1.0).
6. **¿Cuánto cuesta?** Una evaluación, 0.032 s. Una corrida de GA, 0.87 min. El estudio completo se estimó en 1.0 h y cerró dentro de las 6 h. La inferencia por centroide está en microsegundos. FAISS, con este tamaño, es más lento que NumPy.
7. **¿Qué cambiar para otro dataset?** El esquema y el experimento viven en YAML. Cambiar el CSV obliga a repetir la auditoría y la confirmación humana. Con este CSV, repetir 40 × 40 no va a acercar el ARI a la etiqueta: k = 5 sobre estas cuatro columnas ya está resuelto por enumeración.

## 6. Qué decir y qué no decir en el IEEE

Se puede escribir:

- El objetivo es ponderar variables para un agrupamiento, con dos aptitudes declaradas antes de correr.
- Train busca, validación elige, prueba se lee una vez. La partición es por aldea.
- El genético mejora el Silhouette de validación frente a los dos baselines.
- El modelo elegido describe cinco patrones de ingreso y trabajo. La prevalencia de `depressed` dentro de ellos permanece cerca de la tasa global.
- El tope `w_max` evitó concentrar el peso en una sola variable. No evitó que cuatro binarias con cinco celdas ocupadas llevaran el Silhouette a 1. El plan ya advertía ese colapso.

No se sostiene:

- Que el método detecte o prediga depresión.
- Que Silhouette 1.0 sea un ajuste del indicador clínico.
- Que la cuarta variable sea `farm_expenses`. Es `labor_primary`.
- Que haya un solo vector de pesos, o que el tope casi no haya actuado en las dos variantes. En V2, `labor_primary` baja de 0.331 a 0.300.
- Que el estudio no tuvo potencia para el Silhouette. El p fue 0.002.
- Que la ablación financiera haya puesto a prueba al modelo elegido. Esas columnas ya pesaban cero.
- Que CMA-ES o NSGA-II hayan sido el modelo reportado.

Amenazas que el plan y el README ya reconocen, y que esta corrida confirma:

- La construcción de `depressed` no está documentada. ARI y AUC no son validación clínica.
- Maximizar Silhouette puede describir una partición geométrica sin relación con la etiqueta. Aquí ocurrió.
- Un cluster puede reflejar una combinación de ingreso. No se concluye causalidad.
- Elegir en el mismo conjunto que se reporta inflaría el resultado. Eso se evitó para k y para los pesos. El desempate flotante entre semillas 47 y 50 no usa la prueba, pero sí muestra que el vector no es único.

## 7. Qué sigue

En este orden.

1. **Redactar el reporte desde las secciones 5 y 6.** Las cifras ya están en el clon. No hace falta volver a correr el estudio para citarlas.
2. **Cerrar la ficha de datos en el texto:** mapa 0/1 de la ficha Kaggle, instrumento desconocido, 1 429 filas frente a las 1 432 anunciadas, licencia por confirmar contra el GPL-2 de esa ficha.
3. **Tratar los pesos como ejemplos del soporte.** Si se publica un vector, decir que la semilla 47 queda a 10⁻¹⁰ de Silhouette y tiene mayor `N_eff`.
4. **Verificación por otra persona (paso 9.4).** `pytest` en un clon limpio, y una lectura de `results/real/chosen.json` contra este documento. Reejecutar las 1 h solo si se borran los checkpoints a propósito.
5. **No repetir la malla 40 × 40 sobre este CSV** si la pregunta sigue siendo el Silhouette de estas variables. El catálogo de cinco patrones ya es el óptimo de k = 5.
6. **Si el curso pide otra corrida de método**, cambiar la aptitud antes de gastar cómputo: penalizar soluciones cuyo espacio ponderado tenga tantas celdas distintas como k, y volver a elegir en validación. Eso sí es otra pregunta.
7. **Huecos opcionales, solo si el reporte los exige:** volcar Davies–Bouldin, Calinski–Harabasz, NMI y chi-cuadrado a la tabla; evaluar en validación una solución declarada del frente NSGA-II; quitar el corte de 12 niveles en la edad. No son el siguiente experimento principal.
8. **FastAPI, texto y embeddings** siguen fuera de este avance, como dice el plan (D13, D14, 9.2).

## 8. Mapa corto de `results/real/`

| Archivo | Contenido |
|---|---|
| `selection_criterion.txt` | Criterio escrito antes de la prueba |
| `chosen.json` | V1, V2 y el k = 3 de reporte, con pesos antes y después del tope |
| `comparison.csv` | Validación de todas las corridas y las seis filas de prueba |
| `comparison.md` | Medias y Wilcoxon |
| `splits.json` | Índices y conteo de positivos |
| `bootstrap_test.json` | Intervalo por aldea en la prueba |
| `stability.json` | Spearman, Jaccard y ARI de asignaciones, V1, k = 5 |
| `robustness.csv` | 5 particiones × 5 semillas, k = 3 y k = 5, solo validación |
| `cluster_profiles.csv` | Perfil de train del V1 elegido |
| `cost_estimate.csv` | Sonda de tiempo y presupuesto |
| `faiss_report.json` | Vecinos, empates y latencia |
| `cma_es.json`, `nsga2.json` | Comparadores. No eligen el modelo |
| `representation_ablation.csv` | Educación nominal y aldea nominal, pesos iguales |
| `ablation.csv` | Bloque financiero quitado sobre los pesos ya elegidos |
| `supervised_reference.json` | Logística y árbol |
| `checkpoints/` | Una corrida por archivo. La segunda ejecución se la salta |
| `artifacts/` | Transformador, pesos, centroides, índice FAISS, filas de train |
| `figures/` | Convergencia V1 en k = 3, Pareto, barras del V1 elegido |
