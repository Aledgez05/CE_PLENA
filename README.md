# PLENA

Prototipo académico de **cómputo evolutivo**. Un algoritmo genético de dominio real elige y pondera variables para agrupar datos tabulares asociados a depresión en población general.

No diagnostica depresión. Los grupos se llaman `cluster_0`, `cluster_1`, … y no son niveles clínicos de riesgo. El pivote desde depresión posparto, motivo de la primera etapa del método, queda como antecedente de desarrollo: este avance estudia depresión en población general.

Silhouette no es comparable entre valores distintos de `k`. Por eso se recorre `k ∈ {2,3,4,5}` y **k = 3 se reporta siempre**, sin fijarlo como único resultado. La propuesta inicial hablaba de riesgo bajo, medio y alto; esa lectura clínica no se usa.

## Entorno

Python 3.10 o superior. En Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,faiss,extra]"
.\.venv\Scripts\python.exe -m pytest
```

`requirements.lock` guarda el `pip freeze` de la máquina en la que se cerró este avance. FAISS, pymoo, cma y kmedoids tienen rueda para este entorno. Si FAISS o kmedoids fallan en otra máquina, el código usa fuerza bruta NumPy e implementación alternante de k-medoids y lo anota en el informe. FAISS no se usa para acelerar la distancia de Gower.

## Datos

Los CSV originales no deben subirse. Están en `.gitignore`. El esquema confirmado vive en YAML: cambiar de dataset es editar `config/schema.yaml`, no el código.

| Archivo | Papel |
|---|---|
| `config/schema.yaml` | Esquema sintético confirmado, sirve para reconstruir el experimento de prueba |
| `config/schema.synthetic.yaml` | El mismo generador, reescrito por `python -m plena.synthetic` |
| `config/schema.draft.yaml` | Borrador del CSV real. Tiene `UNCONFIRMED` y **no se usa para entrenar** |
| `config/experiment.yaml` | Presupuesto del avance (10 semillas, población 40, 40 generaciones) |
| `config/experiment.synthetic.yaml` | Presupuesto reducido para el sintético |

El CSV real se audita así:

```powershell
.\.venv\Scripts\python.exe -m plena.audit --config config/audit_real.yaml
```

El experimento sintético:

```powershell
.\.venv\Scripts\python.exe -m plena.synthetic
.\.venv\Scripts\python.exe scripts\run_experiments.py --schema config/schema.yaml --experiment config/experiment.synthetic.yaml --output results/synthetic
```

Cuando el esquema real quede sin ningún `UNCONFIRMED`, el mismo comando con `config/schema.yaml` apuntando al CSV real y `config/experiment.yaml` corre el estudio. El conjunto de prueba entra una sola vez, después de elegir `k`, individuo y variante en validación.

## Decisiones de diseño

El cromosoma es real, en `[-1, 1]`, con una posición por variable original. Si el gen no supera `tau` (0 por defecto), la variable sale. Si lo supera, el peso es el gen. Los pesos activos suman 1, se recortan a `w_max` y se renormalizan. Hay un mínimo de variables activas. El número efectivo `N_eff = 1 / Σ w²` penaliza la concentración: sin ese tope, Silhouette empuja el peso hacia una o dos variables.

El peso entra como `sqrt(w_j)` en el bloque de columnas de la variable. En one-hot cada columna del bloque se escala por `sqrt(w_j / n_cols_j)`, de modo que `d² = Σ w_j (x_j − y_j)²` y una nominal de muchos niveles no domina por tener más columnas.

La ruta principal es densa: imputación con mediana o moda ajustada solo en train, `RobustScaler` o `StandardScaler`, one-hot con `handle_unknown="ignore"`. K-Means es el agrupamiento principal. La ruta de Gower con k-medoids es una comparación de representación, no el método que se indexa.

Hay dos variantes, siempre etiquetadas:

- **V1**, no supervisada: Silhouette en el espacio ponderado, menos una penalización por el número de variables activas y por clusters degenerados o `N_eff` bajo. RI y ARI se calculan después, no dentro de la aptitud.
- **V2**, semisupervisada: mezcla Silhouette y ARI. El ARI usa `depressed` **solo en train**.

`depressed` no entra en la matriz de características. La “precisión” del agrupamiento es un mapeo: cada cluster recibe la etiqueta mayoritaria de train y se mide balanced accuracy y F1 en validación y en prueba. Otra inferencia es KNN sobre los vecinos de train, con FAISS `IndexFlatL2` en los vectores ponderados **sin** normalizarlos a norma unitaria. Ese índice devuelve distancia al cuadrado.

Los baselines son pesos iguales, búsqueda aleatoria con el mismo presupuesto de evaluaciones (`población × generaciones`) y una referencia por permutación. El genético no se detiene al superar un baseline.

## Qué se comprobó

En el sintético, tres variables generan los clusters y el resto es ruido, incluida una nominal de 25 niveles y una variable de escala enorme. El criterio, fijado para la prueba automática, es que en la mayoría de las semillas al menos el 70 % del peso caiga en las variables informativas. Esa prueba pasa: las cinco semillas de la configuración de `tests/test_audit_and_signal.py` dejan el peso en esas variables.

La corrida de `config/experiment.synthetic.yaml` (5 semillas, población 16, 8 generaciones, submuestra de 220 filas) está en `results/synthetic/`. No es el presupuesto de `experiment.yaml`.

En validación, el Silhouette medio de V1 queda muy por encima de los pesos iguales (por ejemplo, k = 3: 0.706 ± 0.074 frente a 0.235; k = 4: 0.836 ± 0.069 frente a 0.299). Con 5 parejas, el Wilcoxon bilateral no baja de p = 0.0625 aunque todas las semillas ganen, así que **la declaración formal es empate** frente a pesos iguales y frente a la búsqueda aleatoria. La diferencia media contra el azar de la búsqueda es pequeña (a veces a favor del GA, a veces en contra). No se afirma que el GA mejore al azar con el mismo presupuesto.

El individuo V1 elegido en validación usa k = 4, cuatro variables activas y `N_eff` ≈ 3.57. El 90 % del peso cae en `signal_x`, `signal_y` y `signal_flag`. En la prueba, leída una sola vez, su Silhouette es 0.886 y el ARI es 0.327. Los pesos iguales en esa misma prueba tienen Silhouette 0.318 y ARI 0.007. La búsqueda aleatoria elegida en validación queda en Silhouette 0.802 y ARI 0.337. El mapeo cluster→etiqueta da balanced accuracy 0.895 en este sintético porque la etiqueta se plantó a partir del cluster; no es una cifra clínica.

V2, en la prueba, queda en k = 2, Silhouette 0.636 y ARI 0.676. La búsqueda aleatoria de esa variante llega al mismo ARI. Otra vez: empate formal con el azar, y una asociación con la etiqueta mayor que la de V1 en este diseño, porque V2 optimiza ARI en train.

FAISS `IndexFlatL2` reprodujo los vecinos de la fuerza bruta en el 99.4 % de las consultas de validación. El acuerdo entre asignación por centroide y por KNN tiene ARI 1 en esa corrida. La latencia media por consulta fue del orden de 1e-5 s en ambos backends: con este tamaño, FAISS no aporta una ventaja práctica.

Entre semillas de V1 en el k elegido, Spearman medio de pesos 0.50, Jaccard medio de máscaras 0.64 y ARI medio entre asignaciones 0.47. Hay estabilidad parcial, no identidad.

k = 3 se reporta en la tabla aunque la validación de V1 haya preferido k = 4.

Los modelos supervisados quedan en `supervised_reference.json`. No se comparan con Silhouette. En este sintético la etiqueta se construyó desde el cluster generativo, así que una balanced accuracy alta no se traslada al CSV real.

## Amenazas a la validez

- El significado de 0 y 1 en `depressed` sigue sin confirmar en el CSV real. No se infiere por el nombre.
- Maximizar Silhouette puede describir una partición geométrica sin relación con la etiqueta. Por eso se reportan ARI, RI, AMI y la prevalencia por cluster, y existe la referencia por permutación.
- Elegir el modelo con el mismo conjunto que se reporta infla el resultado. Train busca, validación elige, prueba se lee una vez.
- Un cluster puede reflejar ciudad o demografía. Las ablaciones y la revisión por subgrupos existen para mirarlo; no se concluye causalidad.
- Con pocos cientos de filas, FAISS puede ser igual o más lento que la fuerza bruta. El informe de latencia no esconde ese caso.
- El presupuesto sintético es menor que `experiment.yaml`. No se extrapola esa corrida al CSV real.
- NSGA-II y CMA-ES están implementados como extensión y no forman parte del resultado principal.

## Verificación independiente

Otra persona puede clonar el repositorio, crear el entorno, ejecutar `pytest` y el comando sintético de arriba, y comparar `results/synthetic/comparison.csv` con una segunda corrida en la misma máquina: la semilla está en el YAML y el historial debe repetirse. Los CSV de personas no viajan en el repositorio.

## Qué habría que cambiar para el dataset definitivo

Confirmar `config/schema.draft.yaml` hasta que no quede `UNCONFIRMED`, guardarlo como `config/schema.yaml` y lanzar `config/experiment.yaml`. Si el costo de una evaluación supera el presupuesto, se baja `fitness_sample_size`, la población o las generaciones en ese YAML y se deja constancia en `results/cost_estimate.csv`. No hace falta tocar el código mientras las columnas sigan descritas en el esquema.
