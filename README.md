# PLENA

Prototipo académico de cómputo evolutivo. Un algoritmo genético de dominio real elige y pondera variables para agrupar una encuesta de condiciones de vida en zonas rurales con un indicador binario de depresión. La lectura cerrada del avance, las cifras y lo que sigue están en `PLENA_Estado_Resultados.md`.

No diagnostica depresión. Los grupos se llaman `cluster_0`, `cluster_1`, … y no son niveles clínicos de riesgo. ARI y AUC no se interpretan como validación clínica: la construcción del indicador `depressed` no está documentada. En la descripción pública de OpenML (id 43419) 0 es no deprimido y 1 es deprimido.

Silhouette no es comparable entre valores distintos de `k`. Se recorre `k ∈ {2,3,4,5}` y k = 3 se reporta siempre. El criterio de elección queda escrito en `results/` antes de optimizar: V1 por Silhouette de validación, V2 por la mezcla de Silhouette y ARI en validación. El conjunto de prueba no participa en esa elección.

## Reproducir

Python 3.10 o superior. En Windows, desde el repositorio clonado:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,faiss,extra]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_experiments.py --schema config/schema.yaml --experiment config/experiment.yaml --output results/real
```

No se usa GPU. K-Means, Silhouette y `faiss-cpu` corren en CPU. `requirements.lock` guarda el `pip freeze` de la máquina en la que se cerró este avance. Si FAISS o kmedoids fallan en otra máquina, el código usa fuerza bruta NumPy e implementación alternante de k-medoids y lo anota en el informe.

Para repartir el trabajo: `--k`, `--variant v1|v2` y `--seeds A:B` (índices semiabiertos de la lista de semillas). Cada corrida terminada queda en `results/real/checkpoints/` y una segunda ejecución se la salta. `scripts/merge_results.py carpeta_a carpeta_b --output results/real` une checkpoints y tablas sin duplicar la clave método, variante, k, semilla y partición.

## Datos y configuración

`data/b_depressed.csv` viaja en el repositorio para que el equipo pueda clonar y trabajar. Procedencia y límites: `data/README.md`. El esquema confirmado es `config/schema.yaml`. `config/schema.draft.yaml` es un borrador automático y no se usa para entrenar. `config/schema.synthetic.yaml` describe el generador de juguete.

| Archivo | Papel |
|---|---|
| `config/schema.yaml` | Esquema confirmado del CSV real |
| `config/experiment.yaml` | Presupuesto del estudio real |
| `config/experiment.synthetic.yaml` | Presupuesto reducido del sintético |
| `results/` | Auditoría, estudio real y sintético, ya versionados |

El sintético, con señal débil y al menos 20 variables:

```powershell
.\.venv\Scripts\python.exe -m plena.synthetic
.\.venv\Scripts\python.exe scripts\run_experiments.py --schema config/schema.synthetic.yaml --experiment config/experiment.synthetic.yaml --output results/synthetic --planted-only
```

Las cifras y su lectura están en `results/` y en `PLENA_Estado_Resultados.md`.

## Método, en corto

El cromosoma es real, en `[-1, 1]`, con una posición por variable original. `w_max` es configurable. Los pesos se reportan antes y después del tope. El peso entra como `sqrt(w_j / n_cols_j)` en el bloque de la variable, de modo que una nominal de muchos niveles no domina.

La ruta principal imputa con mediana o moda ajustada solo en train. Las columnas de escala grande usan `log1p` y después `StandardScaler`. `education_level` es ordinal con el orden natural 1–19; el 15 ausente no se rellena. `ville_id` no entra en X: parte los datos para que ninguna aldea caiga en dos particiones. Hay una comparación aparte con `education_level` nominal y otra con `ville_id` nominal.

V1 no usa la etiqueta en la aptitud. V2 mezcla Silhouette y ARI, y el ARI usa `depressed` solo en train. Los baselines son pesos iguales y búsqueda aleatoria con el mismo número de evaluaciones. El contraste publicado usa 10 semillas del genético. `comparison.md` trae además un aviso de plantilla sobre cinco parejas; la lectura de ese aviso está en `PLENA_Estado_Resultados.md`.

La robustez, ya elegida la configuración, repite solo ese k y k = 3 en otras particiones y semillas. No reoptimiza toda la malla. CMA-ES es el comparador siguiente. NSGA-II entra solo si el estudio principal ya cerró y queda tiempo.

## Amenazas a la validez

- La etiqueta es un indicador binario de la fuente. Su construcción no está confirmada, así que no se puede descartar del todo una fuga no documentada. Ningún nombre de columna parece un instrumento de depresión.
- Maximizar Silhouette puede describir una partición geométrica sin relación con la etiqueta.
- Elegir el modelo con el mismo conjunto que se reporta infla el resultado. Train busca, validación elige, prueba se lee una vez.
- Un cluster puede reflejar aldea o demografía. La ablación sirve para mirarlo; no se concluye causalidad.
- Con este tamaño, FAISS puede no ser más rápido que la fuerza bruta. El informe de vecinos separa empates de desacuerdos.
