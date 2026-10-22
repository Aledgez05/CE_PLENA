# PLENA

FAISS agrupa en dos grupos. Un genético pondera las variables, una vez por Silhouette y otra por Rand. Se compara con tres baselines: clase mayoritaria, regresión logística y una red ligera. No diagnostica depresión.

La lectura de esta corrida está en `PLENA_Cuaderno_Resultados.md`. El estudio anterior, con otro protocolo, queda descrito en `PLENA_Estado_Resultados.md` y sus archivos siguen en `results/real/`.

## Correr

Desde la raíz, con el entorno del repositorio. PyTorch se instala aparte porque la rueda de CPU no sale del índice normal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run.py
```

El recorrido es `data.py` (carga y partición), `cluster.py` (FAISS y genético), `baselines.py` (logística, red y clase mayoritaria) y `report.py` (tablas y figuras). `scripts/run.py` llama a `study.py`. La prueba no entra en la búsqueda.

## Resultados

`results/cuaderno/` guarda la tabla de prueba, las matrices de confusión y cuatro figuras: comparación, convergencia, pesos y matrices. Las cifras de esta máquina están en `PLENA_Cuaderno_Resultados.md`.

`data/b_depressed.csv` viaja en el repositorio. Procedencia y límites: `data/README.md`.
