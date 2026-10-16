# Preguntas pendientes del esquema

Respuestas incorporadas. El esquema confirmado está en `config/schema.yaml`. El borrador automático sigue en `config/schema.draft.yaml`.

## 1. Etiqueta `depressed`

Valores observados: 0 en 1191 filas, 1 en 238 filas. Sin faltantes.

Qué significa 0: no deprimido, según la descripción pública del dataset en OpenML id 43419 (republicado desde Kaggle/Frankcc).

Qué significa 1: deprimido, según esa misma descripción.

Cómo se obtuvo la etiqueta: no está documentado. Queda registrado como indicador binario de la fuente, construcción no confirmada. ARI y AUC no se interpretan como validación clínica.

## 2. Tipo de cada columna

Propuesta automática, sin confirmar. Marca el tipo final: continua, discreta, binaria, nominal, ordinal (con el orden) o identificador.

| Columna | Propuesta | Tipo confirmado | Notas |
|---|---|---|---|
| survey_id | identificador, único | identificador, excluido de X | |
| ville_id | 255 códigos | identificador de aldea, excluido de X, group_column | ablación aparte como nominal |
| sex | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| age | continua, 17–91 | continua | |
| married | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| number_children | discreta 0–11 | discreta numérica | no nominal |
| education_level | enteros 1–19, falta el 15 | ordinal, orden natural 1–19 | el 15 ausente no se rellena; ablación aparte como nominal |
| total_members | discreta 1–12 | discreta numérica | no nominal |
| gained_asset | continua, escala grande | continua | log1p y luego StandardScaler |
| durable_asset | continua, escala grande | continua | log1p y luego StandardScaler |
| save_asset | continua, escala grande | continua | log1p y luego StandardScaler |
| living_expenses | continua, escala grande | continua | log1p y luego StandardScaler |
| other_expenses | continua, escala grande | continua | log1p y luego StandardScaler |
| incoming_salary | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| incoming_own_farm | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| incoming_business | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| incoming_no_business | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| incoming_agricultural | continua, escala grande | continua | log1p y luego StandardScaler |
| farm_expenses | continua, escala grande | continua | log1p y luego StandardScaler |
| labor_primary | binaria 0/1 | binaria | solo aparecen 0 y 1 |
| lasting_investment | continua, escala grande | continua | log1p y luego StandardScaler |
| no_lasting_investmen | continua, 20 faltantes, nombre truncado | continua | log1p y luego StandardScaler; 20 faltantes imputados con la mediana de train |

Orden, si alguna es ordinal: solo `education_level`, enteros 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19.

## 3. Columnas que hay que excluir por venir de la etiqueta

Ningún nombre sugiere un instrumento de depresión. El AUC univariado más alto ronda 0.57.

Columnas a excluir, y por qué: ninguna confirmada. No hay documentación de origen para descartarlo del todo; queda como limitación.

## 4. Códigos 99 y 999

`survey_id` tiene un 99 y un 999. `ville_id` tiene cuatro filas con 99. No aparecen -1 ni -9. `no_lasting_investmen` sí tiene 20 celdas vacías.

¿99 y 999 son faltantes o códigos reales? Son identificadores o códigos reales, no faltantes. El detector de centinelas ignora columnas de identificador y de grupo. Los 20 vacíos de `no_lasting_investmen` sí son faltantes y se imputan con la mediana de train.

## 5. `ville_id`

255 niveles. Como nominal, el one-hot añade 255 columnas.

Decisión: excluir / usar solo en ablación / usar como nominal / otra: fuera de X y usada como `group_column`. Ablación aparte como nominal de 255 niveles, solo para comparar.

## 6. Repeticiones por persona u hogar

`survey_id` no se repite. No hay otra columna de persona.

¿Hay que particionar por grupo? Si sí, ¿con qué columna? Sí, por `ville_id`. Ningún grupo en más de una partición. No hay otra columna de persona. Estudio principal: una partición train/val/test con semilla fija e intervalos bootstrap en prueba. Robustez, solo para la configuración elegida y sin reoptimizar el estudio completo: al menos 5 particiones por 5 semillas de GA, en el k elegido y en k=3.

## 7. Presupuesto

Referencia actual: 10 semillas del genético, 30 búsquedas aleatorias, población 40, 40 generaciones.

Semillas: 10 para el GA y 30 búsquedas aleatorias. Población 40, generaciones 40. Laptop sin GPU; K-Means, Silhouette y faiss-cpu en CPU.

Tiempo máximo por corrida: el estudio completo apunta a 6 horas como máximo. Una corrida de GA apunta a 10 minutos como máximo. El tiempo medido queda en `results/cost_estimate.csv`.

## 8. NSGA-II

¿Activarlo si el estudio principal ya está cerrado? Sí / no: sí, condicionado. Primero CMA-ES como comparador. NSGA-II solo si el estudio principal ya está cerrado y queda tiempo.

## 9. Dataset

¿`b_depressed.csv` es el definitivo? Sí / no. Si va a cambiar, en qué: sí, para este avance. Es una encuesta de condiciones de vida en zonas rurales con un indicador binario de depresión, semántica no documentada.
