# Preguntas pendientes del esquema

Respuestas todavía vacías. El borrador está en `config/schema.draft.yaml` y el informe en `results/audit/` (esa carpeta no va en git; se regenera con `python -m plena.audit --config config/audit_real.yaml`).

Hasta llenar esto, `depressed` no entra al agrupamiento y el CSV real no se usa para entrenar.

## 1. Etiqueta `depressed`

Valores observados: 0 en 1191 filas, 1 en 238 filas. Sin faltantes.

Qué significa 0:

Qué significa 1:

Cómo se obtuvo la etiqueta:

## 2. Tipo de cada columna

Propuesta automática, sin confirmar. Marca el tipo final: continua, discreta, binaria, nominal, ordinal (con el orden) o identificador.

| Columna | Propuesta | Tipo confirmado | Notas |
|---|---|---|---|
| survey_id | identificador, único | | |
| ville_id | 255 códigos | | |
| sex | binaria 0/1 | | |
| age | continua, 17–91 | | |
| married | binaria 0/1 | | |
| number_children | discreta 0–11 | | |
| education_level | enteros 1–19, falta el 15 | | |
| total_members | discreta 1–12 | | |
| gained_asset | continua, escala grande | | |
| durable_asset | continua, escala grande | | |
| save_asset | continua, escala grande | | |
| living_expenses | continua, escala grande | | |
| other_expenses | continua, escala grande | | |
| incoming_salary | binaria 0/1 | | |
| incoming_own_farm | binaria 0/1 | | |
| incoming_business | binaria 0/1 | | |
| incoming_no_business | binaria 0/1 | | |
| incoming_agricultural | continua, escala grande | | |
| farm_expenses | continua, escala grande | | |
| labor_primary | binaria 0/1 | | |
| lasting_investment | continua, escala grande | | |
| no_lasting_investmen | continua, 20 faltantes, nombre truncado | | |

Orden, si alguna es ordinal:

## 3. Columnas que hay que excluir por venir de la etiqueta

Ningún nombre sugiere un instrumento de depresión. El AUC univariado más alto ronda 0.57.

Columnas a excluir, y por qué:

## 4. Códigos 99 y 999

`survey_id` tiene un 99 y un 999. `ville_id` tiene cuatro filas con 99. No aparecen -1 ni -9. `no_lasting_investmen` sí tiene 20 celdas vacías.

¿99 y 999 son faltantes o códigos reales?

## 5. `ville_id`

255 niveles. Como nominal, el one-hot añade 255 columnas.

Decisión: excluir / usar solo en ablación / usar como nominal / otra:

## 6. Repeticiones por persona u hogar

`survey_id` no se repite. No hay otra columna de persona.

¿Hay que particionar por grupo? Si sí, ¿con qué columna?

## 7. Presupuesto

Referencia actual: 10 semillas del genético, 30 búsquedas aleatorias, población 40, 40 generaciones.

Semillas:

Tiempo máximo por corrida:

## 8. NSGA-II

¿Activarlo si el estudio principal ya está cerrado? Sí / no:

## 9. Dataset

¿`b_depressed.csv` es el definitivo? Sí / no. Si va a cambiar, en qué:
