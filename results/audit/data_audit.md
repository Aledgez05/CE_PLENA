# Auditoría de datos

Informe automático. No confirma el significado de las columnas.

## Estructura

- Filas: 1429
- Columnas: 23
- Memoria: 263068 bytes
- Filas duplicadas: 0
- Candidatas a identificador: [{'column': 'survey_id', 'unique_ratio': 1.0}]

## Calidad

- Faltantes por columna: {'survey_id': 0, 'ville_id': 0, 'sex': 0, 'age': 0, 'married': 0, 'number_children': 0, 'education_level': 0, 'total_members': 0, 'gained_asset': 0, 'durable_asset': 0, 'save_asset': 0, 'living_expenses': 0, 'other_expenses': 0, 'incoming_salary': 0, 'incoming_own_farm': 0, 'incoming_business': 0, 'incoming_no_business': 0, 'incoming_agricultural': 0, 'farm_expenses': 0, 'labor_primary': 0, 'lasting_investment': 0, 'no_lasting_investmen': 20, 'depressed': 0}
- Filas con algún faltante: 20
- Códigos centinela (solo reportados): {'survey_id': 2, 'ville_id': 4}
- Constantes: []
- Casi constantes: []
- Casi duplicadas: []
- Patrones de faltantes: []

## Etiqueta

- Columna: `depressed`
- Valores: ['0', '1']
- Conteos: {'0': 1191, '1': 238}
- Faltantes: 0
- Semántica: UNCONFIRMED

## Fuga

- Alertas AUC: []
- Sospechosas por nombre: []
- Misma encuesta: UNCONFIRMED: no hay metadatos que indiquen qué variables salen de la misma encuesta que la etiqueta.

## Grupos y geografía

- Claves candidatas: [{'column': 'survey_id', 'unique': 1429, 'duplicated_values': 0, 'unique_ratio': 1.0}, {'column': 'ville_id', 'unique': 255, 'duplicated_values': 1174, 'unique_ratio': 0.17844646606018194}, {'column': 'lasting_investment', 'unique': 982, 'duplicated_values': 447, 'unique_ratio': 0.6871938418474458}, {'column': 'no_lasting_investmen', 'unique': 939, 'duplicated_values': 489, 'unique_ratio': 0.6571028691392582}]
- Geografía: ['ville_id']

## Viabilidad

- D estimada tras one-hot (solo nominales ya inferidas): 22
- Códigos de cardinalidad alta, si se one-hot: [{'column': 'ville_id', 'nunique': 255, 'one_hot_columns_if_nominal': 255, 'note': 'UNCONFIRMED: si se trata como nominal, el one-hot añade estas columnas.'}, {'column': 'age', 'nunique': 71, 'one_hot_columns_if_nominal': 71, 'note': 'UNCONFIRMED: si se trata como nominal, el one-hot añade estas columnas.'}, {'column': 'education_level', 'nunique': 18, 'one_hot_columns_if_nominal': 18, 'note': 'UNCONFIRMED: si se trata como nominal, el one-hot añade estas columnas.'}]
- Memoria de una matriz n×n: 16336328 bytes
- Segundos de una sonda de fitness: 0.1936015999999654
- Tamaño de submuestra sugerido: 1429

## Sesgo y suficiencia

- Columnas sensibles heurísticas: ['ville_id', 'sex', 'age', 'married']
- n/D: 62.130434782608695
- Clase minoritaria: 238

## Hallazgos y decisiones pendientes

- El significado de los códigos de la etiqueta está UNCONFIRMED.
- Los tipos del borrador son propuestas automáticas, no un diccionario confirmado.
- Hay códigos centinela sospechosos. No se convirtieron a faltantes.
- Hay columnas geográficas candidatas. Su uso queda UNCONFIRMED.
- ville_id tiene 255 valores. Tratarla como nominal dispararía el one-hot.
- age tiene 71 valores. Tratarla como nominal dispararía el one-hot.
- education_level tiene 18 valores. Tratarla como nominal dispararía el one-hot.
- No se confirma si hay observaciones repetidas por persona u hogar.

### Preguntas para confirmar el esquema

1. ¿Qué significan los valores de la etiqueta y cómo se obtuvo?
2. ¿Qué columnas son nominales, ordinales (con qué orden), binarias, continuas o identificadores?
3. ¿Hay columnas derivadas de la etiqueta o del mismo instrumento?
4. ¿Los códigos sospechosos son faltantes?
5. ¿La columna geográfica se usa, se excluye o se reserva para ablación?
6. ¿Hay repeticiones por persona u hogar que obliguen a particionar por grupo?
7. ¿Cuántas semillas y qué presupuesto de cómputo hay?
8. ¿Se activa NSGA-II si sobra tiempo?
9. ¿Este dataset es el definitivo?
