# Comparación

El conjunto de prueba no se usó para elegir k, pesos, lambda, alpha ni ningún hiperparámetro. La configuración se congeló con métricas de validación y solo después se calcularon las métricas de prueba, una vez.

Un ARI cercano a 0 es un resultado legítimo: no se presenta como fallo ni como éxito.
Los clusters se nombran cluster_0, cluster_1, … y no son niveles clínicos.

## Media en validación

```
method,variant,k,silhouette,ari,n_active,n_eff,opt_seconds
equal_weights,v1,2,0.241,-0.005,8.0,8.0,0.0
equal_weights,v1,3,0.235,0.005,8.0,8.0,0.0
equal_weights,v1,4,0.299,-0.007,8.0,8.0,0.0
equal_weights,v1,5,0.229,-0.007,8.0,8.0,0.0
equal_weights,v2,2,0.241,-0.005,8.0,8.0,0.0
equal_weights,v2,3,0.235,0.005,8.0,8.0,0.0
equal_weights,v2,4,0.299,-0.007,8.0,8.0,0.0
equal_weights,v2,5,0.229,-0.007,8.0,8.0,0.0
ga,v1,2,0.677 ± 0.000,-0.003 ± 0.001,4.0,3.6263736263736264,1.1474773800000548
ga,v1,3,0.706 ± 0.074,0.132 ± 0.076,4.0,3.6466165413533838,1.4393608399994264
ga,v1,4,0.836 ± 0.069,0.203 ± 0.192,4.0,3.5714285714285707,0.9949377799999638
ga,v1,5,0.775 ± 0.016,0.030 ± 0.020,4.0,3.5714285714285707,1.2106859200004692
ga,v2,2,0.638 ± 0.036,0.430 ± 0.397,4.0,3.6263736263736264,1.6834983599994302
ga,v2,3,0.691 ± 0.033,0.320 ± 0.141,4.0,3.726881301261659,1.5385275800010276
ga,v2,4,0.842 ± 0.096,0.344 ± 0.001,4.0,3.5714285714285707,1.2893780999995215
ga,v2,5,0.732 ± 0.001,0.279 ± 0.001,4.0,3.5714285714285707,1.8451473999997687
random_search,v1,2,0.664 ± 0.028,0.141 ± 0.323,4.0,3.6752995077238757,1.5744849999999133
random_search,v1,3,0.713 ± 0.004,0.168 ± 0.013,4.0,3.670653654286215,1.853800460000275
random_search,v1,4,0.782 ± 0.014,0.274 ± 0.157,4.0,3.6630357506037057,1.2868202200002998
random_search,v1,5,0.753 ± 0.046,0.090 ± 0.108,4.0,3.6263736263736264,1.609962860000087
random_search,v2,2,0.559 ± 0.062,0.719 ± 0.000,4.0,3.6666257163800964,1.5210819800002355
random_search,v2,3,0.713 ± 0.004,0.168 ± 0.013,4.0,3.670653654286215,1.9165795400000207
random_search,v2,4,0.779 ± 0.015,0.344 ± 0.001,4.0,3.7179808055487604,1.5067749800000456
random_search,v2,5,0.728 ± 0.058,0.235 ± 0.103,4.0,3.5714285714285707,1.6913529600002222
```

## Pruebas pareadas de Wilcoxon

Se compara el algoritmo genético contra cada baseline en las semillas emparejadas.
«Mejora», «empata» o «empeora» se decide solo si el contraste está definido.
El umbral es p < 0.05 en Wilcoxon bilateral. Con 5 parejas, el p mínimo cuando todas las diferencias tienen el mismo signo es 0.0625: una ventaja descriptiva puede quedar como empate. La diferencia media se reporta igual.

- v1, k=2, silhouette, GA vs equal_weights: empata (diferencia media 0.4355, p=0.0625).
- v1, k=2, ari, GA vs equal_weights: empata (diferencia media 0.0015, p=0.2500).
- v1, k=2, silhouette, GA vs random_search: empata (diferencia media 0.0128, p=0.2500).
- v1, k=2, ari, GA vs random_search: empata (diferencia media -0.1438, p=1.0000).
- v1, k=3, silhouette, GA vs equal_weights: empata (diferencia media 0.4708, p=0.0625).
- v1, k=3, ari, GA vs equal_weights: empata (diferencia media 0.1276, p=0.1250).
- v1, k=3, silhouette, GA vs random_search: empata (diferencia media -0.0074, p=0.8125).
- v1, k=3, ari, GA vs random_search: empata (diferencia media -0.0358, p=0.5000).
- v1, k=4, silhouette, GA vs equal_weights: empata (diferencia media 0.5366, p=0.0625).
- v1, k=4, ari, GA vs equal_weights: empata (diferencia media 0.2100, p=0.2500).
- v1, k=4, silhouette, GA vs random_search: empata (diferencia media 0.0533, p=0.1875).
- v1, k=4, ari, GA vs random_search: empata (diferencia media -0.0704, p=0.2500).
- v1, k=5, silhouette, GA vs equal_weights: empata (diferencia media 0.5461, p=0.0625).
- v1, k=5, ari, GA vs equal_weights: empata (diferencia media 0.0369, p=0.0625).
- v1, k=5, silhouette, GA vs random_search: empata (diferencia media 0.0215, p=0.6250).
- v1, k=5, ari, GA vs random_search: empata (diferencia media -0.0601, p=0.3125).
- v2, k=2, silhouette, GA vs equal_weights: empata (diferencia media 0.3961, p=0.0625).
- v2, k=2, ari, GA vs equal_weights: empata (diferencia media 0.4343, p=0.2500).
- v2, k=2, silhouette, GA vs random_search: empata (diferencia media 0.0781, p=0.2500).
- v2, k=2, ari, GA vs random_search: empata (diferencia media -0.2896, p=0.5000).
- v2, k=3, silhouette, GA vs equal_weights: empata (diferencia media 0.4556, p=0.0625).
- v2, k=3, ari, GA vs equal_weights: empata (diferencia media 0.3151, p=0.0625).
- v2, k=3, silhouette, GA vs random_search: empata (diferencia media -0.0226, p=0.1875).
- v2, k=3, ari, GA vs random_search: empata (diferencia media 0.1517, p=0.2500).
- v2, k=4, silhouette, GA vs equal_weights: empata (diferencia media 0.5425, p=0.0625).
- v2, k=4, ari, GA vs equal_weights: empata (diferencia media 0.3503, p=0.0625).
- v2, k=4, silhouette, GA vs random_search: empata (diferencia media 0.0623, p=0.3125).
- v2, k=4, ari, GA vs random_search: empata (diferencia media 0.0000, p=1.0000).
- v2, k=5, silhouette, GA vs equal_weights: empata (diferencia media 0.5031, p=0.0625).
- v2, k=5, ari, GA vs equal_weights: empata (diferencia media 0.2862, p=0.0625).
- v2, k=5, silhouette, GA vs random_search: empata (diferencia media 0.0042, p=1.0000).
- v2, k=5, ari, GA vs random_search: empata (diferencia media 0.0444, p=1.0000).
