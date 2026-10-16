# Comparación

El conjunto de prueba no se usó para elegir k, pesos, lambda, alpha ni ningún hiperparámetro. La configuración se congeló con métricas de validación y solo después se calcularon las métricas de prueba, una vez.

Un ARI cercano a 0 es un resultado legítimo: no se presenta como fallo ni como éxito.
Los clusters se nombran cluster_0, cluster_1, … y no son niveles clínicos.

## Media en validación

```
method,variant,k,silhouette,ari,n_active,n_eff,opt_seconds
equal_weights,v1,2,0.165,-0.010,20.0,19.999999999999993,0.0
equal_weights,v1,3,0.136,0.006,20.0,19.999999999999993,0.0
equal_weights,v1,4,0.155,-0.010,20.0,19.999999999999993,0.0
equal_weights,v1,5,0.127,0.018,20.0,19.999999999999993,0.0
equal_weights,v2,2,0.165,-0.010,20.0,19.999999999999993,0.0
equal_weights,v2,3,0.136,0.006,20.0,19.999999999999993,0.0
equal_weights,v2,4,0.155,-0.010,20.0,19.999999999999993,0.0
equal_weights,v2,5,0.127,0.018,20.0,19.999999999999993,0.0
ga,v1,2,0.660 ± 0.056,-0.011 ± 0.013,4.0,3.5740966614546785,66.79848429999947
ga,v1,3,0.839 ± 0.008,-0.011 ± 0.004,4.0,3.574337691344808,82.29546759000041
ga,v1,4,0.916 ± 0.050,-0.003 ± 0.010,4.0,3.5989010989010985,73.50648611999968
ga,v1,5,0.963 ± 0.035,0.003 ± 0.013,4.0,3.634325125609018,79.31965681999972
ga,v2,2,0.668 ± 0.047,-0.008 ± 0.014,4.0,3.5772763360154536,74.30818562000022
ga,v2,3,0.815 ± 0.039,-0.005 ± 0.014,4.0,3.587791277208693,77.09875836000064
ga,v2,4,0.906 ± 0.046,-0.004 ± 0.009,4.0,3.5871010673794936,81.94527413999967
ga,v2,5,0.940 ± 0.038,0.005 ± 0.011,4.0,3.606485852340822,77.48360753999914
random_search,v1,2,0.591 ± 0.033,-0.010 ± 0.012,5.066666666666666,3.978153025973227,130.0124926166672
random_search,v1,3,0.631 ± 0.049,-0.006 ± 0.021,4.966666666666667,3.9616672810049534,149.64415195333288
random_search,v1,4,0.684 ± 0.068,0.001 ± 0.016,5.033333333333333,3.9936414079553106,131.42606764999994
random_search,v1,5,0.700 ± 0.068,0.007 ± 0.012,4.866666666666666,4.008387127440807,139.33373904333337
random_search,v2,2,0.593 ± 0.027,-0.010 ± 0.012,4.866666666666666,3.910508281681444,133.58396303999993
random_search,v2,3,0.629 ± 0.045,-0.004 ± 0.022,5.0,3.9178244700572122,139.6542238833333
random_search,v2,4,0.675 ± 0.068,0.008 ± 0.019,4.7,3.8240397467572538,134.064140413333
random_search,v2,5,0.736 ± 0.106,0.013 ± 0.018,4.6,3.87234613907938,140.27568622666683
```

## Pruebas pareadas de Wilcoxon

Se compara el algoritmo genético contra cada baseline en las semillas emparejadas.
«Mejora», «empata» o «empeora» se decide solo si el contraste está definido.
El umbral es p < 0.05 en Wilcoxon bilateral. Con 5 parejas, el p mínimo
cuando todas las diferencias tienen el mismo signo es 0.0625: una ventaja
descriptiva puede quedar como empate. La diferencia media se reporta igual.

- v1, k=2, silhouette, GA vs equal_weights: mejora (diferencia media 0.4948, p=0.0020).
- v1, k=2, ari, GA vs equal_weights: empata (diferencia media -0.0018, p=0.6484).
- v1, k=2, silhouette, GA vs random_search: mejora (diferencia media 0.0613, p=0.0137).
- v1, k=2, ari, GA vs random_search: empata (diferencia media 0.0010, p=0.5938).
- v1, k=3, silhouette, GA vs equal_weights: mejora (diferencia media 0.7033, p=0.0020).
- v1, k=3, ari, GA vs equal_weights: empeora (diferencia media -0.0172, p=0.0020).
- v1, k=3, silhouette, GA vs random_search: mejora (diferencia media 0.2108, p=0.0020).
- v1, k=3, ari, GA vs random_search: empata (diferencia media -0.0003, p=0.9375).
- v1, k=4, silhouette, GA vs equal_weights: mejora (diferencia media 0.7605, p=0.0020).
- v1, k=4, ari, GA vs equal_weights: mejora (diferencia media 0.0071, p=0.0430).
- v1, k=4, silhouette, GA vs random_search: mejora (diferencia media 0.2406, p=0.0020).
- v1, k=4, ari, GA vs random_search: empata (diferencia media 0.0015, p=0.5703).
- v1, k=5, silhouette, GA vs equal_weights: mejora (diferencia media 0.8364, p=0.0020).
- v1, k=5, ari, GA vs equal_weights: empeora (diferencia media -0.0148, p=0.0098).
- v1, k=5, silhouette, GA vs random_search: mejora (diferencia media 0.2612, p=0.0020).
- v1, k=5, ari, GA vs random_search: empata (diferencia media -0.0072, p=0.3750).
- v2, k=2, silhouette, GA vs equal_weights: mejora (diferencia media 0.5027, p=0.0020).
- v2, k=2, ari, GA vs equal_weights: empata (diferencia media 0.0012, p=0.2188).
- v2, k=2, silhouette, GA vs random_search: mejora (diferencia media 0.0811, p=0.0039).
- v2, k=2, ari, GA vs random_search: empata (diferencia media 0.0059, p=0.3672).
- v2, k=3, silhouette, GA vs equal_weights: mejora (diferencia media 0.6796, p=0.0020).
- v2, k=3, ari, GA vs equal_weights: empata (diferencia media -0.0113, p=0.0781).
- v2, k=3, silhouette, GA vs random_search: mejora (diferencia media 0.1894, p=0.0020).
- v2, k=3, ari, GA vs random_search: empata (diferencia media 0.0076, p=0.6855).
- v2, k=4, silhouette, GA vs equal_weights: mejora (diferencia media 0.7503, p=0.0020).
- v2, k=4, ari, GA vs equal_weights: empata (diferencia media 0.0061, p=0.0820).
- v2, k=4, silhouette, GA vs random_search: mejora (diferencia media 0.2170, p=0.0020).
- v2, k=4, ari, GA vs random_search: empata (diferencia media -0.0179, p=0.0645).
- v2, k=5, silhouette, GA vs equal_weights: mejora (diferencia media 0.8129, p=0.0020).
- v2, k=5, ari, GA vs equal_weights: empeora (diferencia media -0.0130, p=0.0137).
- v2, k=5, silhouette, GA vs random_search: mejora (diferencia media 0.2690, p=0.0020).
- v2, k=5, ari, GA vs random_search: empata (diferencia media -0.0055, p=0.2754).
