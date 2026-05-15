# Plan de Trabajo — ShelfScan
### Sistema de Auditoría Visual de Lineales de Supermercado
**Curso:** Visión por Computadora  
**Grupo:** Diego Valenzuela · Daniel Dubón · Bianca Calderón  
**Catedrático:** Alberto Suriano  
**Versión:** 2.3 — Entrega 2 completa, Entrega Final en progreso

---

## Descripción del Proyecto

Sistema que analiza fotos de estantes de supermercado para auditar automáticamente el estado del stock. A partir de imágenes tomadas con celular o cámara fija, el sistema ejecuta tres líneas de análisis paralelas:

1. **Detección y clasificación** — YOLOv8 detecta productos y zonas vacías; ResNet clasifica por categoría mediante transfer learning
2. **Verificación de planograma** — Homografía completa alinea la imagen real contra el planograma de referencia y calcula un score de cumplimiento mediante matching ORB/SIFT
3. **Análisis temporal** — Secuencias de imágenes del mismo estante a distintas horas permiten detectar patrones de vaciado y predecir cuándo ocurrirá el próximo quiebre de stock

---

## Roles del Equipo

| Integrante | Módulo Principal |
|---|---|
| **Diego Valenzuela** | Detección — Dataset, YOLOv8, evaluación con mAP/IoU/NMS |
| **Daniel Dubón** | Clasificación + Matching + Análisis temporal — ResNet transfer learning, ORB/SIFT para identificación de productos, predicción de quiebres |
| **Bianca Calderón** | Geometría proyectiva + Planograma — Corrección perspectiva, homografía completa, score de cumplimiento, reporte final |

---

## Entrega 1 — 30 de abril de 2026 ✓ COMPLETADA

**Resultado:** mAP@0.5 = 0.436 (objetivo > 0.30 ✓). 36 épocas, early stopping en época 26.

### Resultados por clase
| Clase | mAP@0.5 | Nota |
|---|---|---|
| snacks | 0.973 | ✓ |
| bebidas | 0.846 | ✓ |
| enlatados | 0.647 | ✓ |
| lacteos | 0.444 | ✓ |
| zona_vacia | 0.136 | datos insuficientes |
| aceites | 0.006 | solo 3 instancias en val |
| higiene | 0.000 | solo 5 instancias en val |
| cereales | — | 0 instancias en val |
| limpieza | — | 0 instancias en val |
| confiteria | — | 0 instancias en val |

### Lo que se hizo
- ✓ 10 categorías definidas (`categories.py`)
- ✓ Dataset: Kaggle `humansintheloop/supermarket-shelves-dataset` descargado con `kagglehub`
- ✓ Pre-etiquetado automático con YOLO-World (`autolabel.py`), revisión con makesense.ai
- ✓ Augmentation: rotación ±15°, brillo, blur, flip — pipeline completo (`augmentation.py`)
- ✓ Split 70/20/10 train/val/test (`split_dataset.py`)
- ✓ YOLOv8n entrenado en MPS (Apple M5) con AMP deshabilitado
- ✓ Script de inferencia (`inference.py`)
- ✓ Corrección de perspectiva con selección de 4 puntos (`perspective.py`)
- ✓ Módulo `PlanogramReference` base (`planogram.py`)
- ✓ Notebook Entrega 1 (`entrega1_training.ipynb`)
- ✓ `map_report.txt` con métricas

### Deuda para Entrega 2
- Clases sin datos en val: cereales, limpieza, confiteria — agregar más imágenes de esas categorías
- aceites e higiene con muy pocas instancias — mismo problema
- Galería antes/después de corrección de perspectiva (Bianca — pendiente)
- Documento formal de diseño de planograma y módulo temporal (pendiente)
- Modelo sesga todo hacia `enlatados` — labels de YOLO-World mal balanceados, necesita más data de otras clases

---

## Entrega 2 — 7 de mayo de 2026 ✓ COMPLETADA

**Objetivo:** Pipeline completo de extremo a extremo funcionando. Los tres módulos integrados, medibles y con primeros resultados cuantitativos.

---

### Diego Valenzuela — Detección ✓ COMPLETO (Entrega 2)

- ✓ Modelo YOLOv8 re-entrenado con Open Images + class weights (`best.pt` v3 — NVIDIA)
- ✓ Script de inferencia: `inference.py`
- ✓ Pipeline perspectiva → detección: `detect_on_shelf.py`
- ✓ Eval formal: `evaluate.py`, `map_report_test.txt`
- ✓ NMS activo, conf=0.25
- ✓ `split_dataset.py` actualizado — test set aislado (solo imágenes originales)

**Métricas reales (val set NVIDIA — labels correctos):**
| Clase | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| bebidas | 0.909 ✓ | 0.655 |
| lacteos | 0.952 ✓ | 0.758 |
| snacks | 0.846 ✓ | 0.569 |
| enlatados | 0.911 ✓ | 0.723 |
| aceites | 0.842 ✓ | 0.735 |
| higiene | 0.939 ✓ | 0.671 |
| confiteria | 0.976 ✓ | 0.707 |
| zona_vacia | 0.952 ✓ | 0.871 |
| **all** | **0.916** ✓ | **0.711** |

> Nota: eval local muestra 0.208 — test set local tiene labels basura de YOLO-World (65 latas/imagen). Métricas reales son las de NVIDIA. Para Entrega Final: re-labelear 45 imágenes originales con makesense.ai.

**Entregables:**
- ✓ Modelo .pt (mAP@0.5 = 0.916)
- ✓ `inference.py`
- ✓ `map_report_test.txt`

---

### Daniel Dubón — Clasificación + Análisis Temporal ✓ COMPLETO (Entrega 2)

- ✓ Crops generados desde detecciones YOLO (modelo v3, todas las clases)
- ✓ ResNet-50 fine-tuning corrido (`entrega1_training.ipynb` cells 15–18)
- ✓ Accuracy top-1 + matriz de confusión documentada
- ✓ Módulo temporal v1: secuencia 3+ imágenes → conteo por clase → gráfica de ocupación

**Entregables:**
- ✓ Pipeline: imagen → YOLO → crop → ResNet
- ✓ Métricas clasificador
- ✓ Gráfica temporal de ocupación por categoría

---

### Bianca Calderón — Geometría + Planograma ✓ COMPLETO (Entrega 2)

- ✓ Corrección de perspectiva integrada con detección (`detect_on_shelf.py`)
- ✓ `planogram.py` expandido: `PlanogramZone`, compliance scoring, métricas por zona
- ✓ Score de cumplimiento: % zonas con producto esperado

**Entregables:**
- ✓ Módulo planograma v1 con métricas
- ✓ Pipeline integrado perspectiva → detección → planograma

---

### Integración — Entrega 2 ✓

- ✓ Pipeline completo funcionando: imagen → perspectiva → YOLO → planograma
- ✓ Módulos tres integrados via `detect_on_shelf.py`

---

## Entrega Final — 21 de mayo de 2026

**Objetivo:** Sistema pulido, validado formalmente con ground truth, análisis de resultados completo y presentación lista.

---

### Diego Valenzuela — Detección

- ⬜ Re-labelear 45 imágenes originales con makesense.ai (test set limpio)
- ⬜ Eval formal en test set limpio → `map_report_test.txt` confiable
- ⬜ Análisis de casos de fallo: luz, ángulos, productos similares
- ⬜ Curvas de entrenamiento documentadas (loss, mAP por época)
- ⬜ Slides detección para presentación

**Entregables finales:**
- Modelo final (ya listo: mAP@0.5 = 0.916)
- Test set con labels correctos + métricas formales
- Análisis de errores
- Slides

---

### Daniel Dubón — Clasificación + Matching + Análisis Temporal

- ⬜ Iterar ResNet en categorías con bajo accuracy (mejorar con modelo v3)
- ⬜ **ORB/SIFT matching:** identificar productos conocidos vs. desconocidos por descriptor visual — precisión top-1 y top-5
- ⬜ **Módulo temporal final:**
  - Ampliar secuencia a múltiples días o franjas horarias
  - Detectar qué categorías rotan más rápido y en qué franja horaria se concentran quiebres
  - Predicción de quiebre: calcular error de predicción (T_pred vs. T_real observado)
- ⬜ Slides clasificación + temporal para presentación

**Entregables finales:**
- Clasificador ResNet con accuracy top-1 + matriz de confusión final
- ORB/SIFT matching: métricas de identificación de productos
- Módulo temporal: gráfica de rotación por categoría, franjas horarias, error de predicción
- Slides

---

### Bianca Calderón — Geometría Proyectiva + Planograma

- ⬜ Afinar corrección de perspectiva en casos difíciles (fotos anguladas, distorsión de lente)
- ⬜ **Planograma final:** evaluar score de cumplimiento sobre 10+ pares imagen real / planograma de referencia
- ⬜ Calcular correlación entre score de cumplimiento y quiebre detectado
- ⬜ Validar contra ground truth manual: contar físicamente productos en 10 imágenes, comparar con sistema
- ⬜ Calcular error absoluto: % quiebre y share of shelf vs. conteo manual
- ⬜ Generar reportes visuales finales con los tres ejes integrados
- ⬜ Slides geometría + planograma para presentación

**Entregables finales:**
- Planograma validado sobre 10+ pares reales con score de cumplimiento
- Correlación score cumplimiento vs. quiebre detectado
- Error absoluto en % quiebre y share of shelf vs. ground truth
- Reportes visuales finales
- Slides

---

### Integración Final

- **Informe final:** introducción, marco teórico, metodología por módulo, resultados, análisis, conclusiones
- **Demo:** foto/secuencia entra → reporte sale, mostrando los tres módulos integrados
- **Repositorio limpio:** código comentado, README de instalación y uso, modelos guardados
- **Presentación:** 15–20 minutos divididos equitativamente entre los tres

---

## Resumen de Entregables

| Entrega | Fecha | Qué se entrega |
|---|---|---|
| **Entrega 1** | 30 abril 2026 | Dataset anotado, corrección de perspectiva, primer YOLOv8, pipeline de clasificación base, diseño de módulos de planograma y temporal |
| **Entrega 2** | 7 mayo 2026 | Pipeline completo integrado, métricas formales, módulo de planograma v1, módulo temporal v1 con primeros patrones |
| **Entrega Final** | 21 mayo 2026 | Sistema pulido, validación con ground truth, análisis de los tres módulos, informe, demo y presentación |

---

## Métricas de Evaluación

| Métrica | Qué mide | Responsable |
|---|---|---|
| mAP@0.5 | Calidad del detector YOLO | Diego |
| IoU promedio | Precisión de bounding boxes | Diego |
| Accuracy top-1 | Clasificación por categoría | Daniel |
| Error de predicción temporal | Cuánto se anticipa al quiebre real | Daniel |
| Score de cumplimiento de planograma | Coincidencia real vs. planograma | Bianca |
| Error en % de quiebre vs. ground truth | Validez del análisis de stock | Bianca |
| Error en share of shelf vs. ground truth | Validez del análisis por categoría | Bianca |

---

## Cobertura de Técnicas del Curso

| Técnica del curso | Módulo donde se aplica |
|---|---|
| Filtrado y convolución | Preprocesamiento de imágenes en los tres módulos |
| Transformadas de Fourier | Análisis de textura para detección de zonas vacías |
| Morfología | Limpieza de regiones detectadas |
| Harris / SIFT / ORB | Matching de productos por descriptor visual (Daniel) + alineación planograma (Bianca) |
| Geometría proyectiva y homografías | Corrección de perspectiva y alineación de planograma (Bianca) |
| CNNs y transfer learning | Clasificación por categoría con ResNet (Daniel) |
| Arquitecturas modernas (ResNet) | Backbone del clasificador (Daniel) |
| Detección de objetos (YOLO) | Detección de productos y zonas vacías (Diego) |
| IoU, mAP, NMS | Evaluación y post-procesamiento del detector (Diego) |

---

## Stack Tecnológico

| Herramienta | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| Ultralytics YOLOv8 | Detección de objetos |
| PyTorch + torchvision | Transfer learning con ResNet |
| OpenCV | Homografía, ORB/SIFT, procesamiento de imagen |
| makesense.ai | Anotación del dataset (browser, formato YOLO) |
| YOLO-World | Pre-etiquetado automático zero-shot para acelerar anotación |
| Open Images v7 (fiftyone) | Dataset público con labels — reemplazó SKU110K |
| Pandas + Matplotlib | Análisis temporal y visualización |
| Scikit-learn | Regresión para predicción de quiebres |
| PIL / Matplotlib | Generación de reportes visuales |
| Google Colab / GPU local | Entrenamiento |

---

*Plan versión 2.3 — Actualizado el 14 de mayo de 2026.*