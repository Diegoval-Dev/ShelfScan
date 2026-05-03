# Plan de Trabajo — ShelfScan
### Sistema de Auditoría Visual de Lineales de Supermercado
**Curso:** Visión por Computadora  
**Grupo:** Diego Valenzuela · Daniel Dubón · Bianca Calderón  
**Catedrático:** Alberto Suriano  
**Versión:** 2.0 — Alcance ampliado a 3 integrantes

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
| **Daniel Dubón** | Clasificación + Análisis temporal — ResNet, ORB/SIFT, predicción de quiebres |
| **Bianca Calderón** | Geometría + Planograma — Homografía, score de cumplimiento, reporte final |

---

## Entrega 1 — 30 de abril de 2026

**Objetivo:** Infraestructura base lista. Datos recolectados, corrección de perspectiva funcionando, primer modelo de detección entrenado, y diseño del módulo de planograma definido.

---

### Diego Valenzuela — Detección

- Definir las 8–12 categorías de productos a detectar (bebidas, lácteos, snacks, limpieza, etc.)
- Fotografiar estantes en 1–2 tiendas locales desde distintos ángulos y condiciones de luz — meta: 150–200 fotos propias
- Descargar y explorar SKU110K como complemento
- Anotar dataset con labelImg (mínimo 100 imágenes, formato YOLOv8)
- Aplicar data augmentation: rotación, cambio de brillo, blur, flip horizontal
- Configurar entorno: Python, Ultralytics YOLOv8, CUDA si disponible
- Correr primera versión de entrenamiento YOLOv8 (versión preliminar)

**Entregables:**
- Dataset propio anotado (100+ imágenes)
- Script de augmentation aplicado
- Primer modelo YOLOv8 con reporte de mAP inicial
- README del entorno de entrenamiento

---

### Daniel Dubón — Clasificación + Análisis Temporal

- Investigar y documentar arquitectura ResNet-50 para transfer learning en clasificación de categorías
- Preparar subconjunto de imágenes clasificadas por categoría (usando dataset de Diego)
- Implementar pipeline base de fine-tuning: cargar ResNet preentrenado, congelar capas base, ajustar cabeza clasificadora
- Primera prueba de entrenamiento con las categorías definidas
- **Diseño del módulo temporal:** definir el esquema de recolección de secuencias (cuántas fotos, con qué intervalo, del mismo punto de cámara) y el formato de los datos temporales

**Entregables:**
- Notebook con pipeline de transfer learning funcionando
- Reporte de accuracy en validación (versión inicial)
- Documento de diseño del módulo temporal: esquema de datos, métricas a calcular, approach de predicción

---

### Bianca Calderón — Geometría + Planograma

- Implementar corrección de perspectiva con OpenCV: `getPerspectiveTransform` y `warpPerspective`
- Desarrollar herramienta de selección de 4 puntos semi-asistida (click manual sobre esquinas del estante)
- Probar corrección sobre 10–15 fotos reales y documentar resultados
- **Diseño del módulo de planograma:** definir qué es el planograma de referencia (imagen fotografiada en condiciones controladas), cómo se alinea con la imagen real, y qué métricas de cumplimiento se calcularán
- Investigar approach de matching espacial ORB/SIFT para comparación planograma vs. realidad

**Entregables:**
- Script funcional de corrección de perspectiva con interfaz de selección de puntos
- Galería de 10 imágenes antes/después de la corrección
- Documento de diseño del módulo de planograma: formato de referencia, pipeline de comparación, métricas propuestas

---

### Integración — Entrega 1

- Verificar compatibilidad de formatos de salida entre módulos
- Documento de 1–2 páginas: descripción del dataset, decisiones de arquitectura y diseño de los dos módulos nuevos

---

## Entrega 2 — 7 de mayo de 2026

**Objetivo:** Pipeline completo de extremo a extremo funcionando. Los tres módulos integrados, medibles y con primeros resultados cuantitativos.

---

### Diego Valenzuela — Detección

- Completar entrenamiento YOLOv8 con dataset completo (propio + SKU110K filtrado)
- Implementar y ajustar NMS para limpiar detecciones solapadas en zonas densas
- Evaluar con métricas formales: mAP@0.5, mAP@0.5:0.95, Precision, Recall por clase
- Probar detección sobre imágenes corregidas (salida de Bianca como entrada)
- Identificar clases con peor desempeño y aplicar mejoras

**Entregables:**
- Modelo YOLOv8 entrenado y guardado (.pt)
- Reporte de métricas por clase con análisis de errores
- Script de inferencia: imagen → bounding boxes con clase y confianza

---

### Daniel Dubón — Clasificación + Análisis Temporal

- Completar fine-tuning de ResNet con todas las categorías
- Integrar clasificador con detecciones de YOLO: cada bounding box pasa por ResNet
- **Módulo temporal v1:** recolectar secuencia de imágenes del mismo estante en distintos momentos del día (mínimo 3 momentos distintos), correr el detector sobre cada imagen y construir la serie temporal de ocupación por categoría
- Implementar detección de patrones básicos: qué categorías se vacían más rápido, en qué franja horaria
- Primera versión de predicción simple: regresión lineal o promedio móvil sobre la serie temporal para estimar cuándo habrá quiebre

**Entregables:**
- Pipeline integrado: imagen → detección → clasificación por categoría
- Evaluación del clasificador: accuracy top-1, matriz de confusión
- Módulo temporal funcionando con al menos una secuencia real de datos
- Gráfica de ocupación por categoría a lo largo del tiempo

---

### Bianca Calderón — Geometría + Planograma

- Integrar corrección de perspectiva con salida de detección y clasificación
- **Módulo de planograma v1:** tomar imagen de referencia del estante en condiciones controladas, alinearla con imagen real usando homografía completa, calcular matching ORB/SIFT entre ambas para identificar correspondencias por zona
- Calcular score de cumplimiento: qué porcentaje de zonas tienen el producto esperado según el planograma
- Implementar cálculo de las métricas base: % de quiebre, share of shelf por categoría, score general
- Generar reporte visual integrado con las tres líneas de análisis

**Entregables:**
- Módulo de planograma funcionando con al menos 5 pares imagen real / planograma
- Script de cálculo de métricas integrado
- 5 reportes de ejemplo generados sobre imágenes reales con los tres ejes de análisis

---

### Integración — Entrega 2

- Correr pipeline completo sobre 20 imágenes de prueba
- Documento de avance: métricas obtenidas, problemas encontrados, ajustes al plan
- Verificar integración entre los tres módulos

---

## Entrega Final — 21 de mayo de 2026

**Objetivo:** Sistema pulido, validado formalmente con ground truth, análisis de resultados completo y presentación lista.

---

### Diego Valenzuela — Detección

- Iteración final del modelo: ajuste de hiperparámetros según resultados de entrega 2
- Evaluación formal sobre conjunto de test (nunca visto durante entrenamiento)
- Documentar curvas de entrenamiento (loss, mAP por época)
- Análisis de casos de fallo: condiciones de luz, ángulos, productos similares entre sí
- Preparar sección de detección para informe y presentación

**Entregables finales:**
- Modelo final con métricas formales en test set
- Análisis de errores documentado
- Slides de detección para la presentación

---

### Daniel Dubón — Clasificación + Análisis Temporal

- Iterar clasificador en categorías con bajo accuracy
- **Módulo temporal final:** ampliar la secuencia temporal a múltiples días si es posible, refinar predicción de quiebres, calcular error de predicción vs. quiebre real observado
- Análisis comparativo: ¿en qué zonas o categorías el sistema predice mejor?
- Evaluar matching ORB: precisión en identificación de productos conocidos vs. desconocidos
- Preparar sección de clasificación y análisis temporal para informe y presentación

**Entregables finales:**
- Clasificador final con evaluación completa
- Módulo temporal con análisis de patrones y métricas de predicción
- Slides correspondientes

---

### Bianca Calderón — Geometría + Planograma

- Afinar corrección de perspectiva en casos difíciles (fotos muy anguladas, distorsión de lente)
- **Módulo de planograma final:** evaluar score de cumplimiento sobre 10+ pares reales, calcular correlación entre score de cumplimiento y quiebre detectado
- Validar métricas contra ground truth manual: contar físicamente productos y espacios en 10 imágenes y comparar con lo que el sistema reporta
- Calcular error absoluto en % de quiebre y share of shelf
- Generar conjunto final de reportes para la presentación

**Entregables finales:**
- Validación de métricas con ground truth manual
- Evaluación del módulo de planograma con correlación entre cumplimiento y quiebre
- Conjunto final de reportes visuales
- Slides de geometría y métricas

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
| Harris / SIFT / ORB | Matching planograma vs. realidad (Bianca + Daniel) |
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
| labelImg | Anotación del dataset (local, formato YOLO) |
| YOLO-World | Pre-etiquetado automático zero-shot para acelerar anotación |
| SKU110K | Dataset público complementario |
| Pandas + Matplotlib | Análisis temporal y visualización |
| Scikit-learn | Regresión para predicción de quiebres |
| PIL / Matplotlib | Generación de reportes visuales |
| Google Colab / GPU local | Entrenamiento |

---

*Plan versión 2.0 — Actualizado el 29 de abril de 2026.*