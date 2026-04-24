# Plan de Trabajo — Sistema de Auditoría Visual de Lineales de Supermercado
**Curso:** Visión por Computadora  
**Grupo:** Diego Valenzuela · Daniel Dubón · Bianca Calderón  
**Catedrático:** Alberto Suriano  

---

## Resumen del Proyecto

Sistema que analiza fotos de estantes de supermercado para detectar automáticamente el estado del stock. A partir de una imagen, corrige la perspectiva con homografías, detecta productos y espacios vacíos con YOLOv8, clasifica por categoría con transfer learning sobre ResNet, y genera un reporte con métricas: porcentaje de quiebre, share of shelf por categoría y score general del lineal.

---

## Roles del Equipo

| Integrante | Rol Principal |
|---|---|
| **Diego Valenzuela** | Detección de objetos — Dataset, YOLOv8, evaluación con mAP/IoU |
| **Daniel Dubón** | Clasificación y matching — ResNet, transfer learning, ORB/SIFT |
| **Bianca Calderón** | Geometría y reporte — Homografía, métricas finales, visualización |

> Los roles son principales, no exclusivos. En cada entrega hay puntos de integración donde los tres trabajan juntos.

---

## Entrega 1 — 30 de abril de 2026

**Objetivo:** Tener la base del proyecto funcionando: datos recolectados, pipeline de corrección de perspectiva operativo y primer modelo de detección entrenado aunque sea en versión preliminar.

---

### Diego Valenzuela — Detección

**Semana 1 (23–27 abril)**
- Definir las categorías de productos a detectar (8–12 clases: bebidas, lácteos, snacks, limpieza, etc.)
- Ir a 1–2 tiendas locales (Walmart, La Torre, Paiz) y fotografiar estantes desde distintos ángulos y condiciones de luz — meta: 150–200 fotos propias
- Descargar y explorar el dataset público SKU110K para complementar
- Configurar entorno: Python, Ultralytics YOLOv8, CUDA si hay GPU disponible

**Entregable al 30 de abril:**
- Carpeta con dataset propio anotado (mínimo 100 imágenes en Roboflow)
- Script de data augmentation aplicado (rotación, brillo, blur, flip)
- Primera versión de YOLOv8 entrenada (aunque sea con pocas épocas) con reporte de mAP inicial
- README del entorno de entrenamiento

---

### Daniel Dubón — Clasificación y Matching

**Semana 1 (23–27 abril)**
- Investigar y documentar arquitectura ResNet-50 para transfer learning en clasificación de categorías de productos
- Preparar el subconjunto de imágenes clasificadas por categoría (usar parte del dataset de Diego)
- Implementar pipeline base de fine-tuning con PyTorch o Keras: cargar ResNet preentrenado, congelar capas base, ajustar cabeza clasificadora
- Primera prueba de entrenamiento con las categorías definidas

**Entregable al 30 de abril:**
- Notebook con pipeline de transfer learning funcionando
- Reporte de accuracy en conjunto de validación (aunque sea con pocas clases todavía)
- Investigación documentada sobre ORB vs SIFT para el matching de productos con imagen de referencia (qué se usará y por qué)

---

### Bianca Calderón — Geometría y Reporte

**Semana 1 (23–27 abril)**
- Estudiar e implementar corrección de perspectiva usando OpenCV: `cv2.getPerspectiveTransform` y `cv2.warpPerspective`
- Desarrollar la herramienta de selección de 4 puntos de referencia semi-asistida (click manual sobre la imagen para marcar esquinas del estante)
- Probar la corrección sobre 10–15 fotos reales del dataset y documentar los resultados visualmente
- Definir el formato del reporte final (qué métricas, cómo se visualizan)

**Entregable al 30 de abril:**
- Script funcional de corrección de perspectiva con interfaz de selección de puntos
- Galería de 10 imágenes antes/después de la corrección
- Documento con el diseño del reporte final (mockup o esquema)

---

### Integración — Entrega 1

- Los tres revisan juntos que el formato de salida de cada módulo sea compatible
- Documento de 1–2 páginas con: descripción del dataset, decisiones de arquitectura tomadas y plan actualizado para las siguientes entregas

---

## Entrega 2 — 7 de mayo de 2026

**Objetivo:** Pipeline completo de extremo a extremo funcionando: foto entra, reporte sale. No tiene que ser perfecto, pero tiene que ser funcional y medible.

---

### Diego Valenzuela — Detección

**Semana 2 (28 abril–7 mayo)**
- Completar el entrenamiento de YOLOv8 con el dataset completo (dataset propio + SKU110K filtrado)
- Aplicar NMS (Non-Maximum Suppression) para limpiar detecciones solapadas en zonas densas del estante
- Evaluar el modelo con métricas formales: mAP@0.5, mAP@0.5:0.95, Precision, Recall por clase
- Probar detección sobre imágenes corregidas (usar salida de Bianca como entrada)
- Identificar clases con peor desempeño y aplicar estrategias de mejora (más data, augmentation específico)

**Entregable al 7 de mayo:**
- Modelo YOLOv8 entrenado y guardado (.pt)
- Reporte de métricas por clase con análisis de errores
- Script de inferencia que recibe imagen y devuelve bounding boxes con clase y confianza

---

### Daniel Dubón — Clasificación y Matching

**Semana 2 (28 abril–7 mayo)**
- Completar fine-tuning de ResNet con todas las categorías definidas
- Integrar el clasificador con las detecciones de YOLO: cada bounding box detectado pasa por ResNet para obtener la categoría
- Implementar matching con imagen de referencia usando ORB: extraer descriptores de la imagen actual y de la referencia, calcular distancia entre descriptores para estimar similitud por zona
- Evaluar clasificación: accuracy, matriz de confusión por categoría

**Entregable al 7 de mayo:**
- Pipeline integrado: imagen → detección → clasificación por categoría
- Notebook con evaluación del clasificador (accuracy top-1, matriz de confusión)
- Script de matching ORB con imagen de referencia funcionando sobre al menos 5 pares de imágenes

---

### Bianca Calderón — Geometría y Reporte

**Semana 2 (28 abril–7 mayo)**
- Integrar la corrección de perspectiva con la salida de detección y clasificación
- Implementar el cálculo de las tres métricas principales:
  - **% de quiebre:** (área de bounding boxes vacíos) / (área total del estante)
  - **Share of shelf:** (área por categoría) / (área total detectada con productos)
  - **Score general:** función que combina ambas métricas en un valor de 0–100
- Generar el reporte visual: imagen anotada con bounding boxes coloreados por categoría + tabla de métricas superpuesta o adjunta

**Entregable al 7 de mayo:**
- Script de cálculo de métricas integrado con las detecciones
- Función de generación de reporte visual (imagen anotada exportada como .png o .jpg)
- 5 reportes de ejemplo generados sobre imágenes reales

---

### Integración — Entrega 2

- Correr el pipeline completo de extremo a extremo sobre un set de 20 imágenes de prueba
- Documento de avance: métricas obtenidas hasta ahora, problemas encontrados, ajustes al plan
- Verificar que los tres módulos (detección, clasificación, reporte) estén comunicados correctamente

---

## Entrega 3 (Final) — 21 de mayo de 2026

**Objetivo:** Sistema pulido, evaluado formalmente, con análisis de resultados y presentación lista.

---

### Diego Valenzuela — Detección

**Semanas 3–4 (8–21 mayo)**
- Iteración final del modelo: ajustar hiperparámetros (learning rate, batch size, épocas) según resultados de la entrega 2
- Correr evaluación formal sobre el conjunto de test (nunca visto durante entrenamiento)
- Documentar curvas de entrenamiento (loss, mAP por época)
- Análisis de casos donde el modelo falla: ¿qué tipo de productos se detectan mal? ¿qué condiciones de luz o ángulo afectan?
- Preparar la sección de detección para el informe final y la presentación

**Entregable final:**
- Modelo final entrenado con métricas formales en test set
- Sección de análisis de errores del detector
- Slides de la parte de detección para la presentación

---

### Daniel Dubón — Clasificación y Matching

**Semanas 3–4 (8–21 mayo)**
- Iterar el clasificador si hay categorías con bajo accuracy según resultados de la entrega 2
- Evaluar formalmente el matching ORB: calcular qué tan bien identifica productos conocidos vs. desconocidos
- Explorar si agregar descriptores SIFT mejora el matching en algún caso específico
- Análisis comparativo: ¿dónde ayuda el clasificador ResNet? ¿dónde falla?
- Preparar la sección de clasificación para el informe y la presentación

**Entregable final:**
- Clasificador final con evaluación completa
- Análisis comparativo ORB vs SIFT en el contexto del proyecto
- Sección del informe y slides correspondientes

---

### Bianca Calderón — Geometría y Reporte

**Semanas 3–4 (8–21 mayo)**
- Afinar la corrección de perspectiva en casos difíciles (fotos muy anguladas o con distorsión)
- Validar las métricas contra ground truth manual: contar físicamente productos y espacios vacíos en 10 imágenes y comparar con lo que el sistema reporta
- Calcular error absoluto del sistema en % de quiebre y share of shelf
- Generar el conjunto final de reportes para la presentación (mínimo 10 casos distintos)
- Preparar la sección de métricas y resultados para el informe final y la presentación

**Entregable final:**
- Validación de métricas con ground truth manual
- Conjunto final de reportes visuales generados
- Sección del informe y slides de métricas

---

### Integración Final — Entrega 3

- **Informe final** estructurado con: introducción, marco teórico, metodología, resultados, análisis, conclusiones
- **Demo en video** o en vivo: foto entra → reporte sale, mostrando los 3 módulos integrados
- **Repositorio limpio** con código comentado, README de instalación y uso, y modelos guardados
- **Presentación** de 15–20 minutos dividida equitativamente entre los tres integrantes

---

## Resumen de Entregables por Fecha

| Entrega | Fecha | Qué se entrega |
|---|---|---|
| **Entrega 1** | 30 abril 2026 | Dataset anotado, corrección de perspectiva, primer modelo YOLOv8, pipeline de clasificación base |
| **Entrega 2** | 7 mayo 2026 | Pipeline completo de extremo a extremo, métricas formales, reportes visuales de ejemplo |
| **Entrega Final** | 21 mayo 2026 | Sistema pulido, informe completo, validación con ground truth, demo y presentación |

---

## Métricas de Evaluación del Proyecto

| Métrica | Qué mide | Módulo responsable |
|---|---|---|
| mAP@0.5 | Calidad del detector YOLO | Diego |
| IoU promedio | Precisión de bounding boxes | Diego |
| Accuracy top-1 | Clasificación por categoría | Daniel |
| Distancia de descriptores ORB | Calidad del matching | Daniel |
| Error en % de quiebre vs. ground truth | Validez del análisis | Bianca |
| Error en share of shelf vs. ground truth | Validez del análisis | Bianca |

---

## Stack Tecnológico

| Herramienta | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| Ultralytics YOLOv8 | Detección de objetos |
| PyTorch + torchvision | Transfer learning con ResNet |
| OpenCV | Homografía, ORB, procesamiento de imagen |
| Roboflow | Anotación del dataset |
| SKU110K | Dataset público complementario |
| Matplotlib / PIL | Generación de reportes visuales |
| Google Colab / GPU local | Entrenamiento |

---

*Plan elaborado el 23 de abril de 2026.*