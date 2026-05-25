# ShelfScan — Guía de Presentación Final

**Duración estimada:** 15–20 minutos + Q&A
**Formato:** slides + demo en vivo
**Miembros:** Diego Valenzuela · Daniel Dubón · Bianca Calderón

---

## Estructura general de slides

| # | Slide | Quién |
|---|---|---|
| 1 | Portada: ShelfScan, nombres, fecha | Diego |
| 2 | Problema y objetivo | Diego |
| 3 | Arquitectura del sistema (diagrama de módulos) | Diego |
| 4 | Dataset: fuentes, clases, anotación | Diego |
| 5 | Entrenamiento YOLOv8: corridas, parámetros | Diego |
| 6 | Métricas de detección (tabla por clase) | Diego |
| 7 | DEMO en vivo — detección | Diego |
| 8 | Clasificación ResNet-50: arquitectura y resultados | Daniel |
| 9 | SIFT matching: metodología y precision | Daniel |
| 10 | Análisis temporal: predicción de quiebres | Daniel |
| 11 | DEMO en vivo — clasificación o temporal | Daniel |
| 12 | Corrección de perspectiva + homografía | Bianca |
| 13 | Planograma: estructura y compliance score | Bianca |
| 14 | Resultados planograma sobre 12 imágenes | Bianca |
| 15 | DEMO en vivo — perspectiva | Bianca |
| 16 | Técnicas del curso aplicadas | Diego |
| 17 | Conclusiones y trabajo futuro | Todos |

---

## Slide por slide — contenido

### Slide 1 — Portada
- Título: **ShelfScan: Sistema de Auditoría Visual de Lineales**
- Subtítulo: Visión por Computadora — Entrega Final
- Nombres de los tres integrantes
- Fecha

### Slide 2 — Problema y objetivo
- Foto de un estante desordenado o con quiebres
- 3 bullets:
  - Auditar estantes manualmente es lento y costoso
  - Los quiebres de stock causan pérdidas directas de ventas
  - ShelfScan detecta, clasifica y audita estantes automáticamente desde una foto de celular

### Slide 3 — Arquitectura
- Diagrama horizontal:
```
Imagen entrada → Perspectiva → YOLO Detección → ResNet Clasificación → Planograma
                  (Bianca)       (Diego)           (Daniel)             (Bianca)
                                                      ↓
                                               Análisis temporal (Daniel)
```
- Una oración por módulo

### Slide 4 — Dataset
- Tabla fuentes: Kaggle (45 imgs) + Open Images v7 (~4,500 imgs)
- Lista de 10 clases con ícono o color
- Mencionar: YOLO-World pre-etiquetado → revisión manual en makesense.ai
- Split: originales → test / Open Images → 70/30 train-val

### Slide 5 — Entrenamiento
- Tabla de 3 corridas:

| Corrida | Hardware | mAP@0.5 | Nota |
|---|---|---|---|
| Entrega 1 | Apple M5 MPS | 0.436 | 36 épocas, early stop |
| Entrega Final | NVIDIA GPU | **0.916** | Dataset balanceado, cls=1.5 |
| Re-entrenamiento v2 | Apple M5 CPU | 0.745 | Fine-tuning, cls=2.5 |

- Mencionar: AMP desactivado en MPS por bug PyTorch

### Slide 6 — Métricas detección
- Tabla modelo NVIDIA (mAP@0.5 por clase)
- Highlight: confitería 0.976, higiene 0.939, lacteos 0.952
- Nota al pie: val set NVIDIA ≠ val set local → métricas oficiales son las NVIDIA

### Slide 7 — DEMO detección (ver sección DEMO más abajo)

### Slide 8 — ResNet-50
- Diagrama: imagen → YOLO crop → ResNet-50 → clase
- Arquitectura: ResNet-50 ImageNet → FC(2048→10)
- Mostrar matriz de confusión
- Clases fáciles vs difíciles

### Slide 9 — SIFT matching
- Diagrama: imagen query → SIFT keypoints → BFMatcher → clase ganadora
- Métricas: top-1=0.2203 / top-5=0.2703
- Por qué es bajo y por qué sigue siendo útil: no requiere re-entrenar, identifica productos del catálogo

### Slide 10 — Temporal
- Gráfica `temporal_rotacion_categoria.png`
- Gráfica `temporal_quiebres_franja.png`
- Explicar: regresión lineal → hora estimada de quiebre

### Slide 12 — Perspectiva
- Par de imágenes: foto en ángulo → imagen warped rectificada
- Fórmula de homografía (visual, no matemática)
- `cv2.getPerspectiveTransform` + `cv2.warpPerspective`

### Slide 13 — Planograma
- Diagrama de zonas: grid 3×3 con clases esperadas
- Fórmula compliance: zonas correctas / zonas totales
- Métricas adicionales: share_of_shelf, breakage_by_category

### Slide 14 — Resultados planograma
- Tabla de resultados 12 imágenes
- Share error: 0.053 / Breakage error: 0.130 / Count error: 0.259
- Explicar NaN de Pearson honestamente: mismatch geométrico zonas/imágenes reales → oportunidad de mejora

### Slide 16 — Técnicas del curso
- Tabla con técnica → dónde se usa en ShelfScan
- Convolución, homografías, SIFT, CNNs, transfer learning, YOLO, IoU/mAP/NMS, regresión

### Slide 17 — Conclusiones
- Lo que funciona: detección mAP 0.916, perspectiva robusta, pipeline integrado
- Limitaciones honestas: sesgo enlatados en OI, NaN planograma, SIFT bajo
- Trabajo futuro: más clases, más datos balanceados, deploy móvil

---

## Guión por miembro

---

### Diego Valenzuela — Introducción + Dataset + Detección

**[Slide 1 — Portada]**
> "Buenas tardes. Vamos a presentar ShelfScan, un sistema de auditoría visual de lineales de supermercado. La idea es que con una simple foto de celular, el sistema detecte qué productos hay en el estante, en qué estado está el stock, y si el acomodo coincide con el planograma. Lo dividimos en tres módulos principales, uno por integrante."

**[Slide 2 — Problema]**
> "El problema que atacamos es concreto: auditar estantes manualmente en un supermercado es lento, inconsistente y caro. Un quiebre de stock que no se detecta a tiempo es venta perdida directa. Nuestra solución automatiza eso con visión por computadora."

**[Slide 3 — Arquitectura]**
> "La arquitectura es un pipeline secuencial. Primero Bianca corrige la perspectiva de la foto para que el estante quede recto. Luego mi módulo de detección YOLOv8 ubica y clasifica cada producto. Con esas detecciones, Daniel clasifica más finamente con ResNet y analiza el comportamiento en el tiempo. Y Bianca también compara contra el planograma para calcular el cumplimiento."

**[Slide 4 — Dataset]**
> "Para entrenar necesitábamos datos. Empezamos con 45 imágenes reales de estantes de Kaggle que anotamos manualmente. Para volumetría, descargamos ~4,500 imágenes de Open Images v7, que ya trae labels automáticos. Definimos 10 clases: bebidas, lácteos, snacks, cereales, enlatados, aceites, higiene, confitería y zona vacía. El proceso de anotación fue: pre-etiquetado automático con YOLO-World, corrección manual en makesense.ai. Tomó alrededor de 6 horas para las 45 imágenes originales."

**[Slide 5 — Entrenamiento]**
> "Hicimos tres corridas de entrenamiento. La primera en mi Mac M5 llegó a 0.436 mAP. La corrida oficial en NVIDIA alcanzó 0.916. Y para la presentación de hoy, hice un re-entrenamiento local partiendo del modelo NVIDIA para corregir un sesgo que tenía hacia enlatados en imágenes fuera de distribución — ese llegó a 0.745. Un detalle técnico: en Apple Silicon hay un bug de PyTorch con Automatic Mixed Precision que causa crashes aleatorios, la solución fue desactivar AMP en MPS."

**[Slide 6 — Métricas]**
> "Las métricas formales del modelo NVIDIA son estas. Confitería en 0.976, higiene 0.939, lácteos 0.952. El promedio general es 0.916 mAP@0.5. Una aclaración importante: si evalúan el modelo localmente van a ver un número diferente, 0.211. Esto se debe a que Open Images usa shuffle al descargar — cada máquina baja un subconjunto diferente, entonces el val set de NVIDIA no es el mismo que el local. Las métricas oficiales son las del entrenamiento NVIDIA."

**[Slide 7 — DEMO]**
> "Ahora les muestro en vivo." *(ver sección DEMO)*

**[Slide 16 — Técnicas del curso]**
> "Para cerrar, estas son las técnicas del curso que aplicamos. Convolución y filtrado en el preprocesamiento, homografías para la perspectiva, SIFT para matching, CNNs tanto en YOLO como en ResNet, transfer learning, detección con YOLO, y regresión para predecir quiebres temporales. Prácticamente todo el programa del curso está en este proyecto."

---

### Daniel Dubón — Clasificación + SIFT + Temporal

**[Slide 8 — ResNet-50]**
> "Mi módulo toma las detecciones de YOLO y las refina. El flujo es: la imagen entra, YOLO encuentra los bounding boxes, yo recorto cada producto y lo paso por una ResNet-50. ResNet-50 está preentrenada en ImageNet, reemplacé solo la última capa por una FC de 2048 a 10 clases. Fine-tuning con Adam, learning rate 1e-4, early stopping en val accuracy. [Mostrar matriz de confusión] Las clases más fáciles fueron enlatados y bebidas por sus formas distintivas. Las más difíciles cereales y snacks que tienen packaging muy similar."

**[Slide 9 — SIFT]**
> "El segundo componente es feature matching con SIFT. La idea es diferente a la CNN: dada una imagen de un producto, quiero saber si es un producto que ya vi en mi catálogo, sin re-entrenar nada. Extraigo keypoints SIFT en la imagen de referencia y en la query, hago matching con BFMatcher y ratio test de Lowe en 0.75, y la clase que acumula más matches gana. La precision top-1 es 0.2203 y top-5 es 0.2703. Es bajo, pero esperado: SIFT fue diseñado para matching de la misma escena, no entre empaquetados diferentes de la misma categoría. Superficies lisas como latas o botellas generan pocos keypoints. Su valor es que no requiere datos etiquetados nuevos para reconocer un producto nuevo del catálogo."

**[Slide 10 — Temporal]**
> "El tercer componente es el análisis temporal. La idea: si tengo fotos del mismo estante a distintas horas, puedo detectar qué categorías se vacían y predecir cuándo va a haber un quiebre. [Mostrar gráfica rotacion_categoria] Aquí ven qué categorías se vacían más rápido en el día. [Mostrar gráfica quiebres_franja] Y aquí en qué franjas horarias se concentran los quiebres. El mecanismo de predicción es una regresión lineal sobre la serie de conteo por clase — cuando la recta proyectada cruza el umbral de quiebre, esa es la hora estimada. Con más datos históricos se podría mejorar con modelos de series de tiempo."

**[Slide 11 — DEMO Daniel]**
> *(si hay notebook o resultado visual de ResNet o temporal, mostrarlo aquí)*

---

### Bianca Calderón — Perspectiva + Planograma

**[Slide 12 — Perspectiva]**
> "Mi módulo empieza con el problema geométrico: una foto de celular de un estante casi nunca está perfectamente frontal. Si corres detección sobre una imagen en ángulo, los bounding boxes se distorsionan y el planograma no puede comparar zonas correctamente. La solución es una homografía. Selecciono las 4 esquinas del estante, calculo la transformación proyectiva con cv2.getPerspectiveTransform, y aplico cv2.warpPerspective. [Mostrar par de imágenes antes/después] El resultado es la imagen rectificada como si la cámara estuviera perfectamente perpendicular al estante."

**[Slide 13 — Planograma]**
> "Sobre la imagen rectificada, aplico el análisis de planograma. Un planograma define qué producto debería estar en qué zona del estante. En código, cada PlanogramZone tiene un ID, una clase esperada y un bounding box. El compliance score es simple: zonas donde la detección coincide con la clase esperada, dividido entre zonas totales. También calculo share_of_shelf — qué proporción del espacio ocupa cada categoría — y breakage_by_category."

**[Slide 14 — Resultados planograma]**
> "Corrí el pipeline sobre 12 imágenes reales. Los resultados: share error promedio de 0.053, breakage error de 0.130, count error de 0.259. La correlación de Pearson entre cumplimiento y quiebre resultó NaN. Voy a ser honesta sobre por qué: las zonas del planograma se definen sobre la imagen de referencia sintética, pero las detecciones se calculan sobre la imagen warped real. Hay un mismatch geométrico — las coordenadas no coinciden — entonces ninguna detección cae dentro de una zona y el compliance es cero en todos los casos. Con varianza cero, Pearson está indefinido. Los errores de share y breakage sí son válidos porque se calculan independientemente. Es la principal oportunidad de mejora: calibrar el planograma sobre la imagen warped."

**[Slide 15 — DEMO perspectiva]**
> "Les muestro en vivo la corrección de perspectiva y la detección." *(ver sección DEMO)*

---

## Sección DEMO — comandos listos

### Demo 1: detección con modelo v2 sobre imagen real de Guatemala

Antes de la presentación, tener lista esta imagen ya warped o usar puntos que cubren toda la imagen:

```
python scripts/detect_on_shelf.py --model models/shelfscan_v2/best.pt -i data/dataGT/IMG_6730.jpeg -p 0,0 5712,0 5712,3213 0,3213 -o data/results/demo
```

### Demo 2: inferencia rápida sobre imagen del dataset

```
python scripts/inference.py --model models/shelfscan_v2/best.pt --image data/annotated/images/001.jpg
```

### Demo 3: error analysis con modelo v2

```
python scripts/error_analysis.py --model models/shelfscan_v2/best.pt
```

### Si algo falla en vivo

Tener ya generadas imágenes de resultados en `data/results/dataGT_v2/` y `data/results/error_analysis/` para mostrar como fallback.

---

## Consejos para el Q&A

### Si preguntan por el 0.211 vs 0.916
> "Es por shuffle=True en la descarga de Open Images. Cada máquina descarga imágenes distintas, entonces el val set que usó NVIDIA no tenemos localmente. No hay trampa — el modelo es idéntico, el data de evaluación es diferente."

### Si preguntan por el NaN del planograma
> "El mismatch es geométrico: definimos las zonas en coordenadas de referencia sintética, no sobre la imagen warped. Para la siguiente versión se calibraría el planograma directamente sobre la imagen rectificada."

### Si preguntan por qué SIFT es tan bajo
> "SIFT fue diseñado para reconocer la misma escena desde distintos ángulos. Aquí lo usamos para clasificar entre categorías — superficies lisas como latas generan pocos keypoints. Es complementario a la CNN, no un reemplazo."

### Si preguntan por zona_vacia con 0.277 mAP
> "Zona vacía es la clase más difícil: es el fondo del estante, varía mucho por iluminación y color. Con más ejemplos anotados mejoraría significativamente."

### Si preguntan qué mejorarían
> 1. Corregir sesgo de enlatados con dataset más balanceado o oversampling
> 2. Calibrar zonas del planograma sobre imagen warped, no de referencia
> 3. Agregar clase "desconocido" para productos fuera de catálogo
> 4. Deploy móvil con modelo cuantizado (YOLOv8n → ONNX → CoreML)

---

## Checklist pre-presentación

- [ ] Correr demo localmente y verificar que genera output en `data/results/demo/`
- [ ] Tener imagen abierta de backup en caso de falla de cámara/script
- [ ] Verificar que `models/shelfscan_v2/best.pt` existe
- [ ] Tener slides exportadas a PDF (por si falla el render de markdown)
- [ ] Cada miembro sabe su guión de memoria (no leer de la pantalla)
- [ ] Decidir quién cierra con conclusiones (sugerido: Diego, por ser el que habla primero y último)
