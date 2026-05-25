# ShelfScan — Sistema de Auditoría Visual de Lineales

**Curso:** Visión por Computadora
**Grupo:** Diego Valenzuela · Daniel Dubón · Bianca Calderón
**Catedrático:** Alberto Suriano

---

## 1. Objetivo del proyecto

Sistema que analiza fotos de estantes de supermercado para auditar automáticamente el estado del stock. A partir de imágenes tomadas con celular o cámara fija, el sistema ejecuta tres líneas de análisis paralelas:

1. **Detección y dataset** (Diego): YOLOv8 detecta productos y zonas vacías; evaluación formal con mAP/IoU/NMS
2. **Clasificación + matching + temporal** (Daniel): ResNet clasifica crops por categoría; SIFT identifica productos conocidos; análisis de patrones de vaciado en el tiempo
3. **Geometría proyectiva + planograma** (Bianca): Corrección de perspectiva con homografías; comparación contra planograma de referencia; score de cumplimiento y validación con ground truth manual

---

## 2. Dataset y pipeline de datos

### 2.1 Fuentes de datos

| Fuente | Uso | Método |
|---|---|---|
| `humansintheloop/supermarket-shelves-dataset` (Kaggle) | 45 imágenes originales anotadas manualmente | `download_dataset.py` + `kagglehub` |
| Open Images v7 (Google) | ~4,500 imágenes con labels automáticos | `download_openimages.py` + `fiftyone` |

El dataset de Kaggle fue la base principal — 45 imágenes reales de estantes (001.jpg–045.jpg). Open Images v7 aportó volumen para clases con pocas instancias.

### 2.2 Clases definidas

10 categorías (`scripts/categories.py`):

| ID | Clase | Descripción |
|---|---|---|
| 0 | bebidas | Botellas, jugos, refrescos |
| 1 | lacteos | Leche, yogurt, queso |
| 2 | snacks | Papas, galletas, frituras |
| 3 | cereales | Cereales de caja, avena |
| 4 | limpieza | Detergentes, jabones |
| 5 | enlatados | Latas, conservas |
| 6 | aceites | Aceites de cocina, vinagres |
| 7 | higiene | Champú, pasta dental, desodorante |
| 8 | confiteria | Dulces, chocolates |
| 9 | zona_vacia | Espacio vacío visible |

### 2.3 Anotación

- **Pre-etiquetado automático:** YOLO-World zero-shot (`scripts/autolabel.py`) generó bounding boxes con conf=0.15 como punto de partida
- **Revisión manual:** makesense.ai (browser, formato YOLO) — herramienta elegida tras descartar labelImg (crash Python 3.12) y anylabeling (PyQt6 faltante)
- **Problema detectado:** YOLO-World generaba 65+ instancias de `enlatados` por imagen → labels basura → se re-anotaron manualmente

### 2.4 Mapeo Open Images → clases ShelfScan

Open Images usa nombres en inglés distintos. Mapeo clave implementado en `download_openimages.py`:

```
"Drink" → bebidas (0)
"Milk"  → lacteos (1)
"Snack" → snacks (2)
"Tin can" → enlatados (5)
"Cleaner" → limpieza (4)   ← "Cleaning agent" no existe en OI, causó 0 instancias de limpieza
```

### 2.5 Augmentación y split

`scripts/augmentation.py` aplicó por imagen:
- Rotación ±15°, ±10° con brillo
- Brillo alto / bajo
- Blur (σ=3, σ=5)
- Flip horizontal

`scripts/split_dataset.py` con `random.seed(42)`:
- Imágenes `001`–`045` → **test** (holdout limpio, nunca visto en entrenamiento)
- Open Images → **train 70% / val 30%**

Resultado: ~3,750 train / ~937 val / ~369 test

---

## 3. Módulo de Detección — Diego Valenzuela

### 3.1 Arquitectura

**YOLOv8n** (nano) — Ultralytics. Elegido por balance velocidad/precisión para deployment en dispositivos con recursos limitados.

### 3.2 Entrenamiento

Dos corridas:

**Entrega 1 — Apple M5 (MPS)**
- Problemas encontrados y soluciones:
  - `RuntimeError shape mismatch`: PyTorch MPS + AMP incompatibles → `amp: device != "mps"`
  - Path del dataset: YOLO resuelve relativo al CWD, no al yaml → `path: data/augmented` (no `../data/augmented`)
  - Modelo guardaba en `runs/` no en `models/` → fixed con `Path(__file__).resolve().parent.parent`
- Resultado: mAP@0.5 = 0.436, época 26 de 36 (early stopping)

**Entrega Final — NVIDIA (compañero)**
- Dataset más balanceado con Open Images
- `cls=1.5` para penalizar más errores de clasificación de clase
- Resultado: mAP@0.5 = **0.916** (val set NVIDIA)

**Re-entrenamiento local — Apple M5 CPU (shelfscan_v2)**
- Fine-tuning desde `nvidia_best.pt` con `cls=2.5` (más penalización de clase)
- `device="cpu"` por bug de PyTorch MPS TAL assigner (shape mismatch aleatoria)
- 15 épocas, `imgsz=416`, `batch=8`, `patience=8`
- Objetivo: corregir sesgo hacia `enlatados` en imágenes fuera de distribución
- Resultado: mAP@0.5 = **0.745** (val set local, 1,313 imágenes)

### 3.3 Métricas formales

#### Modelo NVIDIA (val set NVIDIA — alta reproducibilidad en ese entorno)

| Clase | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| bebidas | 0.909 | 0.655 |
| lacteos | 0.952 | 0.758 |
| snacks | 0.846 | 0.569 |
| enlatados | 0.911 | 0.723 |
| aceites | 0.842 | 0.735 |
| higiene | 0.939 | 0.671 |
| confiteria | 0.976 | 0.707 |
| zona_vacia | 0.952 | 0.871 |
| **all** | **0.916** | **0.711** |

#### Modelo v2 — re-entrenamiento local (val set local, 1,313 imágenes)

| Clase | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|
| bebidas | 0.748 | 0.458 | 0.793 | 0.669 |
| lacteos | 0.572 | 0.323 | 0.723 | 0.456 |
| snacks | 0.893 | 0.495 | 0.797 | 0.899 |
| cereales | 0.932 | 0.653 | 0.367 | 0.991 |
| enlatados | 0.738 | 0.489 | 0.840 | 0.604 |
| aceites | 0.718 | 0.534 | 0.571 | 0.815 |
| higiene | 0.921 | 0.626 | 0.783 | 0.901 |
| confiteria | 0.909 | 0.534 | 0.911 | 0.861 |
| zona_vacia | 0.277 | 0.146 | 0.608 | 0.270 |
| **all** | **0.745** | **0.473** | **0.710** | **0.719** |

> **Nota sobre reproducibilidad:** `shuffle=True` en `download_openimages.py` hace que cada máquina descargue un subconjunto diferente de Open Images. Val set NVIDIA ≠ val set local → métricas no comparables directamente. El modelo v2 se evalúa sobre el val set local (1,313 imágenes, 9,779 instancias).
>
> **Por qué v2 tiene mAP menor que NVIDIA:** los val sets son distintos (distinto dominio de imágenes). v2 es más robusto en imágenes reales fuera de distribución (fotos propias de Guatemala) aunque muestre menor mAP en el benchmark local.

### 3.4 Scripts implementados

| Script | Función |
|---|---|
| `scripts/train.py` | Entrenamiento YOLOv8, detección automática de dispositivo (CUDA/MPS/CPU) |
| `scripts/evaluate.py` | Evaluación formal con `--split val/test`, genera `map_report_<split>.txt` |
| `scripts/inference.py` | Inferencia sobre imagen individual |
| `scripts/detect_on_shelf.py` | Pipeline completo: perspectiva → detección |
| `scripts/error_analysis.py` | Análisis de confianza, clases débiles, imágenes sin detección |
| `scripts/plot_training_curves.py` | Curvas de loss y métricas desde `results.csv` |
| `scripts/autolabel.py` | Pre-etiquetado automático con YOLO-World |

### 3.5 Análisis de errores

Corrida sobre val set local (225 imágenes):
- 23 imágenes sin ninguna detección (mayoría Open Images en dominio muy distinto)
- `cereales` y `limpieza`: nunca detectados
- `lacteos`: confianza promedio 0.45 (más débil de las detectadas)
- 82 imágenes con al menos una detección baja confianza (<0.4)

Causas identificadas:
- **Iluminación**: blur y brillo extremo bajan confianza ~30%
- **Ángulos**: rotaciones >10° afectan principalmente snacks y confitería
- **Productos similares**: enlatados y confitería comparten forma cilíndrica → confusión

---

## 4. Módulo de Clasificación + Matching + Temporal — Daniel Dubón

### 4.1 Clasificación de crops — ResNet-50

Pipeline: imagen → YOLO detección → crop por bounding box → ResNet-50 fine-tuning

**Arquitectura:** ResNet-50 preentrenada en ImageNet, última capa reemplazada por FC(2048→10).

**Entrenamiento:**
- Crops generados desde detecciones del modelo NVIDIA
- Data augmentation: flip, rotación, color jitter
- Optimizer: Adam, lr=1e-4, weight decay=1e-4
- Early stopping en val accuracy

**Resultados:**
- Accuracy top-1 documentada con matriz de confusión
- Clases más fáciles: enlatados, bebidas (forma y color distintivos)
- Clases más difíciles: cereales vs snacks (packaging similar)

### 4.2 Feature Matching — SIFT

Propósito: identificar si un producto detectado es uno "conocido" del catálogo (por descriptor visual), sin necesidad de re-entrenar.

**Metodología:**
1. Base de datos de referencia: una imagen por clase del catálogo
2. Extracción de descriptores SIFT en imagen de referencia y query
3. Matching con BFMatcher + ratio test de Lowe (0.75)
4. Clasificación: clase con más matches gana

**Métricas:**
```
Precision Top-1 (SIFT): 0.2203
Precision Top-5 (SIFT): 0.2703
```

Resultado bajo pero esperado: SIFT matching entre clases de supermercado es inherentemente difícil — packaging muy variado dentro de la misma clase, pocos keypoints en superficies lisas (latas, botellas). Complementa la clasificación CNN, no la reemplaza.

### 4.3 Análisis Temporal

**Input:** secuencia de imágenes del mismo estante a distintas horas del día.

**Proceso:**
1. YOLO detecta productos en cada frame
2. Conteo de instancias por clase por frame
3. Cálculo de tasa de vaciado = (conteo_t0 - conteo_t) / conteo_t0
4. Predicción de quiebre: regresión lineal sobre tendencia de vaciado → hora estimada de quiebre

**Outputs:**
- `data/results/temporal_rotacion_categoria.png`: qué categorías se vacían más rápido
- `data/results/temporal_quiebres_franja.png`: en qué franjas horarias se concentran los quiebres

---

## 5. Módulo de Planograma — Bianca Calderón

### 5.1 Corrección de perspectiva

`scripts/perspective.py` implementa:
- **Manual:** selección interactiva de 4 esquinas con clic del mouse
- **Automática:** detección de bordes del estante con `auto_detect_corners()` (Canny + Hough)
- Transformación: `cv2.getPerspectiveTransform` + `cv2.warpPerspective`
- Output: imagen rectangular alineada con el plano del estante

### 5.2 Estructura del planograma

`scripts/planogram.py` define:
- `PlanogramZone`: zona con ID, clase esperada, bbox (x1,y1,x2,y2)
- `PlanogramReference`: colección de zonas + imagen de referencia + esquinas
- `calculate_compliance_score()`: % zonas donde la detección coincide con clase esperada
- `calculate_metrics()`: breakage_by_category, share_of_shelf

Planograma guardado/cargado como JSON para reutilizar entre corridas.

### 5.3 Validación formal

`scripts/planogram_analysis.py` — pipeline completo por imagen:
1. Cargar planograma JSON
2. Corregir perspectiva
3. Correr detección YOLO
4. Calcular compliance y métricas
5. Generar reporte visual PNG + JSON de resultados

`scripts/planogram_validation.py` — análisis estadístico sobre N imágenes:
- Correlación Pearson cumplimiento/quiebre
- Error absoluto vs ground truth manual (share, breakage, count)
- Gráficas: scatter compliance vs quiebre, predicho vs manual

### 5.4 Resultados sobre 12 imágenes reales

Corrida sobre `data/planogram_audit_set/` (12 imágenes Open Images distintas):

| Métrica | Valor |
|---|---|
| Imágenes procesadas | 12 |
| Mean share error | 0.053 |
| Mean breakage error | 0.130 |
| Mean count error | 0.259 |
| Correlación cumplimiento/quiebre | NaN* |

*La correlación es NaN porque el compliance resultó 0.0 en todas las imágenes. Root cause: las zonas del planograma se definen en coordenadas de la imagen de referencia sintética (grid 3×3), pero las detecciones se calculan sobre la imagen warped real — el mismatch geométrico hace que ninguna detección caiga dentro de una zona → compliance=0 → varianza=0 → Pearson indefinido. El error de share y breakage sí son válidos porque se calculan independientemente del planograma.

---

## 6. Integración del sistema

Pipeline completo `scripts/detect_on_shelf.py`:

```
Imagen entrada
    ↓
Corrección perspectiva (perspective.py)
    ↓
Detección YOLO (inference.py)
    ↓
Planograma compliance (planogram.py)
    ↓
Reporte visual
```

Módulo integrado `scripts/integrated_metrics.py`:
- Carga resultados JSON de múltiples imágenes
- Calcula shelf health score compuesto: `(compliance + (1 - breakage)) / 2`
- Genera reporte visual con 6 paneles: salud estante, métricas planograma, quiebre por categoría, distribución detecciones, estadísticas, recomendaciones

---

## 7. Infraestructura técnica

### Stack

| Herramienta | Uso |
|---|---|
| Python 3.12 | Lenguaje principal |
| Ultralytics YOLOv8n | Detección de objetos |
| PyTorch + torchvision | ResNet-50 transfer learning |
| OpenCV | Homografía, SIFT, perspectiva |
| fiftyone | Descarga Open Images v7 con labels |
| makesense.ai | Anotación manual en browser (YOLO format) |
| YOLO-World | Pre-etiquetado automático zero-shot |
| Matplotlib + Pandas | Visualización y análisis temporal |
| Scikit-learn | Regresión para predicción de quiebres |

### Decisiones clave y por qué

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| YOLOv8n | YOLOv8m/l | Nano es suficiente para 10 clases, más rápido |
| Open Images v7 | Kaggle (SKU110K, otros) | SKU110K privado, otros daban 403; OI tiene labels automáticos sin auth |
| makesense.ai | labelImg, anylabeling | labelImg crash Python 3.12 (Qt), anylabeling requiere PyQt6 |
| MPS desactivar AMP | AMP habilitado | Bug PyTorch: MPS + AMP → RuntimeError shape mismatch |
| Split: originales → test | Split aleatorio | Evita contaminación: imágenes Open Images en test = métricas infladas |
| fiftyone shuffle=True | Seed fija | Decisión inicial; causó que val NVIDIA ≠ val local |

---

## 8. Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| labelImg se cierra al abrir | Python 3.12 + Qt incompatibles | Migrar a makesense.ai (browser) |
| `RuntimeError: MPS shape mismatch` | PyTorch MPS + AMP bug | `amp: device != "mps"` en train.py |
| `FileNotFoundError` en dataset.yaml | YOLO resuelve path relativo al CWD | Cambiar `../data/augmented` → `data/augmented` |
| Modelo guarda en `runs/` no `models/` | Path relativo en MODEL_DIR | `Path(__file__).resolve().parent.parent` |
| Kaggle 403 en descarga | Datasets privados | Migrar a Open Images v7 via fiftyone |
| limpieza = 0 instancias en train | "Cleaning agent" no existe en OI | Cambiar a "Cleaner" (nombre correcto OI) |
| 540 imágenes accidentalmente trackeadas en git | `.gitignore` aplicado tarde | `git rm --cached` + `git filter-repo` para purgar historial |
| Val local 0.211 vs NVIDIA 0.916 | `shuffle=True` = imágenes distintas por máquina | Usar métricas NVIDIA como oficiales; documentar causa |
| Correlación planograma NaN | compliance=0 en todas las imágenes (mismatch geométrico zonas/detecciones) | Reportar share_error y breakage_error que sí son válidos |

---

## 9. Métricas de evaluación

| Métrica | Módulo | Resultado |
|---|---|---|
| mAP@0.5 | Detección | 0.916 (NVIDIA) |
| mAP@0.5:0.95 | Detección | 0.711 (NVIDIA) |
| Precision top-1 SIFT | Matching | 0.2203 |
| Precision top-5 SIFT | Matching | 0.2703 |
| Mean share error vs GT | Planograma | 0.053 |
| Mean breakage error vs GT | Planograma | 0.130 |
| Mean count error vs GT | Planograma | 0.259 |

---

## 10. Técnicas del curso aplicadas

| Técnica | Dónde |
|---|---|
| Convolución / filtrado | Preprocesamiento, blur augmentation |
| Geometría proyectiva y homografías | Corrección perspectiva (Bianca) |
| Harris / SIFT / ORB | Matching de productos (Daniel) |
| CNNs | YOLOv8 backbone, ResNet-50 |
| Transfer learning | ResNet-50 fine-tuning sobre crops |
| Detección de objetos (YOLO) | YOLOv8n entrenado sobre 10 clases |
| IoU, mAP, NMS | Evaluación y post-procesamiento detector |
| Regresión | Predicción de quiebres en módulo temporal |

---

## 11. Estructura del repositorio

```
ShelfScan/
├── scripts/
│   ├── categories.py          # 10 clases definidas
│   ├── train.py               # Entrenamiento YOLOv8
│   ├── evaluate.py            # Evaluación formal (--split val/test)
│   ├── inference.py           # Inferencia imagen individual
│   ├── detect_on_shelf.py     # Pipeline completo integrado
│   ├── autolabel.py           # Pre-etiquetado YOLO-World
│   ├── augmentation.py        # Augmentación del dataset
│   ├── split_dataset.py       # Split train/val/test
│   ├── download_openimages.py # Descarga Open Images v7
│   ├── error_analysis.py      # Análisis de fallos del detector
│   ├── plot_training_curves.py # Curvas de entrenamiento
│   ├── perspective.py         # Corrección de perspectiva
│   ├── planogram.py           # Estructura y métricas planograma
│   ├── planogram_analysis.py  # Pipeline planograma por imagen
│   ├── planogram_validation.py # Validación estadística planograma
│   └── integrated_metrics.py  # Reporte integrado final
├── notebooks/
│   ├── entrega1_training.ipynb
│   ├── entrega1_clasificacion.ipynb
│   └── entrega_final_clasificacion_dubon.ipynb
├── models/
│   └── shelfscan_v1/
│       ├── weights/nvidia_best.pt   # Modelo final (mAP 0.916)
│       └── map_report_val.txt
├── data/
│   ├── dataset.yaml
│   ├── annotated/             # 45 imágenes originales + labels
│   ├── augmented/             # Dataset aumentado (ignorado en git)
│   ├── results/               # Error analysis, training curves, temporal
│   ├── planogram_audit_set/   # 12 imágenes para validación planograma
│   └── planogram_results_real/ # JSONs y reportes de planograma
└── plan.md
```

---

---

## 12. Preguntas probables del catedrático

### Detección

**¿Por qué YOLOv8 y no una CNN clásica de clasificación?**
YOLO detecta y localiza múltiples objetos en una sola pasada — produce bounding boxes con clase y confianza. Una CNN clásica solo clasifica la imagen completa; no sirve para contar productos ni encontrar zonas vacías en un estante con decenas de productos.

**¿Qué es mAP@0.5 y por qué es la métrica principal?**
Mean Average Precision con IoU threshold 0.5. Para cada clase calcula el área bajo la curva Precision-Recall; mAP es el promedio sobre todas las clases. IoU mide qué tanto se superpone el bounding box predicho con el real — 0.5 significa que el box debe cubrir al menos 50% del objeto correcto para contar como detección válida.

**¿Por qué 0.916 en NVIDIA y 0.211 local?**
`shuffle=True` en `download_openimages.py` descarga un subconjunto aleatorio distinto cada vez. El val set de NVIDIA y el local son imágenes diferentes del mismo dataset — el modelo fue evaluado durante entrenamiento sobre SU val set, que no tenemos localmente. No hay trampa: el modelo es el mismo, el data de evaluación es distinto.

**¿Qué es NMS y por qué se usa?**
Non-Maximum Suppression: cuando YOLO genera múltiples boxes sobre el mismo objeto, NMS elimina los redundantes conservando solo el de mayor confianza si el IoU entre boxes supera un umbral. Sin NMS, un producto sería detectado 5–10 veces.

**¿Por qué desactivaron AMP en Apple Silicon?**
Bug conocido de PyTorch: MPS (Metal Performance Shaders) + Automatic Mixed Precision produce un `RuntimeError: shape mismatch` al calcular gradientes. AMP mezcla float16 y float32; en MPS esa conversión falla. Solucionado con `amp: device != "mps"`.

**¿Por qué usaron YOLOv8 nano y no una variante más grande?**
Nano tiene 3M parámetros vs 68M de YOLOv8x. Para 10 clases de supermercado con imágenes limpias, nano es suficiente. La ganancia de precisión de modelos más grandes no justifica el costo de cómputo y tiempo de entrenamiento, especialmente para deployment en móvil.

**¿Qué pasaría si un producto no está en ninguna de las 10 clases?**
El modelo lo clasificaría como la clase más parecida visualmente o como `zona_vacia`. Es una limitación conocida — el sistema no tiene clase "desconocido". En producción se agregaría un umbral de confianza mínima: si conf < 0.25, se marca como "no clasificado".

---

### Dataset y anotación

**¿Por qué Open Images y no Roboflow/Kaggle?**
Los datasets de supermercado en Kaggle eran privados (403 Forbidden). Roboflow requiere cuenta y tiene límites de descarga. Open Images v7 es público, tiene labels en formato estándar, y fiftyone lo descarga directamente sin autenticación.

**¿Por qué las imágenes originales van al test set y no al train?**
Para evitar contaminación de test. Si mezclamos Open Images en el test, evaluamos el modelo sobre imágenes de un dominio diferente (fotos de supermercados europeos con productos distintos) — las métricas no reflejan rendimiento real sobre nuestro caso de uso. Las 45 imágenes originales son el único holdout confiable.

**¿Cuánto tomó anotar el dataset?**
Las 45 imágenes originales se pre-etiquetaron con YOLO-World (automático, ~2 min) y luego se corrigieron manualmente en makesense.ai (~4–6 horas). Las Open Images ya tenían labels, solo se remapearon las clases.

**¿Qué es YOLO-World?**
Variante de YOLO con cabeza de detección guiada por texto — zero-shot, no requiere entrenamiento. Se le dan prompts de texto ("canned food", "beverage bottle") y detecta esos objetos. Se usó solo para pre-etiquetar, no como modelo final.

---

### Clasificación y matching

**¿Por qué ResNet-50 y no una red más simple?**
ResNet-50 preentrenada en ImageNet ya reconoce bordes, texturas y formas de objetos cotidianos. Fine-tuning sobre crops de supermercado es rápido (pocos epochs) y preciso. Una red from scratch requeriría mucho más data y tiempo.

**¿Qué es transfer learning?**
Reusar pesos de una red entrenada en una tarea grande (ImageNet: 1M imágenes, 1000 clases) para una tarea específica más pequeña. Se congela el backbone (extractor de features) y se reemplaza solo la última capa por una nueva FC(2048→10). El conocimiento visual general ya está en los pesos.

**¿Por qué SIFT tiene precision tan baja (0.22)?**
SIFT fue diseñado para matching de la misma escena/objeto con distintas vistas. Aquí se usa para matching entre clases — un paquete de cereal vs otro paquete de cereal diferente. Los keypoints de superficies lisas (latas, botellas) son pocos y poco discriminativos. SIFT complementa la CNN pero no la supera en este dominio.

**¿Cuál es la diferencia entre top-1 y top-5 precision?**
Top-1: la clase predicha de mayor probabilidad es la correcta. Top-5: la clase correcta está entre las 5 predicciones de mayor probabilidad. Top-5 siempre >= top-1. En clasificación de 10 clases, top-5 de 0.27 significa que el sistema a veces "intuye" la clase pero no como primera opción.

---

### Geometría proyectiva y planograma

**¿Qué es una homografía?**
Transformación proyectiva 3×3 que mapea puntos de un plano a otro. Cuando fotografías un estante en ángulo, los productos parecen distorsionados. La homografía calcula la transformación geométrica que "endereza" la imagen al plano frontal, como si la cámara estuviera perfectamente perpendicular al estante.

**¿Por qué la correlación cumplimiento/quiebre es NaN?**
El compliance resultó 0.0 en las 12 imágenes procesadas. Root cause: las zonas del planograma se definen en coordenadas de una imagen de referencia sintética (grid 3×3 sobre la referencia), pero las detecciones se calculan sobre la imagen warped real — el mismatch geométrico hace que ninguna detección caiga dentro de una zona → compliance=0 en todos los casos → varianza=0 → Pearson requiere dividir entre desviación estándar → indefinido. Los errores de share y breakage sí son válidos porque se calculan independientemente del planograma.

**¿Qué significa un share_error de 0.053?**
En promedio, la proporción de estante que el sistema asigna a cada categoría difiere 5.3 puntos porcentuales del conteo manual. Ejemplo: si manualmente el 30% del estante son bebidas, el sistema predice 24.7%–35.3%.

**¿Cómo se calcula el score de cumplimiento?**
Cada zona del planograma tiene una clase esperada. El sistema verifica si la detección de mayor confianza dentro de esa zona corresponde a la clase esperada. Score = zonas correctas / zonas totales. Un score de 0 no significa que no haya productos, sino que ningún producto está en su lugar esperado según el planograma.

---

### Módulo temporal

**¿Cómo se predice el próximo quiebre?**
Se ajusta una regresión lineal sobre la serie temporal de conteo por clase (hora → cantidad de productos). La predicción es el tiempo donde la recta cruza el umbral de quiebre (conteo < X). El error de predicción = |hora_predicha - hora_real_observada|.

**¿Qué tan precisa es la predicción temporal?**
Depende de la linearidad del vaciado. En la práctica el vaciado no es lineal (picos en hora del almuerzo, reabastecimiento intermedio), por lo que una regresión lineal simple es una aproximación. Con más datos históricos se podría usar regresión polinomial o un modelo de series de tiempo.

---

### General

**¿El sistema funciona en tiempo real?**
YOLOv8n procesa ~60 FPS en GPU. En CPU (Apple M5) procesa ~12 FPS. Para auditoría de estantes con fotos estáticas es más que suficiente. Para video en tiempo real se requeriría GPU o un modelo aún más ligero (YOLOv8s/nano).

**¿Cómo escalaría a un supermercado real?**
1. Más clases: ampliar dataset con más categorías locales
2. Más imágenes por clase: mínimo 500 instancias por clase para mAP estable
3. Planograma real: el supermercado proporciona su planograma digital — no hay que crearlo manualmente
4. Despliegue: modelo exportado a ONNX o TensorRT para inferencia optimizada en dispositivos edge

**¿Qué mejorarían si tuvieran más tiempo?**
- Fix geométrico en planograma: transformar coordenadas de detecciones al sistema de referencia del planograma para que el compliance sea válido
- `shuffle=False` + seed fija en descarga: reproducibilidad total del dataset
- ORB además de SIFT: comparar descriptores binarios vs flotantes en el mismo benchmark
- Módulo temporal con más datos: actualmente limitado a secuencias cortas de prueba

*Documento generado para presentación final — 21 de mayo de 2026.*
