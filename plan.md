# Plan de Trabajo — ShelfScan
### Sistema de Auditoría Visual de Lineales de Supermercado
**Curso:** Visión por Computadora  
**Grupo:** Diego Valenzuela · Daniel Dubón · Bianca Calderón  
**Catedrático:** Alberto Suriano  
**Versión:** 2.2 — Entrega 2 en progreso

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

## Entrega 2 — 7 de mayo de 2026

**Objetivo:** Pipeline completo de extremo a extremo funcionando. Los tres módulos integrados, medibles y con primeros resultados cuantitativos.

---

### Diego Valenzuela — Detección ✓ COMPLETO (Entrega 2)

- ✓ Modelo YOLOv8 re-entrenado con Open Images (`best.pt` — NVIDIA)
- ✓ Script de inferencia: `inference.py`
- ✓ Pipeline perspectiva → detección: `detect_on_shelf.py`
- ✓ Eval formal en test set: `evaluate.py` corrido, `map_report_test.txt` generado
- ✓ NMS activo en YOLOv8, conf=0.25

**Métricas test set (Entrega 2):**
| Clase | mAP@0.5 |
|---|---|
| enlatados | 0.454 ✓ |
| confiteria | 0.319 ✓ |
| aceites | 0.182 ⚠ |
| higiene | 0.147 ⚠ |
| limpieza | 0.132 ⚠ |
| bebidas | 0.052 ✗ |
| zona_vacia | 0.006 ✗ |
| snacks | 0.003 ✗ |
| lacteos | 0.000 ✗ |
| cereales | 0.000 ✗ |
| **all** | **0.173** |

> Nota: test set ahora incluye Open Images (dominado por enlatados — 58% instancias). Comparación directa con Entrega 1 (0.436) no válida por distribución distinta. Mejora real se verá en Entrega Final con test set homogéneo.

**Entregables:**
- ✓ Modelo .pt
- ✓ `inference.py`
- ✓ `map_report_test.txt`

---

### Daniel Dubón — Clasificación + Análisis Temporal 🔴 INICIO INMEDIATO

**Prerequisito listo:** `best.pt` disponible, crops generables desde test set.

**Paso 1 — Generar crops (10 min):**
```bash
python -c "
from ultralytics import YOLO
import cv2, os, sys
sys.path.insert(0,'scripts')
model = YOLO('models/shelfscan_v1/weights/best.pt')
results = model.predict(source='data/augmented/images/test', conf=0.4, verbose=False)
for i,r in enumerate(results):
    for j,box in enumerate(r.boxes):
        cls=int(box.cls); name=r.names[cls]
        os.makedirs(f'data/classifier_dataset/{name}', exist_ok=True)
        x1,y1,x2,y2=map(int,box.xyxy[0])
        crop=r.orig_img[y1:y2,x1:x2]
        if crop.size>0: cv2.imwrite(f'data/classifier_dataset/{name}/crop_{i}_{j}.jpg',crop)
print('Done')
"
```

**Paso 2 — Fine-tuning ResNet-50:** usar notebook `entrega1_training.ipynb` cells 15–18.

**Paso 3 — Módulo temporal:** usar 3+ imágenes del mismo estante a distintas horas, correr `inference.py` sobre cada una, contar productos por clase, graficar serie temporal.

**Limitación conocida:** crops dominados por `enlatados` y `confiteria` (clases que YOLO detecta bien). ResNet aprenderá principalmente esas. Documentar como limitación del modelo actual.

- ⬜ Crops generados en `data/classifier_dataset/`
- ⬜ ResNet fine-tuning corrido
- ⬜ Accuracy top-1 + matriz de confusión
- ⬜ Módulo temporal v1: secuencia 3+ imágenes → gráfica de ocupación

**Entregables:**
- Pipeline integrado: imagen → detección YOLO → crop → clasificación ResNet
- Evaluación clasificador: accuracy top-1, matriz de confusión
- Módulo temporal: gráfica de ocupación por categoría vs. tiempo

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
| makesense.ai | Anotación del dataset (browser, formato YOLO) |
| YOLO-World | Pre-etiquetado automático zero-shot para acelerar anotación |
| SKU110K | Dataset público complementario |
| Pandas + Matplotlib | Análisis temporal y visualización |
| Scikit-learn | Regresión para predicción de quiebres |
| PIL / Matplotlib | Generación de reportes visuales |
| Google Colab / GPU local | Entrenamiento |

---

*Plan versión 2.2 — Actualizado el 6 de mayo de 2026.*