# ShelfScan — Módulo de Detección (Diego Valenzuela)

Detección de productos y zonas vacías en estantes de supermercado usando YOLOv8.

## Categorías (10 clases)

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | bebidas | agua, refresco, jugos |
| 1 | lacteos | leche, yogurt, queso |
| 2 | snacks | papas, galletas, barras |
| 3 | cereales | cereales de caja, avena |
| 4 | limpieza | detergente, jabón, desinfectante |
| 5 | enlatados | atún, frijoles, sardinas |
| 6 | aceites | aceite, vinagre, salsas |
| 7 | higiene | shampoo, pasta dental, desodorante |
| 8 | confiteria | dulces, chocolate, chicle |
| 9 | zona_vacia | espacio vacío en el estante |

## Entorno

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. CUDA opcional (acelera entrenamiento ~10x).

Verificar GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Estructura

```
ShelfScan/
├── data/
│   ├── raw/              # fotos originales sin anotar
│   ├── annotated/        # imágenes + labels YOLO (.txt) de Roboflow
│   │   ├── images/
│   │   └── labels/
│   ├── augmented/        # salida de augmentation.py
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── dataset.yaml      # config para YOLOv8
├── models/
│   └── shelfscan_v1/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png
│       └── map_report.txt
├── scripts/
│   ├── categories.py     # definición de clases
│   ├── augmentation.py   # augmentation pipeline
│   ├── split_dataset.py  # train/val/test split (70/20/10)
│   ├── train.py          # entrenamiento YOLOv8
│   └── inference.py      # inferencia: imagen → bounding boxes
├── notebooks/
│   └── entrega1_training.ipynb
└── requirements.txt
```

## Pipeline completo

### 1. Anotar dataset en Roboflow
- Subir fotos de `data/raw/` a Roboflow
- Anotar con las 10 clases definidas en `categories.py`
- Exportar en formato YOLOv8 → guardar en `data/annotated/`

### 2. Augmentation
```bash
python scripts/augmentation.py   # genera data/augmented/
python scripts/split_dataset.py  # divide en train/val/test
```

Cada imagen genera 4 variantes: rotación ±15°, brillo alto/bajo, blur, flip horizontal.

### 3. Entrenamiento
```bash
python scripts/train.py
# O en Colab: abrir notebooks/entrega1_training.ipynb
```

Parámetros preliminares (Entrega 1):
- Modelo base: `yolov8n.pt` (nano)
- Epochs: 50 con early stopping (patience=10)
- Imagen: 640×640
- Batch: 16

### 4. Inferencia
```bash
python scripts/inference.py ruta/imagen.jpg
# Output: JSON con class_name, confidence, bbox
```

### 5. Corrección de perspectiva y planograma
```bash
python scripts/perspective.py -i ruta/imagen.jpg
```
- Selecciona manualmente 4 esquinas del estante: superior izquierda, superior derecha, inferior derecha, inferior izquierda.
- El script aplica `cv2.getPerspectiveTransform` y `cv2.warpPerspective` para generar una vista frontal del estante.
- Si prefieres pasar los puntos desde línea de comandos:
  ```bash
  python scripts/perspective.py -i ruta/imagen.jpg -p 100,120 520,130 540,400 90,390
  ```

```bash
python scripts/perspective.py -d data/raw -o data/warped
```
- Procesa un directorio completo de imágenes JPG/JPEG/PNG y guarda las versiones corregidas.

#### ¿Qué es el planograma de referencia?
El planograma de referencia es la imagen del estante tomada en condiciones controladas que representa la distribución ideal de productos. Esta imagen sirve como plantilla para alinear y comparar una imagen real del estante, evaluando el cumplimiento espacial y la presencia de artículos esperados.

## Métricas objetivo

| Métrica | Entrega 1 (preliminar) | Entrega Final |
|---------|------------------------|---------------|
| mAP@0.5 | > 0.30 | > 0.65 |
| mAP@0.5:0.95 | — | > 0.45 |
| Precision | — | > 0.70 |
| Recall | — | > 0.65 |

## Dataset

- **Propio:** 150–200 fotos en tiendas locales (distintos ángulos y luz)
- **Complemento:** SKU110K (filtrado a categorías relevantes)
- **Anotado en:** Roboflow (mínimo 100 imágenes para Entrega 1)
- **Aumentado:** ~5× el tamaño original
